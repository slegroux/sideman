# AbletonLOM - generic Live Object Model engine.
#
# Hot-reloadable: edit this file, send {"op":"reload"}, no Live restart needed.
# Everything here runs ON LIVE'S MAIN THREAD (see __init__.py). Never block.
#
# Conventions verified by the Phase 0 probe against Live 12.2.7 / Python 3.11.6:
#   * get_available_properties_for_type(type_, (3, 0)) -> [MFLProperty, ...]
#   * MFLProperty = (name, format, to_json, from_json, min_epii_version, hidden)
#   * that list mixes PROPERTIES AND FUNCTIONS - classify at runtime, not by name
#   * some members raise RuntimeError per *instance*, not per type
#     (crossfader/cue_volume/song_tempo: main track only;
#      input_meter_*/output_meter_*: not on MIDI tracks;
#      fold_state: only on groupable tracks)

from __future__ import absolute_import, print_function

import traceback

import Live

EPII_VERSION = (3, 0)

# Functions that destroy user work. Require params["confirm"] is True.
WRITE_GUARD = frozenset([
    "delete_track", "delete_return_track", "delete_scene", "delete_device",
    "delete_clip", "delete_cue_point", "remove_all_notes", "crop",
])

_MXD = {"types": None, "utils": None, "error": None}


def _mxd():
    """Lazy, cached, non-fatal import of Ableton's private M4L LOM modules."""
    if _MXD["types"] is None and _MXD["error"] is None:
        try:
            from _MxDCore import LomTypes, LomUtils
            _MXD["types"] = LomTypes
            _MXD["utils"] = LomUtils
        except Exception:
            _MXD["error"] = traceback.format_exc()
    return _MXD


# --------------------------------------------------------------------- roots

def _roots(surface):
    app = Live.Application.get_application()
    return {
        "live_set": surface.song(),
        "live_app": app,
        "app_view": app.view,
        "this_device": None,
    }


def resolve(surface, path):
    """'live_set tracks 0 mixer_device volume' -> the live object.

    Vendored resolver (not _MxDCore's) so a Live update cannot hard-break us.
    """
    if not path or not str(path).strip():
        raise ValueError("empty path")
    comps = str(path).split()
    roots = _roots(surface)
    head = comps[0]
    if head not in roots:
        raise ValueError("unknown root %r; expected one of %s"
                         % (head, sorted(roots)))
    cur = roots[head]
    walked = [head]
    for comp in comps[1:]:
        if cur is None:
            raise ValueError("path %r resolved to None at %r"
                             % (path, " ".join(walked)))
        try:
            if comp.lstrip("-").isdigit():
                cur = cur[int(comp)]
            else:
                cur = getattr(cur, comp)
        except IndexError:
            raise IndexError("index %s out of range at %r"
                             % (comp, " ".join(walked)))
        except AttributeError:
            raise AttributeError("no member %r at %r" % (comp, " ".join(walked)))
        walked.append(comp)
    return cur


# ------------------------------------------------------------- json marshalling

def _type_name(obj):
    t = type(obj)
    mod = getattr(t, "__module__", "") or ""
    return ("%s.%s" % (mod, t.__name__)) if mod else t.__name__


def _is_lom_object(v):
    m = _mxd()
    if m["types"] is not None:
        try:
            return bool(m["types"].is_lom_object(v))
        except Exception:
            pass
    return hasattr(v, "_live_ptr")


def _is_vector(v):
    return (hasattr(v, "__len__") and hasattr(v, "__getitem__")
            and not isinstance(v, (str, bytes, dict)))


def jsonify(v, depth=0):
    """Convert a Live value into something JSON-serialisable, without recursing
    into the whole object graph (a Song reaches everything)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if _is_vector(v):
        try:
            n = len(v)
        except Exception:
            n = None
        if depth == 0 and n is not None and n <= 64:
            try:
                return [jsonify(x, depth + 1) for x in v]
            except Exception:
                pass
        return {"__vector__": True, "count": n}
    if _is_lom_object(v):
        out = {"__lom__": _type_name(v)}
        name = getattr(v, "name", None)
        if isinstance(name, str):
            out["name"] = name
        return out
    # Enums and anything else Live hands back.
    return {"__repr__": str(v), "type": _type_name(v)}


def _coerce(obj, prop, value):
    """Use Ableton's own from_json for this property when it ships one."""
    info = _mfl_index(type(obj)).get(prop)
    if info is not None and getattr(info, "from_json", None):
        try:
            return info.from_json(value)
        except Exception:
            pass
    return value


# ------------------------------------------------------------- member listing

_MFL_CACHE = {}


def _mfl_index(type_):
    """{name: MFLProperty} for a LOM type, from Ableton's own registry."""
    key = type_
    if key in _MFL_CACHE:
        return _MFL_CACHE[key]
    out = {}
    m = _mxd()
    if m["types"] is not None:
        try:
            for p in m["types"].get_available_properties_for_type(type_, EPII_VERSION):
                out[p.name] = p
        except Exception:
            out = {}
    _MFL_CACHE[key] = out
    return out


def _member_names(obj):
    """Union of Ableton's M4L registry and dir().

    These are NOT the same set. The M4L registry is what Max for Live chose to
    expose; the raw Python API can carry more (and occasionally different)
    members. Preferring one would silently cap coverage, which is the exact
    failure mode this project exists to fix - so we merge and report the split.
    """
    mxd = set(_mfl_index(type(obj)))
    raw = set(n for n in dir(obj) if not n.startswith("_"))
    both = mxd | raw
    meta = {
        "mxd_count": len(mxd),
        "dir_count": len(raw),
        "union_count": len(both),
        "mxd_only": sorted(mxd - raw),
        "dir_only": sorted(raw - mxd),
    }
    return sorted(both), ("union" if mxd else "dir"), meta


def describe(surface, path, include_values=True):
    """getinfo, but honest about per-INSTANCE availability."""
    obj = resolve(surface, path)
    names, source, meta = _member_names(obj)
    idx = _mfl_index(type(obj))

    out = {
        "path": path,
        "type": _type_name(obj),
        "source": source,
        "sources": meta,
        "properties": {},
        "children": {},
        "functions": [],
        "unavailable": {},
    }

    for name in names:
        info = idx.get(name)
        if info is not None and getattr(info, "hidden", False):
            continue
        try:
            val = getattr(obj, name)
        except Exception as e:
            # This is the Phase 0 caveat: availability is per-instance.
            out["unavailable"][name] = "%s: %s" % (type(e).__name__, e)
            continue
        if callable(val):
            out["functions"].append(name)
        elif _is_vector(val):
            try:
                out["children"][name] = len(val)
            except Exception:
                out["children"][name] = None
        else:
            entry = {"type": _type_name(val)}
            if include_values:
                entry["value"] = jsonify(val, depth=1)
            out["properties"][name] = entry

    out["counts"] = {
        "properties": len(out["properties"]),
        "children": len(out["children"]),
        "functions": len(out["functions"]),
        "unavailable": len(out["unavailable"]),
    }
    return out


# --------------------------------------------------------------------- verbs

def _undo(surface, label):
    """REENTRANT undo step.

    Every mutating op opens one of these. A transaction wraps N such ops, so
    without a depth counter we would emit N nested begin/end pairs inside the
    outer pair - which is how you get a corrupted or unusable undo history.
    Only the OUTERMOST scope actually begins and ends the step, so a
    transaction collapses to exactly one Cmd-Z.

    The depth lives on the surface instance, not module state, so it survives
    hot-reload (same reasoning as the observer registry).
    """
    song = surface.song()

    class _Step(object):
        def __enter__(self):
            depth = getattr(surface, "_lom_undo_depth", 0)
            if depth == 0:
                try:
                    song.begin_undo_step()
                    surface._lom_undo_begins = getattr(
                        surface, "_lom_undo_begins", 0) + 1
                except Exception as e:
                    surface._lom_undo_err = str(e)
            surface._lom_undo_depth = depth + 1
            return self

        def __exit__(self, *exc):
            depth = max(getattr(surface, "_lom_undo_depth", 1) - 1, 0)
            surface._lom_undo_depth = depth
            if depth == 0:
                try:
                    song.end_undo_step()
                    surface._lom_undo_ends = getattr(
                        surface, "_lom_undo_ends", 0) + 1
                except Exception as e:
                    surface._lom_undo_err = str(e)
            return False

    return _Step()


# ------------------------------------------------- PHASE 5: batch + transaction

def op_get_batch(surface, params):
    """Read many properties in ONE round trip.

    specs: [{"path": ..., "property": ...}, ...]
    Never fails as a whole - each entry carries its own ok/error, because one
    unavailable property (see the per-instance caveat) should not lose the
    other 40 reads.
    """
    out = []
    for spec in params["specs"]:
        try:
            out.append(dict(op_get(surface, spec), ok=True))
        except Exception as e:
            out.append({"ok": False, "path": spec.get("path"),
                        "property": spec.get("property"),
                        "error": "%s: %s" % (type(e).__name__, e)})
    return {"count": len(out), "ok_count": sum(1 for r in out if r["ok"]),
            "results": out}


def op_set_batch(surface, params):
    """Write many properties inside ONE undo step."""
    return _run_ops(surface, [dict(s, op="set") for s in params["specs"]],
                    params.get("stop_on_error", True),
                    params.get("rollback_on_error", False))


def op_transaction(surface, params):
    """Run a list of ops inside ONE undo step, so the whole batch is a single
    Cmd-Z for the user rather than N of them.

    ops: [{"op": "set", "path": ..., "property": ..., "value": ...},
          {"op": "call", "path": ..., "function": ..., "args": [...]}, ...]

    CAVEAT, measured not assumed: grouping does not cover automatable
    parameters. tempo, mixer volume, mute and device parameters each form
    their own undo step in Live even inside an explicit step. Track name,
    color and time signature do group. See UNDO_GROUPING_NOTE.

    NOT a database transaction. Live has no intra-step rollback, so if op 5
    fails, ops 1-4 have already applied. Two honest options:
      stop_on_error   (default true)  - halt at the first failure
      rollback_on_error (default FALSE) - additionally call song.undo() once

    rollback defaults OFF deliberately: it is a MUTATION on an error path. If
    our step captured nothing, undo() would revert whatever the user did
    before, which is worse than the partial application it is trying to fix.
    Either way a single Cmd-Z reverts the batch, which is stated in the result.
    """
    return _run_ops(surface, params["ops"],
                    params.get("stop_on_error", True),
                    params.get("rollback_on_error", False))


def _run_ops(surface, ops, stop_on_error, rollback_on_error):
    results = []
    failed = False
    with _undo(surface, "transaction"):
        for i, spec in enumerate(ops):
            name = spec.get("op", "set")
            fn = OPS.get(name)
            if fn is None:
                results.append({"index": i, "op": name, "ok": False,
                                "error": "unknown op %r" % name})
                failed = True
                if stop_on_error:
                    break
                continue
            try:
                results.append({"index": i, "op": name, "ok": True,
                                "result": fn(surface, spec)})
            except Exception as e:
                results.append({"index": i, "op": name, "ok": False,
                                "error": "%s: %s" % (type(e).__name__, e)})
                failed = True
                if stop_on_error:
                    break

    applied = sum(1 for r in results if r["ok"])
    out = {"count": len(results), "applied": applied, "failed": failed,
           "results": results,
           "_debug": {"begins": getattr(surface, "_lom_undo_begins", 0),
                      "ends": getattr(surface, "_lom_undo_ends", 0),
                      "depth_after": getattr(surface, "_lom_undo_depth", None),
                      "err": getattr(surface, "_lom_undo_err", None)},
           "undo": UNDO_GROUPING_NOTE}

    if failed and rollback_on_error and applied:
        try:
            surface.song().undo()
            out["rolled_back"] = True
        except Exception as e:
            out["rolled_back"] = False
            out["rollback_error"] = str(e)
    elif failed and applied:
        out["rolled_back"] = False
        out["warning"] = ("%d op(s) applied before the failure; they remain "
                          "applied. One Cmd-Z reverts them." % applied)
    return out


def op_get(surface, params):
    obj = resolve(surface, params["path"])
    prop = params["property"]
    try:
        val = getattr(obj, prop)
    except Exception as e:
        raise RuntimeError("%r unavailable on this %s: %s"
                           % (prop, _type_name(obj), e))
    if callable(val):
        raise TypeError("%r is a function; use op 'call'" % prop)
    return {"path": params["path"], "property": prop,
            "value": jsonify(val), "type": _type_name(val)}


def op_set(surface, params):
    path, prop = params["path"], params["property"]
    obj = resolve(surface, path)
    if not hasattr(obj, prop):
        raise AttributeError("%s has no member %r" % (_type_name(obj), prop))
    if callable(getattr(obj, prop)):
        raise TypeError("%r is a function; use op 'call'" % prop)
    value = _coerce(obj, prop, params["value"])
    with _undo(surface, "set %s" % prop):
        setattr(obj, prop, value)
    return {"path": path, "property": prop, "value": jsonify(getattr(obj, prop))}


def op_call(surface, params):
    path, fn = params["path"], params["function"]
    args = params.get("args") or []
    obj = resolve(surface, path)
    target = getattr(obj, fn, None)
    if target is None:
        raise AttributeError("%s has no member %r" % (_type_name(obj), fn))
    if not callable(target):
        raise TypeError("%r is a property; use op 'get'/'set'" % fn)
    if fn in WRITE_GUARD and not params.get("confirm"):
        raise PermissionError(
            "%r is destructive; re-send with confirm=true to proceed" % fn)
    with _undo(surface, "call %s" % fn):
        result = target(*args)
    return {"path": path, "function": fn, "result": jsonify(result)}


def op_count(surface, params):
    obj = resolve(surface, params["path"])
    child = params["child"]
    val = getattr(obj, child)
    if not _is_vector(val):
        raise TypeError("%r is not a list" % child)
    return {"path": params["path"], "child": child, "count": len(val)}


def _is_listener_plumbing(name):
    """add_x_listener / remove_x_listener / x_has_listener.

    Three of these exist per observable property. Counting them as distinct
    "coverage" would inflate the headline number, so they are reported
    separately rather than folded into the substantive total.
    """
    return (name.endswith("_has_listener")
            or (name.startswith("add_") and name.endswith("_listener"))
            or (name.startswith("remove_") and name.endswith("_listener")))


def op_types(surface, params):
    """Full LOM type census - the coverage-harness baseline.

    Uses the UNION of Ableton's M4L registry and dir() on the class. These
    differ substantially: for Clip the registry lists 69 members while dir()
    finds 190, and everything the registry omits (the extended-notes API, the
    automation-envelope API, the whole listener API) is real and callable.
    """
    m = _mxd()
    if m["types"] is None:
        raise RuntimeError("_MxDCore unavailable: %s" % m["error"])

    # get_available_lom_types() registers 43 types, but the reachable object
    # graph is larger: Browser/BrowserItem are navigable (live_app browser ...)
    # yet unregistered. Probe known-reachable paths and fold their types in,
    # otherwise the census understates what the server can actually address.
    extra = []
    for probe in ("live_app browser", "live_app browser instruments",
                  "live_set view", "live_set tracks 0 view",
                  "live_set tracks 0 clip_slots 0",
                  "live_set tracks 0 input_routing_type",
                  "live_set tracks 0 input_routing_channel",
                  "live_set tracks 0 output_routing_type",
                  "live_set tracks 0 output_routing_channel"):
        try:
            extra.append(type(resolve(surface, probe)))
        except Exception:
            pass

    types, totals = {}, {"mxd": 0, "union": 0, "substantive": 0, "listeners": 0}
    registered = list(m["types"].get_available_lom_types())
    seen_ids = set(id(t) for t in registered)
    unregistered = [t for t in extra if id(t) not in seen_ids
                    and not seen_ids.add(id(t))]
    totals["registered_types"] = len(registered)
    totals["unregistered_types"] = len(unregistered)

    for t in registered + unregistered:
        name = _type_name_of_class(t)
        try:
            mxd_names = set(p.name for p in
                            m["types"].get_available_properties_for_type(t, EPII_VERSION))
        except Exception:
            mxd_names = set()
        raw_names = set(n for n in dir(t) if not n.startswith("_"))
        union = mxd_names | raw_names
        listeners = set(n for n in union if _is_listener_plumbing(n))
        substantive = union - listeners

        types[name] = {
            "members": sorted(substantive),
            "mxd_count": len(mxd_names),
            "union_count": len(union),
            "substantive_count": len(substantive),
            "listener_count": len(listeners),
            "hidden_by_mxd": sorted(substantive - mxd_names),
        }
        totals["mxd"] += len(mxd_names)
        totals["union"] += len(union)
        totals["substantive"] += len(substantive)
        totals["listeners"] += len(listeners)

    return {"epii_version": list(EPII_VERSION),
            "type_count": len(types),
            "totals": totals,
            "types": types}


def _type_name_of_class(t):
    mod = getattr(t, "__module__", "") or ""
    return ("%s.%s" % (mod, t.__name__)) if mod else str(t)


def op_describe(surface, params):
    return describe(surface, params["path"],
                    include_values=params.get("include_values", True))


# ------------------------------------------------------------- PHASE 3: notes
#
# Generic get/set cannot express notes: get_notes_extended returns a
# MidiNoteVector of MidiNote objects, and writes need MidiNoteSpecification.
# jsonify() would render those as opaque {"__lom__": ...} handles. Hence typed
# wrappers - the one place hand-written code genuinely earns its keep.
#
# This is the Live 11+ "extended" API, which the M4L registry does not list.

NOTE_FIELDS = ("pitch", "start_time", "duration", "velocity", "mute",
               "probability", "velocity_deviation", "release_velocity")


def _note_to_dict(n):
    out = {"note_id": getattr(n, "note_id", None)}
    for f in NOTE_FIELDS:
        out[f] = getattr(n, f, None)
    return out


def _require_midi_clip(surface, path):
    clip = resolve(surface, path)
    if not getattr(clip, "is_midi_clip", False):
        raise TypeError("%s is not a MIDI clip" % path)
    return clip


def op_notes_get(surface, params):
    clip = _require_midi_clip(surface, params["path"])
    from_pitch = int(params.get("from_pitch", 0))
    pitch_span = int(params.get("pitch_span", 128))
    from_time = float(params.get("from_time", 0.0))
    time_span = float(params.get("time_span", clip.length or 0.0))
    notes = clip.get_notes_extended(from_pitch, pitch_span, from_time, time_span)
    return {"path": params["path"], "count": len(notes),
            "notes": [_note_to_dict(n) for n in notes]}


def op_notes_add(surface, params):
    """notes: [{pitch, start_time, duration, velocity?, mute?}, ...]"""
    from Live.Clip import MidiNoteSpecification
    clip = _require_midi_clip(surface, params["path"])
    specs = []
    for n in params["notes"]:
        specs.append(MidiNoteSpecification(
            pitch=int(n["pitch"]),
            start_time=float(n["start_time"]),
            duration=float(n["duration"]),
            velocity=float(n.get("velocity", 100)),
            mute=bool(n.get("mute", False)),
        ))
    with _undo(surface, "add notes"):
        clip.add_new_notes(tuple(specs))
    return {"path": params["path"], "added": len(specs)}


def op_notes_remove(surface, params):
    clip = _require_midi_clip(surface, params["path"])
    from_pitch = int(params.get("from_pitch", 0))
    pitch_span = int(params.get("pitch_span", 128))
    from_time = float(params.get("from_time", 0.0))
    time_span = float(params.get("time_span", clip.length or 0.0))
    before = len(clip.get_notes_extended(from_pitch, pitch_span,
                                         from_time, time_span))
    with _undo(surface, "remove notes"):
        clip.remove_notes_extended(from_pitch, pitch_span, from_time, time_span)
    return {"path": params["path"], "removed": before}


def op_notes_modify(surface, params):
    """Edit existing notes in place. Each entry needs note_id plus the fields
    to change; note_id comes from notes_get."""
    clip = _require_midi_clip(surface, params["path"])
    wanted = {}
    for n in params["notes"]:
        if n.get("note_id") is None:
            raise ValueError("each note needs a note_id (from notes_get)")
        wanted[int(n["note_id"])] = n

    live_notes = clip.get_notes_extended(0, 128, 0.0, clip.length or 0.0)
    touched = 0
    for ln in live_notes:
        spec = wanted.get(getattr(ln, "note_id", None))
        if spec is None:
            continue
        for f in NOTE_FIELDS:
            if f in spec:
                setattr(ln, f, spec[f])
        touched += 1
    if touched:
        with _undo(surface, "modify notes"):
            clip.apply_note_modifications(live_notes)
    missing = sorted(set(wanted) - set(getattr(n, "note_id", None)
                                       for n in live_notes))
    return {"path": params["path"], "modified": touched,
            "unmatched_note_ids": missing}


# ----------------------------------------------------------- PHASE 3: browser

def _browser(surface):
    app = Live.Application.get_application()
    b = getattr(app, "browser", None)
    if b is None:
        raise RuntimeError("Application.browser unavailable")
    return b


def _browser_item(it):
    return {"name": getattr(it, "name", None),
            "is_loadable": bool(getattr(it, "is_loadable", False)),
            "is_folder": bool(getattr(it, "is_folder", False)),
            "is_device": bool(getattr(it, "is_device", False)),
            "uri": getattr(it, "uri", None),
            "children": len(getattr(it, "children", []) or [])}


BROWSER_ROOTS = ("instruments", "sounds", "drums", "audio_effects",
                 "midi_effects", "plugins", "clips", "samples", "packs",
                 "user_library", "current_project", "max_for_live")


def op_browser_list(surface, params):
    """path is browser-relative: '' for roots, or 'plugins/My Plugin/...'."""
    b = _browser(surface)
    rel = (params.get("path") or "").strip("/")
    if not rel:
        out = []
        for r in BROWSER_ROOTS:
            node = getattr(b, r, None)
            if node is not None:
                d = _browser_item(node)
                d["name"] = d["name"] or r
                d["root"] = r
                out.append(d)
        return {"path": "", "items": out}

    parts = rel.split("/")
    node = getattr(b, parts[0], None)
    if node is None:
        raise ValueError("unknown browser root %r; expected one of %s"
                         % (parts[0], list(BROWSER_ROOTS)))
    for p in parts[1:]:
        match = None
        for c in (getattr(node, "children", []) or []):
            if getattr(c, "name", None) == p:
                match = c
                break
        if match is None:
            raise ValueError("no browser child %r under %r" % (p, node.name))
        node = match
    return {"path": rel, "item": _browser_item(node),
            "items": [_browser_item(c)
                      for c in (getattr(node, "children", []) or [])]}


def op_browser_load(surface, params):
    """Load a browser item onto the SELECTED track. Set `select_track` to a
    track index first if you need a specific destination."""
    b = _browser(surface)
    rel = params["path"].strip("/")
    parts = rel.split("/")
    node = getattr(b, parts[0], None)
    if node is None:
        raise ValueError("unknown browser root %r" % parts[0])
    for p in parts[1:]:
        match = None
        for c in (getattr(node, "children", []) or []):
            if getattr(c, "name", None) == p:
                match = c
                break
        if match is None:
            raise ValueError("no browser child %r under %r" % (p, node.name))
        node = match
    if not getattr(node, "is_loadable", False):
        raise ValueError("%r is not loadable" % rel)

    song = surface.song()
    idx = params.get("track_index")
    if idx is not None:
        song.view.selected_track = song.tracks[int(idx)]
    with _undo(surface, "load browser item"):
        b.load_item(node)
    return {"loaded": rel,
            "track": getattr(song.view.selected_track, "name", None)}


# -------------------------------------------------- PHASE 3: automation envelopes
#
# Envelopes hang off a (clip, DeviceParameter) pair, so both must be addressed
# by path. Parameter paths look like:
#   live_set tracks 0 devices 0 parameters 1
#   live_set tracks 0 mixer_device volume

def _envelope(surface, params, create=False):
    clip = resolve(surface, params["path"])
    param = resolve(surface, params["parameter"])
    env = clip.automation_envelope(param)
    if env is None and create:
        env = clip.create_automation_envelope(param)
    return clip, param, env


def op_envelope_get(surface, params):
    """Sample an envelope at the given times (defaults to 8 points over the clip)."""
    clip, param, env = _envelope(surface, params)
    if env is None:
        return {"path": params["path"], "parameter": params["parameter"],
                "exists": False, "points": []}
    times = params.get("times")
    if not times:
        n = int(params.get("samples", 8))
        length = float(clip.length or 0.0)
        times = [length * i / max(n - 1, 1) for i in range(n)]
    return {"path": params["path"], "parameter": params["parameter"],
            "exists": True,
            "parameter_name": getattr(param, "name", None),
            "min": getattr(param, "min", None), "max": getattr(param, "max", None),
            "points": [{"time": float(t), "value": env.value_at_time(float(t))}
                       for t in times]}


def op_envelope_insert_step(surface, params):
    """Write a flat step into an envelope, creating it if needed."""
    clip, param, env = _envelope(surface, params, create=True)
    if env is None:
        raise RuntimeError("could not create an envelope for that parameter")
    with _undo(surface, "insert envelope step"):
        env.insert_step(float(params["time"]), float(params["length"]),
                        float(params["value"]))
    return {"path": params["path"], "parameter": params["parameter"],
            "time": params["time"], "length": params["length"],
            "value": params["value"]}


def op_envelope_clear(surface, params):
    clip = resolve(surface, params["path"])
    with _undo(surface, "clear envelope"):
        if params.get("parameter"):
            clip.clear_envelope(resolve(surface, params["parameter"]))
            return {"cleared": params["parameter"]}
        clip.clear_all_envelopes()
    return {"cleared": "all"}


# ------------------------------------------------- PHASE 3: arrangement view
#
# The gap in jpoindexter's 128 tools: it can navigate the arrangement but not
# author clips into it.

def op_arrangement_create_clip(surface, params):
    """Create a clip directly in the Arrangement. kind: 'midi' | 'audio'."""
    track = resolve(surface, params["path"])
    start = float(params["start_time"])
    kind = params.get("kind", "midi")
    with _undo(surface, "create arrangement clip"):
        if kind == "midi":
            clip = track.create_midi_clip(start, float(params["length"]))
        elif kind == "audio":
            clip = track.create_audio_clip(params["file_path"], start)
        else:
            raise ValueError("kind must be 'midi' or 'audio'")
    return {"path": params["path"], "kind": kind, "start_time": start,
            "clip": jsonify(clip)}


def op_arrangement_duplicate_clip(surface, params):
    """Copy a session clip into the Arrangement at a given time."""
    track = resolve(surface, params["path"])
    clip = resolve(surface, params["clip"])
    with _undo(surface, "duplicate clip to arrangement"):
        result = track.duplicate_clip_to_arrangement(
            clip, float(params["destination_time"]))
    return {"track": params["path"], "clip": params["clip"],
            "destination_time": params["destination_time"],
            "result": jsonify(result)}


def op_arrangement_list(surface, params):
    track = resolve(surface, params["path"])
    clips = getattr(track, "arrangement_clips", []) or []
    return {"path": params["path"], "count": len(clips),
            "clips": [{"index": i, "name": getattr(c, "name", None),
                       "start_time": getattr(c, "start_time", None),
                       "end_time": getattr(c, "end_time", None),
                       "is_midi": getattr(c, "is_midi_clip", None)}
                      for i, c in enumerate(clips)]}


# ------------------------------------------------------ PHASE 6: observers
#
# Live's listener API is add_<prop>_listener / remove_<prop>_listener /
# <prop>_has_listener. Callbacks take NO arguments - they are bare
# notifications - so the current value must be read inside the callback.
#
# Two failure modes drive this design:
#
# 1. LEAK ACROSS RELOAD. importlib.reload re-executes this module, so a
#    registry in module globals would be reset to empty while Live still holds
#    every callback - unremovable, firing forever. The registry therefore lives
#    on the ControlSurface INSTANCE, which survives reload.
#
# 2. DANGLING OBJECTS. A listener keeps a reference to its Live object. If the
#    user deletes that track/clip, touching the object can crash Live. Every
#    callback re-checks liveobj_valid before reading.

MAX_EVENTS = 2000

# MEASURED against Live 12.2.7, not assumed. begin_undo_step/end_undo_step
# groups many mutations into one Cmd-Z, but NOT all: automatable/mappable
# parameters get their own undo step regardless.
#
#   grouped  : track name, color_index, signature_numerator
#   separate : tempo, mixer volume, mute
#
# So a transaction is "one Cmd-Z for the groupable ops, plus one per
# automatable parameter touched". Claiming a flat single-undo would be false.
UNDO_GROUPING_NOTE = (
    "Groupable ops collapse into ONE Cmd-Z. Automatable parameters "
    "(tempo, mixer volume, mute, device parameters) each form their OWN undo "
    "step in Live regardless of grouping, so reverting those needs one extra "
    "undo apiece. Measured on Live 12.2.7."
)


def _registry(surface):
    reg = getattr(surface, "_lom_observers", None)
    if reg is None:
        reg = {"listeners": {}, "events": [], "seq": 0, "dropped": 0}
        surface._lom_observers = reg
    return reg


def _valid(obj):
    m = _mxd()
    if m["types"] is not None:
        try:
            return bool(m["types"].liveobj_valid(obj))
        except Exception:
            pass
    return obj is not None


def _key(path, prop):
    return "%s\x00%s" % (path, prop)


def op_observe_add(surface, params):
    path, prop = params["path"], params["property"]
    reg = _registry(surface)
    k = _key(path, prop)
    if k in reg["listeners"]:
        return {"path": path, "property": prop, "already_observing": True}

    obj = resolve(surface, path)
    adder = getattr(obj, "add_%s_listener" % prop, None)
    remover = getattr(obj, "remove_%s_listener" % prop, None)
    if adder is None or remover is None:
        raise AttributeError(
            "%s has no listener for %r (expected add_%s_listener). Use "
            "lom_describe to see which properties are observable."
            % (_type_name(obj), prop, prop))

    def _callback():
        try:
            if not _valid(obj):
                return
            try:
                value = jsonify(getattr(obj, prop), depth=1)
            except Exception as e:
                value = {"error": "%s: %s" % (type(e).__name__, e)}
            reg["seq"] += 1
            reg["events"].append({"seq": reg["seq"], "path": path,
                                  "property": prop, "value": value})
            # Bounded: drop oldest, and record that we did rather than
            # silently losing events.
            over = len(reg["events"]) - MAX_EVENTS
            if over > 0:
                del reg["events"][:over]
                reg["dropped"] += over
        except Exception:
            # A raising callback inside Live's notification loop is a good way
            # to destabilise the GUI. Never propagate.
            pass

    adder(_callback)
    reg["listeners"][k] = {"cb": _callback, "obj": obj, "remover": remover,
                           "path": path, "property": prop}
    return {"path": path, "property": prop, "observing": True,
            "active_listeners": len(reg["listeners"])}


def _remove_one(entry):
    """-> 'removed' | 'object_gone' | 'failed:<reason>'

    'object_gone' is NOT a leak: Live destroyed the object and its listener
    list with it, so there is nothing left to detach from. Distinguished from
    a genuine failure so a real leak cannot hide behind a benign one.
    """
    if not _valid(entry["obj"]):
        return "object_gone"
    try:
        entry["remover"](entry["cb"])
        return "removed"
    except Exception as e:
        return "failed:%s: %s" % (type(e).__name__, e)


def op_observe_remove(surface, params):
    reg = _registry(surface)
    k = _key(params["path"], params["property"])
    entry = reg["listeners"].pop(k, None)
    if entry is None:
        return {"path": params["path"], "property": params["property"],
                "was_observing": False}
    return {"path": params["path"], "property": params["property"],
            "was_observing": True, "outcome": _remove_one(entry),
            "active_listeners": len(reg["listeners"])}


def op_observe_clear(surface, params):
    reg = _registry(surface)
    outcomes = {}
    failures = []
    for entry in list(reg["listeners"].values()):
        r = _remove_one(entry)
        bucket = r if r in ("removed", "object_gone") else "failed"
        outcomes[bucket] = outcomes.get(bucket, 0) + 1
        if bucket == "failed":
            failures.append({"path": entry["path"],
                             "property": entry["property"], "reason": r})
    reg["listeners"].clear()
    return {"attempted": sum(outcomes.values()), "outcomes": outcomes,
            "failures": failures,
            "leaked": len(failures)}


def op_observe_list(surface, params):
    reg = _registry(surface)
    return {"active_listeners": len(reg["listeners"]),
            "buffered_events": len(reg["events"]),
            "dropped_events": reg["dropped"],
            "max_events": MAX_EVENTS,
            "listeners": [{"path": e["path"], "property": e["property"],
                           "object_valid": _valid(e["obj"])}
                          for e in reg["listeners"].values()]}


def op_observe_poll(surface, params):
    """Drain buffered events. Pass `since` (a seq) to avoid re-reading.

    MCP has no server->client push, so this is a pull with a ring buffer
    behind it - events accumulate in Live and are collected on demand.
    """
    reg = _registry(surface)
    since = params.get("since")
    events = reg["events"]
    if since is not None:
        events = [e for e in events if e["seq"] > int(since)]
    limit = int(params.get("limit", 500))
    out = events[-limit:] if limit and len(events) > limit else events
    if params.get("consume"):
        reg["events"] = []
    return {"count": len(out),
            "latest_seq": reg["seq"],
            "dropped_events": reg["dropped"],
            "active_listeners": len(reg["listeners"]),
            "events": out}


# ------------------------------------------- PHASE 2: search + canonical path
#
# The generic API is complete but not discoverable: knowing tempo lives at
# "live_set tempo" is easy, knowing which index holds the track called "bass"
# is not. search closes that gap without adding per-feature tools.

# Collections worth walking. Deliberately EXCLUDES the browser: samples alone
# has 6343 children, and a naive walk would hang Live's main thread.
WALKABLE = (
    "tracks", "return_tracks", "scenes", "clip_slots", "devices", "chains",
    "drum_pads", "parameters", "arrangement_clips", "take_lanes", "cue_points",
    "sends",
)
# Mixer parameters are DeviceParameter objects hanging off singular attributes,
# not off a `parameters` collection - so a collection-only walk silently misses
# every track volume and pan. Some of these raise per instance (crossfader is
# main-track only); the walk already guards getattr.
def _identity(obj):
    """Stable identity for a Live object.

    id() is NOT safe here: Live returns a fresh Python wrapper per getattr, and
    once a wrapper is garbage-collected its address is reused - so a later,
    different object gets mistaken for one already visited and the walk silently
    truncates. _live_ptr is the underlying C++ pointer and is stable.
    """
    ptr = getattr(obj, "_live_ptr", None)
    return ("ptr", ptr) if ptr is not None else ("id", id(obj))


SINGULAR = (
    "master_track", "mixer_device", "clip", "view", "sample",
    "volume", "panning", "track_activator", "panning_mode",
    "crossfade_assign", "crossfader", "cue_volume", "song_tempo",
    "left_split_stereo", "right_split_stereo",
)


def op_search(surface, params):
    """Find LOM paths by name and/or type, breadth-first from a root.

    query   substring, case-insensitive, matched against `name`
    type    substring matched against the type, e.g. "Track" or "DeviceParameter"

    Bounded by max_results and max_nodes because this runs on Live's main
    thread - an unbounded graph walk would freeze the GUI.
    """
    query = (params.get("query") or "").lower()
    want_type = (params.get("type") or "").lower()
    if not query and not want_type:
        raise ValueError("give at least one of `query` or `type`")
    max_results = int(params.get("max_results", 50))
    max_nodes = int(params.get("max_nodes", 4000))
    root = params.get("root", "live_set")

    start = resolve(surface, root)
    queue = [(root, start, 0)]
    seen = set()
    keepalive = []          # holds refs so wrapper addresses cannot be recycled
    results = []
    nodes = 0
    truncated = False

    while queue:
        path, obj, depth = queue.pop(0)
        if nodes >= max_nodes or len(results) >= max_results:
            truncated = True
            break
        nodes += 1
        oid = _identity(obj)
        if oid in seen:
            continue
        seen.add(oid)
        keepalive.append(obj)

        tname = _type_name(obj)
        name = getattr(obj, "name", None)
        name = name if isinstance(name, str) else None
        hit = True
        if query:
            hit = hit and name is not None and query in name.lower()
        if want_type:
            hit = hit and want_type in tname.lower()
        if hit and path != root:
            results.append({"path": path, "type": tname, "name": name})

        if depth >= int(params.get("max_depth", 6)):
            continue
        for attr in WALKABLE:
            try:
                coll = getattr(obj, attr, None)
            except Exception:
                continue
            if coll is None or not _is_vector(coll):
                continue
            try:
                for i, child in enumerate(coll):
                    queue.append(("%s %s %d" % (path, attr, i), child, depth + 1))
            except Exception:
                continue
        for attr in SINGULAR:
            try:
                child = getattr(obj, attr, None)
            except Exception:
                continue
            if child is not None and _is_lom_object(child):
                queue.append(("%s %s" % (path, attr), child, depth + 1))

    return {"query": params.get("query"), "type": params.get("type"),
            "root": root, "count": len(results), "nodes_visited": nodes,
            "truncated": truncated, "results": results}


def op_canonical_path(surface, params):
    """Resolve an alias path to its canonical location.

    Useful because many paths point at the same object: "live_set view
    selected_track" is whichever track is selected right now, and the canonical
    form ("live_set tracks 3") is what you want to store or reuse.
    """
    path = params["path"]
    obj = resolve(surface, path)
    target = _identity(obj)

    # Walk the same bounded graph and report where this object actually lives.
    found = None
    queue = [("live_set", resolve(surface, "live_set"), 0)]
    seen = set()
    keepalive = []
    nodes = 0
    while queue and found is None and nodes < 6000:
        p, o, d = queue.pop(0)
        nodes += 1
        oid = _identity(o)
        if oid in seen:
            continue
        seen.add(oid)
        keepalive.append(o)
        if oid == target and p != path:
            found = p
            break
        if d >= 6:
            continue
        for attr in WALKABLE:
            try:
                coll = getattr(o, attr, None)
            except Exception:
                continue
            if coll is not None and _is_vector(coll):
                try:
                    for i, c in enumerate(coll):
                        queue.append(("%s %s %d" % (p, attr, i), c, d + 1))
                except Exception:
                    pass
        for attr in SINGULAR:
            try:
                c = getattr(o, attr, None)
            except Exception:
                continue
            if c is not None and _is_lom_object(c):
                queue.append(("%s %s" % (p, attr), c, d + 1))

    return {"path": path, "type": _type_name(obj),
            "canonical_path": found or path,
            "is_alias": bool(found and found != path),
            "nodes_visited": nodes}


OPS = {
    "describe": op_describe,
    "search": op_search,
    "canonical_path": op_canonical_path,
    "get_batch": op_get_batch,
    "set_batch": op_set_batch,
    "transaction": op_transaction,
    "observe_add": op_observe_add,
    "observe_remove": op_observe_remove,
    "observe_clear": op_observe_clear,
    "observe_list": op_observe_list,
    "observe_poll": op_observe_poll,
    "get": op_get,
    "set": op_set,
    "call": op_call,
    "count": op_count,
    "types": op_types,
    "envelope_get": op_envelope_get,
    "envelope_insert_step": op_envelope_insert_step,
    "envelope_clear": op_envelope_clear,
    "arrangement_create_clip": op_arrangement_create_clip,
    "arrangement_duplicate_clip": op_arrangement_duplicate_clip,
    "arrangement_list": op_arrangement_list,
    "notes_get": op_notes_get,
    "notes_add": op_notes_add,
    "notes_remove": op_notes_remove,
    "notes_modify": op_notes_modify,
    "browser_list": op_browser_list,
    "browser_load": op_browser_load,
}


def dispatch(surface, op, params):
    fn = OPS.get(op)
    if fn is None:
        raise ValueError("unknown op %r; known: %s" % (op, sorted(OPS)))
    return fn(surface, params)


def teardown(surface):
    """Called on disconnect AND before reload. Must leave no listener behind:
    Live holds the callback, so anything not removed here fires forever with
    no way to reach it."""
    _MFL_CACHE.clear()
    try:
        op_observe_clear(surface, {})
    except Exception:
        pass
