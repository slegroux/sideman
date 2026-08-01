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
    """Authoritative member list, with a dir() fallback."""
    idx = _mfl_index(type(obj))
    if idx:
        return sorted(idx), "mxd"
    return sorted(n for n in dir(obj) if not n.startswith("_")), "dir"


def describe(surface, path, include_values=True):
    """getinfo, but honest about per-INSTANCE availability."""
    obj = resolve(surface, path)
    names, source = _member_names(obj)
    idx = _mfl_index(type(obj))

    out = {
        "path": path,
        "type": _type_name(obj),
        "source": source,
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
    song = surface.song()

    class _Step(object):
        def __enter__(self):
            try:
                song.begin_undo_step()
            except Exception:
                pass
            return self

        def __exit__(self, *exc):
            try:
                song.end_undo_step()
            except Exception:
                pass
            return False

    return _Step()


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


def op_types(surface, params):
    """Full LOM type census - the coverage-harness baseline."""
    m = _mxd()
    if m["types"] is None:
        raise RuntimeError("_MxDCore unavailable: %s" % m["error"])
    out = {}
    for t in m["types"].get_available_lom_types():
        try:
            props = m["types"].get_available_properties_for_type(t, EPII_VERSION)
            out[_type_name_of_class(t)] = sorted(p.name for p in props)
        except Exception as e:
            out[_type_name_of_class(t)] = {"error": str(e)}
    return {"epii_version": list(EPII_VERSION), "types": out,
            "type_count": len(out),
            "member_total": sum(len(v) for v in out.values()
                                if isinstance(v, list))}


def _type_name_of_class(t):
    mod = getattr(t, "__module__", "") or ""
    return ("%s.%s" % (mod, t.__name__)) if mod else str(t)


def op_describe(surface, params):
    return describe(surface, params["path"],
                    include_values=params.get("include_values", True))


OPS = {
    "describe": op_describe,
    "get": op_get,
    "set": op_set,
    "call": op_call,
    "count": op_count,
    "types": op_types,
}


def dispatch(surface, op, params):
    fn = OPS.get(op)
    if fn is None:
        raise ValueError("unknown op %r; known: %s" % (op, sorted(OPS)))
    return fn(surface, params)


def teardown(surface):
    _MFL_CACHE.clear()
