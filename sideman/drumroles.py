"""The hits inside one drum stem, each with its place on the grid and its group: hits that sound alike.

A drum stem from a separator holds several sounds (every hat of a track lands in `drums_hh`, a clap in
`drums_snare`). Electronic drums are a few samples repeated on a grid, so hits that fall at the same 16ths and sound
alike are one sound. `hit_groups` finds the hits (oneshot's hit detection), places each on the grid, groups them with
`spectral_distance` and reports each group's pattern, level and one clean hit. Local only; needs the `listen` extra.

Choices that are not obvious from the code:

- Where a hit sits: the grid line nearest its onset (bar from t0, 16th 0..15 from the bar's start). `offset_ms` is the
  onset minus that line, so a hit 8 ms early reads -8 and keeps its 16th. A hit more than GRID_LEAD_S (20 ms, the
  tolerance the reference-profile windows use) from every line is `on_grid: false`: it keeps its group but counts in no
  pattern. A pattern is the share of the scope's bars (the stem: first to last bar with a hit; a section: its `bars`)
  in which the group plays at that 16th; `unmeasured_pattern` is the same for the hits that could not be grouped.
- Onsets: oneshot's, so hit detection is `oneshot._segments` (a new hit rises 15 dB within 10 ms, to within 6 dB of its
  peak, and 6 dB over the 50 ms before it, at least 50 ms after the last). The first hit's onset is read with
  `_hit_onset` (12 dB over the bed before it), the others with `hit_cut`'s (the first rise out of the tail, which it
  reads as the loudest ms over 20 ms: one ms of a low tom's tail is loud or quiet with the phase it falls on, and read
  from 1 ms the onset came 5 ms early).
- What is compared: each hit's cut runs from its onset to 2 ms before the next hit's onset, or `max_cut_s` after it, and
  two cuts are compared over the shorter of the two (both ended with a 5 ms fade, so the cut itself is not a click). What
  follows a hit never decides its group, and a hit is only compared as far as it is alone: an open hat and the closed
  hat a 16th after it are compared over that 16th, loud against fallen.
- Hits that cannot be measured are listed with their reason, never dropped and never put in a group:
  `too_short` (under MIN_CUT_S before the next hit: a grace note or flam; the distance was measured down to 60 ms),
  `no_lead` (the stem starts just before it: nothing says what lies under it), `overlapped` (the 10 ms before its onset
  are less than OVERLAP_DB under its loudest ms: the tail of an earlier hit or a bleed bed), `faint` (FAINT_DB under
  the median of the hits around it: with a bleed bed under the whole stem `_segments` reports the bed before the first
  hit as a hit), `silent`, `ungrouped` (more than `max_groups` sounds: not forced into the nearest). OVERLAP_DB is
  measured: a closed noise hat (tau 12 ms) over the tail of an open one, against a clean closed hat, reads 0.8-0.9 with
  the tail 35 dB under it, 1.0-1.1 at 30, 1.2-1.3 at 27, 1.4-1.5 at 24 and 1.6-1.7 at 21 (the threshold below is
  1.5); a constant bed that every hit shares does not separate them until about 20 dB (within 0.8 and between 2.5 at
  40 dB, review round 2), but one rule cannot tell a shared bed from an earlier hit's tail, so a bed at -30 dB or
  louder under the hits makes them overlapped, and the report says so.
  What this cannot report: a hit that does not rise out of a tail as `_segments` asks (16th hats under an open hat of tau
  120 ms, whose tail is only 9 dB down one 16th later) is part of the earlier hit. Each hit lists the 16th lines its own tail still
  sounds through (`through`), which is where such a hit could hide, and the result says so.
- Grouping: agglomerative, average linkage, cut at `threshold` (1.5). Average linkage because single linkage chains (a
  velocity gradient or a bleed bed links closed to open through the hits between them: 0, 1.4 and 2.8 are one group
  at 1.5) and complete linkage splits one sound on its one worst pair, which for a noise hit is the noise realisation.
  The number is measured on noise hats (tau 12 and 120 ms, cuts of 60 to 500 ms): two realisations of one hat read
  0.55-0.74 (closed) and 0.65-0.82 (open); a hat with a 50 % longer decay 0.7-1.3; a closed hat against an open one
  2.8-4.9; and the review of the distance measured at least 2.5 between two sounds over a bleed bed. 1.5 is 1.8 times the
  worst pair of one sound and 1.7 times under the nearest of two. It is coarser on low pitched drums, where the
  distance moves little with pitch (the review, linear scale: a 50 Hz kick reads 0.38 for 1.5 semitones and 0.78 for an octave; on the
  Flare stems a 45 Hz kick against a 136 Hz tom read 1.06), so two toms a fifth apart can still share a group.
- Cost: every distance is one `spectral_distance` (about 2 ms on a 120 ms cut, 6 ms on 500 ms). The hits are not
  compared pairwise: a deterministic sample of SAMPLE_HITS hits (evenly spaced in time) is grouped (S^2 / 2 distances),
  every other hit goes to the nearest of the sample's medoids if within `threshold` of it (n * k distances for k
  groups), and what is left over is sampled again, until nothing is or `max_groups` groups exist. About 40-120 ms a hit
  on real stems (a nine-minute stem of 3000 hits: a few minutes). The hits are put in time order first, so the order
  they come in never matters; ties go to the earlier hit. Memory: the stem as float64.
- Sections do not regroup: a section's groups are the stem's groups restricted to the hits whose bar lies in the section
  (the same ids in every section; the count, pattern, level, spread and clean hit are the section's own).
"""
from __future__ import annotations

import math
import pathlib

import numpy as np

from . import oneshot
from .listen import GRID_LEAD_S, _check_channels, _check_sections, load

THRESHOLD = 1.5          # average-linkage distance (spectral_distance, mel) at which groups stop merging
MAX_CUT_S = 0.5          # a hit's cut is at most this long
END_GUARD_S = 0.002      # a cut ends this long before the next hit's onset
MIN_CUT_S = 0.060        # a hit with less room than this before the next one is too short to compare
FADE_S = 0.005           # the shared cut ends with a half-Hann fade of this length (at most a quarter of it)
LEAD_WINDOW_S = 0.010    # what came before a hit is read over this, ending 1 ms before the onset
OVERLAP_DB = -30.0       # a hit is overlapped when the 10 ms before it are louder than this under its loudest ms
SAMPLE_HITS = 32         # hits grouped pairwise at a time
MAX_GROUPS = 24
MAX_CANDIDATES = 40      # members tried for the clean hit, nearest the medoid first
FAINT_DB = 20.0          # a hit this far under the median of its 12 neighbours is a bed or bleed, not a hit
NEIGHBOURS = 6
THROUGH_DB = 40.0        # a tail still sounds through a grid line within this of its hit's loudest ms


def _fade(x: np.ndarray, sr: int) -> np.ndarray:
    """x [n, channels] with its last FADE_S (at most a quarter of it) faded out by a half Hann window."""
    n = min(int(round(FADE_S * sr)), len(x) // 4)
    if n < 2:
        return x
    x = x.copy()
    x[-n:] *= (0.5 + 0.5 * np.cos(np.pi * (np.arange(n) + 1) / n))[:, None]
    return x


def _db_rms(x: np.ndarray, sr: int) -> float:
    """dBFS of the loudest 1 ms of x [n, channels] (power mean over the channels)."""
    return 20.0 * math.log10(max(oneshot._loudest_ms(x, sr), 1e-6))


def _onset_after_tail(x: np.ndarray, sr: int, start: int) -> int:
    """Sample of the hit whose segment begins at `start` (up to 10 ms before it, after an earlier hit): hit_cut's
    onset, the first rise out of the tail with the tail read as its loudest ms over 20 ms, so the phase a low tone's
    tail is at when the segment begins does not move the onset."""
    onset = oneshot.hit_cut(x, sr, start / sr)[1]
    return start if onset is None else int(round(onset * sr))


def _agglomerate(dm: np.ndarray, threshold: float) -> list[list[int]]:
    """Average-linkage clusters of the points whose pairwise distances are dm [m, m] (symmetric): the two
    clusters with the least mean distance between their members merge until the least is over `threshold`.
    Ties go to the lowest indices, so the same distances in the same order always give the same clusters."""
    clusters = [[i] for i in range(len(dm))]
    d = np.array(dm, dtype=np.float64)
    np.fill_diagonal(d, np.inf)
    while len(clusters) > 1:
        i, j = divmod(int(np.argmin(d)), len(clusters))
        if d[i, j] > threshold:
            break
        ni, nj = len(clusters[i]), len(clusters[j])
        d[i, :] = d[:, i] = (ni * d[i, :] + nj * d[j, :]) / (ni + nj)       # Lance-Williams, average linkage
        d[i, i] = np.inf
        d = np.delete(np.delete(d, j, axis=0), j, axis=1)
        clusters[i] += clusters.pop(j)
    return clusters


def group_cuts(cuts: list, keys: list, sr: int, threshold: float = THRESHOLD, sample: int = SAMPLE_HITS,
               max_groups: int = MAX_GROUPS) -> tuple[list[dict], list[int]]:
    """Group hit cuts ([n, channels] arrays, each from its hit's onset) by how alike they sound: (groups,
    ungrouped). `keys` are the hits' onset times (any numbers, distinct): the hits are put in that order first,
    so the order of `cuts` never changes the answer. Two cuts are compared over the shorter of the two (spectral_distance,
    mel, 60 dB floor). Each round: a deterministic sample of `sample` hits, evenly spaced in time, is grouped
    (average linkage, cut at `threshold`); every other remaining hit joins the group with the nearest medoid when
    it is within `threshold` of it; the hits left over go round again, until none is left or `max_groups` groups
    exist (the rest are `ungrouped`, not forced). A group is {"members": [indices into cuts], "medoid": index (the
    sample member with the least mean distance to the others), "distance": {index: distance to the medoid}}."""
    order = sorted(range(len(cuts)), key=lambda i: keys[i])
    seen: dict = {}

    def dist(i: int, j: int) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            n = min(len(cuts[a]), len(cuts[b]))
            seen[(a, b)] = oneshot.spectral_distance(_fade(cuts[a][:n], sr), _fade(cuts[b][:n], sr), sr)
        return seen[(a, b)]

    groups, remaining = [], order
    while remaining and len(groups) < max_groups:
        m = len(remaining)
        pick = remaining if m <= sample else [remaining[int(round(k * (m - 1) / (sample - 1)))] for k in range(sample)]
        pick = list(dict.fromkeys(pick))
        dm = np.zeros((len(pick), len(pick)))
        for a in range(len(pick)):
            for b in range(a + 1, len(pick)):
                dm[a, b] = dm[b, a] = dist(pick[a], pick[b])
        new = []
        for cl in _agglomerate(dm, threshold)[:max_groups - len(groups)]:
            mean = [np.mean([dm[a, b] for b in cl]) if len(cl) > 1 else 0.0 for a in cl]
            medoid = pick[cl[int(np.argmin(mean))]]
            new.append({"members": [pick[a] for a in cl], "medoid": medoid,
                        "distance": {pick[a]: dist(pick[a], medoid) for a in cl}})
        taken = {i for g in new for i in g["members"]}
        left = []
        for i in remaining:
            if i in taken:
                continue
            near = min(new, key=lambda g: (dist(i, g["medoid"]), g["medoid"]))
            d = dist(i, near["medoid"])
            if d <= threshold:
                near["members"].append(i)
                near["distance"][i] = d
            else:
                left.append(i)
        groups += new
        remaining = left
        if len(new) == 0:
            break
    return groups, remaining


def _clean_hit(x: np.ndarray, sr: int, hits: list, ids: list, dist: dict) -> tuple[dict | None, str | None]:
    """(the clean hit, or None and why) among hits `ids` of one group: oneshot._clean_hit's rule on the group's
    members. Candidates, nearest their group's medoid first (at most MAX_CANDIDATES), must be complete (their cut
    falls 40 dB before it ends, oneshot._measure's `truncated`); of those the ones within CLEAN_BAND_DB of the loudest,
    then the ones with a quiet lead-in (the 20 ms before the onset LEAD_DB under the peak), if any, then the ones with
    at least ROOM_FRAC of the median room, then the ones within TYPICAL_DB of the median peak, if any. Of the
    last the one nearest the medoid wins (oneshot's rule takes the earliest; here the group is already known, so
    the most typical sound is taken)."""
    order = sorted(ids, key=lambda i: (dist[i], hits[i]["start"]))[:MAX_CANDIDATES]
    cands = []
    for i in order:
        h = hits[i]
        try:
            got = oneshot._measure(x[h["start"]:h["end"]], sr, "the cut ends", quick=True)
        except ValueError:
            continue
        if got["truncated"]:
            continue
        lead = np.abs(x[max(0, h["start"] - int(oneshot.LEAD_S * sr)):h["start"]]).max(axis=1)
        quiet = not len(lead) or 20 * math.log10(max(float(lead.max()), 1e-12)) <= got["peak_db"] - oneshot.LEAD_DB
        cands.append({"i": i, "peak": got["peak_db"], "room": h["room"], "quiet": quiet})
    if not cands:
        return None, ("none of the %d member%s tried is complete: each is cut off by the next hit or by the %g s "
                      "limit before it has fallen 40 dB" % (len(order), "" if len(order) == 1 else "s", MAX_CUT_S))
    loud = max(c["peak"] for c in cands)
    cands = [c for c in cands if c["peak"] >= loud - oneshot.CLEAN_BAND_DB]
    cands = [c for c in cands if c["quiet"]] or cands
    med_room = float(np.median([c["room"] for c in cands]))
    cands = [c for c in cands if c["room"] >= oneshot.ROOM_FRAC * med_room]
    med_peak = float(np.median([c["peak"] for c in cands]))
    cands = [c for c in cands if abs(c["peak"] - med_peak) <= oneshot.TYPICAL_DB] or cands
    h = hits[cands[0]["i"]]                     # candidates are in order of distance to the medoid
    return {"hit": h["id"], "start": h["start"], "end": h["end"], "onset_s": h["onset_s"], "level_db": h["level_db"],
            "audio": x[h["start"]:h["end"]].copy()}, None


def _scope(x: np.ndarray, sr: int, hits: list, found: list, ids: set, bars: int, steps: int) -> dict:
    """The groups of `found` (group_cuts output, each member an index into hits) restricted to the hits in `ids`,
    over `bars` bars: {"hits": n, "bars": bars, "groups": [...], "unmeasured_pattern": [...], "unmeasured": [...]}. Ordered by count, then by the
    first member's time; "id" is the group's number in the whole stem."""
    groups = []
    for gid, g in found:
        mine = [i for i in g["members"] if i in ids]      # in time order
        if not mine:
            continue
        on = [hits[i] for i in mine if hits[i]["on_grid"]]
        by_step = [{h["bar"] for h in on if h["step"] == s} for s in range(steps)]
        clean, why = _clean_hit(x, sr, hits, mine, g["distance"])
        dists = [g["distance"][i] for i in mine]
        groups.append({
            "id": gid, "count": len(mine), "bars_with": len({hits[i]["bar"] for i in mine}),
            "pattern": [len(b) / bars if bars else 0.0 for b in by_step],
            "level_db": float(np.median([hits[i]["level_db"] for i in mine])),
            "spread": {"median": float(np.median(dists)), "max": float(max(dists))},
            "off_grid": len(mine) - len(on), "members": [hits[i]["id"] for i in mine],
            "clean_hit": clean, "clean_note": why})
    groups.sort(key=lambda g: (-g["count"], g["members"][0]))
    lost = [hits[i] for i in ids if hits[i]["status"] in ("overlapped", "too_short", "no_lead", "ungrouped") and hits[i]["on_grid"]]
    return {"hits": len(ids), "bars": bars, "groups": groups,
            "unmeasured_pattern": [len({h["bar"] for h in lost if h["step"] == s}) / bars if bars else 0.0 for s in range(steps)],
            "unmeasured": [{"hit": i, "status": hits[i]["status"], "reason": hits[i]["reason"]}
                           for i in sorted(ids) if hits[i]["status"] != "grouped"]}


def hit_groups(stem, tempo: float, t0: float, sections: list | None = None, sr: int | None = None,
               beats_per_bar: int = 4, threshold: float = THRESHOLD, max_cut_s: float = MAX_CUT_S,
               sample: int = SAMPLE_HITS, max_groups: int = MAX_GROUPS) -> dict:
    """Every hit in one drum stem, with its place on the grid, grouped by how alike the hits sound (see the module
    docstring for the choices). `stem` is a path or an array [n] / [n, channels] with its rate in `sr`; bar 0 starts
    at `t0` seconds, 16ths are a quarter of a beat of `tempo` (4 * beats_per_bar to a bar); `sections` are
    {name, start_bar, bars} (as reference_profile takes them).

    Returns {"sample_rate", "tempo", "t0", "threshold", "linkage": "average", "max_cut_s", "hits", "groups",
    "unmeasured_pattern" (per 16th, the share of bars with a hit there that could not be grouped), "unmeasured"
    ([{"hit", "status", "reason"}]), "sections", "notes"}. A hit is {"id", "onset_s", "bar", "step" (16th, 0-15, of the nearest grid line),
    "offset_ms" (onset minus that line), "on_grid" (within 20 ms of it), "level_db" (loudest ms, dBFS), "lead_db" (the
    10 ms before the hit against its loudest ms; None at the start of the stem), "through" (16th lines inside its cut
    that its own tail still sounds through), "start", "end" (cut, in samples of the stem), "room" (samples to the
    next hit), "status" ("grouped", "too_short", "no_lead", "faint", "overlapped", "silent" or "ungrouped"), "reason" (why not grouped),
    "group", "distance" (to its group's medoid)}. A group is {"id", "count", "bars_with" (bars it plays in),
    "pattern" (per 16th, the share of the scope's bars it plays at, on-grid hits only), "level_db" (median),
    "spread" ({"median", "max"} of the members' distances to the medoid), "off_grid", "members" (hit ids),
    "clean_hit" ({"hit", "start", "end", "onset_s", "level_db", "audio"} or None), "clean_note" (why None)}.
    `groups` is the whole stem (its bars run from the first hit's bar to the last's); each of `sections` has the same
    groups restricted to its bars, with the section's own count, pattern over its `bars`, level, spread and clean hit."""
    if not (tempo > 0 and math.isfinite(tempo)) or not math.isfinite(t0):
        raise ValueError("tempo must be above 0 and t0 a number of seconds, got %r and %r" % (tempo, t0))
    if beats_per_bar < 1 or sample < 2 or max_groups < 1 or threshold <= 0:
        raise ValueError("beats_per_bar and max_groups need at least 1, sample at least 2, threshold above 0")
    if sections is not None:
        _check_sections(sections)
    if isinstance(stem, (str, pathlib.Path)):
        x, sr = load(stem)
    else:
        if sr is None:
            raise ValueError("an array needs its sample rate: pass sr=")
        x = np.asarray(stem)
        x = x[:, None] if x.ndim == 1 else x
    _check_channels(x)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("the stem holds NaN or infinite samples")
    sr = int(sr)
    x = oneshot._undc(x, sr)
    out = {"sample_rate": sr, "tempo": tempo, "t0": t0, "threshold": threshold, "linkage": "average",
           "max_cut_s": max_cut_s, "hits": [], "groups": [], "unmeasured_pattern": [], "unmeasured": [], "sections": [],
           "notes": []}
    if not len(x) or float(np.abs(x).max()) <= 0:
        out["notes"].append("the stem is silent: no hits")
        return out
    segs = oneshot._segments(x, sr)
    onsets = [oneshot._hit_onset(x[s:e], sr) if s == 0 else _onset_after_tail(x, sr, s) for s, e in segs]
    step_s, steps = 60.0 / tempo / 4.0, 4 * beats_per_bar
    guard, lead_end, lead_len = (int(round(v * sr)) for v in (END_GUARD_S, 0.001, LEAD_WINDOW_S))
    hits, cuts, keys = [], [], []
    frame, glead = max(1, int(round(0.001 * sr))), int(round(GRID_LEAD_S * sr))
    for k, o in enumerate(onsets):
        last = k + 1 == len(onsets)
        nxt = len(x) if last else onsets[k + 1]
        end = min(nxt if last else nxt - guard, o + int(round(max_cut_s * sr)))
        rel = o / sr - t0
        line = int(round(rel / step_s))
        bar, step = divmod(line, steps)
        off = rel - line * step_s
        h = {"id": k, "onset_s": o / sr, "bar": bar, "step": step, "offset_ms": round(1000 * off, 2),
             "on_grid": abs(off) <= GRID_LEAD_S, "level_db": None, "lead_db": None, "through": 0, "start": o,
             "end": end, "room": nxt - o, "status": "grouped", "reason": None, "group": None, "distance": None}
        hits.append(h)
        cut = x[o:end]
        if float(np.abs(cut).max()) <= 0:
            h["status"], h["reason"] = "silent", "the cut holds only zeros"
            continue
        h["level_db"] = _db_rms(cut, sr)
        a = max(0, o - lead_end - lead_len)
        if o - lead_end - a >= lead_len // 2:           # something of the stem lies before the hit
            before = float((x[a:o - lead_end] ** 2).mean())
            h["lead_db"] = round(10.0 * math.log10(max(before, 1e-12)) - h["level_db"], 1)
        loud = oneshot._loudest_ms(cut, sr) ** 2
        for t in (t0 + n * step_s for n in range(line + 1, line + steps + 1)):
            i = int(round(t * sr)) - o
            if i + frame > len(cut) - glead:
                break
            if i > glead and float((cut[i:i + frame] ** 2).mean()) >= loud * 10 ** (-THROUGH_DB / 10):
                h["through"] += 1
        if nxt - o < int(round(MIN_CUT_S * sr)):
            h["status"], h["reason"] = "too_short", "the next hit starts %.0f ms after it (under %.0f ms)" % (
                1000 * (nxt - o) / sr, 1000 * MIN_CUT_S)
        elif h["lead_db"] is None:
            h["status"], h["reason"] = "no_lead", (
                "it starts within %.0f ms of the start of the stem: a tail or bed under it cannot be ruled out"
                % (1000 * (lead_end + lead_len // 2) / sr))
        elif h["lead_db"] > OVERLAP_DB:
            h["status"], h["reason"] = "overlapped", (
                "it starts out of a tail or bed only %.0f dB under its own loudest ms (it needs %.0f)"
                % (-h["lead_db"], -OVERLAP_DB))
    for h in hits:
        if h["status"] != "grouped":
            continue
        around = hits[max(0, h["id"] - NEIGHBOURS):h["id"]] + hits[h["id"] + 1:h["id"] + NEIGHBOURS + 1]
        near = [a["level_db"] for a in around if a["level_db"] is not None]
        if len(near) >= 3 and h["level_db"] < float(np.median(near)) - FAINT_DB:
            h["status"], h["reason"] = "faint", "%.0f dB under the hits around it (over %.0f dB): bed or bleed, not a hit" % (
                float(np.median(near)) - h["level_db"], FAINT_DB)
        else:
            h["_cut"] = len(cuts)
            cuts.append(x[h["start"]:h["end"]])
            keys.append(h["start"])
    found, rest = group_cuts(cuts, keys, sr, threshold, sample, max_groups) if cuts else ([], [])
    index = {h["_cut"]: h["id"] for h in hits if "_cut" in h}
    for g in found:                                      # from cut numbers to hit ids
        g["members"] = sorted(index[i] for i in g["members"])
        g["medoid"] = index[g["medoid"]]
        g["distance"] = {index[i]: d for i, d in g["distance"].items()}
    found.sort(key=lambda g: (-len(g["members"]), g["members"][0]))
    found = list(enumerate(found))
    for gid, g in found:
        for i, d in g["distance"].items():
            hits[i]["group"], hits[i]["distance"] = gid, d
    for c in rest:
        h = hits[index[c]]
        h["status"], h["reason"] = "ungrouped", "more than %d distinct sounds: not forced into a group" % max_groups
    for h in hits:
        h.pop("_cut", None)
    out["hits"] = hits
    if not hits:
        return out
    real = [h["bar"] for h in hits if h["status"] not in ("faint", "silent", "no_lead")]
    first, last = (min(real), max(real)) if real else (0, 0)
    whole = _scope(x, sr, hits, found, {h["id"] for h in hits}, last - first + 1, steps)
    out.update({k: whole[k] for k in ("groups", "unmeasured_pattern", "unmeasured")})
    for sec in sections or []:
        a, b = sec["start_bar"], sec["start_bar"] + sec["bars"]
        got = _scope(x, sr, hits, found, {h["id"] for h in hits if a <= h["bar"] < b}, sec["bars"], steps)
        out["sections"].append({"name": sec.get("name"), "start_bar": a, "bars": sec["bars"], **got})
    if any(h["through"] for h in hits):
        out["notes"].append("a hit played inside another hit's tail, without rising 15 dB out of it, is not found: "
                            "`through` lists the 16th lines each tail still sounds through")
    return out
