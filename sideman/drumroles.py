"""The hits inside one drum stem, each with its place on the grid and its group: hits that sound alike.

A drum stem from a separator holds several sounds (every hat of a track lands in `drums_hh`, a clap in
`drums_snare`). Electronic drums are a few samples repeated on a grid, so hits that fall at the same 16ths and sound
alike are one sound. `hit_groups` finds the hits (oneshot's hit detection), places each on the grid, groups them with
`spectral_distance` and reports each group's pattern, level and one clean hit. Local only; needs the `listen` extra.

Choices that are not obvious from the code:

- Where a hit sits: the grid line nearest its onset (bar from t0, 16th 0..15 from the bar's start). `offset_ms` is the
  onset minus that line, so a hit 8 ms early reads -8 and keeps its 16th. A hit more than GRID_LEAD_S (20 ms, the
  tolerance the reference-profile windows use) from every line is `on_grid: false`: it keeps its group but counts in no
  pattern. A pattern is the share of the scope's bars (the stem: first to last bar with a real hit; a section: its `bars`)
  in which the group plays at that 16th.
- Onsets: oneshot's, so hit detection is `oneshot._segments` (a new hit rises 15 dB within 10 ms, to within 6 dB of its
  peak, and 6 dB over the 50 ms before it, at least 50 ms after the last), and it is relative to the stem's loudest
  moment: a hit more than 40 dB under the loudest 5 ms of the stem is not found (a stem with one very loud hit loses
  its quiet ones). The first hit's onset is read with `_hit_onset` (12 dB over the bed before it), the others with
  `hit_cut`'s (the first rise out of the tail, which it reads as the loudest ms over 20 ms: one ms of a low tom's tail
  is loud or quiet with the phase it falls on, and read from 1 ms the onset came 5 ms early).
- What is compared: each hit's cut runs from its onset to 2 ms before the next hit's onset, or `max_cut_s` after it, and
  two cuts are compared over the shorter of the two (both ended with a 5 ms fade, so the cut itself is not a click). What
  follows a hit never decides its group, and a hit is only compared as far as it is alone: two hats a 16th apart are
  compared over that 16th.
- Hits that cannot be measured are listed with their reason, never dropped and never put in a group:
  `too_short` (under MIN_CUT_S before the next hit: a grace note or flam; the distance was measured down to 60 ms),
  `no_lead` (the stem starts just before it: nothing says what lies under it), `overlapped` (the 10 ms before its onset
  are less than OVERLAP_DB under its loudest ms: the tail of an earlier hit or a bleed bed), `faint` (FAINT_DB under
  the median of the hits around it: with a bleed bed under the whole stem `_segments` reports the bed before the first
  hit as a hit, and a ghost note reads the same, so faint hits are counted in `unmeasured_pattern`), `silent`,
  `ungrouped` (more than `max_groups` sounds: not forced into the nearest).
- Overlapped hits and groups: an overlapped hit never forms or moves a group. Once the groups exist it is compared with
  each group's medoid, over the whole cut when its tail is ASSIGN_FULL_DB (-24 dB) or more under it, over its first 40 ms
  when the tail is between -24 and -14 dB under it, not at all when louder; it goes to the nearest medoid when that is
  within the threshold and 15 % nearer than the second nearest, else it stays "overlapped" with the numbers in its
  reason. An assigned hit has status "assigned", counts in its group's `pattern` (and in `assigned_pattern`, the same for
  the assigned hits alone), `bars_with` and `assigned`, never in `count`, `level_db`, `spread`, the medoid or the clean hit.
  Measured on 24 stems of 8 bars (open hats of tau 80, 120, 160 and 200 ms on the offbeats, closed hats of tau 12 ms
  on the other 16ths, 3 seeds, the search off and on): no hat in a group of the other kind; the closed hats placed
  went from 2 to about 50 (tau 120) and 14 (200) without the search and 61-63 and 29-33 with it. The 25 ms window the
  review also tried (tail louder than -14 dB) placed 368 right and 69 wrong, so it is not used. OVERLAP_DB is measured: a closed noise hat (tau 12 ms) over the tail of an open
  one, against a clean closed hat, reads 0.8-0.9 with the tail 35 dB under it, 1.0-1.1 at 30, 1.2-1.3 at 27, 1.4-1.5 at
  24 and 1.6-1.7 at 21 (the threshold below is 1.5); a bed that every hit shares does not separate them until about
  20 dB under (within 0.8, between 2.5 at 40 dB, review round 2), but one rule cannot tell a shared bed from an earlier
  hit's tail, so a bed at -30 dB or louder under the hits makes them overlapped, and the report says so.
- Hits the detector cannot see: a hit that does not rise out of a tail as `_segments` asks (a closed hat on the 16th
  after an open hat of tau 120 ms, whose tail is only 9 dB down there) is part of the earlier hit. `through` lists the
  16th lines each hit's tail still sounds through (within 40 dB), and `hidden_pattern` (per 16th, the share of bars in
  which that line is inside a tail with no hit found at it) says where they could be, even without the search.
  `unmeasured_pattern` leaves them out, and every hit that was never found. `grid_onsets=True` looks at every 16th line
  with no onset within 50 ms for a rise of 6 dB or more (the first-difference power, 3 ms after against 5 ms before, the
  best point within 20 ms of the line, no more than 60 dB under the stem's peak) and adds it as a hit with
  `source: "grid"`. It is meant for hat-like stems (`hat_like`: the share of energy above 4 kHz; a hat stem reads
  0.9 and more, kick, snare and toms stems well under 0.2): on the others it finds bleed and the phase of a low tone's
  tail as hits (118-355 hits on the Flare kick, snare and toms stems, review round 1; a synthetic kick stem here reads
  the kick's own tail at every line after a kick), which `grid_hits` and `source` count and mark. It reads only a
  window of the stem around each line, never the whole stem at once.
- Grouping: agglomerative, average linkage, cut at `threshold` (1.5). Average linkage because single linkage chains (a
  velocity gradient or a bleed bed links closed to open through the hits between them: 0, 1.4 and 2.8 are one group
  at 1.5) and complete linkage splits one sound on its one worst pair, which for a noise hit is the noise realisation.
  The number is measured on noise hats (tau 12 and 120 ms, cuts of 60 to 500 ms): two realisations of one hat read
  0.55-0.74 (closed) and 0.65-0.82 (open); a hat with a 50 % longer decay 0.7-1.3; a closed hat against an open one
  2.8-4.9; and the review of the distance measured at least 2.5 between two sounds over a bleed bed. 1.5 is 1.8 times the
  worst pair of one sound and 1.7 times under the nearest of two.
  Material whose distances lie around the threshold (a continuum of decays, a velocity gradient: 48 of 247 hat member
  distances on Flare lie between 1.3 and 1.5) is cut where the sample puts the cut: a different sample, or the same
  bars inside a longer stem, gives other groups for it (the review: ARI 0.65-0.95 for the sample at 16 to 128 hits). Well
  separated sounds do not move. `stability` says which groups moved (see hit_groups), and groups under 4 measured
  hits are `tentative`.
- Pitch: the distance moves little with pitch on low drums, so a group whose members' body pitches (`profile_array`'s
  `pitch_body_hz` on each member's cut) fall in two clusters more than 3 semitones apart is split after grouping, each
  side keeping at least two hits; members with no pitch stay with the larger cluster. Measured as the same drum
  shifted up by 7 / 12 / 19 semitones, the distance reads 0.44 / 0.61 / 0.98 (45 Hz kick), 0.65 / 1.06 / 1.55 (100 Hz
  tom), 0.87 / 1.37 / 1.87 (150 Hz tom, review round 1): toms an octave apart and a kick a fifth to an octave up share a
  group without the split. A group is read only when at least half of 8 members spread over it have a pitch (a hat
  group costs 8 reads of 7 ms; a tom, 150 ms for each member, which is most of the run time of a kick or toms stem).
- Cost: every distance is one `spectral_distance` (about 2 ms on a 120 ms cut, 6 ms on 500 ms). The hits are not
  compared pairwise: a deterministic sample of SAMPLE_HITS hits (evenly spaced in time) is grouped (S^2 / 2 distances),
  every other hit goes to the nearest of the sample's medoids if within `threshold` of it (n * k distances for k
  groups), and what is left over is sampled again, until nothing is or `max_groups` groups exist. Overlapped hits cost
  one distance per group. `stability_runs` regroupings with a shifted sample reuse every distance already measured, and
  cost about as much again for each. The hits are put in time order first, so the order they come in never matters;
  ties go to the earlier hit. Memory: the stem as float64.
- Sections do not regroup: a section's groups are the stem's groups restricted to the hits whose bar lies in the section
  (the same ids in every section; the count, pattern, level, spread and clean hit are the section's own, `stability`
  is the stem's).
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
ASSIGN_FULL_DB = -24.0   # an overlapped hit whose tail is this far under it or more is compared over its whole cut...
ASSIGN_SHORT_DB = -14.0  # ...one with a tail between this and that over its first ASSIGN_SHORT_S, a louder tail: not at all
ASSIGN_SHORT_S = 0.040
ASSIGN_MARGIN = 1.15     # the nearest group must be this much nearer than the second nearest
GRID_RISE_DB = 6.0       # grid_onsets: a rise of this much (3 ms after against 5 ms before) within GRID_LEAD_S of a 16th line
GRID_FLOOR_DB = 60.0     # ...to a level no further than this under the stem's peak
GRID_QUIET_S = 0.050     # ...at a line with no onset within this
HAT_HZ = 4000.0          # `hat_like`: the share of the stem's energy above this
PITCH_SPLIT_ST = 3.0     # a group whose members' body pitches fall in two clusters further apart than this is split
PITCH_SCREEN = 8         # members whose pitch is read first; a group under half pitched is not read further
STABILITY_RUNS = 2       # regroupings with a shifted sample that `stability` is measured against
TENTATIVE_N = 4          # a group under this many hits is tentative


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


def _pair_distance(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """spectral_distance of two cuts over the shorter of the two, both ended by a fade."""
    n = min(len(a), len(b))
    return oneshot.spectral_distance(_fade(a[:n], sr), _fade(b[:n], sr), sr)


def group_cuts(cuts: list, keys: list, sr: int, threshold: float = THRESHOLD, sample: int = SAMPLE_HITS,
               max_groups: int = MAX_GROUPS, shift: float = 0.0, cache: dict | None = None) -> tuple[list[dict], list[int]]:
    """Group hit cuts ([n, channels] arrays, each from its hit's onset) by how alike they sound: (groups,
    ungrouped). `keys` are the hits' onset times (any numbers, distinct): the hits are put in that order first,
    so the order of `cuts` never changes the answer. Two cuts are compared over the shorter of the two (spectral_distance,
    mel, 60 dB floor). Each round: a deterministic sample of `sample` hits, evenly spaced in time (`shift`, in
    spacings, moves them along: 0 starts at the first hit), is grouped (average linkage, cut at `threshold`); every
    other remaining hit joins the group with the nearest medoid when it is within `threshold` of it; the hits left
    over go round again, until none is left or `max_groups` groups exist (the rest are `ungrouped`, not forced).
    A group is {"members": [indices into cuts], "medoid": index (the sample member with the least mean distance to
    the others), "distance": {index: distance to the medoid}}. `cache` ({(i, j): distance}, i < j) is filled and can
    be passed to a later call on the same cuts."""
    order = sorted(range(len(cuts)), key=lambda i: keys[i])
    seen = {} if cache is None else cache

    def dist(i: int, j: int) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        if (a, b) not in seen:
            seen[(a, b)] = _pair_distance(cuts[a], cuts[b], sr)
        return seen[(a, b)]

    groups, remaining = [], order
    while remaining and len(groups) < max_groups:
        m = len(remaining)
        pick = remaining if m <= sample else [
            remaining[int(round((k + shift) * (m - 1) / (sample - 1 + shift)))] for k in range(sample)]
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


def _hat_like(x: np.ndarray, sr: int) -> float:
    """Share of the stem's energy above HAT_HZ, from 8192-sample blocks (never the whole stem in one FFT)."""
    size = 8192
    freqs = np.fft.rfftfreq(size, 1.0 / sr)
    window = np.hanning(size)
    total = np.zeros(len(freqs))
    for a in range(0, max(1, len(x) - size + 1), size * 128):
        blocks = x[a:a + size * 128]
        m = len(blocks) // size
        if m == 0:
            blocks = np.concatenate([blocks, np.zeros((size - len(blocks), x.shape[1]))])
            m = 1
        mono = blocks[:m * size].mean(axis=1).reshape(m, size)
        total += (np.abs(np.fft.rfft(mono * window, axis=1)) ** 2).sum(axis=0)
    return float(total[freqs >= HAT_HZ].sum() / total.sum()) if total.sum() > 0 else 0.0


def _grid_onsets(x: np.ndarray, sr: int, onsets: list, t0: float, step_s: float) -> list[int]:
    """Onsets (samples) of hits that oneshot._segments cannot see because they start out of a tail: at each 16th
    line from the first onset on, with no onset within GRID_QUIET_S, the rise of the first-difference power (the
    mean over the channels of the squared sample-to-sample difference, which favours hats) from the 5 ms before a
    point to the 3 ms after it (1 ms apart), looked for every 0.5 ms within GRID_LEAD_S of the line; the best point
    counts as a hit when the rise is at least GRID_RISE_DB, the 3 ms after it are no more than GRID_FLOOR_DB under
    the stem's peak, and no onset lies within GRID_QUIET_S of it. Only a window of the stem around each line is
    read."""
    if not len(onsets):
        return []
    on = np.array(sorted(onsets))
    post, pre, gap, q, g, quiet = (max(1, int(round(v * sr))) for v in (0.003, 0.005, 0.001, 0.0005, GRID_LEAD_S, GRID_QUIET_S))
    peak2 = float(np.abs(x).max()) ** 2 * 10 ** (-GRID_FLOOR_DB / 10)
    found = []
    n0 = int(math.ceil((on[0] / sr - t0) / step_s))
    n1 = int(math.floor((len(x) / sr - t0) / step_s))
    for n in range(n0, n1 + 1):
        line = int(round((t0 + n * step_s) * sr))
        if float(np.abs(on - line).min()) < quiet:
            continue
        a, b = line - g - pre - gap - 1, line + g + post + 1
        if a < 0 or b > len(x):
            continue
        w = x[a:b]
        d2 = (np.diff(w, axis=0) ** 2).mean(axis=1)          # d2[k] is the step into sample a + k + 1
        p2 = (w[1:] ** 2).mean(axis=1)
        cd, cp = (np.concatenate([[0.0], np.cumsum(v)]) for v in (d2, p2))
        idx = np.arange(pre + gap, 2 * g + pre + gap + 1, q)   # candidate points, as offsets into w[1:]
        after = (cd[idx + post] - cd[idx]) / post
        before = (cd[idx - gap] - cd[idx - gap - pre]) / pre
        level = (cp[idx + post] - cp[idx]) / post
        ratio = np.where(level > peak2, after / np.maximum(before, 1e-30), 0.0)
        k = int(np.argmax(ratio))
        point = a + 1 + int(idx[k])
        if ratio[k] >= 10 ** (GRID_RISE_DB / 10) and float(np.abs(on - point).min()) >= quiet:
            found.append(point)
    return found


def _body_pitch(cut: np.ndarray, sr: int) -> float | None:
    """oneshot's body pitch of one hit's cut, None when it has none (unpitched, or too short to measure)."""
    try:
        return oneshot.profile_array(cut, sr, hit=0, single=True)["pitch_body_hz"]
    except ValueError:
        return None


def _split_by_pitch(x: np.ndarray, sr: int, hits: list, groups: list, pitch: dict) -> list:
    """Groups with those split whose members' body pitches (oneshot's, on each member's cut) fall in clusters more than
    PITCH_SPLIT_ST apart: the sorted pitches are cut at their largest gap over that, each side keeping at least two
    hits, again within each side. A group is read only when at least half of PITCH_SCREEN members spread over it have a
    pitch. Members without a pitch stay with the largest cluster. Each new group's medoid is the member whose pitch is
    nearest its median (the old medoid, where it stays) and its distances are measured to that."""
    out = []
    for g in groups:
        ids = g["members"]
        n = len(ids)
        screen = ids if n <= PITCH_SCREEN else [ids[int(round(k * (n - 1) / (PITCH_SCREEN - 1)))] for k in range(PITCH_SCREEN)]
        for i in screen:
            if i not in pitch:
                pitch[i] = _body_pitch(x[hits[i]["start"]:hits[i]["end"]], sr)
        if n < 4 or sum(pitch[i] is not None for i in screen) * 2 < len(screen):
            out.append(g)
            continue
        for i in ids:
            if i not in pitch:
                pitch[i] = _body_pitch(x[hits[i]["start"]:hits[i]["end"]], sr)
        clusters = [sorted((12 * math.log2(pitch[i]), i) for i in ids if pitch[i])]
        done = []
        while clusters:
            c = clusters.pop()
            best = max(((c[k + 1][0] - c[k][0], k) for k in range(1, len(c) - 2)), default=None)
            if best is not None and best[0] > PITCH_SPLIT_ST:           # each side keeps at least two hits
                clusters += [c[:best[1] + 1], c[best[1] + 1:]]
            else:
                done.append(c)
        if len(done) < 2:
            out.append(g)
            continue
        done.sort(key=lambda c: (-len(c), c[0][0]))
        unpitched = [i for i in ids if not pitch[i]]
        for j, c in enumerate(done):
            mem = sorted([i for _, i in c] + (unpitched if j == 0 else []))
            if g["medoid"] in mem:
                medoid = g["medoid"]
            else:
                mid = float(np.median([st for st, _ in c]))
                medoid = min((i for _, i in c), key=lambda i: (abs(12 * math.log2(pitch[i]) - mid), i))
            mc = x[hits[medoid]["start"]:hits[medoid]["end"]]
            out.append({"members": mem, "medoid": medoid, "distance": {
                i: (g["distance"][i] if medoid == g["medoid"] else _pair_distance(x[hits[i]["start"]:hits[i]["end"]], mc, sr))
                for i in mem}})
    return out


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


def _scope(x: np.ndarray, sr: int, hits: list, found: list, hidden: set, ids: set, first: int, bars: int, steps: int) -> dict:
    """The groups of `found` (hit ids; see hit_groups) restricted to the hits in `ids`, over the `bars` bars from bar
    `first`. Ordered by count, then by the first member's time; "id" is the group's number in the whole stem.
    Assigned hits (overlapped hits placed in a group) count in `pattern`, `bars_with`, `off_grid`; not in `count`,
    `level_db`, `spread` or the clean hit."""
    groups = []
    for gid, g in found:
        mine = [i for i in g["members"] if i in ids]      # in time order
        extra = [i for i in g["assigned"] if i in ids]
        if not mine and not extra:
            continue
        on = [hits[i] for i in mine + extra if hits[i]["on_grid"]]
        on_assigned = [hits[i] for i in extra if hits[i]["on_grid"]]
        clean, why = _clean_hit(x, sr, hits, mine, g["distance"]) if mine else (None, "no measured member in this scope")
        dists = [g["distance"][i] for i in mine]
        groups.append({
            "id": gid, "count": len(mine), "assigned": len(extra), "tentative": len(mine) < TENTATIVE_N,
            "stability": g["stability"], "bars_with": len({hits[i]["bar"] for i in mine + extra}),
            "pattern": [len({h["bar"] for h in on if h["step"] == s}) / bars if bars else 0.0 for s in range(steps)],
            "assigned_pattern": [len({h["bar"] for h in on_assigned if h["step"] == s}) / bars if bars else 0.0
                                 for s in range(steps)],
            "level_db": float(np.median([hits[i]["level_db"] for i in mine])) if mine else None,
            "spread": {"median": float(np.median(dists)), "max": float(max(dists))} if mine else None,
            "off_grid": len(mine) + len(extra) - len(on), "members": mine, "assigned_members": extra,
            "clean_hit": clean, "clean_note": why})
    groups.sort(key=lambda g: (-g["count"], (g["members"] + g["assigned_members"])[0]))
    lost = [h for h in (hits[i] for i in ids) if h["on_grid"] and first <= h["bar"] < first + bars and h["status"] in (
        "overlapped", "too_short", "no_lead", "ungrouped", "faint")]
    pat = lambda hs: [len({h["bar"] for h in hs if h["step"] == s}) / bars if bars else 0.0 for s in range(steps)]  # noqa: E731
    return {"hits": len(ids), "bars": bars, "groups": groups, "unmeasured_pattern": pat(lost),
            "hidden_pattern": [len({b for b in range(first, first + bars) if (b, s) in hidden}) / bars if bars else 0.0
                               for s in range(steps)],
            "unmeasured": [{"hit": i, "status": hits[i]["status"], "reason": hits[i]["reason"]}
                           for i in sorted(ids) if hits[i]["status"] not in ("grouped", "assigned")]}


def _assign_overlapped(x: np.ndarray, sr: int, hits: list, found: list, threshold: float) -> None:
    """Place overlapped hits in groups that already exist, never forming or moving one. A hit whose tail is
    ASSIGN_FULL_DB or more under it is compared with each group's medoid over its whole cut, one whose tail is between
    that and ASSIGN_SHORT_DB over its first ASSIGN_SHORT_S (the tail is lost under a decaying hit's body in the first
    tens of ms and keeps growing against it after), a louder tail: not at all. The hit goes to the nearest medoid when
    that is within `threshold` and ASSIGN_MARGIN times nearer than the second nearest; its status becomes "assigned"."""
    for h in hits:
        if h["status"] != "overlapped" or h["lead_db"] is None:
            continue
        if h["lead_db"] > ASSIGN_SHORT_DB:
            h["reason"] += "; the tail is too loud to compare it with a group"
            continue
        window = None if h["lead_db"] <= ASSIGN_FULL_DB else int(round(ASSIGN_SHORT_S * sr))
        cut = x[h["start"]:h["end"]]
        near = []
        for gid, g in found:
            m = hits[g["medoid"]]
            mc = x[m["start"]:m["end"]]
            n = min(len(cut), len(mc)) if window is None else min(len(cut), len(mc), window)
            near.append((oneshot.spectral_distance(_fade(cut[:n], sr), _fade(mc[:n], sr), sr), gid, n))
        near.sort()
        if not near:
            continue
        if near[0][0] > threshold:
            h["reason"] += "; not assigned: the nearest group is %.2f away (over %.2f)" % (near[0][0], threshold)
        elif len(near) > 1 and near[1][0] < ASSIGN_MARGIN * near[0][0]:
            h["reason"] += "; not assigned: groups %d and %d are %.2f and %.2f away (not %.0f %% apart)" % (
                near[0][1], near[1][1], near[0][0], near[1][0], 100 * (ASSIGN_MARGIN - 1))
        else:
            d, gid, n = near[0]
            h["status"], h["group"], h["distance"] = "assigned", gid, d
            h["reason"] = "overlapped (%s); assigned to group %d over its first %.0f ms (distance %.2f%s)" % (
                h["reason"].split(";")[0].removeprefix("it starts out of "), gid, 1000 * n / sr, d,
                "" if len(near) == 1 else ", the next group %.2f" % near[1][0])
            dict(found)[gid]["assigned"].append(h["id"])


def hit_groups(stem, tempo: float, t0: float, sections: list | None = None, sr: int | None = None,
               beats_per_bar: int = 4, threshold: float = THRESHOLD, max_cut_s: float = MAX_CUT_S,
               sample: int = SAMPLE_HITS, max_groups: int = MAX_GROUPS, grid_onsets: bool = False,
               stability_runs: int = STABILITY_RUNS) -> dict:
    """Every hit in one drum stem, with its place on the grid, grouped by how alike the hits sound (see the module
    docstring for the choices). `stem` is a path or an array [n] / [n, channels] with its rate in `sr`; bar 0 starts
    at `t0` seconds, 16ths are a quarter of a beat of `tempo` (4 * beats_per_bar to a bar); `sections` are
    {name, start_bar, bars} (as reference_profile takes them). `grid_onsets` (off by default) also looks for hits at
    16th lines that `oneshot._segments` cannot see, and is meant for hat-like stems (see `hat_like`): on a kick, snare or
    toms stem it reports bleed and tails as hits (118-355 of them on the Flare stems). `stability_runs` is the number of
    regroupings with a shifted sample that `stability` is measured against (0: not measured).

    Returns {"sample_rate", "tempo", "t0", "threshold", "linkage": "average", "max_cut_s", "grid_onsets", "hat_like"
    (the share of the stem's energy above 4 kHz), "grid_hits" (hits found by `grid_onsets`), "hits", "groups",
    "unmeasured_pattern", "hidden_pattern", "unmeasured", "sections", "notes"}.
    A hit is {"id", "source" ("segment" or "grid"), "onset_s", "bar", "step" (16th, 0-15, of the nearest grid line),
    "offset_ms" (onset minus that line), "on_grid" (within 20 ms of it), "level_db" (loudest ms, dBFS), "lead_db" (the
    10 ms before the hit against its loudest ms; None at the start of the stem), "through" (16th lines inside its cut
    that its own tail still sounds through), "start", "end" (cut, in samples of the stem), "room" (samples to the
    next hit), "status" ("grouped", "assigned" (overlapped, placed in an existing group), "too_short", "no_lead",
    "faint", "overlapped", "silent" or "ungrouped"), "reason" (why not grouped), "group", "distance" (to its group's
    medoid, over the window that was compared)}.
    A group is {"id", "count" (measured members), "assigned" (overlapped hits placed in it), "tentative" (under 4 measured
    members), "stability" (see below), "bars_with" (bars it plays in), "pattern" (per 16th, the share of the scope's
    bars it plays at, on-grid hits, assigned ones included), "assigned_pattern" (the same for the assigned hits alone),
    "level_db" (median), "spread" ({"median", "max"} of the members' distances to the medoid), "off_grid", "members"
    (hit ids), "assigned_members", "clean_hit" ({"hit", "start", "end", "onset_s", "level_db", "audio"} or None; never
    an assigned hit), "clean_note" (why None)}.
    `stability` is how the group holds when the sample changes: the stem is grouped again `stability_runs` times
    with the sample moved by 1/3 and 2/3 of a spacing (same threshold, same pitch split); for each run the Jaccard index
    (members in both / members in either) of the group with the group of that run it shares most members with, and
    the mean over the runs. 1.0: the same members every time; a group that a run splits or merges with another reads
    lower. Material whose distances lie around the threshold (a continuum of decays, a velocity gradient) is cut
    where the sample puts the cut, and reads well under 1; the figure is for the stem as a whole, section groups carry it
    unchanged.
    `groups` is the whole stem (its bars run from the first real hit's bar to the last's); each of `sections` has the same
    groups restricted to its bars, with the section's own count, pattern over its `bars`, level, spread and clean hit.
    `unmeasured_pattern` (per 16th, the share of the scope's bars with an on-grid hit there that is overlapped and not
    assigned, too short, without a lead, faint or ungrouped) leaves out the hits that were never found: a hit inside another's
    tail is not a hit to the detector. `hidden_pattern` (per 16th, the share of the scope's bars in which that line falls
    inside the tail of a hit, `through`, with no hit found at it) is where such hits could be; with `grid_onsets` on it
    is what is left after the search."""
    if not (tempo > 0 and math.isfinite(tempo)) or not math.isfinite(t0):
        raise ValueError("tempo must be above 0 and t0 a number of seconds, got %r and %r" % (tempo, t0))
    if beats_per_bar < 1 or sample < 2 or max_groups < 1 or threshold <= 0 or stability_runs < 0:
        raise ValueError("beats_per_bar and max_groups need at least 1, sample at least 2, threshold above 0, "
                         "stability_runs 0 or more")
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
           "max_cut_s": max_cut_s, "grid_onsets": grid_onsets, "hat_like": None, "grid_hits": 0, "hits": [],
           "groups": [], "unmeasured_pattern": [], "hidden_pattern": [], "unmeasured": [], "sections": [], "notes": []}
    if not len(x) or float(np.abs(x).max()) <= 0:
        out["notes"].append("the stem is silent: no hits")
        return out
    out["hat_like"] = round(_hat_like(x, sr), 4)
    segs = oneshot._segments(x, sr)
    onsets = [(oneshot._hit_onset(x[s:e], sr) if s == 0 else _onset_after_tail(x, sr, s), "segment") for s, e in segs]
    step_s, steps = 60.0 / tempo / 4.0, 4 * beats_per_bar
    if grid_onsets:
        onsets = sorted(onsets + [(o, "grid") for o in _grid_onsets(x, sr, [o for o, _ in onsets], t0, step_s)])
        out["grid_hits"] = sum(src == "grid" for _, src in onsets)
    guard, lead_end, lead_len = (int(round(v * sr)) for v in (END_GUARD_S, 0.001, LEAD_WINDOW_S))
    hits, cuts, keys, tails = [], [], [], {}
    frame, glead = max(1, int(round(0.001 * sr))), int(round(GRID_LEAD_S * sr))
    for k, (o, source) in enumerate(onsets):
        last = k + 1 == len(onsets)
        nxt = len(x) if last else onsets[k + 1][0]
        end = min(nxt if last else nxt - guard, o + int(round(max_cut_s * sr)))
        rel = o / sr - t0
        line = int(round(rel / step_s))
        bar, step = divmod(line, steps)
        off = rel - line * step_s
        h = {"id": k, "source": source, "onset_s": o / sr, "bar": bar, "step": step, "offset_ms": round(1000 * off, 2),
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
        for n in range(line + 1, line + steps + 1):
            i = int(round((t0 + n * step_s) * sr)) - o
            if i + frame > len(cut) - glead:
                break
            if i > glead and float((cut[i:i + frame] ** 2).mean()) >= loud * 10 ** (-THROUGH_DB / 10):
                h["through"] += 1
                tails.setdefault(k, []).append(n)
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
            h["status"], h["reason"] = "faint", (
                "%.0f dB under the hits around it (over %.0f dB): bed, bleed or a ghost note; it is counted in "
                "`unmeasured_pattern`" % (float(np.median(near)) - h["level_db"], FAINT_DB))
        else:
            h["_cut"] = len(cuts)
            cuts.append(x[h["start"]:h["end"]])
            keys.append(h["start"])
    index = {h["_cut"]: h["id"] for h in hits if "_cut" in h}
    seen, pitch = {}, {}

    def cluster(shift: float):
        groups, rest = group_cuts(cuts, keys, sr, threshold, sample, max_groups, shift, seen)
        for g in groups:                                  # from cut numbers to hit ids
            g["members"] = sorted(index[i] for i in g["members"])
            g["medoid"] = index[g["medoid"]]
            g["distance"] = {index[i]: d for i, d in g["distance"].items()}
        return _split_by_pitch(x, sr, hits, groups, pitch), [index[c] for c in rest]

    groups, rest = cluster(0.0) if cuts else ([], [])
    runs = [cluster((r + 1) / (stability_runs + 1))[0] for r in range(stability_runs)] if cuts else []
    groups.sort(key=lambda g: (-len(g["members"]), g["members"][0]))
    found = list(enumerate(groups))
    for gid, g in found:
        g["assigned"] = []
        shares = []
        for other in runs:
            best = max(other, key=lambda o: len(set(o["members"]) & set(g["members"])))
            both = len(set(best["members"]) & set(g["members"]))
            shares.append(both / (len(g["members"]) + len(best["members"]) - both))
        g["stability"] = round(float(np.mean(shares)), 3) if shares else None
        for i, d in g["distance"].items():
            hits[i]["group"], hits[i]["distance"] = gid, d
    for i in rest:
        hits[i]["status"], hits[i]["reason"] = "ungrouped", "more than %d distinct sounds: not forced into a group" % max_groups
    _assign_overlapped(x, sr, hits, found, threshold)
    for h in hits:
        h.pop("_cut", None)
    out["hits"] = hits
    if not hits:
        return out
    hidden = {divmod(n, steps) for ns in tails.values() for n in ns}      # a tail's cut ends at the next hit: no line with a hit
    real = [h["bar"] for h in hits if h["status"] not in ("faint", "silent", "no_lead")]
    first, last = (min(real), max(real)) if real else (0, 0)
    whole = _scope(x, sr, hits, found, hidden, {h["id"] for h in hits}, first, last - first + 1, steps)
    out.update({k: whole[k] for k in ("groups", "unmeasured_pattern", "hidden_pattern", "unmeasured")})
    for sec in sections or []:
        a, b = sec["start_bar"], sec["start_bar"] + sec["bars"]
        got = _scope(x, sr, hits, found, hidden, {h["id"] for h in hits if a <= h["bar"] < b}, a, sec["bars"], steps)
        out["sections"].append({"name": sec.get("name"), "start_bar": a, "bars": sec["bars"], **got})
    if any(h["through"] for h in hits):
        out["notes"].append("a hit played inside another hit's tail, without rising 15 dB out of it, is not found "
                            "(grid_onsets=True looks for it at the 16th lines): `through` lists the 16th lines each "
                            "tail still sounds through, `hidden_pattern` where they fall")
    return out
