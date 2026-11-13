"""Match a synth patch to a one-shot by search, against any renderer.

`search` minimises `oneshot.spectral_distance(candidate, target)` over a patch's free parameters. It knows
nothing about what renders: `render(patches)` returns one hit per patch, and in practice that will be Ableton
Live recording a batch of candidates in one take, so renders are expensive and come in batches. Numpy only.

The method is a (mu, lambda) evolution strategy with weighted recombination and a step size per parameter:
a round is one batch of `population` renders and there are only about eight rounds, which suits a small
sample-and-select method with no gradient and no model to fit (a full covariance, as CMA-ES keeps, cannot be
estimated from so few samples); steps adapt per parameter by whether the selected moves agree in direction.
The search is judged by the score it reaches, not by the parameters it returns: parameters that trade off against
each other (a noise layer's decay and level) are not individually recovered. In `operator_guess`, a noise layer
whose reading is weak (under NOISE_FREE_DB) or lacks evidence (the gate failed) has both its level and its decay
free, so the two can trade off along the ridge a longer, quieter layer shares with a shorter, louder one, and the
search cannot separate them; only a strong layer with evidence has its level pinned.
Choices that are not obvious from the code:

- A patch is a dict of name -> float. `bounds` maps a name to (low, high) or (low, high, "log"); times and
  frequencies are searched on a log scale (a step is a ratio), the rest linearly. A name in `start` but not in
  `bounds`, or with low == high, is pinned: it is rendered as given and never changes. A narrow range around a
  measured value lets a parameter move a little: the distance guides the pitch of a low drum only within about
  half a semitone and is flat to about two (docs/CONVENTIONS.md, "Drum one-shots"), so the caller bounds the
  pitch near its measurement. Steps never exceed half a parameter's range, and candidates are reflected back
  into it, so a bound is never left.
- Round 0 renders the start plus `population` - 1 candidates around it. Every later round renders the best
  patch so far (the incumbent) again, plus `population` - 1 new candidates around the weighted mean of the best
  quarter of the round before (the incumbent counted among them, so an unproductive round does not walk away
  from it). Samples come in mirrored pairs (z and -z): fewer random directions are wasted on a small batch.
- The incumbent is the patch with the lowest mean score over all the times it was rendered, and a challenger
  takes over only by beating it by `MARGIN` repeat spreads. A candidate has one score, and the best of 15 noisy
  scores is partly luck; it is re-rendered next round, and if its luck was noise it does not stay. The margin
  works both ways: an incumbent up to one spread worse than a challenger keeps its place, so a better newcomer
  can wait a round. The repeat scores also measure the renderer's own noise: `repeat_spread` is the standard
  deviation of one patch's score over its repeats, pooled over every patch rendered more than once (in round 1
  it rests on one pair; to be re-baselined when real takes are measured). A noise-like hit never repeats
  exactly (two takes of one hat read 0.5 to 0.6 from each other), so the best distance on such a target has a
  floor that is not zero, and a gain smaller than `repeat_spread` is not a gain.
- The search stops early, and `stalled` is set, when the best distance has improved by no more than `repeat_spread`
  over the last `patience` rounds (4: with 2, runs stopped after 5 or 6 rounds and read all three of decay,
  noise decay and noise level right in 42 to 50 % of kick-like and tom-like runs, against 67 to 92 % unstopped).
  `stalled` means the stop rule fired; it never fires in the last round, where stopping changes nothing.
- A candidate the renderer fails on (an array that is silent, non-finite, shorter than `min_ms`, or that
  `spectral_distance` refuses) is not scored, so it is never the best nor a parent of the next batch (worst
  of the batch), and it is listed in `failures`; the search goes on. A failure of the incumbent's own render
  is listed and left out of its mean. An exception from `render` stops the search: the result so far is
  returned with the exception in `error` (and a whole-batch entry, index None, in `failures`); when nothing
  was scored yet it is raised.
- Results are deterministic for a given `seed` and a renderer that is.

What the distance can and cannot tell the search (measured on the numpy stand-in of tests/test_patch.py: a
tone with a pitch glide and a noise layer, a start 30 % off, 20 seeds, 8 rounds of 16; numbers in the item-3 report):

- The tone's decay is recovered to within 15 % in every run of the kick-like and tom-like targets, and the
  pitch to within 0.3 semitone in 45 to 85 % (its bound is +-0.5). The noise layer is the weak part: its decay
  within 15 % in 55 to 80 % of the runs on those targets (a layer 10 to 12 dB under the tone: 15 % on its
  decay moves the score by about 0.01, one repeat spread), its level within 3 dB in 65 to 95 %. In the
  failures the two trade off (a longer layer, quieter). The distance a patch keeps is the mean of its renders,
  one of which won its selection (a bias of about 0.003 on the kick-like and tom-like targets, 0.02 on the hat);
  re-rendered with noise it never saw, the final patch still scores 0.036 above the true parameters on the
  kick-like and tom-like targets (3 to 4 repeat spreads), and the gap is real. It closes at least 62 % of the way
  from the start's distance to the truth's in every run, 74 to 87 % at the median on those targets (hat: 61 %,
  74 to 94 %; report, "Score reached").
  With 40 rounds instead of 8 the same targets reach 88 to 100 %, so this is the budget against the noise of
  the score, not a parameter the score cannot see.
- A tone masked by noise as loud as itself (the hat-like target, tone at 7 kHz) is one the score cannot see in
  8 rounds: 30 % off in its decay moves the distance by under 0.02, and it is recovered in 35 to 45 % of the
  runs (25 % at 40 rounds with a fresh noise draw per render, 88 % when the renders repeat exactly).
- Narrowing the glide's bounds to a factor of 1.5 around the truth did not help consistently (the same
  tolerances, kick 50 % of runs, tom 80 %, against 70 and 45 %).
"""
from __future__ import annotations

import functools
import json
import math
import numbers
import pathlib
import re
from dataclasses import dataclass, field

import numpy as np

from . import oneshot
from .listen import BANDS
from .oneshot import DIST_FLOOR_DB, spectral_distance

STEP_LOG = 0.15          # first step of a log-scale parameter: 15 % (ln units)
STEP_LIN = 0.05          # first step of a linear parameter: this share of its range
MARGIN = 1.0            # a candidate takes over as incumbent by beating it by this many repeat spreads
MU_FRAC = 0.25          # the best quarter of a batch is recombined into the next mean
STEP_MIN = 1e-3          # a step never falls under this share of the range...
STEP_MAX = 0.5           # ...nor rises over this share
STEP_DAMP = 0.5          # how strongly a step follows agreement of the selected moves
STEP_RATE = 2.0          # a step changes at most by this factor a round
MIN_MS = 10.0            # a render shorter than this failed
EXPECTED_ABS_Z = math.sqrt(2.0 / math.pi)     # E|N(0, 1)|


@dataclass
class SearchResult:
    best: dict                  # the best patch (every name of `start`), by mean score over its renders
    distance: float             # its mean distance to the target
    start_distance: float       # the start's distance, in round 0 (inf when its render failed)
    per_round: list             # the best distance after each round
    rounds: int                 # rounds run (render batches)
    renders: int                # patches rendered in all (rounds * population)
    repeat_spread: float | None  # std of one patch's score over its repeat renders; None before a repeat
    stalled: bool               # the stop rule fired: the last `patience` rounds gained no more than repeat_spread
    failures: list = field(default_factory=list)   # {"round", "index", "reason"} for each failed render
    steps: dict = field(default_factory=dict)      # the final step of each searched parameter (log: ln units)
    evaluated: list = field(default_factory=list)  # (patch, score) of every render that scored, in order
    error: str | None = None    # the exception that stopped the search (render raised), after the rounds above


class _Space:
    """The searched parameters in the space the steps are taken in (ln for log scale, plain for linear)."""

    def __init__(self, start: dict, bounds: dict):
        for name in bounds:
            if name not in start:
                raise ValueError("bounds names %r, which the start patch does not have" % name)
        self.names, self.log, lo, hi = [], [], [], []
        for name, b in bounds.items():
            if len(b) not in (2, 3) or (len(b) == 3 and b[2] not in ("log", "lin")):
                raise ValueError("bounds[%r] must be (low, high) or (low, high, \"log\"|\"lin\"), got %r" % (name, b))
            low, high, log = float(b[0]), float(b[1]), len(b) == 3 and b[2] == "log"
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError("bounds[%r] = %r is not a finite range low <= high" % (name, b))
            if log and low <= 0:
                raise ValueError("bounds[%r] is on a log scale, so its low end must be positive" % name)
            if not low <= start[name] <= high:
                raise ValueError("start[%r] = %r is outside its bounds (%r, %r)" % (name, start[name], low, high))
            if high > low:
                self.names.append(name)
                self.log.append(log)
                lo.append(math.log(low) if log else low)
                hi.append(math.log(high) if log else high)
        self.lo, self.hi = np.array(lo), np.array(hi)
        self.span = self.hi - self.lo

    def to_u(self, patch: dict) -> np.ndarray:
        return np.array([math.log(patch[n]) if lg else patch[n] for n, lg in zip(self.names, self.log)], dtype=float)

    def to_patch(self, u: np.ndarray, base: dict) -> dict:
        out = dict(base)
        for n, lg, v in zip(self.names, self.log, u):
            out[n] = math.exp(v) if lg else float(v)
        return out

    def reflect(self, u: np.ndarray) -> np.ndarray:
        """u folded back into [lo, hi] (a triangle wave, so a step past a bound comes back by the same distance)."""
        w = (u - self.lo) % (2 * self.span)
        return self.lo + np.where(w > self.span, 2 * self.span - w, w)


def _score(out, target: np.ndarray, sr: int, min_ms: float, kw: dict) -> tuple[float | None, str | None]:
    """(distance, None) for a good render, (None, why) for one that failed."""
    try:
        x = np.asarray(out, dtype=np.float64)
    except (TypeError, ValueError):
        return None, "not an array of samples"
    if x.ndim not in (1, 2):
        return None, "wrong shape (%d dimensions, expected 1 or 2)" % x.ndim
    if not x.size:
        return None, "empty"
    if len(x) < min_ms * sr / 1000.0:
        return None, "too short (%.1f ms)" % (1000.0 * len(x) / sr)
    if not np.isfinite(x).all():
        return None, "NaN or infinite samples"
    if np.abs(x).max() <= 0:
        return None, "silent"
    try:
        d = spectral_distance(x, target, sr, **kw)
    except ValueError as e:
        return None, str(e)
    return (d, None) if math.isfinite(d) else (None, "distance is not finite")


def _mean(scores: list) -> float:
    return sum(scores) / len(scores) if scores else math.inf


def _spread(entries: list) -> float | None:
    """Pooled std of the scores of every patch rendered more than once (None when none was)."""
    ss = [(float(np.var(e["scores"], ddof=1)), len(e["scores"]) - 1) for e in entries if len(e["scores"]) > 1]
    if not ss:
        return None
    return math.sqrt(sum(v * k for v, k in ss) / sum(k for _, k in ss))


def search(start: dict, bounds: dict, render, target: np.ndarray, sr: int, rounds: int = 8, population: int = 16,
           seed: int = 0, step: dict | None = None, patience: int = 4, min_ms: float = MIN_MS,
           floor_db: float | None = DIST_FLOOR_DB, scale: str = "mel", match_level: bool = True,
           probes: list | None = None) -> SearchResult:
    """Refine the patch `start` towards `target` ([n] or [n, channels] at `sr`) for at most `rounds` rounds of
    `population` renders each (see the module docstring for the method).

    `bounds`: name -> (low, high) or (low, high, "log"); names of `start` that are not in it, or whose
    low == high, are pinned. `render(patches: list[dict]) -> list[array]` is called once per round with exactly
    `population` patches (all names of `start`, pinned ones included) and must return that many hits at `sr`.
    `step` overrides a parameter's first step (ln units for a log parameter, else the parameter's own); it must
    name searched parameters and be finite and above 0, else ValueError.
    `probes`: patches rendered once in round 0, in place of the last random candidates (at most population - 1), as
    candidates like the others: a place the random steps would not reach in time, for example a free noise layer at
    its lower bound ("off"), which a start at -15 dB and a step of 3 dB leave only slowly (see search_arguments).
    A probe must have every name of `start` and lie inside the bounds (pinned names are taken from `start`).
    `floor_db`, `scale` and `match_level` go to `oneshot.spectral_distance`. Raises ValueError for a start
    outside its bounds, a population under 2, a bad `step`, a bad probe, or a render that returns the wrong number of hits;
    RuntimeError when no render ever scored. An exception from `render` ends the search with the rounds before
    it: the result carries it in `error` (and `failures` a whole-batch entry with index None); when nothing had
    scored yet it is raised."""
    if population < 2 or rounds < 1:
        raise ValueError("population must be at least 2 and rounds at least 1")
    space = _Space(start, bounds)
    kw = {"floor_db": floor_db, "scale": scale, "match_level": match_level}
    rng = np.random.default_rng(seed)
    n = len(space.names)
    first = step or {}
    for nm, v in first.items():
        if nm not in space.names:
            raise ValueError("step names %r, which is not a searched parameter" % nm)
        if not (isinstance(v, numbers.Real) and math.isfinite(v) and v > 0):
            raise ValueError("step[%r] = %r must be a finite number above 0" % (nm, v))
    sigma = np.array([min(first.get(nm, STEP_LOG if lg else STEP_LIN * sp), STEP_MAX * sp)
                      for nm, lg, sp in zip(space.names, space.log, space.span)], dtype=float)
    probe_u = []
    if probes and len(probes) > population - 1:
        raise ValueError("at most population - 1 = %d probes, got %d" % (population - 1, len(probes)))
    for pr in probes or ():
        miss = [nm for nm in start if nm not in pr]
        if miss:
            raise ValueError("a probe lacks %s" % miss)
        u = space.to_u(pr)
        tol = 1e-9 * (1.0 + np.abs(space.span))
        if np.any(u < space.lo - tol) or np.any(u > space.hi + tol):
            raise ValueError("a probe lies outside the bounds: %r" % (pr,))
        probe_u.append(np.clip(u, space.lo, space.hi))
    mu = max(1, int(population * MU_FRAC))
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()

    best = {"patch": dict(start), "u": space.to_u(start), "scores": []}
    entries = [best]
    mean = best["u"].copy()
    failures, evaluated, per_round, renders, stalled, start_distance = [], [], [], 0, False, math.inf
    error = None

    for r in range(rounds):
        # the incumbent first, then population - 1 candidates around the mean, in mirrored pairs
        z = rng.standard_normal((population // 2, n))
        z = np.concatenate([z, -z])[:population - 1]
        cands = []
        for zi in z:
            u = space.reflect(mean + sigma * zi)
            cands.append({"patch": space.to_patch(u, start), "u": u, "scores": []})
        if r == 0 and probe_u:
            cands[len(cands) - len(probe_u):] = [{"patch": space.to_patch(u, start), "u": u, "scores": []} for u in probe_u]
        incumbent = best
        try:
            out = render([dict(best["patch"])] + [dict(c["patch"]) for c in cands])
        except Exception as e:
            if not any(en["scores"] for en in entries):
                raise
            error = "%s: %s" % (type(e).__name__, e)
            failures.append({"round": r, "index": None, "reason": "render raised " + error})
            break
        renders += population
        if len(out) != population:
            raise ValueError("render returned %d hits for %d patches" % (len(out), population))
        scored = [_score(o, target, sr, min_ms, kw) for o in out]
        failures += [{"round": r, "index": i, "reason": why} for i, (d, why) in enumerate(scored) if why]
        if scored[0][0] is not None:
            incumbent["scores"].append(scored[0][0])
            if r == 0:
                start_distance = scored[0][0]
        evaluated += [(dict(p), d) for p, (d, _) in zip([incumbent["patch"]] + [c["patch"] for c in cands], scored)
                      if d is not None]
        for c, (d, _) in zip(cands, scored[1:]):
            if d is not None:
                c["scores"].append(d)
        entries += cands
        leader = min((e for e in entries if e["scores"]), key=lambda e: _mean(e["scores"]), default=best)
        spread = _spread(entries)
        if leader is not best and (not best["scores"] or _mean(leader["scores"]) < _mean(best["scores"]) - MARGIN * (spread or 0.0)):
            best = leader
        per_round.append(_mean(best["scores"]))

        # recombine the best quarter of the batch (candidates and the incumbent) into the next mean; adapt the steps
        # from the candidates alone: selected moves that agree in direction grow a step, ones that scatter shrink it
        good = sorted((c for c in cands if c["scores"]), key=lambda c: c["scores"][0])[:mu]
        pool = sorted([(c["scores"][0], c) for c in good] + ([(_mean(incumbent["scores"]), incumbent)]
                      if incumbent["scores"] else []), key=lambda p: p[0])[:mu]
        if pool and n:
            wp = w[:len(pool)] / w[:len(pool)].sum()
            new_mean = sum(wi * e["u"] for wi, (_, e) in zip(wp, pool))
            if good:
                g = w[:len(good)] / w[:len(good)].sum()
                zbar = sum(gi * (c["u"] - mean) / sigma for gi, c in zip(g, good))
                mu_eff = 1.0 / float((g ** 2).sum())
                factor = np.exp(STEP_DAMP * (np.abs(zbar) * math.sqrt(mu_eff) - EXPECTED_ABS_Z))
                sigma = np.clip(sigma * np.clip(factor, 1 / STEP_RATE, STEP_RATE),
                                STEP_MIN * space.span, STEP_MAX * space.span)
            mean = space.reflect(new_mean)

        # the spread rests on a single pair in round 1 and the stop rule on it: both to be re-baselined on real takes
        if spread is not None and patience <= r < rounds - 1 and per_round[r - patience] - per_round[r] <= spread:
            stalled = True
            break

    if not best["scores"]:
        raise RuntimeError("no candidate could be scored; failures: %r" % failures[:5])
    return SearchResult(best=dict(best["patch"]), distance=_mean(best["scores"]), start_distance=start_distance,
                        per_round=per_round, rounds=len(per_round), renders=renders, repeat_spread=_spread(entries),
                        stalled=stalled, failures=failures, evaluated=evaluated, error=error,
                        steps=dict(zip(space.names, map(float, sigma))))


# ------------------------------------------------------------------------------------------ the first guess
#
# operator_guess: a one-shot's profile -> a physical patch for ONE Operator voice (a sine carrier on Osc A with
# a pitch envelope, a noise layer on Osc B, an amp envelope each). to_live / from_live: physical values <-> Live's
# parameter values through a units table read from Live (not in this repository yet). Three layers, so each
# can be tested alone and no unit is guessed silently:
#   profile --operator_guess--> patch (semantic, physical units: Hz, ms, dB, semitones)
#   patch --operator_live--> every Live parameter the voice owns, as a physical value (the formulas below)
#   that --to_live(units)--> Live parameter name -> Live's value

FALL20_PER_T = 0.227     # Operator Ae Decay T: the amp envelope of a device from insert_device falls 20 dB in 0.227 T
#                          (Live 12.4.6, 2026-11-12: Ae Decay 300 ms read 20 dB in 68.3 ms, 1000 ms in 226.6 ms: 0.2277
#                          and 0.2266; exponential, linear in dB; noise, Be Decay 500 ms: 0.216; live-assumptions report).
#                          The 0.29 of 2026-11-09 and 11 was measured on a kit built through the preset XML with
#                          DecaySlope 1; the slope is not a LOM parameter, so the device a LOM session gets has its own.
FALL20_PER_T_NOISE = 0.216   # the same for Be Decay (the noise's amp envelope): ONE measurement, 2026-11-12: Be Decay 500 ms read a
#                          20 dB fall of 108 ms. The tone's 0.227 has two (300 and 1000 ms); the noise decays use this one.
TAU_PER_FALL20 = 1.0 / math.log(10.0)       # tau = fall_20 / ln(10), for an exponential amplitude decay
T_PER_TAU = math.log(10.0) / FALL20_PER_T   # Operator T = 10.1 tau: the amp curve is 88 dB down at T
PE_DECAY_PER_TAU = 7.94  # Pe Decay = tau * this, measured 2026-11-12 on a scratch device and NOT the amp envelope's
#                          constant (10.1): Pe Init = Peak 12 st, Pe Decay 400 ms read 11.0 / 8.5 / 5.85 / 2.6 / 1.69 /
#                          0.56 / 0.18 st at 5 / 20 / 40 / 80 / 100 / 150 / 200 ms, an exponential of tau 58 ms early and
#                          49 ms late (ratio 6.9 to 8.2; 7.0 to 8.4 over the takes); the glide is steeper than one exponential
#                          late and reaches 0 at T. 7.94 sits inside the range the search's glide factor (4) covers.
ATTACK_FLOOR_MS = 1.5    # a tone attack of 0.1 ms clicks (measured 2026-11-11): never less
NOISE_ATTACK_FLOOR_MS = 0.1   # noise has no tone to click: the hand-built hats and claps use 0.1 ms
CLICK_MS = 10.0          # a pitch drop over in less than this is a click, not a glide. oneshot's own bound is its
#                          FAST_SETTLE_MS (5 ms); Tom_BOAVISTA's 17 st in 8 ms is a click too, so the guess uses 10
TAIL_MS = 100.0          # track readings from here on, when they agree, are the carrier the glide settled on
PITCH_BOUND_ST = 0.5     # the carrier may move this far either way in the search
PE_MAX_ST = 48.0         # a pitch drop is capped at this (Operator's Pe range is -48 to 48 st: CHECKED)
MIN_CARRIER_HZ = 8.2     # MIDI note 0 (8.18 Hz): lower carriers cannot be played with Coarse 1
NOISE_REF_MS = 15.0      # fallback reference time of the attack noise reading (a profile without attack_window_ms)
NOISE_BODY_FRAC = 0.36   # where the body reading's weight sits, as a share into body_window_ms: 80 ms in 40-150
#                          (calibrated on the stand-in's kick and tom, 79 and 86 ms; to re-check on takes)
ONSET_CLAMP_DB = 12.0    # the extrapolation from the reference time back to the onset is at most this many dB
HF_MARGIN_DB = 5.0       # noise counts only when the attack's power above 2 kHz (click_db) is within this of its level...
FLAT_ATTACK = 0.001      # ...and the attack's spectrum has some flat part (flatness.attack: a tone and a click read 0.0, 3 decimals)
FLAT_BODY = 0.02         # the tail rule also needs a flat body spectrum (flatness.body: a noise snare 0.07, a clave 0.001)
NOISE_FREE_DB = -20.0    # a noise reading under this is near the leakage floor: its level is searched, not pinned
NOISE_OFF_DB = -60.0     # ...down to this, which is off for a voice
BODY_NOISE_DB = -9.0     # the tail rule needs a body at least this noisy (12 % of its energy): a tonal tail is not noise
DEFAULT_NOISE_FRACTION = 0.2    # with no usable body reading the noise decays in a fifth of the tone's time
FILTER_CLEAR_DB = 10.0   # a band within this of the densest is "clear" (the noise reaches down to it)
TAIL_RATIO = 2.5         # a hit whose 40 dB fall is this many times its 20 dB fall has a tail longer than an exponential
CONSISTENT = 1.25        # tone + noise may fall 40 dB this much later than the measured fall_40_ms
DECAY_FACTOR = 1.5       # free decays may move by this factor either way
NOISE_DECAY_FACTOR = 3.0
DEFAULT_NOISE_FACTOR = 4.0      # ...4 when the noise decay is a default
SLOW_NOISE_LOW_FACTOR = 3.0     # the tone decay may fall by this factor when the noise outlasts the tone (its 20 dB fall is the noise's)
GLIDE_FACTOR = 4.0       # the glide's tau may move by this factor: either reading of the Pe Decay mapping is inside it
DROP_WIDTH_ST = 2.0      # the free pitch drop may move by max(this, 40 %) either way
PEAK_DB_AT_0DB = {"tone": 3.7, "noise": 8.0}    # peak of a voice at osc level 0 dB over Volume (dB), 2026-11-12, Volume -12 dB:
#                          a sine reads +3.7 (two takes, +3.6 to +4.0); noise +7.5 and +8.6 (two takes: 8 is their middle)
NOTE_VELOCITY = 100      # the velocity every level was measured at
Z_AMPLITUDE_DB = 20.0 / math.log(10.0)      # 8.686 dB per neper of amplitude

CHECKED = (          # read or measured on a scratch Operator (Live 12.4.6) on 2026-11-12; live-assumptions-2026-11-12.md
    "Amp decay: exponential and linear in dB; 20 dB in 0.227 T (Ae Decay 300 ms: fall 10/20/40 dB = 34.2 / 68.3 / 135.9 "
    "ms; 1000 ms: 113.7 / 226.6 / 452.5 ms), for noise 0.216 T (Be Decay 500 ms: 108 ms, ONE measurement; FALL20_PER_T_NOISE). The decay slope is not a LOM "
    "parameter; this is the device insert_device gives (the 0.29 of the XML kit, DecaySlope 1, does not apply).",
    "Pe Decay = tau * 7.94 holds within the search's range (7.0 to 8.4): Pe Init = Peak 12 st, Pe Decay 400 ms read 11.0 / "
    "8.5 / 5.85 / 2.6 / 1.69 / 0.56 / 0.18 st at 5 / 20 / 40 / 80 / 100 / 150 / 200 ms (tau 58 falling to 49 ms); the "
    "glide is steeper than one exponential late and reaches 0 at T.",
    "Pe levels are semitones at Pe Amount 100 % and Pe Amt A 100 %, with Osc-A < Pe on: the table of Pe Init is -48 to "
    "48 st, and a 12 st drop read 11.8 st at 3 ms.",
    "Osc-B Wave 21 is `Noise White` (20 is `Noise Looped`): band density flat within about 2 dB from 20 Hz to 20 kHz, "
    "flatness 0.56; the noise does not change with the note (notes 60 and 36: same spectrum, fall 107.9 and 108.3 ms).",
    "Osc-A Wave 0 is `Sine`; Fix On 0 is ratio mode (note 60 plays 261.62 Hz).",
    "Coarse and Fine: ratio = Coarse * (1 + Fine/1000) (Coarse 2, Fine 386, note 48: 362.61 Hz measured and expected). "
    "A Coarse raw 0 displays 0.5, raw 1 displays 1: the units table has both, so to_live reads their display numbers.",
    "Algorithm raw 10 is `Alg. 11`, all four in parallel: tone and noise both present, noise reading -12.5 dB with Osc-B "
    "Level -10 dB.",
    "Filter Type 1 is the high-pass (types: Lowpass, Highpass, Bandpass, Notch, Morph): Filter Freq 2000 Hz read -70.7 dB "
    "under 200 Hz and -11.2 dB at 200 to 1000 Hz.",
    "Ae Sustain must be written: the device default is 0 dB (raw 1.0), and operator_live writes -inf, which gives a "
    "decaying hit on a held note. Defaults read: Ae Init -inf dB, Ae Peak 0 dB, Pe D Slope 1.0, Pe A Slope 0.",
    "Output: a sine at Osc-A Level 0 dB peaks at Volume + 3.7 dB (Volume -12 dB: -8.3 and -7.9 dBFS); noise at Osc-B "
    "Level 0 dB peaks at Volume + 7.5 and + 8.6 dB (-4.5 and -3.4 dBFS: two takes, so PEAK_DB_AT_0DB uses 8).",
    "The units table (sideman/data/operator_units.json): the range, default and display values (49 points) of every "
    "parameter, 132 continuous and 53 quantized; every table is accepted by to_live's curve reader. Takes repeat for a "
    "sine (two reads of one take: identical onset, attack and falls). The first 0.9 ms of a sine reads 0.45 st high.",
)

ASSUMPTIONS = (      # NOT checked in Live (what the report lists as checked is in CHECKED)
    "Two results that held only roughly: the profile's attack_ms reads Operator's Ae Attack low (1.5 ms reads 1.0 to "
    "1.2 ms, 20 ms reads 16.5 ms: the profile reads to within 1 dB of the peak, so a slow attack reads short), and "
    "Release = Decay does not make the note length immaterial: a note released after an eighth of a second lengthens "
    "the fall of Decay 1000 ms by 7 % (20 dB: 241.6 ms against 226.6) to 10 % (40 dB: 500.0 against 452.5), so the "
    "voice is held for at least the longest Decay and the take runs past it.",
    "Osc-B < Pe and Pe Amt B = 0 (and Pe Dst B): that the pitch envelope does not reach Osc B. Osc-B < Pe raw 0 is "
    "assumed to be off; only Osc-A < Pe = 1 and Pe Amt A = 100 were read (through the pitch of the tone).",
    "Pe Mode, Pe Loop, Pe Retrig, Pe Release and Pe End are left at the device's defaults: they matter after the "
    "note-off (a pitch envelope that loops, or whose release moves the pitch again, would change the tail), and the "
    "guess holds the note for the longest decay. Not read; confirm they are off / zero on the Init preset.",
    "Ae Mode / Ae Loop (no loop), Ae/Be Retrig, Be Init and Be Peak are left at the device's defaults; Be Sustain is "
    "-inf and Be Release = Be Decay as for Ae (only Ae Init and Peak were read).",
    "Velocity: the levels assume the default velocity response at velocity 100 (Osc level < velocity and < key are left "
    "at their defaults; bzkit.py set 35 % level < velocity). Time < Key = 0 % and global Time = 0 % leave every "
    "envelope time as set (Time scales times by about e^(0.023 per %): measured 2026-11-09 on the kit, not on this device).",
    "LFO On = 0 (written by operator_live) is assumed to silence the LFO; the LFO's own settings are not read. "
    "Transpose 0, Tone, Spread, Glide On, Shaper and Pan are left at the device's defaults or the Set's state: a "
    "scratch Operator must start from the Init preset.",
    "Osc-C and Osc-D are switched off, and the unused oscillator's envelope and levels are placeholders (UNUSED_MS, 0 dB): "
    "that they stay silent is assumed, not read.",
    "Filter types other than the high-pass (band-pass, notch) and the filter's resonance: the one filter sits after both "
    "oscillators, so it is used only on a noise-led voice (tone off), and only as a high-pass. The pads' band-pass "
    "filters in bzkit.py are not reproduced.",
    "Reading times: the attack noise reading's weight is taken at the middle of attack_window_ms (the profile's "
    "own window; a Hann window) and the body's at 0.36 of body_window_ms, calibrated on the stand-in; the "
    "extrapolation back to the onset is clamped to +-12 dB. A noise layer is accepted only when the attack's "
    "energy above 2 kHz (click_db) is within 5 dB of its level and its flatness is at least 0.001 (a tone's and a click's reads 0.0): then the level is pinned. "
    "Flatness is taken over 20 Hz to 20 kHz, so noise that is band-limited or low-passed also reads 0.0 (and a "
    "low-passed one has little above 2 kHz): the gate cannot tell it from window leakage. When it fails but the "
    "reading is above the floor, the layer is kept with its level free down to NOISE_OFF_DB, so the search can switch "
    "a phantom layer off and keep a real one (it cannot add one back); the same goes for the tail rule's flat body: a "
    "noisy body with a long tail and no flat spectrum keeps the two-reading decay, and the noise decay's free range "
    "is widened up to the tail's value. All of it is heuristics to re-check on takes.",
)

STATE_SWITCHES = ("Osc-A On", "Osc-B On", "Osc-C On", "Osc-D On", "A Fix On ", "B Fix On ", "Pe On", "Osc-A < Pe",
                  "Osc-B < Pe", "Filter On", "LFO On")
RAW_NAMES = frozenset(STATE_SWITCHES) | {"Algorithm", "Osc-A Wave", "Osc-B Wave", "Filter Type"}   # Live's own value
RAW_ENUMS = frozenset({"Algorithm", "Osc-A Wave", "Osc-B Wave", "Filter Type"})      # whole numbers, not 0/1
QUANTIZED = frozenset({"A Coarse", "A Fine", "B Coarse", "B Fine"})    # whole numbers; display number = raw unless tabled
EXPECTED_DIMENSION = {                        # what a parameter's display must be, so a wrong unit is refused
    **dict.fromkeys(("Ae Attack", "Ae Decay", "Ae Release", "Be Attack", "Be Decay", "Be Release", "Pe Decay"), "time"),
    **dict.fromkeys(("Volume", "Osc-A Level", "Osc-B Level", "Ae Sustain", "Be Sustain"), "db"),
    **dict.fromkeys(("Pe Init", "Pe Peak", "Pe Sustain", "Transpose"), "st"),
    **dict.fromkeys(("Pe Amount", "Pe Amt A", "Pe Amt B", "Time", "Time < Key"), "pct"),
    "Filter Freq": "freq",
}
UNUSED_MS = 100.0        # what an unused envelope time is set to (its oscillator is off)


@functools.lru_cache(maxsize=1)
def operator_parameters() -> tuple:
    """Operator's 195 parameter names as read from Live 12.4.6, in index order (sideman/data/operator_parameters.json).
    Some carry a trailing space ("A Fix On ")."""
    path = pathlib.Path(__file__).parent / "data" / "operator_parameters.json"
    return tuple(json.loads(path.read_text(encoding="utf-8"))["names"])


@functools.lru_cache(maxsize=1)
def _units_file() -> dict:
    path = pathlib.Path(__file__).parent / "data" / "operator_units.json"
    return json.loads(path.read_text(encoding="utf-8"))["units"]


def operator_units() -> dict:
    """The units table of Operator in Live 12.4.6 (sideman/data/operator_units.json, read from a scratch device with
    scripts/operator_units.py), in the form `to_live`, `from_live` and `search_arguments` take: {parameter name:
    [[live_value, display_number, unit], ...]} for the 132 continuous parameters (49 points each, ends included).
    The file stores a "-inf" display as the string "-inf"; here it is the float. A new copy on each call."""
    return {name: [[float(live), float(disp), unit] for live, disp, unit in rows] for name, rows in _units_file().items()}


def _entry(value, reading: str, confidence: str) -> tuple:
    return value, {"reading": reading, "confidence": confidence}


def _pitch_envelope(p: dict) -> dict:
    """Carrier, drop and time constant from the profile's pitch readings: the body pitch, the start, and the track
    (pitch_track_hz at 0, 5, 10, 20, 40, 80, 150, 300 ms), fitted as semitones = s_inf + drop * exp(-t / tau)
    (for each tau on a grid the other two are linear, weighted by t + 5 ms; the best tau wins). The fit gives the carrier
    the glide settles on (the body pitch alone sits on the glide's tail), and the drop at the onset when the start was
    read late. A poor fit or fewer than three points falls back to the measured drop and settle time."""
    f_body = p["pitch_body_hz"]
    pts = {}
    if p.get("pitch_start_hz") and p.get("pitch_start_at_ms") is not None:
        pts[float(p["pitch_start_at_ms"])] = p["pitch_start_hz"]
    for k, v in (p.get("pitch_track_hz") or {}).items():
        if v:
            pts.setdefault(float(k), v)
    drop = p.get("pitch_drop_st") or 0.0
    out = {"carrier_hz": f_body, "drop_st": 0.0, "tau_ms": 20.0,
           "reading": "pitch_body_hz; no drop above %.1f st" % oneshot.DROP_ST, "confidence": "measured"}
    early = [abs(12 * math.log2(v / f_body)) for k, v in (p.get("pitch_track_hz") or {}).items() if v and float(k) <= 20.0]
    if abs(drop) <= oneshot.DROP_ST and max(early, default=0.0) <= oneshot.DROP_ST:
        return out                  # (a start read at the wrong moment can hide a drop the track still shows)
    settle = p.get("pitch_settle_ms")
    if settle is not None and settle < CLICK_MS:
        out["reading"] = "pitch_body_hz; the %.1f st drop is over in %.1f ms: a click, not a glide" % (drop, settle)
        out["click"] = True
        return out
    late = [v for k, v in (p.get("pitch_track_hz") or {}).items() if v and float(k) >= TAIL_MS]
    tail = None
    if len(late) >= 2 and abs(12 * math.log2(max(late) / min(late))) <= oneshot.PITCH_ST:
        tail = float(np.exp(np.mean(np.log(late))))            # the glide has settled: the late readings are the carrier
    carrier = tail or f_body
    s_tail = 12 * math.log2(carrier / f_body)
    start = p.get("pitch_start_hz")
    drop0 = 12 * math.log2(start / carrier) if start else drop
    fallback = {"carrier_hz": carrier, "drop_st": drop0, "confidence": "derived",
                "tau_ms": max(1.0, (settle if settle else 20.0) / math.log(max(abs(drop0), 2 * oneshot.SETTLE_ST) /
                                                                          oneshot.SETTLE_ST)),
                "reading": "pitch_start_hz, pitch_settle_ms (tau = settle / ln(drop / %.1f st)), %s"
                           % (oneshot.SETTLE_ST, "carrier from the late track" if tail else "carrier = pitch_body_hz")}
    if len(pts) < 3:
        return fallback
    t_all = np.array(sorted(pts))
    st_all = np.array([12 * math.log2(pts[x] / f_body) for x in t_all])
    keep = np.ones(len(t_all), dtype=bool)
    for _ in range(3):             # again without the worst point (up to two): an early reading in noise must not sink the fit
        t, st = t_all[keep], st_all[keep]
        w = (t + 5.0) / 50.0       # early points are the least reliable (a fast glide in short frames): weight by time
        best = None
        for tau in np.geomspace(1.0, 600.0, 160):
            e = np.exp(-t / tau)
            if tail:
                amp = float(((st - s_tail) * e * w ** 2).sum() / max((e * e * w ** 2).sum(), 1e-12))
                coef = np.array([s_tail, amp])
            else:
                coef = np.linalg.lstsq(np.stack([np.ones_like(t), e], axis=1) * w[:, None], st * w, rcond=None)[0]
            rss = float(((((coef[0] + coef[1] * e) - st) * w) ** 2).sum())
            if best is None or rss < best[0]:
                best = (rss, tau, coef)
        _, tau, (s_inf, amp) = best
        res = s_inf + amp * np.exp(-t / tau) - st
        rss = float((res ** 2).sum())
        if rss / len(t) <= 0.5 ** 2 or len(t) <= 4:
            break
        keep[np.nonzero(keep)[0][int(np.argmax(np.abs(res)))]] = False
    if rss / len(t) > 0.5 ** 2 or abs(amp) <= oneshot.DROP_ST or tau in (1.0, 600.0) or abs(s_inf) > 2.0:
        return fallback
    return {"carrier_hz": f_body * 2 ** (s_inf / 12), "drop_st": float(amp), "tau_ms": float(tau),
            "reading": "pitch_track_hz and pitch_start_hz, fitted as s_inf + drop * exp(-t / tau)",
            "confidence": "derived"}


def _noise_filter(p: dict):
    """(Hz, reading) of a high-pass for a noise-led hit, or None: the lowest of the six bands whose power per Hz
    is within FILTER_CLEAR_DB of the densest one, when that is the 'mid' band (500 Hz) or above. White noise is
    equally dense in all bands and gets no filter; a hat that is empty under 4 kHz gets one at the lowest band
    edge that still holds it (a coarse edge: the six bands are wide)."""
    bands, sr = p.get("bands_db"), p.get("sample_rate")
    if not bands or not sr:
        return None
    nyq = sr / 2.0
    dens = {}
    for name, lo, hi in BANDS:
        hi = min(hi, nyq)
        if hi > lo and bands.get(name) is not None:
            dens[name] = (bands[name] - 10 * math.log10(hi - lo), lo)
    if not dens:
        return None
    top = max(d for d, _ in dens.values())
    lowest = min(lo for d, lo in dens.values() if d >= top - FILTER_CLEAR_DB)
    return (lowest, "bands_db (lowest band within %.0f dB of the densest, per Hz)" % FILTER_CLEAR_DB) \
        if lowest >= BANDS[3][1] else None


def noise_onset_db(patch: dict) -> float:
    """The noise layer's level against the tone at the onset, dB: the level pinned at noise_ref_ms, carried back to
    the onset along the two decays, at most ONSET_CLAMP_DB either way (a long reach back from a decay estimate is not
    to be trusted: the reference times are a window's centre, and a short hit has a short window)."""
    tau_t = patch["tone_decay_ms"] * FALL20_PER_T * TAU_PER_FALL20
    tau_n = patch["noise_decay_ms"] * FALL20_PER_T_NOISE * TAU_PER_FALL20
    back = Z_AMPLITUDE_DB * patch.get("noise_ref_ms", NOISE_REF_MS) * (1.0 / tau_n - 1.0 / tau_t)
    return patch["noise_level_ref_db"] + max(-ONSET_CLAMP_DB, min(ONSET_CLAMP_DB, back))


def _model_fall_ms(tau_t: float, tau_n: float, onset_db: float, drop_db: float = 40.0) -> float:
    """When tone + noise (power sum of two exponentials, noise at onset_db against the tone) is last within drop_db
    of its onset level."""
    r2 = 10 ** (onset_db / 10.0)
    t = np.linspace(0.0, 12.0 * max(tau_t, tau_n), 6000)
    power = np.exp(-2 * t / tau_t) + r2 * np.exp(-2 * t / tau_n)
    above = np.nonzero(power >= power.max() * 10 ** (-drop_db / 10.0))[0]
    return float(t[above[-1]]) if len(above) else 0.0


def operator_guess(profile: dict) -> dict:
    """A first Operator patch for the one-shot `profile` (oneshot.profile_array / sample_analyze output), in
    physical units, for one voice: a sine carrier on Osc A with a pitch envelope and a noise layer on Osc B,
    or, for an unpitched hit (hat, clap), noise only. Raises ValueError for a profile with no 20 dB fall
    (missing, or 0) or, when pitched, no body pitch.

    Returns {"patch": {name: float}, "structure": {...}, "sources": {name: {"reading", "confidence"}},
    "free": {name: (low, high, "log"|"lin")}, "pinned": {name: why}, "notes": [...], "assumptions": [...] (not
    checked in Live), "checked": [...] (read on a scratch device on 2026-11-12)}.
    `confidence` is "measured" (a reading used as it is), "derived" (computed from readings, with a
    formula) or "default" (no reading; a choice). Feed it to `search_arguments` and `operator_live`.

    The patch (a name is absent when the voice has no such part):
      carrier_hz     body pitch, or the pitch the glide settles on (fit of the pitch track)
      pitch_drop_st  semitones above carrier_hz at the onset (0 = no pitch envelope: under oneshot.DROP_ST, or a
                     click: over in under CLICK_MS), at most PE_MAX_ST
      pitch_tau_ms   time constant of the glide, exponential
      tone_attack_ms attack_ms, at least ATTACK_FLOOR_MS (0.1 ms clicks)
      tone_decay_ms  Operator's Decay time T = fall_20_ms / FALL20_PER_T (0.227: a device from insert_device, measured)
      tone_level_db  0 (the reference: the noise is set against it)
      noise_level_ref_db  the noise layer's level against the tone at noise_ref_ms: the attack-window reading
                     (noise_db.attack, a share of the total) as a power ratio, share / (1 - share)
      noise_ref_ms   half of attack_window_ms, the centre of the reading's Hann window (15 ms for a 30 ms window; a
                     short hit has a shorter window: the profile says where it ended)
      noise_decay_ms Operator Decay time of the noise. In order: (1) a body at least 12 % noise (BODY_NOISE_DB) with a flat
                     spectrum (flatness.body at least FLAT_BODY), whose fall_40_ms is over TAIL_RATIO times its fall_20_ms: the tail, (fall_40 - fall_20) / ln 10;
                     (2) a usable body reading (above oneshot.NOISE_FLOOR_DB, a tone's window leakage): the two
                     readings, at noise_ref_ms and 0.36 into body_window_ms, give the rate of the noise against the
                     tone, (ratio_attack_dB - ratio_body_dB) / (8.686 * the time between), and 1/tau_noise = that
                     + 1/tau_tone; (3) otherwise a fifth of the tone's (default, searched within a factor of 4). The
                     result is shortened until tone + noise fall 40 dB no more than 25 % after the measured fall_40_ms
                     (unless the tone alone does: a gated shape). Noise-led: the larger of fall_20_ms and
                     fall_40_ms / 2, over 0.216 (FALL20_PER_T_NOISE: one measurement; when the hit falls more slowly later the late fall is the noise's)
      noise_attack_ms attack_ms, at least NOISE_ATTACK_FLOOR_MS
      noise_filter_hz  high-pass for a noise-led hit whose bands are empty low (see _noise_filter)
      output_db      Volume: the sample's peak_db minus the peak of one oscillator at 0 dB (+3.7 tone, +8 noise)
    A noise layer is added to a pitched hit whenever its attack reading is above the leakage floor. Its level is PINNED
    only with evidence: the attack's energy above 2 kHz (click_db) within HF_MARGIN_DB of the reading, and a flat
    part in the spectrum (flatness.attack), and the reading itself over NOISE_FREE_DB. Without the evidence (a glide, a
    gated tone or a click on a tone read "noise" from window leakage: the Bodzin kick, low tom and clave; but band-limited
    or low-passed noise reads the same, flatness 0.0 over 20 Hz to 20 kHz) the level is FREE, from NOISE_OFF_DB (off)
    to 6 dB over the reading: the search switches a phantom layer off and keeps a real one, and it cannot add one back,
    which is why the layer is never removed. A reading under NOISE_FREE_DB is free too. The start is the reading either way.
    The decay's range is x/÷3 around its value (4 for a default), and, when the body is noisy with a long tail but not
    flat (band-passed noise, or a tonal tail: the tail rule needs flatness.body), widened up to the tail's decay.
    Why the level is pinned and the decay free: decay and level lie on a ridge (a longer layer, quieter, scores like
    a shorter, louder one: the search report), so one must be held. The level at its reference time is a direct
    reading, the decay a two-reading derivation under a leakage floor, so the level (at the reading's own time, not
    at the onset, so it does not move when the decay does) is pinned and the decay searched.
    Why ratio mode and not a fixed frequency: Coarse/Fine from a played note is what was measured and built
    (2026-11-11); whether a fixed-frequency oscillator follows the pitch envelope is not known."""
    p = profile
    fall20 = p.get("fall_20_ms")
    if not fall20 or fall20 <= 0:
        raise ValueError("the profile has no 20 dB fall (it is missing or 0: the take ends before the hit has "
                         "fallen): record a longer take")
    fall20 = float(fall20)
    pitched = bool(p.get("pitched")) and bool(p.get("pitch_body_hz"))
    patch, src, free, pinned, notes = {}, {}, {}, {}, []

    def put(name, value, reading, confidence):
        patch[name], src[name] = _entry(float(value), reading, confidence)

    attack = p.get("attack_ms") or 0.0
    peak_db = p.get("peak_db")
    peak_db = 0.0 if peak_db is None else peak_db
    tau_tone = fall20 * TAU_PER_FALL20
    f40 = p.get("fall_40_ms")
    if f40 and not 1.6 <= f40 / fall20 <= TAIL_RATIO:
        notes.append("the decay is not exponential (fall_40_ms / fall_20_ms = %.1f, an exponential reads 2.0): Operator's "
                     "amp envelope has no slope parameter in the LOM, so the guess matches the first 20 dB and %s"
                     % (f40 / fall20, "rings on past the 40 dB point" if f40 / fall20 < 1.6 else "falls short of the tail"))
    structure = {"tone_on": pitched, "noise_on": False, "pitch_env_on": False, "filter_on": False}
    if pitched:
        pe = _pitch_envelope(p)
        if pe.get("click"):
            notes.append(pe["reading"])
        if abs(pe["drop_st"]) > PE_MAX_ST:
            notes.append("the pitch drop (%.1f st) is capped at %.0f st (Operator's range, to confirm)" % (pe["drop_st"], PE_MAX_ST))
            pe["drop_st"] = math.copysign(PE_MAX_ST, pe["drop_st"])
        if pe["carrier_hz"] < MIN_CARRIER_HZ:
            notes.append("the carrier (%.1f Hz) is under MIDI note 0: operator_live will refuse it" % pe["carrier_hz"])
        put("carrier_hz", pe["carrier_hz"], pe["reading"], pe["confidence"])
        put("pitch_drop_st", pe["drop_st"], pe["reading"], pe["confidence"])
        put("pitch_tau_ms", pe["tau_ms"], pe["reading"], pe["confidence"])
        structure["pitch_env_on"] = abs(pe["drop_st"]) > oneshot.DROP_ST
        put("tone_attack_ms", max(ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms (0.1 ms clicks)" % ATTACK_FLOOR_MS,
            "measured" if attack >= ATTACK_FLOOR_MS else "default")
        put("tone_decay_ms", fall20 / FALL20_PER_T, "fall_20_ms / %.3f" % FALL20_PER_T, "derived")
        put("tone_level_db", 0.0, "reference level", "default")
        nz = p.get("noise_db") or {}
        a, b, click = nz.get("attack"), nz.get("body"), p.get("click_db")
        flat = p.get("flatness") or {}
        noise_wanted = a is not None and a > oneshot.NOISE_FLOOR_DB
        evidence = noise_wanted and not (click is None or click < a - HF_MARGIN_DB or (flat.get("attack") or 0.0) < FLAT_ATTACK)
        if noise_wanted and not evidence:
            notes.append("noise_db.attack is %.1f dB but the attack has %s dB above 2 kHz (click_db) and a flatness of %s: "
                         "that is what window leakage of a glide, a gated tone or a click reads, and also what noise that is "
                         "band-limited or low-passed reads (flatness 0.0): the layer is kept with its level free down to "
                         "%.0f dB (off), for the search to decide" % (a, "None" if click is None else "%.1f" % click,
                                                                       flat.get("attack"), NOISE_OFF_DB))
        if noise_wanted:
            structure["noise_on"] = True
            aw = p.get("attack_window_ms")
            t_a = aw[1] / 2.0 if aw and aw[1] > 0 else NOISE_REF_MS
            share = 10 ** (a / 10.0)
            ratio_a = 10 * math.log10(share / max(1.0 - share, 1e-3))
            put("noise_level_ref_db", ratio_a, "noise_db.attack as a power ratio to the tone, at %.1f ms (half of "
                "attack_window_ms)" % t_a, "derived")
            put("noise_ref_ms", t_a, "attack_window_ms / 2" if aw else "default %.0f ms (no attack_window_ms)" % NOISE_REF_MS,
                "derived" if aw else "default")
            bw = p.get("body_window_ms") or [40.0, 150.0]
            t_b = bw[0] + NOISE_BODY_FRAC * (bw[1] - bw[0])
            tau_noise, default_decay, tail_reach = None, False, None
            noisy_tail = b is not None and b >= BODY_NOISE_DB and f40 and f40 > TAIL_RATIO * fall20
            if noisy_tail and (flat.get("body") or 0.0) < FLAT_BODY:        # a flat body is noise; a band-limited one reads 0.0 too
                tail_reach = (f40 - fall20) / FALL20_PER_T_NOISE            # the decay a flat tail would give, kept in reach
            if noisy_tail and tail_reach is None:
                tau_noise = (f40 - fall20) * TAU_PER_FALL20
                reading, conf = "fall_40_ms - fall_20_ms (a body %.0f dB noisy, and a tail %.1f times the first 20 dB, "\
                                "is the noise's)" % (b, f40 / fall20), "derived"
            elif b is not None and b > oneshot.NOISE_FLOOR_DB and t_b - t_a >= 5.0:
                shb = 10 ** (b / 10.0)
                ratio_b = 10 * math.log10(shb / max(1.0 - shb, 1e-3))
                k = (ratio_a - ratio_b) / (Z_AMPLITUDE_DB * (t_b - t_a))        # 1/ms
                inv = max(k + 1.0 / tau_tone, 1.0 / (3.0 * tau_tone))
                tau_noise = min(1.0 / inv, 3.0 * tau_tone)
                reading, conf = "noise_db.attack and noise_db.body, at %.0f and %.0f ms, with the tone's fall_20_ms" % (t_a, t_b), "derived"
            if tau_noise is None:
                tau_noise, default_decay = tau_tone * DEFAULT_NOISE_FRACTION, True
                reading = "none (no usable noise reading in the body): a fifth of the tone's decay"
                conf = "default"
            if f40 and 2 * fall20 <= CONSISTENT * f40:                        # the tone alone is no later than fall_40
                tone_only = {"tone_decay_ms": fall20 / FALL20_PER_T, "noise_level_ref_db": ratio_a, "noise_ref_ms": t_a}
                shrunk = False
                for _ in range(20):
                    tone_only["noise_decay_ms"] = tau_noise / (FALL20_PER_T_NOISE * TAU_PER_FALL20)
                    onset = noise_onset_db(tone_only)
                    if _model_fall_ms(tau_tone, tau_noise, onset) <= CONSISTENT * f40:
                        break
                    tau_noise *= 0.8
                    shrunk = True
                if shrunk:
                    reading += "; shortened so tone + noise fall 40 dB within %.0f %% of fall_40_ms" % (100 * (CONSISTENT - 1))
            put("noise_decay_ms", tau_noise / (FALL20_PER_T_NOISE * TAU_PER_FALL20), reading, conf)
            put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS,
                "measured" if attack >= NOISE_ATTACK_FLOOR_MS else "default")
            if a > NOISE_FREE_DB and evidence:
                pinned["noise_level_ref_db"] = "decay and level of the noise layer lie on a ridge; the level at its reference time is read directly"
            else:
                free["noise_level_ref_db"] = (NOISE_OFF_DB, ratio_a + 6.0, "lin")
                notes.append("noise_db.attack is %.1f dB, near the %.0f dB leakage floor: the noise level is searched "
                             "(down to %.0f dB, off), not pinned" % (a, oneshot.NOISE_FLOOR_DB, NOISE_OFF_DB))
            nfactor = DEFAULT_NOISE_FACTOR if default_decay else NOISE_DECAY_FACTOR
            free["noise_decay_ms"] = (patch["noise_decay_ms"] / nfactor, patch["noise_decay_ms"] * nfactor, "log")
            if tail_reach is not None and tail_reach > free["noise_decay_ms"][1]:
                free["noise_decay_ms"] = (free["noise_decay_ms"][0], tail_reach, "log")
                notes.append("the body is %.0f dB noisy and falls 40 dB in %.1f times its first 20 dB, but its spectrum has no "
                             "flat part (flatness.body %s: a tonal tail, or band-limited noise): the tail is not taken as the "
                             "noise's decay, and the noise decay's range is widened up to it (%.0f ms) so the search can reach it"
                             % (b, f40 / fall20, flat.get("body"), tail_reach))
            pinned["noise_ref_ms"] = pinned["noise_attack_ms"] = "set from the reading, not searched"
        elif a is None or a <= oneshot.NOISE_FLOOR_DB:
            notes.append("noise_db.attack is %s: no noise layer" % ("missing" if a is None else "%.1f dB, under the %.0f dB floor "
                                                                    "(a decaying tone reads that from window leakage)" % (a, oneshot.NOISE_FLOOR_DB)))
        put("output_db", min(0.0, peak_db - PEAK_DB_AT_0DB["tone"]),
            "peak_db minus %.1f dB (a sine at Osc-A Level 0 dB peaks at Volume + 3.7 dB: measured)" % PEAK_DB_AT_0DB["tone"], "default")
        free["carrier_hz"] = (patch["carrier_hz"] * 2 ** (-PITCH_BOUND_ST / 12), patch["carrier_hz"] * 2 ** (PITCH_BOUND_ST / 12), "log")
        if structure["pitch_env_on"]:
            d = patch["pitch_drop_st"]
            w = max(DROP_WIDTH_ST, 0.4 * abs(d))
            free["pitch_drop_st"] = (d - w, d + w, "lin")
            free["pitch_tau_ms"] = (patch["pitch_tau_ms"] / GLIDE_FACTOR, patch["pitch_tau_ms"] * GLIDE_FACTOR, "log")
        else:
            pinned["pitch_drop_st"] = pinned["pitch_tau_ms"] = "no pitch drop above %.1f st was measured" % oneshot.DROP_ST
        low = SLOW_NOISE_LOW_FACTOR if structure["noise_on"] and (patch["noise_decay_ms"] >= 0.8 * patch["tone_decay_ms"]) else DECAY_FACTOR
        free["tone_decay_ms"] = (patch["tone_decay_ms"] / low, patch["tone_decay_ms"] * DECAY_FACTOR, "log")
        pinned["tone_attack_ms"] = pinned["tone_level_db"] = pinned["output_db"] = "set from the reading, not searched"
    else:
        structure["noise_on"] = True
        late = f40
        decay_ms = max(fall20, late / 2.0 if late else 0.0) / FALL20_PER_T_NOISE
        put("noise_decay_ms", decay_ms, "the larger of fall_20_ms and fall_40_ms / 2, over %.3f (when the hit falls more "
            "slowly later, the late fall is the noise's: a tone in the first 20 dB decays faster)" % FALL20_PER_T_NOISE,
            "derived")
        put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS,
            "measured" if attack >= NOISE_ATTACK_FLOOR_MS else "default")
        flt = _noise_filter(p)
        if flt:
            structure["filter_on"] = True
            put("noise_filter_hz", flt[0], flt[1], "derived")
            pinned["noise_filter_hz"] = "set from the band balance, not searched"
        put("output_db", min(0.0, peak_db - PEAK_DB_AT_0DB["noise"]),
            "peak_db minus %.0f dB (noise at Osc-B Level 0 dB peaks at Volume + 7.5 to 8.6 dB: two takes)" % PEAK_DB_AT_0DB["noise"], "default")
        free["noise_decay_ms"] = (patch["noise_decay_ms"] / DECAY_FACTOR, patch["noise_decay_ms"] * DECAY_FACTOR, "log")
        pinned["noise_attack_ms"] = pinned["output_db"] = "set from the reading, not searched"
        notes.append("unpitched: the voice is noise only (tone off); the profile's partials are not rendered")
    return {"patch": patch, "structure": structure, "sources": src, "free": free, "pinned": pinned,
            "notes": notes, "assumptions": list(ASSUMPTIONS), "checked": list(CHECKED)}


def search_arguments(guess: dict, units: dict | None = None) -> dict:
    """The `start`, `bounds` and `probes` arguments of `search` from an operator_guess result: start is the guess's
    patch, bounds the guess's free parameters; every other name is pinned (rendered as given, never changed).
    `probes` holds one patch, the start with a free noise level at its lower bound (off), when the guess has one: a
    layer without evidence may be a phantom, and a start at -15 dB with a step of 3 dB reaches "off" too slowly and
    leaves a remnant at -30 to -40 dB (measured: report, "phantom layers"); rendered once in round 0 it is
    chosen at once when the layer is a phantom and dropped when it is real. With a
    units table, each bound is pulled in toward the start (halving the distance, up to 30 times) until the voice
    at that bound converts through to_live: one candidate out of the table's range would stop the search
    (a render that raises ends it), so the search is never offered one. Raises ValueError when the start itself
    does not convert. The check is one parameter at a time, the others at their start values: a combination of
    bounds (two decays both at their long end) is not checked, and may still fall outside a table. The corners are
    not checked: 2^n voices for n free parameters, and pulling in one bound at a time would not say which to pull."""
    start, bounds = dict(guess["patch"]), dict(guess["free"])
    if units is None:
        return {"start": start, "bounds": bounds, **_off_probe(start, bounds)}

    def converts(patch: dict) -> bool:
        try:
            to_live(operator_live(patch, guess["structure"])["params"], units)
            return True
        except ValueError:
            return False

    if not converts(start):
        raise ValueError("the guess itself does not convert through the units table (a value outside a table's range)")
    for name, (lo, hi, scale) in list(bounds.items()):
        for end in (0, 1):
            v = (lo, hi)[end]
            for _ in range(30):
                if converts(dict(start, **{name: v})):
                    break
                a, b = start[name], v
                v = math.sqrt(a * b) if scale == "log" else (a + b) / 2.0
            else:
                v = start[name]
            lo, hi = (v, hi) if end == 0 else (lo, v)
        bounds[name] = (lo, hi, scale)
    return {"start": start, "bounds": bounds, **_off_probe(start, bounds)}


def _off_probe(start: dict, bounds: dict) -> dict:
    """{"probes": [the start with the noise level at its lower bound]} when that level is searched, else {}."""
    if "noise_level_ref_db" not in bounds:
        return {}
    return {"probes": [dict(start, noise_level_ref_db=bounds["noise_level_ref_db"][0])]}


def operator_live(patch: dict, structure: dict) -> dict:
    """The complete Operator state that plays `patch` (operator_guess's patch, or a search's candidate), as
    {"params": {Live parameter name: physical value}, "note": MIDI note, "velocity": NOTE_VELOCITY,
    "note_length_ms": how long to hold it}. The same keys for every structure: every oscillator's On, the pitch
    envelope's switches and routing, Filter On, LFO On, global Time, Osc A and B's levels, envelopes and waves;
    a part the voice does not have is switched off and its numbers are placeholders (UNUSED_MS, 0 dB, 0 st), which
    still need a row in the units table. Raw switches and enumerations (RAW_NAMES) hold Live's own value; the
    rest are Hz, ms, dB, semitones or % (the unit to_live checks). Noise-only voices are played at note 60 and
    a carrier under MIDI note 0 (8.18 Hz) raises ValueError. The formulas are the ones in operator_guess's docstring and
    CHECKED and ASSUMPTIONS; only what CHECKED lists has been read in Live."""
    q = patch
    tone, noise = bool(structure.get("tone_on")), bool(structure.get("noise_on"))
    pe_on = tone and bool(structure.get("pitch_env_on"))
    filt = bool(structure.get("filter_on"))
    par = {"Algorithm": 10, "Volume": q.get("output_db", 0.0), "Time": 0.0, "Time < Key": 0.0,
           "Osc-C On": 0, "Osc-D On": 0, "LFO On": 0}
    note = 60
    lev_tone, lev_noise = 0.0, 0.0
    coarse, fine = 1, 0
    if tone:
        f = q["carrier_hz"]
        if f < MIN_CARRIER_HZ:
            raise ValueError("the carrier (%.2f Hz) is under MIDI note 0 (8.18 Hz): it cannot be played" % f)
        floor_note = int(math.floor(69 + 12 * math.log2(f / 440.0)))
        note = max(0, min(127, floor_note - 1))
        ratio = f / (440.0 * 2 ** ((note - 69) / 12.0))
        coarse = max(1, min(48, int(math.floor(ratio))))
        fine = max(0, min(999, int(round((ratio / coarse - 1) * 1000))))
        lev_tone = q.get("tone_level_db", 0.0)
        if noise:
            onset = noise_onset_db(q)
            lev_tone, lev_noise = lev_tone + min(0.0, -onset), min(0.0, onset)
    par.update({"Osc-A On": int(tone), "A Fix On ": 0, "A Coarse": coarse, "A Fine": fine, "Osc-A Wave": 0,
                "Osc-A Level": lev_tone,
                "Ae Attack": q.get("tone_attack_ms", ATTACK_FLOOR_MS), "Ae Decay": q.get("tone_decay_ms", UNUSED_MS),
                "Ae Sustain": -math.inf, "Ae Release": q.get("tone_decay_ms", UNUSED_MS)})
    par.update({"Osc-B On": int(noise), "B Fix On ": 0, "Osc-B Wave": 21, "Osc-B Level": lev_noise,
                "Be Attack": q.get("noise_attack_ms", NOISE_ATTACK_FLOOR_MS),
                "Be Decay": q.get("noise_decay_ms", UNUSED_MS), "Be Sustain": -math.inf,
                "Be Release": q.get("noise_decay_ms", UNUSED_MS)})
    if pe_on:
        d = q["pitch_drop_st"]
        par.update({"Pe Init": d, "Pe Peak": d, "Pe Decay": q["pitch_tau_ms"] * PE_DECAY_PER_TAU, "Pe Sustain": 0.0,
                    "Pe Amount": 100.0, "Pe Amt A": 100.0, "Pe Amt B": 0.0})
    else:
        par.update({"Pe Init": 0.0, "Pe Peak": 0.0, "Pe Decay": UNUSED_MS, "Pe Sustain": 0.0, "Pe Amount": 0.0,
                    "Pe Amt A": 0.0, "Pe Amt B": 0.0})
    par.update({"Pe On": int(pe_on), "Osc-A < Pe": int(pe_on), "Osc-B < Pe": 0,
                "Filter On": int(filt), "Filter Type": 1, "Filter Freq": q.get("noise_filter_hz", 1000.0)})
    longest = max(par["Ae Decay"] if tone else 0.0, par["Be Decay"] if noise else 0.0, UNUSED_MS)
    return {"params": par, "note": note, "velocity": NOTE_VELOCITY, "note_length_ms": longest}


# ---- physical <-> Live values through a units table: {name: [[live_value, display_number, unit], ...]}

_UNITS = {"ms": ("time", 1.0), "s": ("time", 1000.0), "Hz": ("freq", 1.0), "kHz": ("freq", 1000.0), "dB": ("db", 1.0),
          "%": ("pct", 1.0), "st": ("st", 1.0), "": ("none", 1.0)}


class _Curve:
    """One parameter's table: Live value <-> display number (in ms, Hz, dB, %, st or plain), piecewise linear. Consecutive
    rows (in live-value order) with the same display number (display rounding: two live values that both read "0.10
    ms") are collapsed into one at their mean live value; rows that share a number but are not adjacent are not merged,
    so a table that goes up and comes back down is refused. Each segment is interpolated in the log of the number when the table's positive rows
    span a decade or more and both ends of the segment are positive, linearly otherwise (a 0 ms row does not make
    the whole table linear). -inf (a row whose display is -inf) is its own point. Raises ValueError for a table that
    is not strictly monotone once plateaus are collapsed, mixes dimensions, or has under two finite rows."""

    def __init__(self, name: str, rows: list):
        pts, self.floor, dims = [], None, set()
        for row in rows:
            if len(row) != 3:
                raise ValueError("units[%r]: a row is [live_value, display_number, unit], got %r" % (name, row))
            live, disp, unit = float(row[0]), float(row[1]), str(row[2]).strip()
            dim, mul = _UNITS.get(unit, ("other:" + unit, 1.0))
            dims.add(dim)
            if disp == -math.inf and math.isfinite(live):
                self.floor = live
            elif math.isfinite(disp) and math.isfinite(live):
                pts.append((live, disp * mul))
            else:
                raise ValueError("units[%r]: row %r is not finite" % (name, row))
        if len(pts) < 2:
            raise ValueError("units[%r]: at least two finite rows are needed" % name)
        if len(dims) != 1:
            raise ValueError("units[%r]: the rows mix units %s" % (name, sorted(dims)))
        self.dim = dims.pop()
        want = EXPECTED_DIMENSION.get(name)
        if want and self.dim != want:
            raise ValueError("units[%r]: its display is in %s, a physical value of this parameter is %s" % (name, self.dim, want))
        pts.sort()
        if len(set(p[0] for p in pts)) != len(pts):
            raise ValueError("units[%r]: two rows have the same live value" % name)
        runs = []                      # plateaus: consecutive rows (live order) with one display number, as one row
        for live, disp in pts:
            if runs and runs[-1][1] == disp:
                runs[-1][0].append(live)
            else:
                runs.append(([live], disp))
        pts = [(sum(v) / len(v), d) for v, d in runs]               # at the mean live value
        self.live = np.array([p[0] for p in pts])
        self.disp = np.array([p[1] for p in pts])
        d = np.diff(self.disp)
        if len(pts) < 2 or not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError("units[%r] is not monotone: the display number must rise (or fall) with the live value" % name)
        pos = self.disp[self.disp > 0]
        self.decades = bool(len(pos) >= 2 and pos.max() / pos.min() >= 10.0)
        self.name = name

    def _log(self, i: int) -> bool:
        return self.decades and self.disp[i] > 0 and self.disp[i + 1] > 0

    def to_live(self, x: float) -> float:
        if x == -math.inf and self.floor is not None:
            return self.floor
        lo, hi = self.disp.min(), self.disp.max()
        if not (math.isfinite(x) and lo - 1e-9 * max(1.0, abs(lo), abs(hi)) <= x <= hi + 1e-9 * max(1.0, abs(lo), abs(hi))):
            raise ValueError("%s: %g is outside the table's range %g to %g" % (self.name, x, lo, hi))
        x = min(max(x, lo), hi)
        order = np.argsort(self.disp)
        d, lv = self.disp[order], self.live[order]
        i = int(min(max(np.searchsorted(d, x, side="right") - 1, 0), len(d) - 2))
        seg_log = self.decades and d[i] > 0 and d[i + 1] > 0
        u = (lambda v: math.log(v)) if seg_log else (lambda v: v)
        f = (u(x) - u(d[i])) / (u(d[i + 1]) - u(d[i]))
        return float(lv[i] + f * (lv[i + 1] - lv[i]))

    def from_live(self, v: float) -> float:
        if self.floor is not None and v == self.floor:
            return -math.inf
        lo, hi = self.live.min(), self.live.max()
        if not (lo <= v <= hi):
            raise ValueError("%s: live value %g is outside the table's range %g to %g" % (self.name, v, lo, hi))
        i = int(min(max(np.searchsorted(self.live, v, side="right") - 1, 0), len(self.live) - 2))
        f = (v - self.live[i]) / (self.live[i + 1] - self.live[i])
        if self._log(i):
            return float(math.exp(math.log(self.disp[i]) + f * (math.log(self.disp[i + 1]) - math.log(self.disp[i]))))
        return float(self.disp[i] + f * (self.disp[i + 1] - self.disp[i]))


def _convert(values: dict, units: dict, direction: str) -> dict:
    names, out, curves = set(operator_parameters()), {}, {}
    for name in units:
        if name not in names:
            raise ValueError("units names %r, which is not an Operator parameter" % name)
    for name, v in values.items():
        if name not in names:
            raise ValueError("%r is not an Operator parameter" % name)
        if name in RAW_NAMES:
            if not (isinstance(v, numbers.Real) and float(v) == int(v) and v >= 0
                    and (name in RAW_ENUMS or v in (0, 1))):
                raise ValueError("%s: %r is not a raw Live value (%s)" % (name, v, "a whole number from 0" if name in RAW_ENUMS else "0 or 1"))
            out[name] = int(v)
        elif name in units:
            if name not in curves:
                curves[name] = _Curve(name, units[name])           # only the tables a conversion needs: a bad unused one is not an error
            x = float(round(v)) if name in QUANTIZED and direction == "to_live" else float(v)
            out[name] = getattr(curves[name], direction)(x)
        elif name in QUANTIZED:
            out[name] = int(round(v))
        else:
            raise ValueError("no units table for %r: it cannot be converted" % name)
    return out


def to_live(values: dict, units: dict) -> dict:
    """{Live parameter name: Live's value} from {name: physical value} (ms, Hz, dB, %, semitones; -inf allowed where
    the table has a -inf row) through `units` ({name: [[live_value, display_number, unit], ...]}, from Live's
    str_for_value over each range). Interpolated piecewise linearly, per segment in the log of the display number
    where the positive rows span a decade (see _Curve); plateaus from display rounding are collapsed. Raises
    ValueError for a name that is not in Operator's parameter list, a name with no table, a table that is not
    monotone, in mixed units or in a unit that is not the parameter's (EXPECTED_DIMENSION; only the tables of the
    names converted are read), and a value outside the table's range: it is refused and not clamped, because a
    clamp would let the search believe in a value Live never rendered. Raw switches (STATE_SWITCHES: 0 or 1) and
    enumerations (whole numbers) pass as they are and are refused otherwise; Coarse and Fine are whole numbers
    (rounded), passed raw if untabled."""
    return _convert(values, units, "to_live")


def from_live(values: dict, units: dict) -> dict:
    """The inverse of to_live: {name: physical value} from {name: Live's value}. Same refusals."""
    return _convert(values, units, "from_live")


# ------------------------------------------------------------------------------------------ Live as the renderer
#
# LiveRenderer is the `render` callable for `search` when the synth is a device in Ableton Live: it plays candidates
# on a scratch MIDI track and records them on a scratch audio track. It is synth-agnostic: `to_params` and `convert`
# say what a patch is for the device. OperatorRenderer fills them in for Operator (operator_live + to_live). The
# client is sideman.client.Live (or anything with get, set, call, get_batch, transaction, notes_add, notes_remove).
# Nothing here opens a socket itself.

PARAM_KEYS = ("peak_db", "attack_ms", "fall_20_ms", "fall_40_ms", "pitch_body_hz", "click_db")   # readings repeat_check spreads


QUANT_BEATS = (0.0, 32.0, 16.0, 8.0, 4.0, 2.0, 4.0 / 3, 1.0, 2.0 / 3, 0.5, 1.0 / 3, 0.25, 1.0 / 6, 0.125)    # Song.clip_trigger_quantization 0..13 in beats (4/4)
_TRACK = re.compile(r"live_set tracks \d+")


class LiveRenderer:
    """`render(patches) -> [hit]` for `search`, one hit (an [n, channels] array at `sample_rate`) per patch, played on
    the device at `device` by the one-note-per-candidate MIDI clip at `clip` and recorded on the audio track
    `record_track` ("live_set tracks N"; its input is the MIDI track's Post FX). A candidate that could not be
    rendered comes back as an empty array (search's convention: it is listed in `failures` as "empty"), with the
    reason in `takes[-1]["candidates"][i]["reason"]`; it is never another candidate's audio.

    `to_params(patch) -> (params, note, note_length_ms)`: the device's physical values, the MIDI note to play and how
    long to hold it. `convert(params) -> {name: Live value}` defaults to to_live(params, units); `names` is the
    device's parameter list in index order (default Operator's), which gives each written name its path
    "<device> parameters <index>".

    Identity. Paths are positions: a track inserted or deleted above the scratch tracks moves them. The record
    track's name, the MIDI track's name and the device's class_name are captured when the renderer is built
    (`expect={"record":, "midi":, "device":}` gives them instead; capture=False leaves them to the first take's
    state batch), read again in every take's state batch, and the take is refused before anything is written
    unless all three are as captured. They are read once more, with every track's solo and arm, just before the
    cleanup writes, which are skipped (and the take raises after the transport is stopped) if the names moved.
    The constructor checks that the device and the clip are on one track and that the record track is another.
    A track inserted DURING a take is caught at the next write: every transaction that writes parameters (the setup
    one included) starts with a guard op, a harmless call on the device's first parameter carrying
    expect={"id": its _live_ptr, read at capture}, and is sent with stop_on_error: Live checks `expect` before the
    call (remote_script handlers._guard, _target_identity) and a mismatch halts the transaction before any set. The take
    is then aborted and the cleanup raises "moved"; no parameter reaches another track's device.

    EVERY request to Live costs about 0.4 to 0.7 s of wall time whatever it holds (measured 2026-11-13: one `get`
    599 ms, a `transaction` of 1 to 38 sets 400 to 710 ms). The schedule is built on that:
    - during a take there is exactly ONE request per candidate, its parameter write; nothing is re-read between
      candidates. The wall clock is synchronised to the clip once, after firing (see below), and the writes are
      timed from it;
    - the request time m is the LARGEST of the three guard reads at the start of the take (is_playing, tempo, the
      state batch);
    - a candidate's write STARTS `lead` before its note, lead = max(lead_factor * m, m + lead_floor_s) + lead_extra_s
      (1.5 and 0.5 s: 1.14 s for m = 0.6, 1.26 for m = 0.76), so it returns at least 0.5 s before the note; it
      counts as late when it returns less than `min_lead_s` before the note, and then fails that candidate;
    - the write starts after the previous note has ended (the voice has sounded out: the note is held for the
      candidate's decay T, where Operator's amp curve is 88 dB down), so
      spacing = max(longest note + lead + margin, hit_s + margin), rounded up to 1/16 beat, margin =
      max(0.2 s, repoll_ratio * m, the last take's sync poll) (1.25 m: the sync's uncertainty, below). The first note is at max(start_beat,
      3 m + 0.3 s) so the sync is done before the second candidate's write;
    - `lead_extra_s` starts at 0 and adapts: when more than a quarter of a batch's writes were late it is raised by
      the largest shortfall plus 0.1 s (logged in the take); after a take without late writes it falls back by half.

    Sync: after both slots are fired the clip is polled (is_playing, playing_position, length in one batch) until it
    plays; beat 0 is at the wall time the request was SENT at, minus position x seconds per beat. Live read the
    position somewhere between sending and returning (unknown where), so the true beat 0 is between that estimate
    and the estimate plus the poll's duration d (logged as `sync_poll_s`): the notes sound NO EARLIER than believed,
    which makes the late check a lower bound on the real lead. A poll longer than repoll_ratio * m (1.25) is
    repeated once and the shorter one used. The early side is guarded: write k must not start before the believed
    note k-1 + d + note k-1's length, else candidate k-1 fails ("could have ended"): its voice may have been
    sounding when the parameters landed. A drift estimate comes from the take itself afterwards:
    `offset_median_ms` and `offset_trend_ms` (found against expected onsets), not from extra requests.

    The file. Launch quantization starts the record slot and the clip on the grid (`clip_trigger_quantization`,
    read in the state batch), and the two fires are two requests apart: the record slot can start one or more grid
    steps before the clip (or, with no quantization, by the time between the two fires). The first onset in the file
    is note 0 (the record slot is fired first). Its shift against the expected time is applied only when it is no more
    than the time the two fires took (`fire_window_s`) or within `shift_tol_s` (30 ms) of a whole number of grid steps and, when it
    is as long as a spacing or longer (which is also what a missing note 0 looks like: the first onset is then note 1),
    every note has an onset where the shift puts it (a count of onsets would be defeated by one extra); otherwise the take is rejected with a reason, every
    candidate fails, and nothing is cut at a guessed place. (A rule of "under half a spacing" would reject the usual
    straddle: with a bar of 2 s and a spacing of 3 s the shift is 2 s.)

    Batch mode (default): ONE take per batch, the clip rewritten with one note per candidate (its pitch and length,
    velocity `velocity`), each cut with oneshot.hit_cut at its expected time: an onset must be found from
    `onset_early_s` (40 ms) before to `onset_late_s` (100 ms) after it, the file must run hit_s past it, the cut must
    not be silent. Single mode (batch=False): one take per candidate, the manual way, each at the next empty slot.

    Requests per take: 3 guards (+ 2 counts the first time), notes_remove, notes_add, one transaction (record
    track solo and arm, clip looping and length, and candidate 0's parameters), 2 fires, the sync (1 or 2), one
    write per further candidate, then the identity / solo / arm re-read, stop of the record slot, stop_playing, a
    restore transaction (only the differences from the state before the take) and file_path. The clip slot is
    not stopped separately: stop_playing stops it.

    Guards, each before anything is changed: the Set must not be playing (read twice: the second time in the state
    batch, a few seconds before the first write) and Arrangement Record (`record_mode`) must be off; the identity
    must match; the record track needs an empty clip slot (the first one is used; none left raises RuntimeError
    naming the track, and nothing is deleted: clear_takes() does that, on request). Any error in the setup
    transaction aborts the take, and so does a clip shorter than its notes. The stops and the restores run
    after any exception, including KeyboardInterrupt, each step guarded separately (a second Ctrl-C does not skip
    the rest); the first exception is re-raised after all of them. A failed restore (checked per operation)
    raises after the cleanup, and the next take restores to the state before the failed one. Solo and ARM
    of every track are read at the start (Exclusive Arm can disarm another track when the record track is
    armed) and restored where they differ; track and slot counts are read at the first take and kept (rebuild the
    renderer if tracks are added). `sleep` and `clock` are injectable.

    `takes` is the log, one dict per take: file, slot, tempo, request_s (m), request_samples, lead_s, margin_s,
    start_s, spacing_beats, spacing_s, planned_s, lead_extra_s, lead_raised_s, sync_poll_s, sync_repolled,
    clip_length_beats, quantization_beats, file_shift_s, onsets_found, rejected, setup_errors, cleanup_errors,
    sample_rate, offset_median_ms, offset_trend_ms, and per candidate: index, note, expected_s, found_s, offset_ms,
    write_ms, lead_ms (note time minus the return of its write), early_ms (the start of its write against the
    earliest it could be), ok, reason."""

    def __init__(self, client, device: str, clip: str, record_track: str, units: dict, to_params, *, names=None,
                 convert=None, batch: bool = True, hit_s: float = 1.0, start_beat: float = 1.0, lead_factor: float = 1.5,
                 lead_floor_s: float = 0.5, min_lead_s: float = 0.03, margin_s: float | None = None,
                 spacing_beats: float | None = None, onset_early_s: float = 0.04, onset_late_s: float = 0.10,
                 settle_s: float = 0.3, sync_timeout_s: float | None = None, velocity: int = 100,
                 repoll_ratio: float = 1.25, shift_tol_s: float = 0.03, expect: dict | None = None, capture: bool = True,
                 sleep=None, clock=None):
        if hit_s <= 0 or start_beat <= 0 or lead_factor < 1.0 or lead_floor_s < 0 or min_lead_s < 0 or repoll_ratio < 1.0:
            raise ValueError("need hit_s > 0, start_beat > 0, lead_factor >= 1, lead_floor_s >= 0, min_lead_s >= 0 and repoll_ratio >= 1")
        mrec, mdev, mclip = _TRACK.fullmatch(record_track.strip()), _TRACK.match(device), _TRACK.match(clip)
        if not (mrec and mdev and mclip):
            raise ValueError("record_track must be 'live_set tracks N' and device and clip paths must start with 'live_set tracks N'")
        if mdev.group(0) != mclip.group(0):
            raise ValueError("the device (%s) and the MIDI clip (%s) are on different tracks" % (mdev.group(0), mclip.group(0)))
        if mdev.group(0) == mrec.group(0):
            raise ValueError("the record track is the MIDI track (%s): they must be two tracks" % mrec.group(0))
        import time as _time
        self.client, self.device, self.clip, self.record_track = client, device, clip, mrec.group(0)
        self.midi_track = mdev.group(0)
        self.guard_path = "%s parameters 0" % device.strip()        # the device's first parameter: the identity guard of every write
        self.units, self.to_params = units, to_params
        self.names = list(names) if names is not None else list(operator_parameters())
        self.convert = convert or (lambda params: to_live(params, units))
        self.batch, self.hit_s, self.start_beat = batch, hit_s, start_beat
        self.lead_factor, self.lead_floor_s, self.min_lead_s, self.margin_s = lead_factor, lead_floor_s, min_lead_s, margin_s
        self.spacing_beats, self.repoll_ratio, self.shift_tol_s = spacing_beats, repoll_ratio, shift_tol_s
        self.onset_early_s, self.onset_late_s, self.settle_s, self.sync_timeout_s = onset_early_s, onset_late_s, settle_s, sync_timeout_s
        self.velocity = velocity
        self.sleep = sleep or _time.sleep
        self.clock = clock or _time.monotonic
        self.lead_extra_s = 0.0
        self.takes: list = []
        self.sample_rate: int | None = None
        self.request_s: float | None = None
        self.expect: dict | None = dict(expect) if expect is not None else None
        self._ntracks: int | None = None
        self._nslots: int | None = None
        self._baseline = None                        # (solo, arm) of every track before the take, kept across a failed restore
        self._restore_failed = False
        self._poll_s = 0.0                           # the last take's sync poll: the margin is never shorter than the poll was
        if self.expect is None and capture:
            self.capture()

    # ---------------------------------------------------------------- the public face
    def capture(self) -> dict:
        """Read the record track's name, the MIDI track's name, the device's class_name and the identity (_live_ptr) of
        the device's first parameter (one request) and keep them as the identity every take is checked against, and
        every write is guarded by."""
        res = self.client.get_batch([{"path": self.record_track, "property": "name"}, {"path": self.midi_track, "property": "name"},
                                     {"path": self.device, "property": "class_name"},
                                     {"path": self.guard_path, "property": "_live_ptr"}])["results"]
        got = {"record": _value(res[0]), "midi": _value(res[1]), "device": _value(res[2]), "device_id": _value(res[3])}
        if any(v is None for v in got.values()):
            raise RuntimeError("could not read the track names and the device class (%s): the paths are wrong, or Live refused" % got)
        self.expect = got
        return got

    def render(self, patches: list) -> list:
        """One hit per patch (an empty array for one that failed), in order: one take for the batch, or one per patch."""
        patches = list(patches)
        out = [np.zeros((0, 1))] * len(patches)
        groups = [list(range(len(patches)))] if self.batch else [[i] for i in range(len(patches))]
        for g in groups:
            if g:
                for i, hit in zip(g, self._take([patches[i] for i in g])):
                    out[i] = hit
        return out

    __call__ = render

    def plan(self, note_lengths_s: list, request_s: float, tempo: float) -> dict:
        """The schedule of a take of candidates held for `note_lengths_s` seconds, with requests costing `request_s`
        seconds, at `tempo` BPM: lead_s, margin_s, start_s, spacing_beats, spacing_s and planned_s (the length of the
        clip's timeline: the first note, the spacings, the last hit; the requests before and after it are extra)."""
        spb = 60.0 / tempo
        m = request_s
        lead = max(self.lead_factor * m, m + self.lead_floor_s) + self.lead_extra_s
        margin = self.margin_s if self.margin_s is not None else max(0.2, self.repoll_ratio * m, self._poll_s)
        longest = max(note_lengths_s)
        need = max(longest + lead + margin, self.hit_s + margin)
        if self.spacing_beats is not None:
            if self.spacing_beats * spb < need:
                raise ValueError("spacing_beats %g is %.2f s: this take needs %.2f s (longest note %.2f s + lead %.2f s + margin %.2f s)"
                                 % (self.spacing_beats, self.spacing_beats * spb, need, longest, lead, margin))
            spacing_beats = self.spacing_beats
        else:
            spacing_beats = math.ceil(need / spb * 16.0) / 16.0
        start_beat = math.ceil(max(self.start_beat * spb, 3.0 * m + 0.3) / spb * 16.0) / 16.0
        n = len(note_lengths_s)
        planned = start_beat * spb + (n - 1) * spacing_beats * spb + max(self.hit_s, note_lengths_s[-1]) + 0.1
        return {"lead_s": lead, "margin_s": margin, "start_beat": start_beat, "start_s": start_beat * spb,
                "spacing_beats": spacing_beats, "spacing_s": spacing_beats * spb, "planned_s": planned}

    def clear_takes(self, confirm: bool = False) -> int:
        """Delete every clip on the record track (the recordings stay on disk). Only with confirm=True, only when the
        track names and the device class are as captured, and each delete carries expect=<the clip's name>, so a
        shifted index is refused by Live. Returns how many."""
        if not confirm:
            raise ValueError("clear_takes deletes the clips of %s: pass confirm=True" % self.record_track)
        if self.expect is None:
            raise RuntimeError("clear_takes needs the identity of the scratch tracks and nothing was captured: call capture() first "
                               "(or build the renderer with capture=True or expect=)")
        count = self.client.get(self.record_track, "clip_slots", limit=1)["value"]["count"]
        specs = [{"path": self.record_track, "property": "name"}, {"path": self.midi_track, "property": "name"},
                 {"path": self.device, "property": "class_name"}, {"path": self.guard_path, "property": "_live_ptr"}]
        specs += [{"path": "%s clip_slots %d" % (self.record_track, k), "property": "has_clip"} for k in range(count)]
        specs += [{"path": "%s clip_slots %d clip" % (self.record_track, k), "property": "name"} for k in range(count)]
        res = self.client.get_batch(specs)["results"]
        self._check_identity({"record": _value(res[0]), "midi": _value(res[1]), "device": _value(res[2]), "device_id": _value(res[3])})
        doomed = [(k, _value(res[4 + count + k])) for k in range(count) if _value(res[4 + k])]
        unnamed = [k for k, name in doomed if name is None]
        if unnamed:
            raise RuntimeError("could not read the name of the clip in slot(s) %s: nothing was deleted" % unnamed)
        for k, name in doomed:
            self.client.call("%s clip_slots %d" % (self.record_track, k), "delete_clip", confirm=True, expect=name)
        return len(doomed)

    # ---------------------------------------------------------------------- helpers
    def _check_identity(self, got: dict) -> None:
        if self.expect is None:
            if any(v is None for v in got.values()):
                raise RuntimeError("could not read the track names and the device class (%s)" % got)
            self.expect = dict(got)
        elif self.expect.get("device_id") is None and got.get("device_id") is not None and \
                {k: v for k, v in got.items() if k != "device_id"} == {k: v for k, v in self.expect.items() if k != "device_id"}:
            self.expect["device_id"] = got["device_id"]           # an expect given without the id: the first read supplies it
        elif got != self.expect:
            raise RuntimeError("the scratch tracks are not where they were: expected %s, found %s (a track was inserted, deleted or "
                               "renamed above them?): nothing was written" % (self.expect, got))

    def _timed(self, fn, *a, **k):
        t0 = self.clock()
        r = fn(*a, **k)
        return r, self.clock() - t0

    def _param_ops(self, live_values: dict) -> list:
        return [{"op": "set", "path": "%s parameters %d" % (self.device, self.names.index(n)), "property": "value", "value": v}
                for n, v in live_values.items()]

    def _guard_op(self) -> dict:
        """The first op of every write transaction: a harmless call (str_for_value 0.0) on the device's first parameter that
        carries expect={"id": its _live_ptr}. Live checks `expect` before the call, and the transaction (stop_on_error)
        halts at a mismatch before any set: a track inserted above the scratch tracks, which moves the paths, redirects
        nothing."""
        return {"op": "call", "path": self.guard_path, "function": "str_for_value", "args": [0.0],
                "expect": {"id": self.expect["device_id"]}}

    def _write(self, live_values: dict) -> tuple[bool, str | None, bool]:
        """(ok, why, refused): refused when the identity guard stopped the transaction (the device is not the one captured)."""
        ops = [self._guard_op()] + self._param_ops(live_values)
        try:
            t = self.client.transaction(ops, stop_on_error=True)
        except Exception as e:                                       # a refusal or a timeout: this candidate only
            return False, "parameter write raised %s: %s" % (type(e).__name__, e), False
        res = t.get("results", [])
        if not res or not res[0].get("ok"):
            return False, "the identity guard refused the write: %s" % (res[0].get("error") if res else "no result"), True
        bad = [r for r in res if not r.get("ok")]
        if bad or len(res) < len(ops):
            return False, "%d of %d parameter writes failed (first: %s)" % (
                len(bad) or len(ops) - len(res), len(ops) - 1, bad[0].get("error") if bad else "halted"), False
        return True, None, False

    def _position(self) -> tuple[bool, float, float | None, float, float]:
        """(playing, position in beats, clip length in beats, wall time the request was sent at, its duration)."""
        t0 = self.clock()
        r = self.client.get_batch([{"path": self.clip, "property": "is_playing"},
                                   {"path": self.clip, "property": "playing_position"},
                                   {"path": self.clip, "property": "length"}])["results"]
        d = self.clock() - t0
        length = _value(r[2])
        return bool(_value(r[0])), float(_value(r[1]) or 0.0), None if length is None else float(length), t0, d

    def _clip_slot(self) -> str:
        """The clip slot that holds the MIDI clip: its path without the trailing ' clip'."""
        if not self.clip.endswith(" clip"):
            raise ValueError("clip must be a clip path ('live_set tracks N clip_slots K clip'), got %r" % self.clip)
        return self.clip[:-len(" clip")]

    def _state_specs(self) -> list:
        nt, ns = self._ntracks, self._nslots
        specs = [{"path": self.record_track, "property": "name"}, {"path": self.midi_track, "property": "name"},
                 {"path": self.device, "property": "class_name"}, {"path": self.guard_path, "property": "_live_ptr"},
                 {"path": "live_set", "property": "is_playing"}, {"path": "live_set", "property": "record_mode"},
                 {"path": "live_set", "property": "clip_trigger_quantization"}]
        specs += [{"path": "live_set tracks %d" % i, "property": "solo"} for i in range(nt)]
        specs += [{"path": "live_set tracks %d" % i, "property": "arm"} for i in range(nt)]
        specs += [{"path": "%s clip_slots %d" % (self.record_track, k), "property": "has_clip"} for k in range(ns)]
        return specs

    def _take(self, patches: list) -> list:
        import soundfile as sf
        empty = np.zeros((0, 1))
        cands = [{"index": i, "ok": False, "reason": None, "note": None} for i in range(len(patches))]
        plans = []                                   # (candidate, live values, note, length s)
        for c, p in zip(cands, patches):
            try:
                params, note, length_ms = self.to_params(p)
                values = self.convert(params)
                for n in values:
                    if n not in self.names:
                        raise ValueError("%r is not a parameter of the device" % n)
            except (ValueError, KeyError) as e:
                c["reason"] = "does not convert to the device's values: %s" % e
                continue
            c["note"] = int(note)
            plans.append((c, values, int(note), max(0.0, float(length_ms)) / 1000.0))
        if not plans:
            self.takes.append({"file": None, "slot": None, "candidates": cands, "failures": [(c["index"], c["reason"]) for c in cands]})
            return [empty] * len(patches)
        cl = self.client
        # the guards, which also measure the request time (three requests the take needs anyway)
        playing, t1 = self._timed(cl.get, "live_set", "is_playing")
        if playing["value"]:
            raise RuntimeError("the Set is playing: not starting a take (stop it first)")
        tempo_r, t2 = self._timed(cl.get, "live_set", "tempo")
        tempo = float(tempo_r["value"])
        if self._ntracks is None:
            self._ntracks = cl.get("live_set", "tracks", limit=1)["value"]["count"]
            self._nslots = cl.get(self.record_track, "clip_slots", limit=1)["value"]["count"]
        nt = self._ntracks
        state, t3 = self._timed(cl.get_batch, self._state_specs())
        v = [_value(r) for r in state["results"]]
        ident = {"record": v[0], "midi": v[1], "device": v[2], "device_id": v[3]}
        if v[4]:
            raise RuntimeError("the Set is playing: not starting a take (it started after the first check)")
        if v[5]:
            raise RuntimeError("Arrangement Record is on (live_set record_mode): a take would record into the arrangement; turn it off")
        self._check_identity(ident)
        quant_enum = v[6]
        quant = QUANT_BEATS[int(quant_enum)] if isinstance(quant_enum, (int, float)) and 0 <= int(quant_enum) < len(QUANT_BEATS) else None
        now_solo = [bool(x) for x in v[7:7 + nt]]
        now_arm = [None if x is None else bool(x) for x in v[7 + nt:7 + 2 * nt]]
        slots = [bool(x) for x in v[7 + 2 * nt:]]
        if all(slots):
            raise RuntimeError("no empty clip slot on %s (all %d hold clips): delete the takes, or add scenes (clear_takes(confirm=True) "
                               "deletes the clips; the recordings stay on disk)" % (self.record_track, len(slots)))
        rec_index = int(self.record_track.split()[-1])
        if rec_index >= nt or len(now_solo) != nt:
            raise RuntimeError("the record track index %d is outside the %d tracks counted at the first take: rebuild the renderer" % (rec_index, nt))
        if self._baseline is None or not self._restore_failed:
            self._baseline = (now_solo, now_arm)                  # what the cleanup restores: kept across a failed restore
        base_solo, base_arm = self._baseline
        slot_index = slots.index(False)
        m = self.request_s = max(t1, t2, t3)
        spb = 60.0 / tempo
        sched = self.plan([length for *_, length in plans], m, tempo)
        lead, spacing_beats = sched["lead_s"], sched["spacing_beats"]
        beats = [sched["start_beat"] + k * spacing_beats for k in range(len(plans))]
        notes = [{"pitch": n, "start_time": b, "duration": max(length / spb, 1.0 / 16.0), "velocity": self.velocity}
                 for (_, _, n, length), b in zip(plans, beats)]
        end_beat = beats[-1] + math.ceil((max(self.hit_s, plans[-1][3]) + sched["margin_s"]) / spb) + 1
        slot = "%s clip_slots %d" % (self.record_track, slot_index)
        take = dict(sched, file=None, slot=slot_index, tempo=tempo, request_s=m, request_samples=[round(x, 3) for x in (t1, t2, t3)],
                    notes=notes, candidates=cands, sample_rate=None, failures=[], lead_extra_s=self.lead_extra_s, lead_raised_s=0.0,
                    setup_errors=[], cleanup_errors=[], offset_median_ms=None, offset_trend_ms=None, clip_length_beats=None,
                    quantization_beats=quant, fire_window_s=None, file_shift_s=None, onsets_found=None, rejected=None, sync_poll_s=None,
                    sync_repolled=False)
        self.takes.append(take)
        # what the cleanup writes if its own re-read fails: the record track's solo and arm, which this take changes
        blind_restore = []
        if not base_solo[rec_index]:
            blind_restore.append({"op": "set", "path": self.record_track, "property": "solo", "value": False})
        if base_arm[rec_index] is False:
            blind_restore.append({"op": "set", "path": self.record_track, "property": "arm", "value": False})
        started = touched = False
        pending = None
        try:
            try:
                cl.notes_remove(self.clip)
            except Exception:
                pass                                                  # an empty clip may refuse
            cl.notes_add(self.clip, notes)
            ops = []
            if not base_solo[rec_index]:
                ops.append({"op": "set", "path": self.record_track, "property": "solo", "value": True})
            if base_arm[rec_index] is False:
                ops.append({"op": "set", "path": self.record_track, "property": "arm", "value": True})
            ops += [{"op": "set", "path": self.clip, "property": "looping", "value": False},
                    {"op": "set", "path": self.clip, "property": "loop_end", "value": float(end_beat)},
                    {"op": "set", "path": self.clip, "property": "end_marker", "value": float(end_beat)}]
            nsetup = len(ops)
            ops += self._param_ops(plans[0][1])                       # candidate 0's parameters, before the clip is fired
            ops.insert(0, self._guard_op())                           # the identity guard: first, and the transaction halts on it
            touched = True
            t_w = self.clock()
            tr = cl.transaction(ops, stop_on_error=True)
            t_d = self.clock()
            results = tr.get("results", [])
            if not results or not results[0].get("ok"):
                raise RuntimeError("the scratch tracks moved during the take: the device guard refused the setup transaction (%s); nothing was written"
                                   % (results[0].get("error") if results else "no result"))
            take["setup_errors"] = [r.get("error") for r in results[1:1 + nsetup] if not r.get("ok")]
            if take["setup_errors"] or len(results) < 1 + nsetup:
                raise RuntimeError("setting up the take failed (%s): nothing was played" % (take["setup_errors"][:2] or "halted"))
            bad0 = [r for r in results[1 + nsetup:] if not r.get("ok")]
            short0 = len(results) < len(ops)
            writes = [(not bad0 and not short0, None if not (bad0 or short0) else "%d of %d parameter writes failed (first: %s)" % (
                len(bad0) or len(ops) - len(results), len(ops) - 1 - nsetup, bad0[0].get("error") if bad0 else "halted"), t_d - t_w, t_d)]
            t_fire = self.clock()
            cl.call(slot, "fire")
            started = True
            cl.call(self._clip_slot(), "fire")
            take["fire_window_s"] = round(self.clock() - t_fire, 3)
            limit = self.sync_timeout_s if self.sync_timeout_s is not None else max(3.0, 8.0 * m)
            deadline = self.clock() + limit
            while True:
                playing, pos, length, t_sent, d = self._position()
                if playing:
                    break
                if self.clock() > deadline:
                    raise RuntimeError("the MIDI clip did not start playing within %.1f s of being fired" % limit)
                self.sleep(0.005)
            if d > self.repoll_ratio * m:                              # a slow poll is a poor sync: once more, keep the shorter
                playing2, pos2, length2, t_sent2, d2 = self._position()
                take["sync_repolled"] = True
                if playing2 and d2 < d:
                    pos, length, t_sent, d = pos2, length2, t_sent2, d2
            take["sync_poll_s"] = round(d, 3)
            self._poll_s = d
            take["clip_length_beats"] = length
            if length is not None and length < beats[-1] + 0.1:
                raise RuntimeError("the MIDI clip is %g beats long, the last note starts at %g: it could not be lengthened" % (length, beats[-1]))
            beat0 = t_sent - pos * spb                                # believed wall time of beat 0; the true one is up to d later
            for k in range(1, len(plans)):
                note_wall = beat0 + beats[k] * spb
                self.sleep(max(0.0, note_wall - lead - self.clock()))
                t_w = self.clock()
                earliest = beat0 + beats[k - 1] * spb + d + plans[k - 1][3]
                plans[k][0]["early_ms"] = round(1000.0 * (t_w - earliest), 1)
                if t_w < earliest - 1e-9 and not plans[k - 1][0]["reason"]:
                    plans[k - 1][0]["reason"] = ("the next candidate's write started %.0f ms before this hit could have ended (clock sync "
                                                 "uncertainty %.0f ms + note %.0f ms)" % (1000 * (earliest - t_w), 1000 * d, 1000 * plans[k - 1][3]))
                ok, why, refused = self._write(plans[k][1])
                t_d = self.clock()
                if refused:
                    raise RuntimeError("the scratch tracks moved during the take: the device guard refused the write of candidate %d "
                                       "(%s); that write and the ones after it were not sent" % (k, why))
                writes.append((ok, why, t_d - t_w, t_d))
                plans[k][0]["late_start_ms"] = round(1000.0 * max(0.0, t_w - (note_wall - lead)), 1)
            for k, (ok, why, took, ended) in enumerate(writes):
                c = plans[k][0]
                got = beat0 + beats[k] * spb - ended
                c["write_ms"], c["lead_ms"] = round(1000.0 * took, 1), round(1000.0 * got, 1)
                if c["reason"]:
                    continue
                if not ok:
                    c["reason"] = why
                elif k and got < self.min_lead_s:
                    c["reason"] = "parameters written only %.0f ms before the note (at least %.0f ms)" % (1000 * got, 1000 * self.min_lead_s)
            self.sleep(max(0.0, beat0 + beats[-1] * spb + max(self.hit_s, plans[-1][3]) + 0.1 - self.clock()))
        except BaseException as e:                                    # cleanup first, then this is raised again
            pending = e
        errors = self._cleanup(take, slot, started, touched, base_solo, base_arm, blind_restore, nt, rec_index)
        first = pending if pending is not None else (errors[0] if errors else None)
        if first is not None:
            if take["warnings"]:                                      # the raised exception says what may still be running
                first.args = ((str(first.args[0]) if first.args else str(first)) + " [%s]" % "; ".join(take["warnings"]),) + tuple(first.args[1:])
            raise first
        late = [c for c, *_ in plans[1:] if c.get("lead_ms") is not None and c["lead_ms"] / 1000.0 < self.min_lead_s]
        if late and len(late) > len(plans) / 4.0:                     # adapt: the next batch starts earlier
            short = max(self.min_lead_s - c["lead_ms"] / 1000.0 for c in late)
            take["lead_raised_s"] = round(short + 0.1, 3)
            self.lead_extra_s += short + 0.1
        elif not late and self.lead_extra_s:
            self.lead_extra_s = 0.0 if self.lead_extra_s < 0.02 else self.lead_extra_s / 2.0
        try:
            path = cl.get(slot + " clip", "file_path")["value"]
        except Exception as e:
            raise RuntimeError("no recording in slot %d of %s: %s" % (slot_index, self.record_track, e))
        take["file"] = path
        for _ in range(5):
            if path and pathlib.Path(path).exists() and pathlib.Path(path).stat().st_size > 44:
                break
            self.sleep(self.settle_s)
        x, sr = sf.read(path, dtype="float64", always_2d=True)
        self.sample_rate = take["sample_rate"] = int(sr)
        out = [empty] * len(patches)
        shift = self._file_shift(take, x, sr, [b * spb for b in beats], spacing_beats * spb, quant)
        if shift is None:
            for c, *_ in plans:
                c["reason"] = c["reason"] or "the take was rejected: %s" % take["rejected"]
            take["failures"] = [(c["index"], c["reason"]) for c in cands if c["reason"]]
            return out
        for k, (c, _, _, _) in enumerate(plans):
            c["expected_s"] = round(beats[k] * spb + shift, 4)
            if c["reason"]:
                continue
            cut, onset = oneshot.hit_cut(x, sr, beats[k] * spb + shift, self.hit_s)
            if onset is None:
                c["reason"] = "no hit near %.3f s in the take (nothing rises out of the previous hit's tail there)" % c["expected_s"]
                continue
            c["found_s"], c["offset_ms"] = round(onset, 4), round(1000.0 * (onset - c["expected_s"]), 1)
            if not -self.onset_early_s <= onset - c["expected_s"] <= self.onset_late_s:
                c["reason"] = "the onset is %.0f ms from where the note was placed (allowed -%.0f to +%.0f ms)" % (
                    c["offset_ms"], 1000 * self.onset_early_s, 1000 * self.onset_late_s)
            elif len(x) - int(round(onset * sr)) < int(self.hit_s * sr) - 1:
                c["reason"] = "the take ends %.0f ms after the onset, %.0f ms are needed" % (
                    1000.0 * (len(x) / sr - onset), 1000 * self.hit_s)
            elif not np.isfinite(cut).all() or np.abs(cut).max() < 1e-7:
                c["reason"] = "the hit is silent"
            else:
                c["ok"] = True
                out[c["index"]] = cut
        found = [c["offset_ms"] for c, *_ in plans if "offset_ms" in c]
        if found:
            take["offset_median_ms"] = float(np.median(found))
            take["offset_trend_ms"] = found[-1] - found[0]
        take["failures"] = [(c["index"], c["reason"]) for c in cands if c["reason"]]
        return out

    def _file_shift(self, take: dict, x: np.ndarray, sr: int, expected: list, spacing_s: float, quant: float | None):
        """How far into the file the clip's first note sounds, against where the clip puts it (0 within the late
        tolerance), or None with take["rejected"] set when it cannot be trusted: see the class docstring."""
        mag = np.abs(x).max(axis=1)
        if not len(mag) or mag.max() <= 0:
            take["rejected"] = "the recording is silent"
            return None
        shift = oneshot._onset(x, mag, sr, False) / sr - expected[0]
        take["file_shift_s"] = round(shift, 4)
        take["onsets_found"] = len(oneshot._segments(x, sr))
        if -self.onset_early_s <= shift <= self.onset_late_s:
            return 0.0
        if shift < 0:
            take["rejected"] = ("the first sound is %.2f s before where the clip puts note 0: the recording started after the clip, "
                                "or the clip started earlier than believed" % -shift)
            return None
        step = quant * 60.0 / take["tempo"] if quant else None
        window = take["fire_window_s"] or 0.0
        on_grid = bool(step) and abs(shift - round(shift / step) * step) <= self.shift_tol_s
        if shift > window + self.shift_tol_s and not on_grid:
            take["rejected"] = ("the first sound is %.2f s after where the clip puts note 0: more than the %.2f s between the two fires and "
                                "not a whole number of launch-quantization steps (%s): note 0 may be missing from the recording, so no hit "
                                "is cut by a guessed shift" % (shift, window, "no quantization" if not step else "step %.2f s" % step))
            return None
        if shift >= spacing_s - 0.1:
            # a shift of about a spacing or more is also what a MISSING note 0 looks like (the first onset is then note 1),
            # and one extra onset (a ghost) would defeat a count: so EVERY note must have an onset where the shift puts it
            for k, e in enumerate(expected):
                _, onset = oneshot.hit_cut(x, sr, e + shift, self.hit_s)
                if onset is None or not -self.onset_early_s <= onset - (e + shift) <= self.onset_late_s:
                    take["rejected"] = ("the first sound is %.2f s late, about a spacing (%.2f s): note 0 may be missing, and with the shift "
                                        "note %d has no onset near %.2f s, so the shift cannot be trusted" % (shift, spacing_s, k, e + shift))
                    return None
        return shift

    def _cleanup(self, take: dict, slot: str, started: bool, touched: bool, base_solo: list, base_arm: list, blind_restore: list,
                 nt: int, rec_index: int) -> list:
        """Re-read the identity and every track's solo and arm, stop the record slot and the transport, restore what this
        take changed. Every step runs whatever happened to the others, BaseException included; the caught exceptions
        are returned in the order they happened (the caller raises the first, with `warnings` appended to its message
        when a stop could not be made: the transport or the recording may still be running).

        Restores are limited to what the take did: the record track's solo and arm if the take set them, and, only
        then, the solo / arm of other tracks that were set before and are now cleared (Live's exclusive solo and
        exclusive arm do that when the record track is soloed or armed). A solo or arm the user changed during the take
        is not reverted."""
        cl = self.client
        errors: list = []
        ctx: dict = {"moved": False}
        warnings: list = []

        def reread():
            specs = [{"path": self.record_track, "property": "name"}, {"path": self.midi_track, "property": "name"},
                     {"path": self.device, "property": "class_name"}, {"path": self.guard_path, "property": "_live_ptr"}]
            specs += [{"path": "live_set tracks %d" % i, "property": "solo"} for i in range(nt)]
            specs += [{"path": "live_set tracks %d" % i, "property": "arm"} for i in range(nt)]
            r = [_value(x) for x in cl.get_batch(specs)["results"]]
            ctx["ident"] = {"record": r[0], "midi": r[1], "device": r[2], "device_id": r[3]}
            ctx["solo"] = [bool(x) for x in r[4:4 + nt]]
            ctx["arm"] = [None if x is None else bool(x) for x in r[4 + nt:4 + 2 * nt]]
            if ctx["ident"] != self.expect:
                ctx["moved"] = True
                self._restore_failed = True                           # nothing is restored: the next take must not take this as its baseline
                raise RuntimeError("the scratch tracks moved during the take (expected %s, found %s): the record slot and the solo/arm "
                                   "states were NOT written to; the real record track may still be soloed and armed" % (self.expect, ctx["ident"]))

        def twice(fn):
            """fn, and once more if it raised (a Ctrl-C or a timeout may have cost the request): the recording and the
            transport must not be left running. The first failure is kept; a second one propagates."""
            try:
                return fn()
            except BaseException as e:
                errors.append(e)
                return fn()

        def stop_slot():
            if started and not ctx["moved"]:
                try:
                    twice(lambda: cl.call(slot, "stop"))
                except BaseException:
                    warnings.append("the recording on %s may still be running" % slot)
                    raise

        def stop_transport():
            if touched:
                try:
                    twice(lambda: cl.call("live_set", "stop_playing"))
                except BaseException:
                    warnings.append("the transport may still be running: stop_playing failed twice")
                    raise

        def restore():
            if not touched or ctx["moved"]:
                return
            if "solo" in ctx:
                ours_solo, ours_arm = not base_solo[rec_index], base_arm[rec_index] is False
                ops = []
                if ours_arm and ctx["arm"][rec_index]:
                    ops.append({"op": "set", "path": self.record_track, "property": "arm", "value": False})      # first: then re-arm the others
                if ours_solo and ctx["solo"][rec_index]:
                    ops.append({"op": "set", "path": self.record_track, "property": "solo", "value": False})
                if ours_solo:
                    ops += [{"op": "set", "path": "live_set tracks %d" % i, "property": "solo", "value": True}
                            for i in range(nt) if i != rec_index and base_solo[i] and not ctx["solo"][i]]
                if ours_arm:
                    ops += [{"op": "set", "path": "live_set tracks %d" % i, "property": "arm", "value": True}
                            for i in range(nt) if i != rec_index and base_arm[i] is True and ctx["arm"][i] is False]
            else:
                ops = blind_restore                                   # the re-read failed: at least the record track
            if ops:
                res = twice(lambda: cl.transaction(ops, stop_on_error=False)).get("results", [])
                bad = [r for r in res if not r.get("ok")]
                if bad or len(res) < len(ops):
                    self._restore_failed = True
                    raise RuntimeError("restoring solo/arm failed (%d of %d operations: %s): the Set's track states may be wrong" % (
                        len(bad) or len(ops) - len(res), len(ops), bad[0].get("error") if bad else "no result"))
            self._restore_failed = False

        for step in ((reread if touched else None), stop_slot, stop_transport, restore):
            if step is None:
                continue
            try:
                step()
            except BaseException as e:                                # a second Ctrl-C included: the other steps still run
                errors.append(e)
                if step is restore:
                    self._restore_failed = True
        take["cleanup_errors"] = [str(e) for e in errors]
        take["warnings"] = warnings
        return errors


def _value(r):
    """The value of one get_batch result, None when the read failed."""
    return r.get("value") if r.get("ok", True) else None


def operator_to_params(structure: dict):
    """to_params for Operator: patch -> (operator_live's physical values, its note, its note length in ms)."""
    def to_params(p: dict):
        live = operator_live(p, structure)
        return live["params"], live["note"], live["note_length_ms"]
    return to_params


class OperatorRenderer(LiveRenderer):
    """LiveRenderer for Operator. `structure` (from operator_guess) can be given later with bind(): a guess is needed
    before the voice's structure is known."""

    def __init__(self, client, device: str, clip: str, record_track: str, structure: dict | None = None,
                 units: dict | None = None, **kw):
        super().__init__(client, device, clip, record_track, units if units is not None else operator_units(), self._params, **kw)
        self.structure = structure

    def _params(self, p: dict):
        if self.structure is None:
            raise RuntimeError("no voice structure: bind(operator_guess(...)['structure']) first")
        return operator_to_params(self.structure)(p)

    def bind(self, structure: dict) -> "OperatorRenderer":
        self.structure = structure
        return self


# ---- patch_render, repeat_check, patch_match: the functions on top of a renderer

def patch_render(patches: list, renderer, structure: dict | None = None) -> list:
    """One hit per semantic patch through `renderer` (a LiveRenderer; `structure` binds an OperatorRenderer first).
    Failed candidates are empty arrays (the reasons are in renderer.takes[-1])."""
    if structure is not None:
        renderer.bind(structure)
    return renderer.render(patches)


def repeat_check(renderer, patch: dict, n: int = 16, structure: dict | None = None, target: np.ndarray | None = None,
                 sr: int | None = None, **distance_kw) -> dict:
    """Render the same patch `n` times in one batch and measure how much the renders differ: the number `search`'s
    margin has to be read against. Returns {"n", "ok" (hits rendered), "pairwise": {"median", "max", "mean", "min",
    "count"} (spectral_distance over every pair), "readings": {key: {"median", "std", "min", "max"}} (profile_array
    of each hit: peak_db, attack_ms, fall_20/40_ms, pitch_body_hz, click_db), "repeat_spread" (the std of the distance
    to `target`, which is what search's repeat_spread is, when a target is given, else None), "failures", "take"}.
    Raises RuntimeError when fewer than 2 hits came back."""
    hits = patch_render([patch] * n, renderer, structure)
    take = renderer.takes[-1] if getattr(renderer, "takes", None) else None
    sr = sr or getattr(renderer, "sample_rate", None)
    good = [h for h in hits if len(h)]
    if len(good) < 2 or not sr:
        raise RuntimeError("repeat_check got %d hit(s) of %d: %s" % (len(good), n, take["failures"] if take else "no log"))
    d = np.array([spectral_distance(a, b, sr, **distance_kw) for i, a in enumerate(good) for b in good[i + 1:]])
    readings = {}
    profs = [oneshot.profile_array(h, sr, single=True) for h in good]
    for key in PARAM_KEYS:
        v = np.array([p[key] for p in profs if p.get(key) is not None], dtype=float)
        if len(v):
            readings[key] = {"median": float(np.median(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                             "min": float(v.min()), "max": float(v.max())}
    spread = None
    if target is not None:
        scores = [spectral_distance(h, target, sr, **distance_kw) for h in good]
        spread = float(np.std(scores, ddof=1))
    return {"n": n, "ok": len(good), "pairwise": {"median": float(np.median(d)), "max": float(d.max()), "mean": float(d.mean()),
                                                  "min": float(d.min()), "count": len(d)},
            "readings": readings, "repeat_spread": spread, "failures": take["failures"] if take else [], "take": take}


MIN_FLOOR_DB = 20.0        # patch_match never sets the distance's floor under this (it would ignore most of the spectrum)
FLOOR_MARGIN_DB = 3.0


def noise_floor_db(x: np.ndarray, sr: int, profile: dict) -> tuple[float, str]:
    """(floor_db for spectral_distance, how it was estimated) for a sample file x [n, channels]. The floor is how far
    under the loudest STFT magnitude the distance's log term clamps; to leave the sample's own noise bed unmatched it
    is set FLOOR_MARGIN_DB shallower than the bed's depth under the hit's loudest millisecond (so the clamp sits that
    far over the bed), between MIN_FLOOR_DB and 60. The bed is read, in this order, from: the file's tail (the last
    0.1 s, when the profile has a fall_40_ms and the file runs 3 x fall_40_ms + 0.15 s past the onset: by then an
    exponential hit is 120 dB down and what is left is the bed), the lead before the onset (when at least 50 ms long),
    else there is no bed to measure and the floor stays 60. Depth is taken in
    the time domain; for a tonal hit the bed is deeper in the spectrum than here (a tone concentrates in a bin),
    so the clamp sits higher over it, which is the safe side."""
    mono = np.asarray(x, dtype=np.float64).mean(axis=1)
    frame = max(1, int(round(0.001 * sr)))
    c = np.concatenate([[0.0], np.cumsum(mono ** 2)])
    rms = np.sqrt((c[frame:] - c[:-frame]) / frame) if len(mono) > frame else np.array([math.sqrt(c[-1] / max(1, len(mono)))])
    peak = float(rms.max())
    onset = (profile.get("onset_ms") or 0.0) / 1000.0
    f40 = profile.get("fall_40_ms")
    bed, how = None, "no bed to measure (no tail past the hit, under 50 ms of lead): the floor stays 60 dB"
    if f40 is not None and onset + 3.0 * f40 / 1000.0 + 0.15 <= len(mono) / sr:
        seg = mono[-int(0.1 * sr):]
        bed, how = float(np.sqrt(np.mean(seg ** 2))), "the file's last 0.1 s, well after the hit fell"
    elif onset - 0.01 >= 0.05:
        seg = mono[:int((onset - 0.01) * sr)]
        bed, how = float(np.sqrt(np.mean(seg ** 2))), "the %.0f ms before the onset" % (1000 * (onset - 0.01))
    if not bed or bed <= 0 or peak <= 0:
        return 60.0, how if bed is None else "the bed is digital silence: the floor stays 60 dB"
    depth = 20.0 * math.log10(peak / bed)
    floor = max(MIN_FLOOR_DB, min(60.0, depth - FLOOR_MARGIN_DB))
    return floor, "%s: %.1f dB under the loudest millisecond, floor = that - %.0f dB, within %.0f to 60" % (
        how, depth, FLOOR_MARGIN_DB, MIN_FLOOR_DB)


def patch_match(sample_path, renderer, rounds: int = 8, population: int = 16, seed: int = 0, sr: int | None = None,
                hit_s: float | None = None, floor_db: float | None = None, units: dict | None = None,
                verify: int = 4) -> dict:
    """Match an Operator patch to the one-shot at `sample_path` by search through `renderer`: profile the sample,
    operator_guess, search_arguments (start, bounds, probes), the distance's floor_db from the sample's own noise
    bed (noise_floor_db, at most 60; `floor_db` overrides), then `search` with the renderer's `render`, then ONE
    verification take.
    `renderer` is a LiveRenderer / OperatorRenderer (bound to the guess's structure here) or any callable
    render(patches) -> hits (then `sr`, the renders' sample rate, is required unless it has a `sample_rate`
    attribute set; a renderer that does not know its rate yet is asked for one hit of the start first). The hit
    is compared over `hit_s` seconds (default: the sample's duration, at most oneshot.DIST_MAX_S); a renderer's own
    `hit_s` is set for the run and put back afterwards.
    A RUN CAN END EARLY: when the renderer raises (no empty slot left, the Set started playing, a track moved) the
    search returns what it had and the exception is in the top-level "error" (None when the run completed); check
    it before reading the rest. A failed verification take is in "verification_error".
    The scores are NOT comparable as they stand: "start_distance" (alias "start_distance_round0") is ONE render of
    the start in round 0, "distance" (alias "distance_search") is the mean over the renders of the best patch, one of
    which won its selection, and a noise-like hit never repeats exactly (the spread is 0.01 to 0.02). The comparable
    pair is "start_distance_verified" and "distance_verified": `verify` fresh renders of the start and of the best in
    ONE take, with their std ("..._std"); gain = start - best, to be read against those stds.
    Returns {"best" (semantic patch), "live" (Live's values for it, through to_live), "error", "distance",
    "distance_search", "start_distance", "start_distance_round0", "start_distance_verified",
    "start_distance_verified_std", "distance_verified", "distance_verified_std", "verification_error", "per_round",
    "rounds", "renders", "takes" (the renderer's takes used, the verification included), "repeat_spread", "stalled",
    "floor_db", "floor_reading", "readings" (oneshot.compare of a best render against the sample: past-threshold
    lines), "guess", "search" (the SearchResult), "best_render"}."""
    import soundfile as sf
    units = units if units is not None else operator_units()
    profile = oneshot.profile(sample_path)
    guess = operator_guess(profile)
    args = search_arguments(guess, units)
    x, sr0 = sf.read(str(sample_path), dtype="float64", always_2d=True)
    onset = (profile.get("onset_ms") or 0.0) / 1000.0
    dur = profile.get("duration_ms")
    if hit_s is None:
        hit_s = min(oneshot.DIST_MAX_S, (dur / 1000.0 + 0.1) if dur else max(0.1, len(x) / sr0 - onset))
    if hasattr(renderer, "bind"):
        renderer.bind(guess["structure"])
    had_hit_s = getattr(renderer, "hit_s", None)
    if hasattr(renderer, "hit_s"):
        renderer.hit_s = hit_s
    try:
        takes0 = len(getattr(renderer, "takes", []))
        render = renderer.render if hasattr(renderer, "render") else renderer
        sr = sr or getattr(renderer, "sample_rate", None)
        if sr is None:
            render([args["start"]])                                # one hit of the start, to learn the recording's rate
            sr = getattr(renderer, "sample_rate", None)
            if sr is None:
                raise ValueError("the renderer does not know its sample rate: pass sr=")
        cut, _ = oneshot.hit_cut(x, sr0, onset, hit_s)
        target = oneshot._resample(cut, sr0, sr) if sr != sr0 else cut
        if floor_db is None:
            floor_db, how = noise_floor_db(x, sr0, profile)
        else:
            how = "given"
        cache = {}

        def key(p):
            return tuple(sorted(p.items()))

        def recording(patches):
            hits = render(patches)
            for p, h in zip(patches, hits):
                if len(h):
                    cache[key(p)] = h
            return hits

        res = search(args["start"], args["bounds"], recording, target, sr, rounds=rounds, population=population, seed=seed,
                     floor_db=floor_db, probes=args.get("probes"))
        best_hit = cache.get(key(res.best))
        out = {"start_distance_verified": None, "start_distance_verified_std": None, "distance_verified": None,
               "distance_verified_std": None, "verification_error": None}
        if verify and res.error:
            out["verification_error"] = "skipped: the search ended with an error (%s)" % res.error
        elif verify:
            try:
                hits = render([args["start"]] * verify + [res.best] * verify)
                for name, part in (("start_distance_verified", hits[:verify]), ("distance_verified", hits[verify:])):
                    d = [spectral_distance(h, target, sr, floor_db=floor_db) for h in part if len(h)]
                    if d:
                        out[name] = float(np.mean(d))
                        out[name + "_std"] = float(np.std(d, ddof=1)) if len(d) > 1 else None
                best_hit = next((h for h in hits[verify:] if len(h)), best_hit)
            except Exception as e:                                 # the search's result is still worth returning
                out["verification_error"] = "%s: %s" % (type(e).__name__, e)
        if best_hit is None and not res.error:                     # after an error no further take is fired
            try:
                best_hit = next((h for h in render([res.best]) if len(h)), None)
            except Exception as e:
                out["verification_error"] = out["verification_error"] or "%s: %s" % (type(e).__name__, e)
        readings = None
        if best_hit is not None:
            readings = oneshot.compare(oneshot.profile_array(best_hit, sr, single=True), profile)["readings"]
        live = to_live(operator_live(res.best, guess["structure"])["params"], units)
        return dict(out, best=res.best, live=live, error=res.error, distance=res.distance, distance_search=res.distance,
                    start_distance=res.start_distance, start_distance_round0=res.start_distance, per_round=res.per_round,
                    rounds=res.rounds, renders=res.renders, takes=len(getattr(renderer, "takes", [])) - takes0,
                    repeat_spread=res.repeat_spread, stalled=res.stalled, floor_db=floor_db, floor_reading=how,
                    readings=readings, guess=guess, search=res, best_render=best_hit)
    finally:
        if hasattr(renderer, "hit_s"):
            renderer.hit_s = had_hit_s
