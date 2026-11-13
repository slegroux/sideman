"""Match a synth patch to a one-shot by search, against any renderer.

`search` minimises `oneshot.spectral_distance(candidate, target)` over a patch's free parameters. It knows
nothing about what renders: `render(patches)` returns one hit per patch, and in practice that will be Ableton
Live recording a batch of candidates in one take, so renders are expensive and come in batches. Numpy only.

The method is a (mu, lambda) evolution strategy with weighted recombination and a step size per parameter:
a round is one batch of `population` renders and there are only about eight rounds, which suits a small
sample-and-select method with no gradient and no model to fit (a full covariance, as CMA-ES keeps, cannot be
estimated from so few samples); steps adapt per parameter by whether the selected moves agree in direction.
The search is judged by the score it reaches, not by the parameters it returns: parameters that trade off against
each other (a noise layer's decay and level) are not individually recovered.
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
           floor_db: float | None = DIST_FLOOR_DB, scale: str = "mel", match_level: bool = True) -> SearchResult:
    """Refine the patch `start` towards `target` ([n] or [n, channels] at `sr`) for at most `rounds` rounds of
    `population` renders each (see the module docstring for the method).

    `bounds`: name -> (low, high) or (low, high, "log"); names of `start` that are not in it, or whose
    low == high, are pinned. `render(patches: list[dict]) -> list[array]` is called once per round with exactly
    `population` patches (all names of `start`, pinned ones included) and must return that many hits at `sr`.
    `step` overrides a parameter's first step (ln units for a log parameter, else the parameter's own); it must
    name searched parameters and be finite and above 0, else ValueError.
    `floor_db`, `scale` and `match_level` go to `oneshot.spectral_distance`. Raises ValueError for a start
    outside its bounds, a population under 2, a bad `step`, or a render that returns the wrong number of hits;
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

FALL20_PER_T = 0.29      # Operator Ae Decay T: the amp envelope falls 20 dB in 0.29 T, measured 2026-11-09 and 11 on
#                          the Bodzin kit (memory match-synth-to-sample-by-recording; bzkit.py: "T ~ 3.5 x the
#                          measured -20 dB time"). Measured with the decay slope set to 1 through the preset XML;
#                          whether the LOM-only device has that slope by default is an assumption (ASSUMPTIONS 1).
TAU_PER_FALL20 = 1.0 / math.log(10.0)       # tau = fall_20 / ln(10), for an exponential amplitude decay
T_PER_TAU = math.log(10.0) / FALL20_PER_T   # Operator T = 7.94 tau: the curve is about 69 dB down at T
PE_DECAY_PER_TAU = T_PER_TAU                # Pe Decay = tau * this, an ASSUMPTION (2) checked against one take
ATTACK_FLOOR_MS = 1.5    # a tone attack of 0.1 ms clicks (measured 2026-11-11): never less
NOISE_ATTACK_FLOOR_MS = 0.1   # noise has no tone to click: the hand-built hats and claps use 0.1 ms
CLICK_MS = 10.0          # a pitch drop over in less than this is a click, not a glide. oneshot's own bound is its
#                          FAST_SETTLE_MS (5 ms); Tom_BOAVISTA's 17 st in 8 ms is a click too, so the guess uses 10
TAIL_MS = 100.0          # track readings from here on, when they agree, are the carrier the glide settled on
PITCH_BOUND_ST = 0.5     # the carrier may move this far either way in the search
PE_MAX_ST = 48.0         # a pitch drop is capped at this (ASSUMPTIONS 4: Operator's Pe range, to read from Live)
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
PEAK_DB_AT_0DB = {"tone": 3.0, "noise": 11.0}   # peak of a voice at osc level 0 dB, velocity 100 (measured 2026-11-11)
NOTE_VELOCITY = 100      # the velocity every level was measured at
Z_AMPLITUDE_DB = 20.0 / math.log(10.0)      # 8.686 dB per neper of amplitude

ASSUMPTIONS = (
    "Ae/Be Decay: the amp envelope's decay slope is not a parameter in the LOM list (only Pe and Fe have slopes). "
    "FALL20_PER_T = 0.29 was measured with DecaySlope 1 written through the preset XML, so the guess ASSUMES the "
    "LOM-only device has that slope by default. Confirm: Ae Decay 100 ms, one note at velocity 100, read the 20 dB "
    "fall: expect 29 ms.",
    "Pe Decay = tau * 7.94 (the amp envelope's constant), with Init = Peak = the drop and Sustain 0. Evidence, not "
    "a measurement of Pe Decay: the recorded kick (take_final.json) reads 52.7, 49.8, 47.6 Hz at 40, 80, 150 ms "
    "against 47.1 Hz, with Pe Decay 400 ms in bzkit.py (see the report: tau 38 to 57 ms, ratio 7 to 10). Confirm "
    "by setting Pe Decay and reading pitch_settle_ms.",
    "Noise: Osc-B Wave raw value 21 is a white noise wave (from bzkit.py's XML value, not the LOM). Confirm its "
    "name and index with str_for_value over Osc-B Wave, that it is white, and that in ratio mode the noise does "
    "not change with the note (the guess plays noise-only voices at note 60).",
    "Pe levels (Pe Init, Pe Peak, Pe Sustain) are read in semitones when Pe Amount and Pe Amt A are 100 % "
    "(bzkit.py wrote 14.0 and 4.5 for a 14 st kick); the range is assumed to reach +-48 st (PE_MAX_ST). "
    "to_live refuses a table whose unit is not 'st'.",
    "Osc-A < Pe, Osc-B < Pe: raw 1 sends the pitch envelope to that oscillator, 0 not (confirm they are switches and "
    "not amounts); Pe Amt A = 100 and Pe Amt B = 0 are the routing amounts; Pe D Slope and Pe Dst B are left at the "
    "device's defaults and the guess does not set them.",
    "Osc-A Wave raw 0 is a sine. Osc-A/B Fix On = 0 is ratio mode (Fix On = 1 would be fixed frequency).",
    "Algorithm raw 10 is 'all four oscillators parallel' (bzkit.py: XML Algorithm 10 = algorithm 11). Confirm the "
    "LOM index and that Osc A and B then both reach the output without modulating each other. Osc-C and Osc-D are "
    "switched off; their other parameters and the unused oscillator's envelope are not set.",
    "Coarse and Fine: ratio = Coarse * (1 + Fine/1000) (measured 2026-11-11). The note played is 1 to 2 semitones "
    "under the carrier, Coarse 1, Fine 59 to 122; Coarse 2 or more only above about 25 kHz. A Coarse and A Fine are "
    "passed as raw values if the units table has no row for them: confirm their display numbers equal the raw ones.",
    "Global Time = 0 % and Time < Key = 0 % leave every envelope time as set (Time scales them by about e^(0.023 per "
    "%): measured 2026-11-09); Transpose 0; Tone, Spread, Glide On, Shaper, LFO On = 0 and Pan are left at the "
    "device's defaults or the Set's state, not read: a scratch Operator must start from the Init preset.",
    "Ae/Be Init and Peak levels, Ae Mode / Ae Loop (no loop), and Ae/Be Retrig are left at the device's defaults; "
    "Sustain is -inf and Release = Decay as in bzkit.py. Osc level < velocity and < key are left at their defaults "
    "(bzkit.py set 35 % level < velocity): the levels assume the default velocity response at velocity 100.",
    "The profile's attack_ms equals Operator's Attack time: the profile reads the envelope's rise over whole periods "
    "and a linear ramp reads 89 % of its length, while Operator's attack may not be linear.",
    "Filter On / Filter Type / Filter Freq: type 1 is the high-pass (bzkit.py: 0 LP, 1 HP, 2 BP in the XML). "
    "The one filter sits after both oscillators, so it is used only on a noise-led voice (tone off).",
    "Output: Volume = the sample's peak minus 3 dB (tone-led) or 11 dB (noise-led), from the peaks of single "
    "oscillators at 0 dB and velocity 100 (+3 dBFS sine, +9 to +13 noise: 2026-11-11). Confirm with one take.",
    "Reading times: the attack noise reading's weight is taken at the middle of attack_window_ms (the profile's "
    "own window; a Hann window) and the body's at 0.36 of body_window_ms, calibrated on the stand-in; the "
    "extrapolation back to the onset is clamped to +-12 dB. A noise layer is accepted only when the attack's "
    "energy above 2 kHz (click_db) is within 5 dB of its level and its flatness is at least 0.001 (a tone's and a click's reads 0.0). All three are heuristics to re-check on takes.",
    "The note is held for at least the longest Decay and the take runs past it (Release = Decay): the length "
    "operator_live returns is a plan for the renderer, not something Live has been asked for.",
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
    tau_n = patch["noise_decay_ms"] * FALL20_PER_T * TAU_PER_FALL20
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
    "free": {name: (low, high, "log"|"lin")}, "pinned": {name: why}, "notes": [...], "assumptions": [...]}.
    `confidence` is "measured" (a reading used as it is), "derived" (computed from readings, with a
    formula) or "default" (no reading; a choice). Feed it to `search_arguments` and `operator_live`.

    The patch (a name is absent when the voice has no such part):
      carrier_hz     body pitch, or the pitch the glide settles on (fit of the pitch track)
      pitch_drop_st  semitones above carrier_hz at the onset (0 = no pitch envelope: under oneshot.DROP_ST, or a
                     click: over in under CLICK_MS), at most PE_MAX_ST
      pitch_tau_ms   time constant of the glide, exponential
      tone_attack_ms attack_ms, at least ATTACK_FLOOR_MS (0.1 ms clicks)
      tone_decay_ms  Operator's Decay time T = fall_20_ms / FALL20_PER_T (0.29: Operator, slope 1, measured)
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
                     fall_40_ms / 2, over 0.29 (when the hit falls more slowly later the late fall is the noise's)
      noise_attack_ms attack_ms, at least NOISE_ATTACK_FLOOR_MS
      noise_filter_hz  high-pass for a noise-led hit whose bands are empty low (see _noise_filter)
      output_db      Volume: the sample's peak_db minus the peak of one oscillator at 0 dB (+3 tone, +11 noise)
    A noise layer is only added to a pitched hit when its attack reading is above the leakage floor AND the attack's
    energy above 2 kHz (click_db) is within HF_MARGIN_DB of it and its spectrum has a flat part (flatness.attack): a glide, a gated tone or a click on a tone read
    "noise" from window leakage, but have no broadband energy to match it (the Bodzin kick, low tom and clave).
    A reading under NOISE_FREE_DB leaves the level free, down to NOISE_OFF_DB; a stronger one pins it.
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
        put("tone_decay_ms", fall20 / FALL20_PER_T, "fall_20_ms / %.2f" % FALL20_PER_T, "derived")
        put("tone_level_db", 0.0, "reference level", "default")
        nz = p.get("noise_db") or {}
        a, b, click = nz.get("attack"), nz.get("body"), p.get("click_db")
        flat = p.get("flatness") or {}
        noise_wanted = a is not None and a > oneshot.NOISE_FLOOR_DB
        if noise_wanted and (click is None or click < a - HF_MARGIN_DB or (flat.get("attack") or 0.0) < FLAT_ATTACK):
            noise_wanted = False
            notes.append("noise_db.attack is %.1f dB but the attack has %s dB above 2 kHz (click_db) and a flatness of %s: "
                         "no broadband noise to match (window leakage of a glide, a gated tone or a click): no noise layer"
                         % (a, "None" if click is None else "%.1f" % click, flat.get("attack")))
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
            tau_noise, default_decay = None, False
            if b is not None and b >= BODY_NOISE_DB and (flat.get("body") or 0.0) >= FLAT_BODY and f40 and f40 > TAIL_RATIO * fall20:
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
                    tone_only["noise_decay_ms"] = tau_noise / (FALL20_PER_T * TAU_PER_FALL20)
                    onset = noise_onset_db(tone_only)
                    if _model_fall_ms(tau_tone, tau_noise, onset) <= CONSISTENT * f40:
                        break
                    tau_noise *= 0.8
                    shrunk = True
                if shrunk:
                    reading += "; shortened so tone + noise fall 40 dB within %.0f %% of fall_40_ms" % (100 * (CONSISTENT - 1))
            put("noise_decay_ms", tau_noise / (FALL20_PER_T * TAU_PER_FALL20), reading, conf)
            put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS,
                "measured" if attack >= NOISE_ATTACK_FLOOR_MS else "default")
            if a > NOISE_FREE_DB:
                pinned["noise_level_ref_db"] = "decay and level of the noise layer lie on a ridge; the level at its reference time is read directly"
            else:
                free["noise_level_ref_db"] = (NOISE_OFF_DB, ratio_a + 6.0, "lin")
                notes.append("noise_db.attack is %.1f dB, near the %.0f dB leakage floor: the noise level is searched "
                             "(down to %.0f dB, off), not pinned" % (a, oneshot.NOISE_FLOOR_DB, NOISE_OFF_DB))
            nfactor = DEFAULT_NOISE_FACTOR if default_decay else NOISE_DECAY_FACTOR
            free["noise_decay_ms"] = (patch["noise_decay_ms"] / nfactor, patch["noise_decay_ms"] * nfactor, "log")
            pinned["noise_ref_ms"] = pinned["noise_attack_ms"] = "set from the reading, not searched"
        elif a is None or a <= oneshot.NOISE_FLOOR_DB:
            notes.append("noise_db.attack is %s: no noise layer" % ("missing" if a is None else "%.1f dB, under the %.0f dB floor "
                                                                    "(a decaying tone reads that from window leakage)" % (a, oneshot.NOISE_FLOOR_DB)))
        put("output_db", min(0.0, peak_db - PEAK_DB_AT_0DB["tone"]),
            "peak_db minus %.0f dB (one sine at 0 dB peaks at +3 dBFS)" % PEAK_DB_AT_0DB["tone"], "default")
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
        decay_ms = max(fall20, late / 2.0 if late else 0.0) / FALL20_PER_T
        put("noise_decay_ms", decay_ms, "the larger of fall_20_ms and fall_40_ms / 2, over %.2f (when the hit falls more "
            "slowly later, the late fall is the noise's: a tone in the first 20 dB decays faster)" % FALL20_PER_T,
            "derived")
        put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS,
            "measured" if attack >= NOISE_ATTACK_FLOOR_MS else "default")
        flt = _noise_filter(p)
        if flt:
            structure["filter_on"] = True
            put("noise_filter_hz", flt[0], flt[1], "derived")
            pinned["noise_filter_hz"] = "set from the band balance, not searched"
        put("output_db", min(0.0, peak_db - PEAK_DB_AT_0DB["noise"]),
            "peak_db minus %.0f dB (noise voices peak at +9 to +13 dBFS at 0 dB)" % PEAK_DB_AT_0DB["noise"], "default")
        free["noise_decay_ms"] = (patch["noise_decay_ms"] / DECAY_FACTOR, patch["noise_decay_ms"] * DECAY_FACTOR, "log")
        pinned["noise_attack_ms"] = pinned["output_db"] = "set from the reading, not searched"
        notes.append("unpitched: the voice is noise only (tone off); the profile's partials are not rendered")
    return {"patch": patch, "structure": structure, "sources": src, "free": free, "pinned": pinned,
            "notes": notes, "assumptions": list(ASSUMPTIONS)}


def search_arguments(guess: dict, units: dict | None = None) -> dict:
    """The `start` and `bounds` arguments of `search` from an operator_guess result: start is the guess's patch,
    bounds the guess's free parameters; every other name is pinned (rendered as given, never changed). With a
    units table, each bound is pulled in toward the start (halving the distance, up to 30 times) until the voice
    at that bound converts through to_live: one candidate out of the table's range would stop the search
    (a render that raises ends it), so the search is never offered one. Raises ValueError when the start itself
    does not convert."""
    start, bounds = dict(guess["patch"]), dict(guess["free"])
    if units is None:
        return {"start": start, "bounds": bounds}

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
    return {"start": start, "bounds": bounds}


def operator_live(patch: dict, structure: dict) -> dict:
    """The complete Operator state that plays `patch` (operator_guess's patch, or a search's candidate), as
    {"params": {Live parameter name: physical value}, "note": MIDI note, "velocity": NOTE_VELOCITY,
    "note_length_ms": how long to hold it}. The same keys for every structure: every oscillator's On, the pitch
    envelope's switches and routing, Filter On, LFO On, global Time, Osc A and B's levels, envelopes and waves;
    a part the voice does not have is switched off and its numbers are placeholders (UNUSED_MS, 0 dB, 0 st), which
    still need a row in the units table. Raw switches and enumerations (RAW_NAMES) hold Live's own value; the
    rest are Hz, ms, dB, semitones or % (the unit to_live checks). Noise-only voices are played at note 60 and
    a carrier under MIDI note 0 (8.18 Hz) raises ValueError. The formulas are the ones in operator_guess's docstring and
    ASSUMPTIONS; nothing here has been checked in Live."""
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
    """One parameter's table: Live value <-> display number (in ms, Hz, dB, %, st or plain), piecewise linear. Rows
    with the same display number (display rounding: two live values that both read "0.10 ms") are collapsed into one
    at their mean live value. Each segment is interpolated in the log of the number when the table's positive rows
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
        groups = {}
        for live, disp in pts:
            groups.setdefault(disp, []).append(live)
        pts = sorted((sum(v) / len(v), d) for d, v in groups.items())      # plateaus: one row at the mean live value
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
