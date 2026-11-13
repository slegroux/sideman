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
#   patch --operator_live--> Live parameter name -> physical value (the formulas below), and the note to play
#   that --to_live(units)--> Live parameter name -> Live's value

FALL20_PER_T = 0.29      # Operator, slope 1: the amp envelope falls 20 dB in 0.29 T (T = its Decay time).
#                          Measured 2026-11-09 and 11 on the Bodzin kit (memory match-synth-to-sample-by-recording;
#                          bzkit.py: "T ~ 3.5 x the measured -20 dB time"). For an exponential 20 dB is ln(10) tau,
#                          so T = 7.94 tau: the Decay time is where the curve is about 69 dB down.
TAU_PER_FALL20 = 1.0 / math.log(10.0)       # tau = fall_20 / ln(10), for an exponential amplitude decay
ATTACK_FLOOR_MS = 1.5    # a tone attack of 0.1 ms clicks (measured 2026-11-11): never less
NOISE_ATTACK_FLOOR_MS = 0.1   # noise has no tone to click: the hand-built hats and claps use 0.1 ms
PITCH_BOUND_ST = 0.5     # the carrier may move this far either way in the search
NOISE_REF_MS = 15.0      # the profile's "attack" noise reading is a Hann window over 0-30 ms: its centre
NOISE_BODY_MS = 80.0     # ...and the "body" reading's centre of weight (40-150 ms of a decaying hit)
PEAK_DB_AT_0DB = {"tone": 3.0, "noise": 11.0}   # peak of a voice at osc level 0 dB, velocity 100 (measured 2026-11-11)
FILTER_CLEAR_DB = 10.0   # a band within this of the densest is "clear" (the noise reaches down to it)
DECAY_FACTOR = 1.5       # free decays may move by this factor either way
NOISE_DECAY_FACTOR = 3.0
DROP_WIDTH_ST = 2.0      # the free pitch drop may move by max(this, 40 %) either way
Z_AMPLITUDE_DB = 20.0 / math.log(10.0)      # 8.686 dB per neper of amplitude

ASSUMPTIONS = (
    "Ae/Be Decay: the envelope's decay slope is not a parameter in the LOM list (only Pe and Fe have slopes), "
    "so Decay is assumed to be the device's default exponential: 20 dB in 0.29 T. Confirm on a scratch Operator "
    "(Ae Decay 100 ms, play one note at velocity 100, read the 20 dB fall: expect 29 ms).",
    "Noise: Osc-B Wave raw value 21 is a white noise wave (taken from bzkit.py's XML value, not from the LOM). "
    "Confirm its name and index with str_for_value over Osc-B Wave, and that it is white.",
    "Algorithm: raw value 10 is 'all four oscillators parallel' (bzkit.py: XML Algorithm 10 = algorithm 11). "
    "Confirm the LOM index and that Osc A and B then both reach the output without modulating each other.",
    "Pitch envelope levels (Pe Init, Pe Peak, Pe Sustain) are read in semitones when Pe Amount and Pe Amt A are "
    "100 % (bzkit.py wrote 14.0 and 4.5 for a 14 st kick). to_live refuses a table whose unit is not 'st'.",
    "Pe Decay is set to the time the glide takes to settle within 0.5 st, tau * ln(|drop| / 0.5), with Init = Peak "
    "= the drop and Sustain 0: confirm that Pe Decay with its default slope gives that settle time (read with "
    "sample_analyze: pitch_settle_ms).",
    "Osc-A < Pe: a raw switch value 1 sends the pitch envelope to Osc A. Confirm it is on/off and not an amount.",
    "Coarse and Fine: ratio = Coarse * (1 + Fine/1000) (measured 2026-11-11). A Coarse and A Fine are passed as "
    "raw values if the units table has no row for them; confirm their display numbers equal the raw ones.",
    "The note: the carrier is played from the MIDI note 1 to 2 semitones under it with Coarse 1 and a Fine of 59 "
    "to 122, so a +-0.5 st move of the carrier stays in Fine's range and Coarse never drops under 1. "
    "Transpose is left at 0.",
    "Filter On / Filter Type / Filter Freq: type 1 is the high-pass (bzkit.py: 0 LP, 1 HP, 2 BP in the XML). "
    "The one filter sits after both oscillators, so it is used only on a noise-led voice (tone off).",
    "Output: Volume = the sample's peak minus 3 dB (tone-led) or 11 dB (noise-led), from the peaks of single "
    "oscillators at 0 dB and velocity 100 (+3 dBFS sine, +9 to +13 noise: 2026-11-11). Confirm with one take.",
    "Ae/Be Sustain -inf and Release = Decay, no loop (as bzkit.py): the raw value of Ae Mode / Ae Loop for "
    "'no loop' is not set here and must be read from the device's defaults.",
)

RAW_NAMES = frozenset({"Algorithm", "Osc-A On", "Osc-B On", "Osc-A Wave", "Osc-B Wave", "A Fix On ", "B Fix On ",
                       "Pe On", "Filter On", "Filter Type", "Osc-A < Pe"})   # Live's own value, no conversion
RAW_IF_NO_TABLE = frozenset({"A Coarse", "A Fine", "B Coarse", "B Fine"})    # display number = raw value, unless tabled
EXPECTED_DIMENSION = {                        # what a parameter's display must be, so a wrong unit is refused
    **dict.fromkeys(("Ae Attack", "Ae Decay", "Ae Release", "Be Attack", "Be Decay", "Be Release", "Pe Decay"), "time"),
    **dict.fromkeys(("Volume", "Osc-A Level", "Osc-B Level", "Ae Sustain", "Be Sustain"), "db"),
    **dict.fromkeys(("Pe Init", "Pe Peak", "Pe Sustain", "Transpose"), "st"),
    **dict.fromkeys(("Pe Amount", "Pe Amt A"), "pct"),
    "Filter Freq": "freq",
}


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
    (for each tau on a grid the other two are linear, weighted by t + 5 ms; the best tau wins). The fit gives the carrier the glide
    settles on (the body pitch alone sits on the glide's tail), and the drop at the onset when the start was read
    late. A poor fit or fewer than three points falls back to the measured drop and settle time."""
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
    if abs(drop) <= oneshot.DROP_ST:
        return out
    settle = p.get("pitch_settle_ms")
    fallback = {"carrier_hz": f_body, "drop_st": drop, "confidence": "derived",
                "tau_ms": max(1.0, (settle if settle else 20.0) / math.log(max(abs(drop), 2 * oneshot.SETTLE_ST) /
                                                                          oneshot.SETTLE_ST)),
                "reading": "pitch_body_hz, pitch_drop_st, pitch_settle_ms (tau = settle / ln(drop / %.1f st))" % oneshot.SETTLE_ST}
    if len(pts) < 3:
        return fallback
    t = np.array(sorted(pts))
    st = np.array([12 * math.log2(pts[x] / f_body) for x in t])
    best = None
    w = (t + 5.0) / 50.0           # early points are the least reliable (a fast glide in short frames): weight by time
    for tau in np.geomspace(1.0, 600.0, 160):
        a = np.stack([np.ones_like(t), np.exp(-t / tau)], axis=1)
        c = np.linalg.lstsq(a * w[:, None], st * w, rcond=None)[0]
        rss = float((((a @ c - st) * w) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, tau, c)
    rss, tau, (s_inf, amp) = best
    rss = float(((np.stack([np.ones_like(t), np.exp(-t / tau)], axis=1) @ np.array([s_inf, amp]) - st) ** 2).sum())
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
    bands, nyq = p.get("bands_db"), p["sample_rate"] / 2.0
    if not bands:
        return None
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


def operator_guess(profile: dict) -> dict:
    """A first Operator patch for the one-shot `profile` (oneshot.profile_array / sample_analyze output), in
    physical units, for one voice: a sine carrier on Osc A with a pitch envelope and a noise layer on Osc B,
    or, for an unpitched hit (hat, clap), noise only. Raises ValueError for a profile with no 20 dB fall
    (the take ends first) or, when pitched, no body pitch.

    Returns {"patch": {name: float}, "structure": {...}, "sources": {name: {"reading", "confidence"}},
    "free": {name: (low, high, "log"|"lin")}, "pinned": {name: why}, "notes": [...], "assumptions": [...]}.
    `confidence` is "measured" (a reading used as it is), "derived" (computed from readings, with a
    formula) or "default" (no reading; a choice). Feed it to `search_arguments` and `operator_live`.

    The patch (a name is absent when the voice has no such part):
      carrier_hz     body pitch, or the pitch the glide settles on (fit of the pitch track)
      pitch_drop_st  semitones above carrier_hz at the onset (0 = no pitch envelope: under oneshot.DROP_ST)
      pitch_tau_ms   time constant of the glide, exponential
      tone_attack_ms attack_ms, at least ATTACK_FLOOR_MS (0.1 ms clicks)
      tone_decay_ms  Operator's Decay time T = fall_20_ms / FALL20_PER_T (0.29: Operator, slope 1, measured)
      tone_level_db  0 (the reference: the noise is set against it)
      noise_level_ref_db  the noise layer's level against the tone at NOISE_REF_MS (15 ms): the attack-window
                     reading (noise_db.attack, a share of the total) as a power ratio, share / (1 - share)
      noise_decay_ms Operator Decay time of the noise: from the two noise readings, whose windows centre on 15
                     and 80 ms, the noise's decay rate against the tone's is
                     (ratio_attack_dB - ratio_body_dB) / (8.686 * 65 ms), and 1/tau_noise = that + 1/tau_tone;
                     with no usable body reading (under oneshot.NOISE_FLOOR_DB: a tone's window leakage) the
                     default is a third of the tone's. Noise-led: fall_40_ms / 2 / 0.29 (the late fall is the
                     noise's: a tone in it decays faster), or fall_20_ms if the take has no 40 dB fall
      noise_attack_ms attack_ms, at least NOISE_ATTACK_FLOOR_MS
      noise_filter_hz  high-pass for a noise-led hit whose bands are empty low (see _noise_filter)
      output_db      Volume: the sample's peak_db minus the peak of one oscillator at 0 dB (+3 tone, +11 noise)
    Why the noise layer's level is pinned and its decay free: decay and level lie on a ridge (a longer layer,
    quieter, scores like a shorter, louder one: the search report), so one must be held. The level at 15 ms is
    a direct reading, the decay a two-reading derivation under a leakage floor, so the level (at the reading's own
    time, not at the onset, so it does not move when the decay does) is pinned and the decay searched.
    Why ratio mode and not a fixed frequency: Coarse/Fine from a played note is what was measured and
    built (2026-11-11); whether a fixed-frequency oscillator follows the pitch envelope is not known."""
    p = profile
    if p.get("fall_20_ms") is None:
        raise ValueError("the profile has no 20 dB fall (the take ends before the hit has fallen): "
                         "record a longer take")
    pitched = bool(p.get("pitched")) and bool(p.get("pitch_body_hz"))
    patch, src, free, pinned, notes = {}, {}, {}, {}, []

    def put(name, value, reading, confidence):
        patch[name], src[name] = _entry(float(value), reading, confidence)

    attack = p.get("attack_ms") or 0.0
    fall20 = float(p["fall_20_ms"])
    tau_tone = fall20 * TAU_PER_FALL20
    structure = {"tone_on": pitched, "noise_on": False, "pitch_env_on": False, "filter_on": False}
    if pitched:
        pe = _pitch_envelope(p)
        put("carrier_hz", pe["carrier_hz"], pe["reading"], pe["confidence"])
        put("pitch_drop_st", pe["drop_st"], pe["reading"], pe["confidence"])
        put("pitch_tau_ms", pe["tau_ms"], pe["reading"], pe["confidence"])
        structure["pitch_env_on"] = abs(pe["drop_st"]) > oneshot.DROP_ST
        put("tone_attack_ms", max(ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms (0.1 ms clicks)" % ATTACK_FLOOR_MS,
            "measured" if attack >= ATTACK_FLOOR_MS else "default")
        put("tone_decay_ms", fall20 / FALL20_PER_T, "fall_20_ms / %.2f" % FALL20_PER_T, "derived")
        put("tone_level_db", 0.0, "reference level", "default")
        nz = p.get("noise_db") or {}
        a, b = nz.get("attack"), nz.get("body")
        if a is not None and a > oneshot.NOISE_FLOOR_DB:
            structure["noise_on"] = True
            share = 10 ** (a / 10.0)
            ratio_a = 10 * math.log10(share / max(1.0 - share, 1e-3))
            put("noise_level_ref_db", ratio_a, "noise_db.attack as a power ratio to the tone, at %.0f ms" % NOISE_REF_MS, "derived")
            if b is not None and b > oneshot.NOISE_FLOOR_DB:
                shb = 10 ** (b / 10.0)
                ratio_b = 10 * math.log10(shb / max(1.0 - shb, 1e-3))
                k = (ratio_a - ratio_b) / (Z_AMPLITUDE_DB * (NOISE_BODY_MS - NOISE_REF_MS))     # 1/ms
                inv = max(k + 1.0 / tau_tone, 1.0 / (3.0 * tau_tone)) if tau_tone > 0 else 1.0
                tau_noise = min(1.0 / inv, 3.0 * tau_tone)
                reading, conf = "noise_db.attack and noise_db.body, with the tone's fall_20_ms", "derived"
            else:
                tau_noise = tau_tone / 3.0
                reading, conf = "none (no noise reading above %.0f dB in the body): a third of the tone's decay" % oneshot.NOISE_FLOOR_DB, "default"
            put("noise_decay_ms", tau_noise / (FALL20_PER_T * TAU_PER_FALL20), reading, conf)
            put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS, "default")
            pinned["noise_level_ref_db"] = "decay and level of the noise layer lie on a ridge; the level at 15 ms is read directly"
        else:
            notes.append("noise_db.attack is %s: no noise layer" % ("missing" if a is None else "%.1f dB, under the %.0f dB floor "
                                                                    "(a decaying tone reads that from window leakage)" % (a, oneshot.NOISE_FLOOR_DB)))
        put("output_db", min(0.0, p.get("peak_db", 0.0) - PEAK_DB_AT_0DB["tone"]),
            "peak_db minus %.0f dB (one sine at 0 dB peaks at +3 dBFS)" % PEAK_DB_AT_0DB["tone"], "default")
        free["carrier_hz"] = (patch["carrier_hz"] * 2 ** (-PITCH_BOUND_ST / 12), patch["carrier_hz"] * 2 ** (PITCH_BOUND_ST / 12), "log")
        if structure["pitch_env_on"]:
            d = patch["pitch_drop_st"]
            w = max(DROP_WIDTH_ST, 0.4 * abs(d))
            free["pitch_drop_st"] = (d - w, d + w, "lin")
            free["pitch_tau_ms"] = (patch["pitch_tau_ms"] / 2.0, patch["pitch_tau_ms"] * 2.0, "log")
        else:
            pinned["pitch_drop_st"] = pinned["pitch_tau_ms"] = "no pitch drop above %.1f st was measured" % oneshot.DROP_ST
        free["tone_decay_ms"] = (patch["tone_decay_ms"] / DECAY_FACTOR, patch["tone_decay_ms"] * DECAY_FACTOR, "log")
        if structure["noise_on"]:
            free["noise_decay_ms"] = (patch["noise_decay_ms"] / NOISE_DECAY_FACTOR, patch["noise_decay_ms"] * NOISE_DECAY_FACTOR, "log")
        pinned["tone_attack_ms"] = pinned["tone_level_db"] = pinned["output_db"] = "set from the reading, not searched"
        if structure["noise_on"]:
            pinned["noise_attack_ms"] = "set from the reading, not searched"
    else:
        structure["noise_on"] = True
        late = p.get("fall_40_ms")
        decay_ms = (late / 2.0 if late else fall20) / FALL20_PER_T
        put("noise_decay_ms", decay_ms, "fall_40_ms / 2 / %.2f (the late fall is the noise's; a tone in the first "
            "20 dB decays faster)" % FALL20_PER_T if late else "fall_20_ms / %.2f" % FALL20_PER_T,
            "derived")
        put("noise_attack_ms", max(NOISE_ATTACK_FLOOR_MS, attack), "attack_ms, floored at %.1f ms" % NOISE_ATTACK_FLOOR_MS,
            "measured" if attack >= NOISE_ATTACK_FLOOR_MS else "default")
        flt = _noise_filter(p)
        if flt:
            structure["filter_on"] = True
            put("noise_filter_hz", flt[0], flt[1], "derived")
            pinned["noise_filter_hz"] = "set from the band balance, not searched"
        put("output_db", min(0.0, p.get("peak_db", 0.0) - PEAK_DB_AT_0DB["noise"]),
            "peak_db minus %.0f dB (noise voices peak at +9 to +13 dBFS at 0 dB)" % PEAK_DB_AT_0DB["noise"], "default")
        free["noise_decay_ms"] = (patch["noise_decay_ms"] / DECAY_FACTOR, patch["noise_decay_ms"] * DECAY_FACTOR, "log")
        pinned["noise_attack_ms"] = pinned["output_db"] = "set from the reading, not searched"
        notes.append("unpitched: the voice is noise only (tone off); the profile's partials are not rendered")
    return {"patch": patch, "structure": structure, "sources": src, "free": free, "pinned": pinned,
            "notes": notes, "assumptions": list(ASSUMPTIONS)}


def search_arguments(guess: dict) -> dict:
    """The `start` and `bounds` arguments of `search` from an operator_guess result: start is the guess's patch,
    bounds the guess's free parameters; every other name is pinned (rendered as given, never changed)."""
    return {"start": dict(guess["patch"]), "bounds": dict(guess["free"])}


def operator_live(patch: dict, structure: dict) -> dict:
    """The Operator settings that play `patch` (operator_guess's patch, or a search's candidate), as
    {"params": {Live parameter name: physical value}, "note": MIDI note to play}. Raw switches and enumerations
    (RAW_NAMES) hold Live's own value; the rest are Hz, ms, dB, semitones or % (the unit to_live checks).
    The formulas are the ones in operator_guess's docstring and ASSUMPTIONS; nothing here has been checked in Live."""
    q, par, note = patch, {}, None
    par.update({"Algorithm": 10, "Volume": q.get("output_db", 0.0)})
    tau_tone = q["tone_decay_ms"] * FALL20_PER_T * TAU_PER_FALL20 if "tone_decay_ms" in q else None
    if structure.get("tone_on"):
        f = q["carrier_hz"]
        floor_note = int(math.floor(69 + 12 * math.log2(f / 440.0)))
        note = max(0, min(127, floor_note - 1))
        ratio = f / (440.0 * 2 ** ((note - 69) / 12.0))
        coarse = max(1, min(48, int(math.floor(ratio))))
        fine = max(0, min(999, int(round((ratio / coarse - 1) * 1000))))
        lev_tone, lev_noise = q.get("tone_level_db", 0.0), 0.0
        if structure.get("noise_on"):
            tau_n = q["noise_decay_ms"] * FALL20_PER_T * TAU_PER_FALL20
            onset = q["noise_level_ref_db"] + Z_AMPLITUDE_DB * NOISE_REF_MS * (1.0 / tau_n - 1.0 / tau_tone)
            lev_tone, lev_noise = lev_tone + min(0.0, -onset), min(0.0, onset)
        par.update({"Osc-A On": 1, "A Coarse": coarse, "A Fine": fine, "A Fix On ": 0, "Osc-A Wave": 0,
                    "Osc-A Level": lev_tone, "Ae Attack": q["tone_attack_ms"], "Ae Decay": q["tone_decay_ms"],
                    "Ae Sustain": -math.inf, "Ae Release": q["tone_decay_ms"]})
        if structure.get("pitch_env_on"):
            d = q["pitch_drop_st"]
            settle = q["pitch_tau_ms"] * math.log(max(abs(d), 2 * oneshot.SETTLE_ST) / oneshot.SETTLE_ST)
            par.update({"Pe On": 1, "Osc-A < Pe": 1, "Pe Init": d, "Pe Peak": d, "Pe Decay": settle, "Pe Sustain": 0.0,
                        "Pe Amount": 100.0, "Pe Amt A": 100.0})
    else:
        par["Osc-A On"] = 0
        lev_noise = 0.0
    if structure.get("noise_on"):
        par.update({"Osc-B On": 1, "Osc-B Wave": 21, "B Fix On ": 0, "Osc-B Level": lev_noise,
                    "Be Attack": q["noise_attack_ms"], "Be Decay": q["noise_decay_ms"], "Be Sustain": -math.inf,
                    "Be Release": q["noise_decay_ms"]})
    if structure.get("filter_on"):
        par.update({"Filter On": 1, "Filter Type": 1, "Filter Freq": q["noise_filter_hz"]})
    return {"params": par, "note": note}


# ---- physical <-> Live values through a units table: {name: [[live_value, display_number, unit], ...]}

_UNITS = {"ms": ("time", 1.0), "s": ("time", 1000.0), "Hz": ("freq", 1.0), "kHz": ("freq", 1000.0), "dB": ("db", 1.0),
          "%": ("pct", 1.0), "st": ("st", 1.0), "": ("none", 1.0)}


class _Curve:
    """One parameter's table: Live value <-> display number (in ms, Hz, dB, %, st or plain), piecewise linear, in the
    log of the number when it is positive and spans a decade or more. -inf (a row whose display is -inf) is its
    own point. Raises ValueError for a table that is not strictly monotone, mixes dimensions, or is empty."""

    def __init__(self, name: str, rows: list):
        pts, self.floor = [], None
        dims = set()
        for row in rows:
            if len(row) != 3:
                raise ValueError("units[%r]: a row is [live_value, display_number, unit], got %r" % (name, row))
            live, disp, unit = float(row[0]), float(row[1]), str(row[2]).strip()
            dim, mul = _UNITS.get(unit, ("other:" + unit, 1.0))
            dims.add(dim)
            if disp == -math.inf:
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
        self.live = np.array([p[0] for p in pts])
        self.disp = np.array([p[1] for p in pts])
        if len(set(self.live)) != len(self.live):
            raise ValueError("units[%r]: two rows have the same live value" % name)
        d = np.diff(self.disp)
        if not (np.all(d > 0) or np.all(d < 0)):
            raise ValueError("units[%r] is not monotone: the display number must rise (or fall) with the live value" % name)
        self.log = bool(np.all(self.disp > 0) and self.disp.max() / self.disp.min() >= 10.0)
        self.name = name

    def _u(self, x):
        return np.log(x) if self.log else np.asarray(x, dtype=float)

    def to_live(self, x: float) -> float:
        if x == -math.inf and self.floor is not None:
            return self.floor
        lo, hi = self.disp.min(), self.disp.max()
        tol = 1e-9 * max(1.0, abs(lo), abs(hi))
        if not (lo - tol <= x <= hi + tol):
            raise ValueError("%s: %g is outside the table's range %g to %g" % (self.name, x, lo, hi))
        u, order = self._u(self.disp), np.argsort(self._u(self.disp))
        return float(np.interp(float(self._u(min(max(x, lo), hi))), u[order], self.live[order]))

    def from_live(self, v: float) -> float:
        if self.floor is not None and v == self.floor:
            return -math.inf
        lo, hi = self.live.min(), self.live.max()
        if not (lo <= v <= hi):
            raise ValueError("%s: live value %g is outside the table's range %g to %g" % (self.name, v, lo, hi))
        u = float(np.interp(v, self.live, self._u(self.disp)))
        return float(np.exp(u) if self.log else u)


def _curves(units: dict) -> dict:
    names = set(operator_parameters())
    for name in units:
        if name not in names:
            raise ValueError("units names %r, which is not an Operator parameter" % name)
    return {name: _Curve(name, rows) for name, rows in units.items()}


def _convert(values: dict, units: dict, direction: str) -> dict:
    curves, names, out = _curves(units), set(operator_parameters()), {}
    for name, v in values.items():
        if name not in names:
            raise ValueError("%r is not an Operator parameter" % name)
        if name in curves and name not in RAW_NAMES:
            out[name] = getattr(curves[name], direction)(float(v))
        elif name in RAW_NAMES or name in RAW_IF_NO_TABLE:
            out[name] = v
        else:
            raise ValueError("no units table for %r: it cannot be converted" % name)
    return out


def to_live(values: dict, units: dict) -> dict:
    """{Live parameter name: Live's value} from {name: physical value} (ms, Hz, dB, %, semitones; -inf allowed where
    the table has a -inf row) through `units` ({name: [[live_value, display_number, unit], ...]}, from Live's
    str_for_value over each range). Interpolated piecewise linearly, in the log of the display number where it is
    positive and spans a decade or more. Raises ValueError for a name that is not in Operator's parameter list,
    a name with no table (raw switches and enumerations, RAW_NAMES, pass as they are, and so do the Coarse and
    Fine of either oscillator when untabled), a table that is not monotone, in mixed units or in a unit that is
    not the parameter's (EXPECTED_DIMENSION), and a value outside the table's range: it is refused and not
    clamped, because a clamp would let the search believe in a value Live never rendered."""
    return _convert(values, units, "to_live")


def from_live(values: dict, units: dict) -> dict:
    """The inverse of to_live: {name: physical value} from {name: Live's value}. Same refusals."""
    return _convert(values, units, "from_live")
