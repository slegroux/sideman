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

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

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
