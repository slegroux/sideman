"""sideman.patch.search against a numpy stand-in for the synth, with parameters known by construction.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_patch.py -q
Skipped (not failed) when numpy or soundfile is missing.

The stand-in (`standin`) is a test double for Ableton Live's Operator and lives here, not in the package: the
search must not know what renders, and the real renderer will be Live. It is a sine carrier with an exponential
pitch envelope, an exponential amplitude envelope, a noise layer with its own level and decay, and a short
attack. Its noise is drawn afresh on every render (like Operator's), so two renders of one patch differ.
"""
import functools
import json
import math
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("soundfile")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import oneshot, patch  # noqa: E402

SR = 24000
SECONDS = 0.3
ATTACK_S = 0.0015
TARGET_SEED = 10 ** 6        # the target's noise is never one a candidate gets

# name -> (low, high, scale), the whole stand-in's parameter set
BOUNDS = {
    "pitch_hz": None,        # set per target: the measurement bounds it to +-0.5 semitone
    "pitch_st": (-12.0, 36.0, "lin"),        # semitones above pitch_hz at the onset
    "pitch_tau": (0.002, 0.2, "log"),        # s, time constant of the glide
    "decay": (0.005, 0.5, "log"),            # s, amplitude time constant of the tone
    "noise_db": (-60.0, 12.0, "lin"),        # noise RMS against the tone's RMS, dB
    "noise_decay": (0.005, 0.5, "log"),      # s, amplitude time constant of the noise
}
TRUE = {
    "kick": {"pitch_hz": 50.0, "pitch_st": 14.0, "pitch_tau": 0.020, "decay": 0.060, "noise_db": -12.0,
             "noise_decay": 0.020},
    "tom": {"pitch_hz": 180.0, "pitch_st": 4.0, "pitch_tau": 0.030, "decay": 0.050, "noise_db": -10.0,
            "noise_decay": 0.030},
    "hat": {"pitch_hz": 7000.0, "pitch_st": 2.0, "pitch_tau": 0.010, "decay": 0.030, "noise_db": 0.0,
            "noise_decay": 0.045},
}
TIMES = ("decay", "noise_decay")      # parameters read as times (within oneshot.TIME_RATIO)


def standin(p: dict, seed: int, sr: int = SR, seconds: float = SECONDS, attack: float = ATTACK_S) -> np.ndarray:
    """One hit, mono [n], from the patch `p`; `seed` is the noise realisation."""
    t = np.arange(int(seconds * sr)) / sr
    f = p["pitch_hz"] * 2.0 ** (p["pitch_st"] * np.exp(-t / p["pitch_tau"]) / 12.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    ramp = np.minimum(1.0, t / attack)
    tone = np.sin(ph) * np.exp(-t / p["decay"]) * ramp
    noise = np.random.default_rng(seed).standard_normal(len(t)) * math.sqrt(0.5)       # RMS of a unit sine
    noise = noise * 10.0 ** (p["noise_db"] / 20.0) * np.exp(-t / p["noise_decay"]) * ramp
    return tone + noise


class Renderer:
    """render(patches): the stand-in once per patch, a fresh noise seed each time (`fresh`) or one fixed
    seed for every render (`not fresh`: the target's own, so a perfect patch reproduces it exactly).
    Counts calls and the size of each batch."""

    def __init__(self, fresh: bool = True, bad=None):
        self.fresh, self.bad, self.batches, self.calls = fresh, bad, [], 0

    def __call__(self, patches):
        self.batches.append(len(patches))
        out = []
        for p in patches:
            self.calls += 1
            seed = self.calls if self.fresh else TARGET_SEED
            out.append(self.bad(len(self.batches) - 1, len(out), p) if self.bad else None)
            if out[-1] is None:
                out[-1] = standin(p, seed)
        return out


def bounds_for(true: dict) -> dict:
    b = dict(BOUNDS)
    b["pitch_hz"] = (true["pitch_hz"] * 2 ** (-0.5 / 12), true["pitch_hz"] * 2 ** (0.5 / 12), "log")
    return b


def start_for(true: dict, off: float = 0.3) -> dict:
    """Every free parameter `off` away from the truth, in alternating directions; the pitch 0.4 semitone off
    (a measurement's error, inside the +-0.5 semitone the search may move it); the noise level 6 dB off (twice
    or half the amplitude: 30 % off in amplitude is 2.3 to 3.1 dB, nearly inside the 3 dB tolerance already)."""
    s = {"pitch_hz": true["pitch_hz"] * 2 ** (0.4 / 12)}
    for k, name in enumerate(n for n in true if n != "pitch_hz"):
        sign = 1 if k % 2 == 0 else -1
        s[name] = true[name] + sign * 6.0 if name == "noise_db" else true[name] * (1 + sign * off)
    return s


def recovered(true: dict, got: dict) -> dict:
    """What each spec'd reading is off by: ratio for times, dB for the noise level, semitones for the pitch."""
    r = {name: got[name] / true[name] for name in TIMES}
    r["noise_db"] = got["noise_db"] - true["noise_db"]
    r["pitch_st_off"] = 12 * math.log2(got["pitch_hz"] / true["pitch_hz"])
    return r


def assert_recovered(true: dict, got: dict, label: str = ""):
    r = recovered(true, got)
    for name in TIMES:
        assert abs(r[name] - 1) <= oneshot.TIME_RATIO, "%s %s: got %.4f, true %.4f" % (label, name, got[name], true[name])
    assert abs(r["noise_db"]) <= oneshot.LEVEL_DB, "%s noise_db: got %.2f, true %.2f" % (label, got["noise_db"], true["noise_db"])


def run(kind: str, seed: int = 0, fresh: bool = True, start: dict | None = None, **kw):
    true = TRUE[kind]
    target = standin(true, TARGET_SEED)
    render = Renderer(fresh)
    res = patch.search(start or start_for(true), bounds_for(true), render, target, SR, seed=seed, **kw)
    return true, res, render


SEEDS = range(10)


def noise_at(p: dict, t: float) -> float:
    """The noise layer's level at time t (s), dB against the tone's peak."""
    return p["noise_db"] - 20 * math.log10(math.e) * t / p["noise_decay"]


@functools.lru_cache(maxsize=None)
def runs_of(kind: str, fresh: bool) -> list:
    return [run(kind, seed=seed, fresh=fresh) for seed in SEEDS]


FRESH_SEEDS = [7 * 10 ** 6 + k for k in range(8)]      # noise for re-rendering: never one the search used


def rerender(p: dict, fresh: bool) -> list:
    return [standin(p, s) for s in (FRESH_SEEDS if fresh else [TARGET_SEED])]


@pytest.mark.parametrize("kind", ["kick", "tom", "hat"])
@pytest.mark.parametrize("fresh", [True, False], ids=["fresh-noise", "same-noise"])
def test_search_reaches_the_score_of_the_truth(kind, fresh):
    """What the search is for is the score it reaches, judged on re-renders of the final patch with noise it never
    saw (the score it kept is the luckiest of its renders), against the true parameters rendered the same way.
    Measured over 20 seeds (report, "Score reached"): the final patch is within the truth's distance plus two
    spreads of it in every seed only on the hat-like target (kick 30 %, tom 20 % of the seeds; its mean gap to
    the truth is 0.036, 3 to 4 spreads), so that is what is asserted strictly, and for the others the share of
    the way from the start's distance to the truth's that it closes: at least 60 % in every seed (measured
    minimum 61 %), 70 % at the median. With one fixed noise the truth scores 0 with no spread, so only the
    share closed is asserted. Readings of the final patch against the target (oneshot.compare): no fall-time
    reading (10 or 20 dB) in any seed of the kick-like and tom-like targets (the hat's readings are noise:
    the truth's own render reads 'attack' and 'sub' differences against its target)."""
    true = TRUE[kind]
    target = standin(true, TARGET_SEED)
    truth = np.array([oneshot.spectral_distance(x, target, SR) for x in rerender(true, fresh)])
    tprof = oneshot.profile_array(target[:, None], SR, single=True)
    closed, within = [], []
    for _, res, _ in runs_of(kind, fresh):
        final = np.array([oneshot.spectral_distance(x, target, SR) for x in rerender(res.best, fresh)]).mean()
        closed.append((res.start_distance - final) / (res.start_distance - truth.mean()))
        if fresh:
            within.append(final <= truth.mean() + 2 * truth.std(ddof=1))
        prof = oneshot.profile_array(rerender(res.best, fresh)[0][:, None], SR, single=True)
        if kind != "hat":
            falls = [t for t in oneshot.compare(prof, tprof)["readings"] if t.startswith(("falls 10 dB", "falls 20 dB"))]
            assert not falls, "%s: %s" % (kind, falls)
    # the 0.6 bar and the spread it rests on are to be re-baselined when real takes are measured
    assert min(closed) >= 0.6 and np.median(closed) >= 0.7, "%s: share of the gap closed per seed %s" % (kind, np.round(closed, 2))
    if fresh and kind == "hat":
        assert all(within), "%s: within truth + 2 spreads: %s" % (kind, within)


@pytest.mark.parametrize("kind", ["kick", "tom", "hat"])
@pytest.mark.parametrize("fresh", [True, False], ids=["fresh-noise", "same-noise"])
def test_search_recovers_known_parameters(kind, fresh):
    """Secondary: parameters that trade off against each other are not individually recovered (the noise layer's
    decay and level: not identifiable in this budget, see the report), so the score above is the primary
    criterion. The phase-21 thresholds (decay within 15 %, noise within 3 dB) from a start 30 % off, in 8 rounds of 16.
    Held to what the distance can tell apart, which the measured numbers set (module docstring of sideman.patch):
    - the tone's decay: every seed (not for the hat-like target, whose tone is masked by noise as loud as it:
      test_distance_is_flat_in_a_masked_tone);
    - the noise layer's decay: every seed for the hat-like target, half of the seeds for the others, where the
      layer is 10 to 12 dB under the tone and 15 % moves the distance by about one repeat spread;
    - the noise layer's level: half of the seeds, and the level it has at the time the tone has fallen 8.7 dB
      (decay and level of the layer lie on a ridge, this does not) within 3 dB at the median seed."""
    runs = runs_of(kind, fresh)
    tone, noise_decay, noise_db, at_decay = [], [], [], []
    for true, res, render in runs:
        assert res.distance < res.start_distance
        assert res.rounds <= 8 and render.batches == [16] * res.rounds
        r = recovered(true, res.best)
        tone.append(abs(r["decay"] - 1) <= oneshot.TIME_RATIO)
        noise_decay.append(abs(r["noise_decay"] - 1) <= oneshot.TIME_RATIO)
        noise_db.append(abs(r["noise_db"]) <= oneshot.LEVEL_DB)
        at_decay.append(abs(noise_at(res.best, true["decay"]) - noise_at(true, true["decay"])))
    if kind != "hat":
        assert all(tone), "%s tone decay within 15 %% in seeds %s" % (kind, tone)
    assert all(noise_decay) if kind == "hat" else np.mean(noise_decay) >= 0.5, "%s noise decay: %s" % (kind, noise_decay)
    assert np.mean(noise_db) >= 0.5, "%s noise level: %s" % (kind, noise_db)
    assert np.median(at_decay) <= oneshot.LEVEL_DB, "%s noise level at the tone's decay time, dB off: %s" % (kind, np.round(at_decay, 1))


@pytest.mark.parametrize("kind", ["kick", "tom", "hat"])
def test_search_from_the_truth_stays_and_stalls(kind):
    true, res, render = run(kind, start=dict(TRUE[kind]))
    assert res.stalled and res.rounds < 8
    r = recovered(true, res.best)
    assert abs(r["noise_decay"] - 1) <= oneshot.TIME_RATIO and abs(r["noise_db"]) <= oneshot.LEVEL_DB
    if kind != "hat":
        assert abs(r["decay"] - 1) <= oneshot.TIME_RATIO


def test_distance_is_flat_in_a_masked_tone():
    """Why the hat-like target's tone decay is not asked for: 30 % off in it moves the distance by far less than
    the same error in the noise layer's decay, and by about the repeat spread of one render."""
    true = TRUE["hat"]
    target = standin(true, TARGET_SEED)

    def mean_distance(p):
        return np.mean([oneshot.spectral_distance(standin(p, s), target, SR) for s in range(1, 9)])
    base = mean_distance(true)
    tone = max(abs(mean_distance(dict(true, decay=true["decay"] * f)) - base) for f in (0.7, 1.3))
    noise = min(mean_distance(dict(true, noise_decay=true["noise_decay"] * f)) - base for f in (0.7, 1.3))
    assert tone < 0.05 and noise > 0.3


def test_search_renders_one_full_batch_per_round_and_is_deterministic():
    a = run("tom")
    b = run("tom")
    assert a[2].batches == [16] * a[1].rounds and a[1].renders == 16 * a[1].rounds
    assert a[1].best == b[1].best and a[1].per_round == b[1].per_round
    other = run("tom", seed=1)[1]
    assert other.best != a[1].best


def test_search_keeps_bounds_and_pins():
    true = TRUE["kick"]
    bounds = bounds_for(true)
    bounds["decay"] = (0.060, 0.060, "log")             # pinned by an empty range
    bounds.pop("noise_decay")                           # pinned by leaving it out
    seen = []

    def render(patches):
        seen.extend(patches)
        return [standin(p, len(seen) + k) for k, p in enumerate(patches)]

    start = start_for(true)
    start["decay"] = 0.060
    res = patch.search(start, bounds, render, standin(true, TARGET_SEED), SR, rounds=3)
    assert len(seen) == 48
    for p in seen:
        assert set(p) == set(start)
        assert p["decay"] == 0.060 and p["noise_decay"] == start["noise_decay"]
        for name, (lo, hi, *_) in bounds.items():
            assert lo <= p[name] <= hi, (name, p[name])
    assert res.best["decay"] == 0.060 and res.best["noise_decay"] == start["noise_decay"]
    assert "decay" not in res.steps and "noise_decay" not in res.steps


def test_search_reports_the_incumbents_repeat_spread():
    true, res, _ = run("hat", rounds=4, start=dict(TRUE["hat"]))
    assert res.repeat_spread is not None and res.repeat_spread > 0         # fresh noise: renders of one patch differ
    _, det, _ = run("hat", rounds=4, fresh=False, start=dict(TRUE["hat"]))
    assert det.repeat_spread == 0


def test_search_survives_a_failing_renderer():
    true = TRUE["kick"]
    short = lambda b, i, p: standin(p, 1)[:50]                              # noqa: E731
    silent = lambda b, i, p: np.zeros(int(SECONDS * SR))                    # noqa: E731
    nan = lambda b, i, p: np.full(int(SECONDS * SR), np.nan)                # noqa: E731

    def bad(b, i, p):
        return {3: short, 5: silent, 7: nan}[i](b, i, p) if i in (3, 5, 7) else None

    render = Renderer(bad=bad)
    res = patch.search(start_for(true), bounds_for(true), render, standin(true, TARGET_SEED), SR, rounds=3)
    assert res.rounds == 3 and math.isfinite(res.distance)
    assert [f["index"] for f in res.failures if f["round"] == 0] == [3, 5, 7]
    assert {f["reason"].split(" ")[0] for f in res.failures} == {"too", "silent", "NaN"}


def test_search_failed_incumbent_is_listed_and_the_search_goes_on():
    true = TRUE["kick"]
    render = Renderer(bad=lambda b, i, p: np.zeros(int(SECONDS * SR)) if (b, i) == (1, 0) else None)
    res = patch.search(start_for(true), bounds_for(true), render, standin(true, TARGET_SEED), SR, rounds=3)
    assert {"round": 1, "index": 0, "reason": "silent"} in res.failures and math.isfinite(res.distance)


def test_search_refuses_a_start_outside_its_bounds_and_a_wrong_batch():
    true = TRUE["kick"]
    start = start_for(true)
    start["pitch_hz"] = true["pitch_hz"] * 2
    with pytest.raises(ValueError, match="outside its bounds"):
        patch.search(start, bounds_for(true), Renderer(), standin(true, TARGET_SEED), SR)
    with pytest.raises(ValueError, match="hits for"):
        patch.search(start_for(true), bounds_for(true), lambda ps: [], standin(true, TARGET_SEED), SR)


# ---- from construction: step adaptation, the margin, a raising renderer, bad shapes, `step` overrides ----

def one_parameter_bounds():
    return {"decay": (0.005, 0.5, "log")}


def spreads_of(batches: list) -> list:
    """The spread (std of ln decay) of the candidates of each batch: what the search's step was in that round
    (mirrored samples, so the batch's std is the step, up to reflection at a bound)."""
    return [float(np.std([math.log(p["decay"]) for p in b[1:]])) for b in batches]


class Capture(Renderer):
    def __call__(self, patches):
        self.seen = getattr(self, "seen", []) + [patches]
        return super().__call__(patches)


def one_parameter_search(decay_start: float, **kw):
    true = dict(TRUE["tom"])
    start = dict(true, decay=decay_start)
    render = Capture()
    bounds = {"decay": (0.002, 2.0, "log")}
    start["pitch_hz"] = true["pitch_hz"]
    res = patch.search(start, bounds, render, standin(true, TARGET_SEED), SR, patience=99, **kw)
    return res, spreads_of(render.seen)


def test_search_grows_its_step_far_from_the_truth_and_shrinks_it_at_the_truth():
    far, far_spreads = one_parameter_search(0.4)                 # 8 times the true decay, in a wide bound
    assert max(far_spreads) > 3 * far_spreads[0], far_spreads    # the step grew on the way (measured: 13 times)
    assert abs(math.log(far.best["decay"] / TRUE["tom"]["decay"])) < abs(math.log(0.4 / TRUE["tom"]["decay"]))
    at, at_spreads = one_parameter_search(TRUE["tom"]["decay"])
    assert at_spreads[-1] < 0.5 * at_spreads[0], at_spreads      # and shrank where there was nothing to find


def scripted(monkeypatch, script):
    """A renderer whose hits are constant arrays that carry the score the script gives for (round, index); the
    distance is the array's value. Returns the renderer, which counts its batches."""
    monkeypatch.setattr(patch, "spectral_distance", lambda x, t, sr, **kw: float(x[0]))
    calls = []

    def render(patches):
        r = len(calls)
        calls.append(patches)
        return [script(r, i) for i in range(len(patches))]
    render.calls = calls
    return render


def hit_scoring(d):
    return np.full(2000, d)


def test_search_keeps_the_incumbent_against_a_gain_within_one_spread(monkeypatch):
    """The incumbent scores 0.50, 0.51, 0.49 (mean 0.50, spread 0.01); a candidate shows a single 0.495 in
    round 2. It is not 0.01 better, so the incumbent keeps its place. (With MARGIN = 0 it would take over.)"""
    incumbent = {0: 0.50, 1: 0.51, 2: 0.49}
    render = scripted(monkeypatch, lambda r, i: hit_scoring(incumbent[r] if i == 0 else 0.495 if (r, i) == (2, 4) else 0.9))
    start = {"decay": 0.05}
    res = patch.search(start, one_parameter_bounds(), render, np.ones(2000), SR, rounds=3)
    assert res.best == start and res.distance == pytest.approx(0.50)
    assert res.repeat_spread == pytest.approx(0.01)
    assert res.rounds == 3 and not res.failures


def test_search_failed_incumbent_render_is_left_out_of_its_mean(monkeypatch):
    incumbent = {0: hit_scoring(0.5), 1: np.zeros(2000), 2: hit_scoring(0.6)}
    render = scripted(monkeypatch, lambda r, i: incumbent[r] if i == 0 else hit_scoring(0.9))
    start = {"decay": 0.05}
    res = patch.search(start, one_parameter_bounds(), render, np.ones(2000), SR, rounds=3)
    assert res.best == start and res.distance == pytest.approx(0.55)          # (0.5 + 0.6) / 2, no inf, no zero
    assert {"round": 1, "index": 0, "reason": "silent"} in res.failures


def test_search_returns_the_rounds_so_far_when_the_renderer_raises():
    true = TRUE["kick"]
    calls = []

    class Boom(Renderer):
        def __call__(self, patches):
            calls.append(1)
            if len(calls) == 4:                    # round 3
                raise RuntimeError("Live went away")
            return super().__call__(patches)

    res = patch.search(start_for(true), bounds_for(true), Boom(), standin(true, TARGET_SEED), SR, rounds=6)
    assert len(calls) == 4                         # not called again
    ref = patch.search(start_for(true), bounds_for(true), Renderer(), standin(true, TARGET_SEED), SR, rounds=3)
    assert res.rounds == 3 and res.best == ref.best and res.distance == ref.distance
    assert res.error == "RuntimeError: Live went away" and res.renders == 48
    assert res.failures[-1] == {"round": 3, "index": None, "reason": "render raised RuntimeError: Live went away"}
    assert ref.error is None


def test_search_raises_when_the_renderer_raises_before_anything_scored():
    true = TRUE["kick"]

    def boom(patches):
        raise RuntimeError("no Live")
    with pytest.raises(RuntimeError, match="no Live"):
        patch.search(start_for(true), bounds_for(true), boom, standin(true, TARGET_SEED), SR)


def test_search_names_wrong_shapes_apart_from_empty_renders():
    true = TRUE["kick"]
    good = lambda p: standin(p, 1)                                          # noqa: E731
    shapes = {2: np.float64(1.0), 3: np.zeros((2, 3, 4)), 4: np.zeros(0), 5: np.zeros((0, 2))}

    def bad(b, i, p):
        return shapes.get(i)

    res = patch.search(start_for(true), bounds_for(true), Renderer(bad=bad), standin(true, TARGET_SEED), SR, rounds=2)
    reasons = {f["index"]: f["reason"] for f in res.failures if f["round"] == 0}
    assert reasons[2].startswith("wrong shape (0 dimensions") and reasons[3].startswith("wrong shape (3 dimensions")
    assert reasons[4] == "empty" and reasons[5] == "empty"
    assert math.isfinite(res.distance) and good


@pytest.mark.parametrize("step", [{"decay": 0.0}, {"decay": -1.0}, {"decay": float("nan")}, {"decay": float("inf")},
                                  {"decay": "big"}, {"nope": 0.1}, {"noise_decay": 0.1}])
def test_search_refuses_a_bad_step_override(step):
    true = TRUE["kick"]
    bounds = bounds_for(true)
    bounds.pop("noise_decay")                                              # pinned: not a searched name
    with pytest.raises(ValueError, match="step"):
        patch.search(start_for(true), bounds, Renderer(), standin(true, TARGET_SEED), SR, step=step)


def test_search_takes_a_step_override():
    _, default = one_parameter_search(0.1)
    _, wide = one_parameter_search(0.1, step={"decay": 0.5})               # same draws, a step 0.5 / 0.15 times larger
    assert wide[0] / default[0] == pytest.approx(0.5 / patch.STEP_LOG, rel=0.1)


# ---- the first guess: operator_guess, operator_live, to_live / from_live, and the bridge to the search ----

AE_20DB_PER_T = 0.227        # a device from insert_device: the amp envelope falls 20 dB in 0.227 T (Live 12.4.6, 2026-11-12)
#                              hard-coded here, not read from patch.FALL20_PER_T, so that a wrong constant moves the guess
TAU_FROM_T = AE_20DB_PER_T / math.log(10.0)            # tau (ms) = Operator Decay time T (ms) * this: exp(-T / tau) = -20 dB at 0.227 T
T_PER_TAU = math.log(10.0) / AE_20DB_PER_T             # and the inverse, Decay time of a given tau
VARIANT = {"pitch_hz": 70.0, "pitch_st": 8.0, "pitch_tau": 0.040, "decay": 0.060, "noise_db": -6.0,
           "noise_decay": 0.030}     # noise half as long as the tone and 6 dB under it: a body noise reading above the floor


def profile_of(true: dict, seed: int, **kw) -> dict:
    return oneshot.profile_array(standin(true, seed, **kw)[:, None], SR, single=True)


noise_onset_db = patch.noise_onset_db


PE_DECAY_PER_TAU = 7.94      # Pe Decay / tau, measured on a scratch device (7.0 to 8.4; not the amp envelope's 10.1),
#                              hard-coded here so that a wrong patch.PE_DECAY_PER_TAU changes what the stand-in plays


def standin_from_live(live: dict) -> dict:
    """The stand-in's parameters for what operator_live asks Live to play (the test double's reading of Operator's
    physical values, under the assumptions in patch.ASSUMPTIONS). The mapping goes through operator_live, so a
    mistake in its formulas (the onset's sign, Pe Decay, Coarse and Fine) moves the stand-in's sound."""
    p = live["params"]
    f = 440.0 * 2 ** ((live["note"] - 69) / 12) * p["A Coarse"] * (1 + p["A Fine"] / 1000)
    out = {"pitch_hz": f, "pitch_st": 0.0, "pitch_tau": 0.01, "decay": 0.0005, "noise_db": -80.0, "noise_decay": 0.05}
    if p["Osc-A On"]:
        out["decay"] = p["Ae Decay"] * TAU_FROM_T / 1000
    if p["Pe On"]:
        out["pitch_st"], out["pitch_tau"] = p["Pe Init"], p["Pe Decay"] / PE_DECAY_PER_TAU / 1000
    if p["Osc-B On"]:
        out["noise_db"] = p["Osc-B Level"] - p["Osc-A Level"] if p["Osc-A On"] else 40.0
        out["noise_decay"] = p["Be Decay"] * TAU_FROM_T / 1000
    return out


GUESS_CASES = {"kick": TRUE["kick"], "tom": TRUE["tom"], "kick-with-long-noise": VARIANT}


@pytest.mark.parametrize("kind", list(GUESS_CASES))
def test_guess_gives_back_the_known_values_within_the_phase_21_thresholds(kind):
    """Pitch within PITCH_ST, drop within DROP_ST, the glide's time constant, the tone's 20 dB fall and the noise's
    decay within TIME_RATIO, the noise level at the onset within LEVEL_DB, from hits with other noise than any the
    guess has seen. The kick's body noise reading is under the leakage floor, so its noise decay is the default (a
    fifth of the tone's), labelled as such and not asked for, and its onset level is asked to 6 dB, not 3."""
    true = GUESS_CASES[kind]
    for seed in range(1, 21):
        g = patch.operator_guess(profile_of(true, seed))
        q = g["patch"]
        assert abs(12 * math.log2(q["carrier_hz"] / true["pitch_hz"])) <= oneshot.PITCH_ST
        assert abs(q["pitch_drop_st"] - true["pitch_st"]) <= oneshot.DROP_ST
        assert abs(q["pitch_tau_ms"] / (true["pitch_tau"] * 1000) - 1) <= oneshot.TIME_RATIO, seed
        assert abs(q["tone_decay_ms"] * AE_20DB_PER_T / (true["decay"] * 1000 * math.log(10)) - 1) <= oneshot.TIME_RATIO
        if kind == "kick":       # the default noise decay: its extrapolation back to the onset is only good to 6 dB
            assert g["sources"]["noise_decay_ms"]["confidence"] == "default"
            assert abs(noise_onset_db(q) - true["noise_db"]) <= 6.0, (seed, noise_onset_db(q))
        else:
            assert abs(noise_onset_db(q) - true["noise_db"]) <= oneshot.LEVEL_DB, (seed, noise_onset_db(q))
            assert g["sources"]["noise_decay_ms"]["confidence"] == "derived"
            assert abs(q["noise_decay_ms"] * TAU_FROM_T / (true["noise_decay"] * 1000) - 1) <= oneshot.TIME_RATIO, seed


def test_guess_reads_the_tone_decay_long_when_the_noise_outlasts_the_tone():
    """A known limit, not a wish: the 20 dB fall is the whole hit's, so a noise layer that decays more slowly than
    the tone sets it. Here the noise (90 ms) is twice the tone's (45 ms): the tone's decay reads about 40 % long."""
    true = dict(VARIANT, decay=0.045, noise_decay=0.090)
    q = patch.operator_guess(profile_of(true, 1))["patch"]
    ratio = q["tone_decay_ms"] * TAU_FROM_T / (true["decay"] * 1000)
    assert 1.25 < ratio < 1.6, ratio


def reading(**kw) -> dict:
    """A hand-made profile of a pitched hit (the keys operator_guess reads), for rules checked from construction."""
    p = {"sample_rate": 48000, "pitched": True, "pitch_body_hz": 100.0, "pitch_drop_st": 0.0, "pitch_settle_ms": None,
         "pitch_start_hz": None, "pitch_start_at_ms": None, "pitch_track_hz": {}, "attack_ms": 3.0, "fall_20_ms": 100.0,
         "fall_40_ms": 200.0, "noise_db": {"attack": None, "body": None}, "peak_db": -6.0, "bands_db": None,
         "click_db": -3.0, "flatness": {"attack": 0.05, "body": 0.05}}
    p.update(kw)
    return p


def test_guess_treats_a_drop_over_in_under_10_ms_as_a_click_and_takes_the_settled_pitch_as_the_carrier():
    click = patch.operator_guess(reading(pitch_drop_st=17.0, pitch_settle_ms=8.0, pitch_start_hz=270.0, pitch_start_at_ms=0.3))
    assert click["patch"]["pitch_drop_st"] == 0.0 and click["structure"]["pitch_env_on"] is False
    assert "click" in click["sources"]["pitch_drop_st"]["reading"]
    glide = patch.operator_guess(reading(pitch_body_hz=51.1, pitch_drop_st=19.9, pitch_settle_ms=74.0, pitch_start_hz=161.7,
                                         pitch_start_at_ms=2.6,
                                         pitch_track_hz={5: 126.8, 10: 90.0, 20: 68.5, 40: 55.7, 80: 51.9, 150: 48.0, 300: 47.9}))
    assert glide["patch"]["carrier_hz"] == pytest.approx(47.95, abs=0.1) and glide["structure"]["pitch_env_on"] is True
    assert "late track" in glide["sources"]["carrier_hz"]["reading"] or "fitted" in glide["sources"]["carrier_hz"]["reading"]


def test_guess_takes_the_noise_decay_from_a_tail_longer_than_an_exponential_and_notes_a_non_exponential_decay():
    snare = patch.operator_guess(reading(fall_20_ms=111.0, fall_40_ms=508.0, noise_db={"attack": -6.9, "body": -6.9},
                                         click_db=-7.8, flatness={"attack": 0.075, "body": 0.069}))
    tail_t = (508.0 - 111.0) / math.log(10) / TAU_FROM_T            # Operator T of the 20 dB between the two falls
    assert snare["patch"]["noise_decay_ms"] == pytest.approx(tail_t, rel=1e-6)
    assert snare["patch"]["tone_decay_ms"] == pytest.approx(111.0 / AE_20DB_PER_T)
    assert any("not exponential" in n and "falls short" in n for n in snare["notes"])
    gated = patch.operator_guess(reading(fall_20_ms=48.5, fall_40_ms=56.3))
    assert any("not exponential" in n and "rings on" in n for n in gated["notes"])
    plain = patch.operator_guess(reading())
    assert not [n for n in plain["notes"] if "exponential" in n]


def test_guess_for_an_unpitched_hit_is_noise_led():
    """20 seeds. In one of them (13) the profile itself calls the hat pitched (its 7 kHz tone stands out of that
    noise draw), and the guess then reads the noise decay 67 % short: a limit of the classification, counted here
    and not hidden (at most 2 of 20 are allowed here, not exactly seed 13: the draw is not the point)."""
    true = TRUE["hat"]
    misclassified = []
    for seed in range(1, 21):
        prof = profile_of(true, seed)
        if prof["pitched"]:
            misclassified.append(seed)
            continue
        g = patch.operator_guess(prof)
        q = g["patch"]
        assert g["structure"]["tone_on"] is False and g["structure"]["noise_on"] is True
        assert not {"carrier_hz", "pitch_drop_st", "tone_decay_ms", "noise_level_ref_db"} & set(q)
        assert abs(q["noise_decay_ms"] * TAU_FROM_T / (true["noise_decay"] * 1000) - 1) <= oneshot.TIME_RATIO, seed
        live = patch.operator_live(q, g["structure"])
        assert live["params"]["Osc-A On"] == 0 and live["params"]["Osc-B On"] == 1 and live["note"] == 60
    assert len(misclassified) <= 2


def test_guess_never_attacks_faster_than_the_floor_and_follows_a_slower_attack():
    fast = patch.operator_guess(profile_of(TRUE["tom"], 1))["patch"]
    assert fast["tone_attack_ms"] == patch.ATTACK_FLOOR_MS and fast["noise_attack_ms"] >= patch.NOISE_ATTACK_FLOOR_MS
    slow = patch.operator_guess(profile_of(TRUE["tom"], 1, attack=0.012))["patch"]
    assert slow["tone_attack_ms"] > 3 * patch.ATTACK_FLOOR_MS            # a slow attack is the measured one
    for q in (fast, slow):
        assert q["tone_attack_ms"] >= patch.ATTACK_FLOOR_MS


def test_guess_labels_every_value_and_free_parameters_stay_inside_their_bounds():
    for kind in ("kick", "tom", "hat"):
        g = patch.operator_guess(profile_of(TRUE[kind], 1))
        assert set(g["sources"]) == set(g["patch"])
        for src in g["sources"].values():
            assert src["confidence"] in ("measured", "derived", "default") and src["reading"]
        assert set(g["free"]) | set(g["pinned"]) >= set(g["patch"]) - {"tone_level_db"} - ({"noise_level_ref_db"} - set(g["pinned"]))
        assert not set(g["free"]) & set(g["pinned"])
        args = patch.search_arguments(g)
        for name, (lo, hi, scale) in args["bounds"].items():
            assert lo < args["start"][name] < hi and scale in ("log", "lin")
        assert g["assumptions"] and "noise_level_ref_db" not in g["free"]       # the level is the pinned one of the pair


def test_guess_refuses_a_profile_without_a_20_db_fall():
    prof = profile_of(TRUE["tom"], 1)
    prof["fall_20_ms"] = None
    with pytest.raises(ValueError, match="20 dB fall"):
        patch.operator_guess(prof)


def test_operator_parameter_names_are_the_195_read_from_live():
    names = patch.operator_parameters()
    assert len(names) == 195 and len(set(names)) == 195
    assert "A Fix On " in names and names[0] == "Device On" and names[194] == "Shaper Drive"
    assert pathlib.Path(patch.__file__).with_name("data").joinpath("operator_parameters.json").is_file()


@pytest.mark.parametrize("kind", ["kick", "tom", "hat"])
def test_every_name_the_guess_sets_is_an_operator_parameter(kind):
    g = patch.operator_guess(profile_of(TRUE[kind], 1))
    live = patch.operator_live(g["patch"], g["structure"])
    assert live["params"] and set(live["params"]) <= set(patch.operator_parameters())
    assert all(isinstance(v, (int, float)) for v in live["params"].values())
    if kind != "hat":
        p = live["params"]
        assert p["A Coarse"] == 1 and 0 <= p["A Fine"] <= 999
        f_note = 440.0 * 2 ** ((live["note"] - 69) / 12)
        assert f_note * p["A Coarse"] * (1 + p["A Fine"] / 1000) == pytest.approx(g["patch"]["carrier_hz"], rel=2e-3)
        assert p["Ae Sustain"] == -math.inf and p["Osc-B Level"] <= 0 <= p["Osc-A Level"] + 60


# ---- to_live / from_live on small hand-made tables

TIME_TABLE = [[0.0, 1.0, "ms"], [0.25, 10.0, "ms"], [0.5, 100.0, "ms"], [0.75, 1.0, "s"], [1.0, 10.0, "s"]]
DB_TABLE = [[0.0, float("-inf"), "dB"], [0.01, -60.0, "dB"], [1.0, 0.0, "dB"]]
UNITS = {"Ae Decay": TIME_TABLE, "Osc-A Level": DB_TABLE, "Pe Init": [[0.0, -48.0, "st"], [1.0, 48.0, "st"]]}


def test_to_live_interpolates_in_the_log_where_the_display_spans_decades_and_linearly_otherwise():
    out = patch.to_live({"Ae Decay": 100.0, "Osc-A Level": 0.0, "Pe Init": 0.0}, UNITS)
    assert out == pytest.approx({"Ae Decay": 0.5, "Osc-A Level": 1.0, "Pe Init": 0.5})
    assert patch.to_live({"Ae Decay": 1000.0}, UNITS)["Ae Decay"] == pytest.approx(0.75)         # 1 s = 1000 ms
    assert patch.to_live({"Ae Decay": 10 ** 1.5}, UNITS)["Ae Decay"] == pytest.approx(0.375)    # log midpoint of 10 and 100
    assert patch.to_live({"Osc-A Level": -30.0}, UNITS)["Osc-A Level"] == pytest.approx(0.505)  # linear in dB
    assert patch.to_live({"Osc-A Level": -math.inf}, UNITS)["Osc-A Level"] == 0.0
    assert patch.to_live({"Pe Init": 12.0}, UNITS)["Pe Init"] == pytest.approx(0.625)


def test_to_live_and_from_live_round_trip_within_the_interpolation():
    values = {"Ae Decay": 37.0, "Osc-A Level": -12.5, "Pe Init": -7.25}
    back = patch.from_live(patch.to_live(values, UNITS), UNITS)
    assert back == pytest.approx(values, rel=1e-9)
    assert patch.from_live({"Osc-A Level": 0.0}, UNITS)["Osc-A Level"] == -math.inf
    live = {"Ae Decay": 0.6}
    assert patch.to_live(patch.from_live(live, UNITS), UNITS)["Ae Decay"] == pytest.approx(0.6)


def test_to_live_passes_raw_switches_and_untabled_coarse_and_fine():
    out = patch.to_live({"Algorithm": 10, "Osc-B Wave": 21, "A Coarse": 1, "A Fine": 59, "Osc-A Level": -6.0}, UNITS)
    assert out["Algorithm"] == 10 and out["Osc-B Wave"] == 21 and out["A Coarse"] == 1 and out["A Fine"] == 59


@pytest.mark.parametrize("values, units, match", [
    ({"Not A Param": 1.0}, UNITS, "not an Operator parameter"),
    ({"Ae Decay": 20000.0}, UNITS, "outside the table's range"),
    ({"Ae Decay": 0.5}, UNITS, "outside the table's range"),
    ({"Ae Attack": 5.0}, UNITS, "no units table"),
    ({"Ae Decay": 5.0}, {"Nope": TIME_TABLE}, "not an Operator parameter"),
    ({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, 1.0, "ms"], [0.5, 100.0, "ms"], [1.0, 50.0, "ms"]]}, "not monotone"),
    ({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, 1.0, "ms"], [1.0, 10.0, "dB"]]}, "mix"),
    ({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, -60.0, "dB"], [1.0, 0.0, "dB"]]}, "display is in db"),
    ({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, 1.0, "ms"], [0.0, 10.0, "ms"]]}, "same live value"),
    ({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, 1.0, "ms"]]}, "two finite rows"),
])
def test_to_live_refuses_what_it_cannot_convert(values, units, match):
    with pytest.raises(ValueError, match=match):
        patch.to_live(values, units)


def test_from_live_refuses_a_value_outside_the_table():
    with pytest.raises(ValueError, match="outside the table's range"):
        patch.from_live({"Ae Decay": 1.5}, UNITS)


def test_guess_start_beats_a_start_30_percent_off_in_the_same_budget():
    """The guess goes through search_arguments (bridge and bounds), the search runs in the guess's own parameters,
    8 rounds of 16, and every candidate is played through operator_live and the stand-in's reading of its
    physical values (standin_from_live), so the formulas of operator_live are in the loop; what is NOT in it is
    Live and the units table. From (a) the guess, (b) the free parameters 30 % off the truth (clipped into the
    bounds; the pinned ones as in the guess, so only what is searched differs). Each final patch is re-rendered
    with noise the search never saw. Measured over 10 seeds in the report."""
    for kind in ("kick", "tom"):
        true = TRUE[kind]
        target = standin(true, TARGET_SEED)
        g = patch.operator_guess(profile_of(true, 1))
        args = patch.search_arguments(g)
        off = dict(args["start"])
        t_true = {"tone_decay_ms": true["decay"] * 1000 / TAU_FROM_T, "noise_decay_ms": true["noise_decay"] * 1000 / TAU_FROM_T,
                  "pitch_drop_st": true["pitch_st"], "pitch_tau_ms": true["pitch_tau"] * 1000, "carrier_hz": true["pitch_hz"]}
        for k, (name, v) in enumerate(t_true.items()):
            if name in args["bounds"]:
                lo, hi, _ = args["bounds"][name]
                off[name] = min(max(v * (1.3 if k % 2 == 0 else 0.7), lo), hi)

        def play(p):
            return standin_from_live(patch.operator_live(p, g["structure"]))

        def render(ps):
            return [standin(play(p), 10 ** 6 + 17 + k) for k, p in enumerate(ps)]

        def final_distance(start, seed):
            res = patch.search(start, args["bounds"], render, target, SR, seed=seed)
            return np.mean([oneshot.spectral_distance(standin(play(res.best), 7 * 10 ** 6 + k), target, SR) for k in range(8)])
        from_guess = np.mean([final_distance(args["start"], s) for s in range(4)])
        from_off = np.mean([final_distance(off, s) for s in range(4)])
        assert from_guess < from_off, (kind, from_guess, from_off)


# ---- review round 1: reference times, spectral evidence for noise, the complete state, Pe Decay, the tail rules,
# ---- table plateaus, quantized and raw values, and operator_live checked against hand-computed values

SR48 = 48000


def band_limited(x, lo, hi, sr=SR48):
    """White noise x restricted to lo..hi Hz by zeroing the other FFT bins, at the rms it had."""
    spec = np.fft.rfft(x)
    hz = np.fft.rfftfreq(len(x), 1.0 / sr)
    spec[(hz < lo) | (hz > hi)] = 0.0
    y = np.fft.irfft(spec, len(x))
    return y * math.sqrt(np.mean(x ** 2) / np.mean(y ** 2))


def sine_hit(f=800.0, tau=0.005, ndb=-10.0, ntau=None, att=0.001, secs=None, seed=0, drop=0.0, ptau=0.02, gate=None,
             band=None):
    """A mono [n, 1] hit at 48 kHz: a sine with an exponential amp envelope (tau s), optional glide, noise at ndb against
    the tone at the onset decaying with ntau (default tau / 2); `gate` ends the tone abruptly (s); `band` = (lo, hi) Hz
    restricts the noise to that band (an FFT mask, before the envelope), at the same power."""
    ntau = ntau or tau / 2
    secs = secs or max(0.3, 12 * max(tau, ntau))
    t = np.arange(int(secs * SR48)) / SR48
    f_t = f * 2 ** (drop * np.exp(-t / ptau) / 12)
    ramp = np.clip(t / att, 0, 1)
    env = np.exp(-t / tau) if gate is None else (t < gate).astype(float)
    tone = np.sin(0.7 + 2 * np.pi * np.cumsum(f_t) / SR48) * env * ramp
    white = np.random.default_rng(seed).standard_normal(len(t)) / math.sqrt(2)
    noise = (white if band is None else band_limited(white, *band)) * 10 ** (ndb / 20) * np.exp(-t / ntau) * ramp
    return (0.5 * (tone + noise))[:, None]


def prof48(x):
    return oneshot.profile_array(x, SR48, single=True)


def test_profile_reports_where_its_attack_noise_reading_was_taken():
    short = prof48(sine_hit(tau=0.003))
    long_ = prof48(sine_hit(tau=0.02))
    assert short["attack_window_ms"][0] == 0.0 and 5.0 < short["attack_window_ms"][1] < 20.0
    assert long_["attack_window_ms"] == [0.0, 30.0]


def test_noise_level_reference_follows_the_profiles_window_on_short_hits():
    """800 Hz sine, noise at -10 dB at the onset, noise decay half the tone's, tone tau 3 to 20 ms. The reference
    time was a fixed 15 ms, so tau 3 / 5 / 8 / 12 ms put the onset at +55 / +24 / +7 / +1 dB and Osc-A down by as
    much. Now the reference is half the profile's attack window and the extrapolation is clamped: the onset is
    within 6 dB of -10 at every tau (the noise decay is a default here, which is what the 6 dB allows for) and
    Osc-A stays within 1 dB of 0. A layer under the profile's floor (-25 dB) is left out: at most 2 of 15 draws."""
    missing = 0
    for tau in (0.003, 0.005, 0.008, 0.012, 0.020):
        for seed in range(3):
            g = patch.operator_guess(prof48(sine_hit(tau=tau, seed=seed)))
            if not g["structure"]["noise_on"]:
                missing += 1
                continue
            q = g["patch"]
            assert abs(patch.noise_onset_db(q) - (-10.0)) <= 6.0, (tau, seed, patch.noise_onset_db(q))
            p = patch.operator_live(q, g["structure"])["params"]
            assert p["Osc-A Level"] >= -1.0 and p["Osc-B Level"] == pytest.approx(min(0.0, patch.noise_onset_db(q)))
            aw = g["patch"]["noise_ref_ms"]
            assert aw == pytest.approx(prof48(sine_hit(tau=tau, seed=seed))["attack_window_ms"][1] / 2, rel=1e-6)
    assert missing <= 2


def test_noise_onset_extrapolation_is_clamped():
    q = {"noise_level_ref_db": -20.0, "noise_ref_ms": 15.0, "tone_decay_ms": 1000.0, "noise_decay_ms": 5.0}
    assert patch.noise_onset_db(q) == pytest.approx(-20.0 + patch.ONSET_CLAMP_DB)
    q["noise_decay_ms"], q["tone_decay_ms"] = 1000.0, 5.0
    assert patch.noise_onset_db(q) == pytest.approx(-20.0 - patch.ONSET_CLAMP_DB)


def test_no_noise_layer_on_a_gliding_sine_a_gated_tone_or_a_click_on_a_tone_and_a_noise_snare_keeps_one():
    """Window leakage of a glide, a gated tone's cliff or a click read 'noise' near the floor but have no
    broadband energy behind it (click_db, flatness): a pinned layer there was 20 to 55 dB louder than any
    energy above 2 kHz (the Bodzin kick, low tom, clave). A real noise snare keeps its layer, pinned."""
    glide = sine_hit(f=50.0, tau=0.08, ndb=-80.0, drop=12.0, ptau=0.025, att=0.0015)
    gated = sine_hit(f=200.0, tau=1.0, ndb=-80.0, gate=0.06)
    click = sine_hit(f=300.0, tau=0.06, ndb=-80.0)
    click[:48, 0] += np.random.default_rng(5).standard_normal(48) * 0.05            # 1 ms noise burst on the onset
    for name, x in (("glide", glide), ("gated", gated), ("click", click)):
        g = patch.operator_guess(prof48(x))
        assert "noise_level_ref_db" not in g["pinned"] and not g["structure"]["noise_on"], name
    snare = patch.operator_guess(prof48(sine_hit(f=190.0, tau=0.06, ndb=-4.0, ntau=0.15)))
    assert snare["structure"]["noise_on"] and "noise_level_ref_db" in snare["pinned"]


def test_a_noise_reading_near_the_floor_frees_the_level_down_to_off():
    g = patch.operator_guess(reading(noise_db={"attack": -22.0, "body": None}, click_db=-20.0))
    assert g["structure"]["noise_on"] and "noise_level_ref_db" in g["free"] and "noise_level_ref_db" not in g["pinned"]
    lo, hi, scale = g["free"]["noise_level_ref_db"]
    assert lo == patch.NOISE_OFF_DB and hi > g["patch"]["noise_level_ref_db"] and scale == "lin"
    stronger = patch.operator_guess(reading(noise_db={"attack": -12.0, "body": None}, click_db=-12.0))
    assert "noise_level_ref_db" in stronger["pinned"]


STATE_KEYS = {"Algorithm", "Volume", "Time", "Time < Key", "Osc-A On", "Osc-B On", "Osc-C On", "Osc-D On", "LFO On",
              "A Fix On ", "B Fix On ", "A Coarse", "A Fine", "Osc-A Wave", "Osc-B Wave", "Osc-A Level", "Osc-B Level",
              "Ae Attack", "Ae Decay", "Ae Sustain", "Ae Release", "Be Attack", "Be Decay", "Be Sustain", "Be Release",
              "Pe On", "Osc-A < Pe", "Osc-B < Pe", "Pe Init", "Pe Peak", "Pe Decay", "Pe Sustain", "Pe Amount", "Pe Amt A",
              "Pe Amt B", "Filter On", "Filter Type", "Filter Freq"}


def structures():
    kick = patch.operator_guess(profile_of(TRUE["kick"], 1))
    tom = patch.operator_guess(profile_of(TRUE["tom"], 1))
    hat = patch.operator_guess(profile_of(TRUE["hat"], 1))
    sine = patch.operator_guess(reading())                          # a tone only: no noise, no glide
    hat_hp = patch.operator_guess(reading(pitched=False, bands_db={"sub": -60, "bass": -60, "low_mid": -60, "mid": -30,
                                                                    "high_mid": -3, "high": -1}))
    return {"kick": kick, "tom": tom, "hat": hat, "tone only": sine, "noise with a filter": hat_hp}


def test_operator_live_returns_the_same_complete_state_for_every_structure():
    seen = structures()
    assert seen["tone only"]["structure"]["noise_on"] is False and seen["noise with a filter"]["structure"]["filter_on"]
    for name, g in seen.items():
        live = patch.operator_live(g["patch"], g["structure"])
        assert set(live["params"]) == STATE_KEYS, name
        assert set(live["params"]) <= set(patch.operator_parameters())
        assert live["velocity"] == patch.NOTE_VELOCITY and 0 <= live["note"] <= 127
        assert live["note_length_ms"] >= max(live["params"]["Ae Decay"] if live["params"]["Osc-A On"] else 0,
                                             live["params"]["Be Decay"] if live["params"]["Osc-B On"] else 0)
        p, s = live["params"], g["structure"]
        assert (p["Osc-A On"], p["Osc-B On"], p["Osc-C On"], p["Osc-D On"]) == (int(s["tone_on"]), int(s["noise_on"]), 0, 0)
        assert (p["Pe On"], p["Osc-A < Pe"], p["Osc-B < Pe"], p["Filter On"], p["LFO On"]) == \
            (int(s["pitch_env_on"]), int(s["pitch_env_on"]), 0, int(s["filter_on"]), 0)
        assert p["Time"] == 0.0 and p["Time < Key"] == 0.0 and p["Algorithm"] == 10
    assert patch.operator_live(seen["tone only"]["patch"], seen["tone only"]["structure"])["params"]["Osc-B On"] == 0
    assert patch.operator_live(seen["hat"]["patch"], seen["hat"]["structure"])["note"] == 60


GENEROUS = {name: [[0.0, 0.1, "ms"], [0.25, 1.0, "ms"], [0.5, 10.0, "ms"], [0.75, 100.0, "ms"], [1.0, 20.0, "s"]]
            for name in ("Ae Attack", "Ae Decay", "Ae Release", "Be Attack", "Be Decay", "Be Release", "Pe Decay")}
GENEROUS.update({name: [[0.0, float("-inf"), "dB"], [0.01, -60.0, "dB"], [1.0, 6.0, "dB"]]
                 for name in ("Volume", "Osc-A Level", "Osc-B Level", "Ae Sustain", "Be Sustain")})
GENEROUS.update({name: [[0.0, -48.0, "st"], [1.0, 48.0, "st"]] for name in ("Pe Init", "Pe Peak", "Pe Sustain")})
GENEROUS.update({name: [[0.0, -100.0, "%"], [1.0, 100.0, "%"]] for name in ("Pe Amount", "Pe Amt A", "Pe Amt B", "Time", "Time < Key")})
GENEROUS["Filter Freq"] = [[0.0, 20.0, "Hz"], [1.0, 22.0, "kHz"]]


def test_every_structure_converts_through_a_generous_table_and_back():
    for name, g in structures().items():
        phys = patch.operator_live(g["patch"], g["structure"])["params"]
        live = patch.to_live(phys, GENEROUS)
        back = patch.from_live(live, GENEROUS)
        for k, v in phys.items():
            assert back[k] == pytest.approx(v, rel=1e-6, abs=1e-6) or (v == -math.inf and back[k] == -math.inf), (name, k)


def test_operator_live_values_against_hand_computed_ones():
    q = {"carrier_hz": 100.0, "pitch_drop_st": 12.0, "pitch_tau_ms": 30.0, "tone_attack_ms": 1.5, "tone_decay_ms": 500.0,
         "tone_level_db": 0.0, "noise_level_ref_db": -12.0, "noise_ref_ms": 10.0, "noise_decay_ms": 250.0,
         "noise_attack_ms": 0.1, "output_db": -6.0}
    s = {"tone_on": True, "noise_on": True, "pitch_env_on": True, "filter_on": False}
    live = patch.operator_live(q, s)
    p = live["params"]
    # note: MIDI 42 (92.50 Hz) is the one 1 to 2 semitones under 100 Hz; ratio 1.0811 = Coarse 1, Fine 81
    assert live["note"] == 42 and p["A Coarse"] == 1 and p["A Fine"] == 81
    # tau_tone = 500 * 0.227 / ln 10 = 49.29 ms, tau_noise = 24.65 ms: back = 8.686 * 10 * (1/24.65 - 1/49.29) = 1.762 dB
    assert p["Osc-B Level"] == pytest.approx(-12.0 + 1.762, abs=0.01) and p["Osc-A Level"] == 0.0
    assert p["Pe Decay"] == pytest.approx(30.0 * 7.94, rel=1e-9)               # 238.2 ms: tau * 7.94, the Pe constant
    assert p["Pe Init"] == p["Pe Peak"] == 12.0 and p["Pe Amt A"] == 100.0 and p["Volume"] == -6.0
    assert p["Ae Decay"] == p["Ae Release"] == 500.0 and p["Be Decay"] == 250.0 and p["Ae Sustain"] == -math.inf
    louder = patch.operator_live(dict(q, noise_level_ref_db=2.0), s)["params"]              # noise above the tone: the tone goes down
    assert louder["Osc-B Level"] == 0.0 and louder["Osc-A Level"] == pytest.approx(-(2.0 + 1.762), abs=0.01)


def test_operator_live_needs_coarse_above_1_only_for_carriers_the_note_range_cannot_reach():
    """The note is 1 to 2 semitones under the carrier (Coarse 1, Fine 59 to 122) up to MIDI 127 (12.54 kHz); above
    about 25 kHz the ratio reaches 2. No profile at 48 kHz gets there, a 96 kHz one could."""
    s = {"tone_on": True, "noise_on": False, "pitch_env_on": False, "filter_on": False}
    for f, coarse, note in ((100.0, 1, 42), (8000.0, 1, 118), (30000.0, 2, 127)):
        p = patch.operator_live({"carrier_hz": f, "tone_attack_ms": 1.5, "tone_decay_ms": 100.0}, s)
        assert p["note"] == note and p["params"]["A Coarse"] == coarse
        ratio = p["params"]["A Coarse"] * (1 + p["params"]["A Fine"] / 1000)
        assert 440.0 * 2 ** ((p["note"] - 69) / 12) * ratio == pytest.approx(f, rel=1e-3)
    with pytest.raises(ValueError, match="MIDI note 0"):
        patch.operator_live({"carrier_hz": 5.0, "tone_attack_ms": 1.5, "tone_decay_ms": 100.0}, s)


def test_the_glide_range_reaches_either_reading_of_the_pe_decay_mapping():
    """Pe Decay = tau * 7.94 (assumption 2) against Pe Decay = the settle time, tau * ln(drop / 0.5): the two
    differ by 7.94 / ln(drop / 0.5), 2 to 4 for 4 to 14 st; the free tau range must contain both."""
    for kind in ("kick", "tom"):
        g = patch.operator_guess(profile_of(TRUE[kind], 1))
        q, (lo, hi, _) = g["patch"], g["free"]["pitch_tau_ms"]
        other = q["pitch_tau_ms"] * PE_DECAY_PER_TAU / math.log(abs(q["pitch_drop_st"]) / oneshot.SETTLE_ST)
        assert lo <= q["pitch_tau_ms"] and lo <= other <= hi and hi >= q["pitch_tau_ms"] * 3.9


def test_a_tonal_tail_is_not_taken_as_the_noises_and_a_flat_one_is():
    tonal = patch.operator_guess(reading(fall_20_ms=26.7, fall_40_ms=118.8, noise_db={"attack": -18.0, "body": -8.0},
                                         click_db=-18.0, flatness={"attack": 0.05, "body": 0.001}))
    flat = patch.operator_guess(reading(fall_20_ms=26.7, fall_40_ms=118.8, noise_db={"attack": -18.0, "body": -8.0},
                                        click_db=-18.0, flatness={"attack": 0.05, "body": 0.07}))
    tail_t = (118.8 - 26.7) / math.log(10) / TAU_FROM_T
    assert flat["patch"]["noise_decay_ms"] == pytest.approx(tail_t, rel=1e-6) and "fall_40_ms - fall_20_ms" in flat["sources"]["noise_decay_ms"]["reading"]
    assert tonal["patch"]["noise_decay_ms"] != pytest.approx(tail_t, rel=0.05)
    assert "fall_40_ms - fall_20_ms" not in tonal["sources"]["noise_decay_ms"]["reading"]


def test_tone_plus_noise_must_not_outlast_the_measured_40_db_fall():
    """A body reading that says the noise barely decays would make tone + noise ring far past fall_40_ms (the
    low tom: noise T 951 ms, 15 dB over the reference at its 40 dB point). The noise decay is shortened until the
    modelled 40 dB fall is within 25 % of the measured one."""
    g = patch.operator_guess(reading(fall_20_ms=50.0, fall_40_ms=100.0, noise_db={"attack": -6.0, "body": -1.0}, click_db=-6.0))
    q = g["patch"]
    tau_t, tau_n = q["tone_decay_ms"] * TAU_FROM_T, q["noise_decay_ms"] * TAU_FROM_T
    assert patch._model_fall_ms(tau_t, tau_n, patch.noise_onset_db(q)) <= patch.CONSISTENT * 100.0 * 1.02
    assert "shortened" in g["sources"]["noise_decay_ms"]["reading"]
    assert tau_n < 3 * tau_t        # the unchecked reading was a noise 3 times the tone's (the clamp)


def test_the_tone_decay_may_fall_by_3_when_the_noise_outlasts_the_tone_and_a_short_noise_is_reachable():
    long_noise = patch.operator_guess(prof48(sine_hit(f=200.0, tau=0.045, ndb=-6.0, ntau=0.090)))
    q, (lo, hi, _) = long_noise["patch"], long_noise["free"]["tone_decay_ms"]
    assert q["noise_decay_ms"] >= 0.8 * q["tone_decay_ms"] and lo == pytest.approx(q["tone_decay_ms"] / patch.SLOW_NOISE_LOW_FACTOR)
    true_t = 0.045 * 1000 * T_PER_TAU
    assert lo <= true_t <= hi                       # the true tone decay is inside the searched range (it was 1.98 to 2.75 off)
    for tau, ntau in ((0.060, 0.006), (0.050, 0.005)):         # noise a tenth of the tone: the default is a fifth
        g = patch.operator_guess(prof48(sine_hit(f=180.0, tau=tau, ndb=-6.0, ntau=ntau)))
        if g["structure"]["noise_on"]:
            lo, hi, _ = g["free"]["noise_decay_ms"]
            assert lo <= ntau * 1000 * T_PER_TAU <= hi, (tau, ntau, lo, hi)


def test_guess_edge_cases():
    for bad in (0.0, -3.0, None):
        with pytest.raises(ValueError, match="20 dB fall"):
            patch.operator_guess(reading(fall_20_ms=bad))
    assert patch.operator_guess(reading(peak_db=None))["patch"]["output_db"] == pytest.approx(0.0 - 3.7)       # a 0 dBFS peak minus the sine's +3.7 dB
    g = patch.operator_guess({k: v for k, v in reading(pitched=False, bands_db={"sub": -30.0}).items() if k != "sample_rate"})
    assert "noise_filter_hz" not in g["patch"]
    capped = patch.operator_guess(reading(pitch_drop_st=72.0, pitch_settle_ms=60.0, pitch_start_hz=6400.0, pitch_start_at_ms=0.5))
    assert abs(capped["patch"]["pitch_drop_st"]) <= patch.PE_MAX_ST
    low = patch.operator_guess(reading(pitch_body_hz=5.0))
    assert any("MIDI note 0" in n for n in low["notes"])
    with pytest.raises(ValueError, match="MIDI note 0"):
        patch.operator_live(low["patch"], low["structure"])


def test_noise_attack_label_follows_the_measured_attack_like_the_tones():
    fast = patch.operator_guess(reading(attack_ms=0.4, noise_db={"attack": -12.0, "body": None}, click_db=-12.0))
    slow = patch.operator_guess(reading(attack_ms=4.0, noise_db={"attack": -12.0, "body": None}, click_db=-12.0))
    assert fast["sources"]["noise_attack_ms"]["confidence"] == "measured" or fast["patch"]["noise_attack_ms"] == 0.4
    assert slow["sources"]["noise_attack_ms"]["confidence"] == "measured" and slow["patch"]["noise_attack_ms"] == 4.0


def test_search_arguments_pulls_bounds_inside_the_units_tables():
    g = patch.operator_guess(profile_of(TRUE["tom"], 1))
    plain = patch.search_arguments(g)
    assert plain["bounds"] == dict(g["free"])
    narrow = dict(GENEROUS)
    narrow["Ae Decay"] = [[0.0, 150.0, "ms"], [1.0, 700.0, "ms"]]            # tone T of the tom is 50 ms * 10.14 = 507 ms: x1.5 leaves the table
    args = patch.search_arguments(g, narrow)
    lo, hi, _ = args["bounds"]["tone_decay_ms"]
    assert 150.0 <= lo <= g["patch"]["tone_decay_ms"] <= hi <= 700.0 and hi < g["free"]["tone_decay_ms"][1]
    assert args["bounds"]["carrier_hz"] == g["free"]["carrier_hz"]
    for end in (lo, hi):
        patch.to_live(patch.operator_live(dict(args["start"], tone_decay_ms=end), g["structure"])["params"], narrow)
    out_of_range = dict(GENEROUS, **{"Ae Decay": [[0.0, 1.0, "ms"], [1.0, 20.0, "ms"]]})
    with pytest.raises(ValueError, match="does not convert"):
        patch.search_arguments(g, out_of_range)


def test_to_live_collapses_display_rounding_plateaus_and_picks_log_or_linear_per_segment():
    plateau = {"Ae Decay": [[0.0, 0.10, "ms"], [0.01, 0.10, "ms"], [0.02, 0.10, "ms"], [0.5, 10.0, "ms"], [1.0, 1000.0, "ms"]]}
    out = patch.to_live({"Ae Decay": 0.10}, plateau)
    assert out["Ae Decay"] == pytest.approx(0.01)                 # the plateau's rows, at their mean live value
    assert patch.to_live({"Ae Decay": 1.0}, plateau)["Ae Decay"] == pytest.approx(0.01 + 0.5 * 0.49)   # log midpoint of 0.1 and 10
    assert patch.to_live({"Ae Decay": 100.0}, plateau)["Ae Decay"] == pytest.approx(0.75)    # log midpoint of 10 and 1000
    assert patch.from_live({"Ae Decay": 0.75}, plateau)["Ae Decay"] == pytest.approx(100.0)
    from_zero = {"Ae Decay": [[0.0, 0.0, "ms"], [0.25, 1.0, "ms"], [0.5, 10.0, "ms"], [0.75, 100.0, "ms"], [1.0, 1000.0, "ms"]]}
    assert patch.to_live({"Ae Decay": 30.0}, from_zero)["Ae Decay"] == pytest.approx(0.5 + 0.25 * math.log(3) / math.log(10))   # log: not 16.7 on a linear scale
    assert patch.to_live({"Ae Decay": 0.5}, from_zero)["Ae Decay"] == pytest.approx(0.125)        # the 0 to 1 ms segment is linear
    assert patch.from_live(patch.to_live({"Ae Decay": 30.0}, from_zero), from_zero)["Ae Decay"] == pytest.approx(30.0)
    with pytest.raises(ValueError, match="not monotone"):
        patch.to_live({"Ae Decay": 5.0}, {"Ae Decay": [[0.0, 1.0, "ms"], [0.5, 100.0, "ms"], [1.0, 50.0, "ms"]]})


def test_to_live_rounds_quantized_values_refuses_bad_raw_ones_and_reads_only_the_tables_it_needs():
    assert patch.to_live({"A Coarse": 1.4, "A Fine": 59.6}, {}) == {"A Coarse": 1, "A Fine": 60}
    for bad in ({"Osc-A On": 7}, {"Osc-A On": 0.5}, {"Filter On": -1}, {"Algorithm": 2.5}, {"Osc-B Wave": -1}):
        with pytest.raises(ValueError, match="raw Live value"):
            patch.to_live(bad, UNITS)
    assert patch.to_live({"Algorithm": 10, "Osc-B Wave": 21, "Osc-A On": 1}, UNITS) == {"Algorithm": 10, "Osc-B Wave": 21, "Osc-A On": 1}
    units = dict(UNITS, **{"Ae Attack": [[0.0, 1.0, "ms"]]})                  # one bad table, for a name not converted
    assert patch.to_live({"Ae Decay": 100.0}, units)["Ae Decay"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="two finite rows"):
        patch.to_live({"Ae Attack": 1.0}, units)


@pytest.mark.parametrize("name, attack, click, flat, pinned", [
    ("Bodzin kick (glide: leakage, little above 2 kHz)", -21.2, -28.8, 0.0, False),
    ("Bodzin low tom (a click on a gated tone)", -19.1, -25.5, 0.0, False),
    ("Bodzin clave (energy above 2 kHz, but a tone's flatness)", -18.5, -18.9, 0.0, False),
    ("Bodzin rim (the same)", -19.2, -18.3, 0.0, False),
    ("Bodzin snare (noise)", -6.9, -7.8, 0.075, True),
])
def test_noise_layer_gate_on_the_readings_of_real_samples(name, attack, click, flat, pinned):
    """The click_db and flatness conditions each matter: the kick and low tom fail the energy above 2 kHz, the clave
    and rim pass it on a hard strike and fail only the flatness. Readings are those of the Bodzin one-shots
    (`~/.omc/reports/sideman-patch-match/scripts/guess_bodzin.out`). All five are above the leakage floor, so all five
    keep a layer; only the snare, which passes the gate, has its level pinned. The others have it free, down to off."""
    g = patch.operator_guess(reading(noise_db={"attack": attack, "body": -8.0}, click_db=click,
                                     flatness={"attack": flat, "body": flat}))
    assert g["structure"]["noise_on"] is True and "noise_level_ref_db" in g["patch"], name
    assert ("noise_level_ref_db" in g["pinned"]) is pinned, name
    if not pinned:
        lo, hi, scale = g["free"]["noise_level_ref_db"]
        assert lo == patch.NOISE_OFF_DB and lo < g["patch"]["noise_level_ref_db"] < hi and scale == "lin", name
        assert any("kept with its level free" in n for n in g["notes"]), name


# ---- review round 2: band-limited noise, the tail, tables with a dip, the search's bound check

NOISE_TOM = dict(f=180.0, tau=0.06, ndb=-10.0, ntau=0.03)
NOISE_CASES = [          # (name, hit, true onset dB, pinned): the noise's colour is the only thing that changes
    ("tom, white noise", dict(NOISE_TOM), -10.0, True),
    ("tom, noise high-passed at 3 kHz", dict(NOISE_TOM, band=(3000.0, 23000.0)), -10.0, True),
    ("tom, noise band-passed 1-5 kHz", dict(NOISE_TOM, band=(1000.0, 5000.0)), -10.0, False),
    ("tom, noise band-passed 2-8 kHz", dict(NOISE_TOM, band=(2000.0, 8000.0)), -10.0, False),
    ("tom, noise low-passed at 1.5 kHz", dict(NOISE_TOM, band=(20.0, 1500.0)), -10.0, False),
    ("snare 200 Hz, noise band-passed 1-5 kHz", dict(f=200.0, tau=0.04, ndb=-3.0, ntau=0.08, band=(1000.0, 5000.0)), -3.0, False),
    ("kick, noise low-passed at 4 kHz", dict(f=55.0, tau=0.15, ndb=-10.0, ntau=0.01, band=(20.0, 4000.0), drop=12.0,
                                             ptau=0.025, att=0.0015), -10.0, False),
]


@pytest.mark.parametrize("name, kw, true_onset, pinned", NOISE_CASES, ids=[c[0] for c in NOISE_CASES])
def test_every_noise_colour_keeps_its_layer_and_only_one_with_evidence_pins_the_level(name, kw, true_onset, pinned):
    """A tone plus noise of a known level and colour (FFT mask). White noise and noise high-passed at 3 kHz are flat
    over some of the spectrum and have energy above 2 kHz at the level of the layer: the gate has its evidence and
    the level is pinned. Band-passed noise and low-passed noise read a flatness of 0.0 (over 20 Hz to 20 kHz), and the
    low-passed also has almost nothing above 2 kHz: the gate has no evidence, but the layer is real, so it is kept with
    the level free from off up to a few dB over the reading, and the true onset is within reach."""
    for seed in range(3):
        prof = prof48(sine_hit(seed=seed, **kw))
        g = patch.operator_guess(prof)
        q = g["patch"]
        assert g["structure"]["noise_on"] is True, (name, seed)
        assert ("noise_level_ref_db" in g["pinned"]) is pinned, (name, seed, prof["flatness"], prof["click_db"])
        if pinned:
            assert prof["flatness"]["attack"] >= patch.FLAT_ATTACK
            assert abs(patch.noise_onset_db(q) - true_onset) <= 3.0, (name, seed, patch.noise_onset_db(q))
        else:
            lo, hi, scale = g["free"]["noise_level_ref_db"]
            assert lo == patch.NOISE_OFF_DB and scale == "lin" and lo < q["noise_level_ref_db"] <= hi
            top = patch.noise_onset_db(dict(q, noise_level_ref_db=hi))
            assert top >= true_onset - 3.0, (name, seed, top)           # the search can reach the true level (and off)
            assert patch.noise_onset_db(dict(q, noise_level_ref_db=lo)) < -40.0           # and switch the layer off
            assert prof["flatness"]["attack"] < patch.FLAT_ATTACK       # why: band-limited noise reads flatness 0.0
            if not name.startswith("kick"):      # the kick's noise decay is a default (no body reading): its start is further off
                assert abs(patch.noise_onset_db(q) - true_onset) <= 3.0, (name, seed, patch.noise_onset_db(q))


def test_a_noisy_body_with_a_long_tail_and_no_flat_spectrum_keeps_the_tail_in_reach_of_the_noise_decay():
    """Body 9 dB down (12 % noise), the 40 dB fall 4.4 times the 20 dB one: a flat body is a noise tail and gives
    the decay; a body with flatness 0.0 may be a tonal ring or a band-passed noise tail. It is not taken
    as the noise's decay, but the decay's free range must reach it: it was x/÷3 around a much shorter decay."""
    kw = dict(fall_20_ms=26.7, fall_40_ms=118.8, noise_db={"attack": -2.0, "body": -9.0}, click_db=-2.0)
    tail_t = (118.8 - 26.7) / math.log(10) / TAU_FROM_T                  # T of the 20 dB between the two falls
    band = patch.operator_guess(reading(flatness={"attack": 0.0, "body": 0.0}, **kw))
    q, (lo, hi, _) = band["patch"], band["free"]["noise_decay_ms"]
    assert q["noise_decay_ms"] * patch.NOISE_DECAY_FACTOR < tail_t          # the range without the widening misses the tail
    assert "fall_40_ms - fall_20_ms" not in band["sources"]["noise_decay_ms"]["reading"]
    assert lo == pytest.approx(q["noise_decay_ms"] / patch.NOISE_DECAY_FACTOR) and hi == pytest.approx(tail_t, rel=1e-6)
    assert any("widened" in n for n in band["notes"])
    flat = patch.operator_guess(reading(flatness={"attack": 0.05, "body": 0.07}, **kw))
    assert "fall_40_ms - fall_20_ms" in flat["sources"]["noise_decay_ms"]["reading"]       # a flat body: the tail is the decay
    assert not any("widened" in n for n in flat["notes"])
    no_tail = patch.operator_guess(reading(flatness={"attack": 0.0, "body": 0.0}, **dict(kw, fall_40_ms=60.0)))
    assert not any("widened" in n for n in no_tail["notes"])            # no tail past 2.5 times the 20 dB fall: nothing to widen


def test_to_live_merges_only_adjacent_rows_with_the_same_display_so_a_table_with_a_dip_is_refused():
    for rows in ([[0, 1, "ms"], [0.4, 50, "ms"], [0.6, 1, "ms"], [1, 100, "ms"]], [[0, 1, "ms"], [0.5, 100, "ms"], [1, 1, "ms"]]):
        with pytest.raises(ValueError, match="not monotone"):
            patch.to_live({"Ae Decay": 20.0}, {"Ae Decay": rows})
        with pytest.raises(ValueError, match="not monotone"):
            patch.from_live({"Ae Decay": 0.7}, {"Ae Decay": rows})
    plateau = {"Ae Decay": [[0, 1, "ms"], [0.2, 10, "ms"], [0.3, 10, "ms"], [0.4, 10, "ms"], [1, 100, "ms"]]}
    assert patch.to_live({"Ae Decay": 10.0}, plateau)["Ae Decay"] == pytest.approx(0.3)      # adjacent rows: one at the mean
    assert patch.from_live({"Ae Decay": 0.3}, plateau)["Ae Decay"] == pytest.approx(10.0)
    falling = {"Ae Decay": [[0, 100, "ms"], [0.5, 10, "ms"], [0.6, 10, "ms"], [1, 1, "ms"]]}
    assert patch.to_live({"Ae Decay": 10.0}, falling)["Ae Decay"] == pytest.approx(0.55)


def test_the_end_to_end_stand_in_reads_pe_decay_by_the_amp_envelopes_constant():
    """standin_from_live divides Pe Decay by a constant written in this file (7.94, measured on a scratch device), so a
    patch.PE_DECAY_PER_TAU that is not 7.94 makes the stand-in glide for the wrong time. A glide of tau 40 ms must
    come back as 40 ms."""
    q = {"carrier_hz": 100.0, "pitch_drop_st": 12.0, "pitch_tau_ms": 40.0, "tone_attack_ms": 1.5, "tone_decay_ms": 300.0,
         "tone_level_db": 0.0, "output_db": -6.0}
    st = {"tone_on": True, "noise_on": False, "pitch_env_on": True, "filter_on": False}
    assert standin_from_live(patch.operator_live(q, st))["pitch_tau"] == pytest.approx(0.040, rel=1e-3)


# ---- review round 2, phantom layers: a free noise layer without evidence, searched against a target with and
# ---- without noise (every candidate played through operator_live and standin_from_live, fresh noise per render)

def test_search_renders_its_probes_in_round_0_in_place_of_the_last_candidates_and_checks_them():
    true = TRUE["tom"]
    bounds = bounds_for(true)
    start = start_for(true)
    probe = dict(start, noise_db=-60.0, decay=0.2)
    render = Capture()
    patch.search(start, bounds, render, standin(true, TARGET_SEED), SR, rounds=2, probes=[probe])
    assert [len(b) for b in render.seen] == [16, 16]
    assert render.seen[0][-1] == pytest.approx(probe) and render.seen[0][0] == start
    assert not any(p == pytest.approx(probe) for p in render.seen[1][1:])
    for bad, match in ((dict(probe, noise_db=-61.0), "outside the bounds"), ({k: v for k, v in probe.items() if k != "decay"}, "lacks")):
        with pytest.raises(ValueError, match=match):
            patch.search(start, bounds, Renderer(), standin(true, TARGET_SEED), SR, rounds=1, probes=[bad])
    with pytest.raises(ValueError, match="at most"):
        patch.search(start, bounds, Renderer(), standin(true, TARGET_SEED), SR, rounds=1, probes=[probe] * 16)


def test_search_arguments_offers_the_noise_layer_at_its_lower_bound_as_a_probe_only_when_its_level_is_free():
    free = patch.operator_guess(reading(noise_db={"attack": -15.0, "body": None}, click_db=-50.0, flatness={"attack": 0.0, "body": 0.0}))
    args = patch.search_arguments(free)
    assert args["probes"] == [dict(args["start"], noise_level_ref_db=patch.NOISE_OFF_DB)]
    pinned = patch.operator_guess(reading(noise_db={"attack": -10.0, "body": None}, click_db=-10.0))
    assert "probes" not in patch.search_arguments(pinned) and "probes" not in patch.search_arguments(patch.operator_guess(reading()))
    narrow = dict(GENEROUS, **{"Osc-B Level": [[0.0, -30.0, "dB"], [1.0, 0.0, "dB"]]})
    pulled = patch.search_arguments(free, narrow)
    assert pulled["probes"][0]["noise_level_ref_db"] == pulled["bounds"]["noise_level_ref_db"][0] > patch.NOISE_OFF_DB


PHANTOM_SILENT = -80.0
PHANTOM_SEEDS = range(5)


def phantom_search(true: dict, target, reading_db, seed: int, profile=None):
    """One search from the free layer a guess gives for a noise reading of `reading_db` dB (None: no layer), through
    search_arguments (start, bounds, probes), against `target`. The profile is that of the noiseless target with the
    reading replaced (a free layer needs: click_db 50 dB down, flatness 0.0: no evidence); the stand-in plays each
    candidate through operator_live. Returns (the level played at the onset, final distance on fresh noise, start's level)."""
    if profile:
        prof = profile
    elif reading_db is None:
        prof = dict(profile_of(dict(true, noise_db=PHANTOM_SILENT), 1), noise_db={"attack": None, "body": None})
    else:
        prof = dict(profile_of(dict(true, noise_db=PHANTOM_SILENT), 1), noise_db={"attack": reading_db, "body": None},
                    click_db=-50.0, flatness={"attack": 0.0, "body": 0.0})
    g = patch.operator_guess(prof)
    args = patch.search_arguments(g)
    assert g["structure"]["noise_on"] is (reading_db is not None or bool(profile)) and ("probes" in args) is g["structure"]["noise_on"]

    def play(p):
        return standin_from_live(patch.operator_live(p, g["structure"]))
    count = [0]

    def render(ps):
        out = []
        for p in ps:
            count[0] += 1
            out.append(standin(play(p), 10 ** 6 + 17 + 1000 * seed + count[0]))
        return out
    res = patch.search(args["start"], args["bounds"], render, target, SR, seed=seed, **({"probes": args["probes"]} if "probes" in args else {}))
    dist = float(np.mean([oneshot.spectral_distance(standin(play(res.best), 7 * 10 ** 6 + k), target, SR) for k in range(8)]))

    def level(p):
        return play(p)["noise_db"] if g["structure"]["noise_on"] else None
    return level(res.best), dist, level(args["start"])


@pytest.mark.parametrize("kind", ["kick", "tom"])
def test_the_search_brings_a_phantom_noise_layer_down_to_where_the_noiseless_start_scores(kind):
    """The target is the tone and its glide with NO noise, so the true level of the layer is "off". The start is a free
    layer at a reading of -15 and of -21 dB (played onsets -6.5 / -12.6 dB kick, -4.2 / -10.3 dB tom), 5 seeds here
    (10 in the report's script) of 8 rounds of 16. Measured over 10 (report, "Review round 2, phantom layers"): the final level is 22 to 43 dB under the
    start in every seed, the median under -37 dB (the played level cannot go under -48 dB: the lower bound of -60 at
    the reference time, plus the 12 dB onset clamp); the median final distance is within 0.003 of the search that has
    no layer at all. Without the probe at the lower bound, the drop was 17 to 46 dB and the median distance 0.007 to
    0.019 over."""
    true = TRUE[kind]
    target = standin(dict(true, noise_db=PHANTOM_SILENT), TARGET_SEED)
    none = np.array([phantom_search(true, target, None, s)[1] for s in PHANTOM_SEEDS])
    for reading_db in (-15.0, -21.0):
        runs = [phantom_search(true, target, reading_db, s) for s in PHANTOM_SEEDS]
        final, dist, start = np.array([r[0] for r in runs]), np.array([r[1] for r in runs]), runs[0][2]
        assert np.all(start - final >= 20.0), (kind, reading_db, start - final)
        assert np.median(final) <= -35.0, (kind, reading_db, np.median(final))
        assert np.median(dist) - np.median(none) <= 0.005, (kind, reading_db, np.median(dist), np.median(none))


def test_the_search_keeps_a_real_band_passed_noise_layer_whose_level_it_may_switch_off():
    """The target is the tom with noise at -10 dB band-passed 1 to 5 kHz (the candidates' noise is white: the stand-in
    has no filter). The guess has a free layer (flatness 0.0), and the probe at off is rendered in round 0. The final
    level stays within 6 dB of the truth in every seed (measured: -9.3 to -13.3, median -12.0): freeing the level does
    not make the search throw a real layer away."""
    true = TRUE["tom"]
    t = np.arange(int(SECONDS * SR)) / SR
    white = np.random.default_rng(TARGET_SEED).standard_normal(len(t)) * math.sqrt(0.5)
    noise = band_limited(white, 1000.0, 5000.0, SR) * 10 ** (true["noise_db"] / 20) * np.exp(-t / true["noise_decay"]) * np.minimum(1.0, t / ATTACK_S)
    target = standin(dict(true, noise_db=PHANTOM_SILENT), TARGET_SEED) + noise
    prof = oneshot.profile_array(target[:, None], SR, single=True)
    assert prof["flatness"]["attack"] < patch.FLAT_ATTACK          # why the layer is free
    g = patch.operator_guess(prof)
    assert g["structure"]["noise_on"] and "noise_level_ref_db" in g["free"]
    levels = np.array([phantom_search(true, target, None, seed, profile=prof)[0] for seed in PHANTOM_SEEDS])
    assert np.all(np.abs(levels - true["noise_db"]) <= 6.0), levels
    assert abs(np.median(levels) - true["noise_db"]) <= 6.0


# ---- the shipped units table (sideman/data/operator_units.json, read from Live 12.4.6 on a scratch Operator)

def test_every_table_of_the_shipped_units_file_is_accepted_and_inf_is_a_float():
    units = patch.operator_units()
    assert len(units) == 132 and set(units) <= set(patch.operator_parameters())
    for name, rows in units.items():
        curve = patch._Curve(name, rows)             # monotone, one unit, the dimension the code expects
        assert all(isinstance(v, float) for row in rows for v in row[:2]), name
        assert len(curve.live) >= 2
    assert units["Volume"][0][:2] == [0.0, -math.inf] and units["Ae Decay"][0][2] == "ms" and units["Ae Decay"][-1][2] == "s"
    assert patch.operator_units() is not units and patch.operator_units() == units          # a new copy each time


def _kick_snare_hat_guesses():
    snare = reading(fall_20_ms=111.0, fall_40_ms=508.0, noise_db={"attack": -6.9, "body": -6.9}, click_db=-7.8,
                    flatness={"attack": 0.075, "body": 0.069}, pitch_body_hz=130.0)
    hat = reading(pitched=False, pitch_body_hz=None, fall_20_ms=60.0, fall_40_ms=120.0, attack_ms=0.4, peak_db=-3.0,
                  bands_db={"sub": -60.0, "low": -50.0, "lowmid": -40.0, "mid": -20.0, "highmid": -6.0, "high": -3.0})
    return {"kick": profile_of(TRUE["kick"], 1), "snare": snare, "hat": hat}


@pytest.mark.parametrize("kind", ["kick", "snare", "hat"])
def test_a_guess_converts_through_the_shipped_table_and_back_within_interpolation_error(kind):
    units = patch.operator_units()
    g = patch.operator_guess(_kick_snare_hat_guesses()[kind])
    phys = patch.operator_live(g["patch"], g["structure"])["params"]
    live = patch.to_live(phys, units)
    back = patch.from_live(live, units)
    assert set(back) == set(phys)
    for name, v in phys.items():
        if name in patch.RAW_NAMES:
            assert back[name] == v, name
        elif name in patch.QUANTIZED:
            assert abs(back[name] - v) <= 0.5 + 1e-9, name       # whole numbers: Coarse 1, Fine to the nearest unit
        elif v == -math.inf:
            assert back[name] == -math.inf, name
        else:
            assert back[name] == pytest.approx(v, rel=1e-6, abs=1e-6), (name, v, back[name])
        if name not in patch.RAW_NAMES and name not in patch.QUANTIZED and v != -math.inf:
            assert 0.0 <= live[name] <= 1.0 or name in ("Time", "Time < Key", "Pe Init", "Pe Peak", "Pe Sustain", "Pe Amount",
                                                         "Pe Amt A", "Pe Amt B"), (name, live[name])
    args = patch.search_arguments(g, units)             # the table feeds the search's bound check directly
    assert args["start"] == g["patch"] and set(args["bounds"]) == set(g["free"])
    for name, (lo, hi, _) in args["bounds"].items():
        assert g["free"][name][0] <= lo <= g["patch"][name] <= hi <= g["free"][name][1]


def _text_display(text: str) -> float:
    """Live's display text -> ms, Hz, dB (read here, apart from the table, to check it)."""
    num, unit = text.split()
    return float(num) * {"ms": 1.0, "s": 1000.0, "Hz": 1.0, "kHz": 1000.0, "dB": 1.0}[unit]


@pytest.mark.parametrize("name, value, written", [
    ("Ae Decay", 300.0, 0.5184374519784173), ("Ae Decay", 1000.0, 0.6278481862400666), ("Pe Decay", 400.0, 0.5446614978544123),
    ("Filter Freq", 2000.0, 0.6538756597411892), ("Volume", -12.0, 0.5416666666666666)])
def test_physical_values_checked_on_the_device_land_on_the_display_live_gave(name, value, written):
    """`written` is the Live value the takes of 2026-11-12 wrote for this physical value and the device played (the
    300 ms and 1000 ms decays fall 20 dB in 68.3 and 226.6 ms). The shipped table reproduces it, and Live's own
    display text at the two table points either side of it, read apart from the table, brackets the value: its
    log (dB: linear) interpolation at `written` is within 1 % of the physical value."""
    units = patch.operator_units()
    assert patch.to_live({name: value}, units)[name] == pytest.approx(written, rel=1e-6)
    assert patch.from_live({name: written}, units)[name] == pytest.approx(value, rel=1e-6)
    raw = json.loads((pathlib.Path(patch.__file__).parent / "data" / "operator_units.json").read_text())["parameters"][name]["display"]
    pts = [(live, _text_display(text)) for live, text in raw if "inf" not in text]
    (l0, d0), (l1, d1) = [(a, b) for a, b in zip(pts, pts[1:]) if a[0] <= written <= b[0]][0]
    f = (written - l0) / (l1 - l0)
    shown = d0 + f * (d1 - d0) if name == "Volume" else math.exp(math.log(d0) + f * (math.log(d1) - math.log(d0)))
    assert shown == pytest.approx(value, rel=0.01)
