"""sideman.patch.search against a numpy stand-in for the synth, with parameters known by construction.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_patch.py -q
Skipped (not failed) when numpy or soundfile is missing.

The stand-in (`standin`) is a test double for Ableton Live's Operator and lives here, not in the package: the
search must not know what renders, and the real renderer will be Live. It is a sine carrier with an exponential
pitch envelope, an exponential amplitude envelope, a noise layer with its own level and decay, and a short
attack. Its noise is drawn afresh on every render (like Operator's), so two renders of one patch differ.
"""
import functools
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


def standin(p: dict, seed: int, sr: int = SR, seconds: float = SECONDS) -> np.ndarray:
    """One hit, mono [n], from the patch `p`; `seed` is the noise realisation."""
    t = np.arange(int(seconds * sr)) / sr
    f = p["pitch_hz"] * 2.0 ** (p["pitch_st"] * np.exp(-t / p["pitch_tau"]) / 12.0)
    ph = 2 * np.pi * np.cumsum(f) / sr
    attack = np.minimum(1.0, t / ATTACK_S)
    tone = np.sin(ph) * np.exp(-t / p["decay"]) * attack
    noise = np.random.default_rng(seed).standard_normal(len(t)) * math.sqrt(0.5)       # RMS of a unit sine
    noise = noise * 10.0 ** (p["noise_db"] / 20.0) * np.exp(-t / p["noise_decay"]) * attack
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
