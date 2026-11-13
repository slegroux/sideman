"""sideman.patch.search against a numpy stand-in for the synth, with parameters known by construction.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_patch.py -q
Skipped (not failed) when numpy or soundfile is missing.

The stand-in (`standin`) is a test double for Ableton Live's Operator and lives here, not in the package: the
search must not know what renders, and the real renderer will be Live. It is a sine carrier with an exponential
pitch envelope, an exponential amplitude envelope, a noise layer with its own level and decay, and a short
attack. Its noise is drawn afresh on every render (like Operator's), so two renders of one patch differ.
"""
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


@pytest.mark.parametrize("kind", ["kick", "tom", "hat"])
@pytest.mark.parametrize("fresh", [True, False], ids=["fresh-noise", "same-noise"])
def test_search_recovers_known_parameters(kind, fresh):
    """The phase-21 thresholds (decay within 15 %, noise within 3 dB) from a start 30 % off, in 8 rounds of 16.
    Held to what the distance can tell apart, which the measured numbers set (module docstring of sideman.patch):
    - the tone's decay: every seed (not for the hat-like target, whose tone is masked by noise as loud as it:
      test_distance_is_flat_in_a_masked_tone);
    - the noise layer's decay: every seed for the hat-like target, half of the seeds for the others, where the
      layer is 10 to 12 dB under the tone and 15 % moves the distance by about one repeat spread;
    - the noise layer's level: half of the seeds, and the level it has at the time the tone has fallen 8.7 dB
      (decay and level of the layer lie on a ridge, this does not) within 3 dB at the median seed."""
    runs = [run(kind, seed=seed, fresh=fresh) for seed in SEEDS]
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
