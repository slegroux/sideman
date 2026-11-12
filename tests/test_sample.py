"""sideman.oneshot on synthetic hits whose answers are known without the code.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_sample.py -q
Skipped (not failed) when numpy or soundfile is missing.
"""
import math
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import oneshot  # noqa: E402

SR = 48000


def hit(freq, tau, seconds=0.6, partials=None, phase=0.0, attack=0.0, sr=SR):
    """A decaying tone, mono [n, 1]. `freq` is Hz or a function of time (s) -> Hz;
    `partials` maps harmonic number -> amplitude; `tau` is the amplitude time
    constant in s; `attack` a linear ramp in s."""
    t = np.arange(int(seconds * sr)) / sr
    f = freq(t) if callable(freq) else np.full(len(t), float(freq))
    ph = 2 * np.pi * np.concatenate([[0.0], np.cumsum(f[:-1]) / sr])
    x = sum(a * np.sin(k * ph + phase) for k, a in (partials or {1: 1.0}).items())
    x = x * np.exp(-t / tau)
    if attack:
        x = x * np.minimum(1.0, t / attack)
    return x[:, None]


def st(a, b):
    return 12 * math.log2(a / b)


# 200 Hz for its first 10 ms, then a raised-cosine glide down to 100 Hz by 40 ms
def sweep(t):
    return np.where(t < 0.010, 200.0,
                    np.where(t < 0.040, 150.0 + 50.0 * np.cos(np.pi * (t - 0.010) / 0.030), 100.0))


# ---------------------------------------------------------------- envelope

def test_envelope_falls_of_a_50ms_time_constant():
    # amplitude e^(-t/tau): it is X dB down after tau * ln(10^(X/20))
    r = oneshot.profile_array(hit(100, 0.050), SR)
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)          # 115.1
    assert r["fall_10_ms"] == pytest.approx(50 * math.log(10 ** 0.5), abs=6)   # 57.6
    assert r["fall_40_ms"] == pytest.approx(50 * math.log(100), abs=6)         # 230.3
    assert r["duration_ms"] == pytest.approx(50 * math.log(1000), abs=10)      # 345.4


def test_envelope_attack_and_peak():
    # a cosine starting at its crest: the peak sample is the amplitude at t = 0
    r = oneshot.profile_array(0.5 * hit(100, 0.1, phase=math.pi / 2), SR)
    assert r["peak_db"] == pytest.approx(20 * math.log10(0.5), abs=0.01)
    assert r["onset_ms"] == 0.0
    assert r["attack_ms"] <= 5.0                       # within half a period of 100 Hz
    # a 10 ms linear attack on a slow decay peaks at 10 ms; 200 Hz -> within 2.5 ms
    r = oneshot.profile_array(hit(200, 0.15, attack=0.010), SR)
    assert r["attack_ms"] == pytest.approx(10.0, abs=2.5)


def test_envelope_and_pitch_ignore_leading_silence():
    x = hit(sweep, 0.15)
    a = oneshot.profile_array(x, SR)
    b = oneshot.profile_array(np.concatenate([np.zeros((int(0.030 * SR), 1)), x]), SR)
    assert b["onset_ms"] == 30.0 and a["onset_ms"] == 0.0
    a.pop("onset_ms"), b.pop("onset_ms")
    assert a == b


def test_envelope_and_pitch_ignore_level():
    x = hit(sweep, 0.15, partials={1: 1.0, 2: 0.3, 3: 0.2})
    a, b = oneshot.profile_array(x, SR), oneshot.profile_array(x * 10 ** (-6 / 20), SR)
    assert b["peak_db"] == pytest.approx(a["peak_db"] - 6.0, abs=0.01)
    for k in ("attack_ms", "fall_10_ms", "fall_20_ms", "fall_40_ms", "pitch_start_hz",
              "pitch_body_hz", "pitch_drop_st", "pitch_settle_ms", "click_db"):
        assert b[k] == pytest.approx(a[k], abs=0.15), k
    for k in a["harmonics_db"]:
        assert b["harmonics_db"][k] == pytest.approx(a["harmonics_db"][k], abs=0.15)


def test_envelope_and_pitch_ignore_sample_rate():
    # the same hit synthesised at 44.1 and 48 kHz
    a = oneshot.profile_array(hit(sweep, 0.05, partials={1: 1.0, 3: 0.3}, sr=48000), 48000)
    b = oneshot.profile_array(hit(sweep, 0.05, partials={1: 1.0, 3: 0.3}, sr=44100), 44100)
    assert b["fall_20_ms"] == pytest.approx(a["fall_20_ms"], abs=2)
    assert b["pitch_body_hz"] == pytest.approx(a["pitch_body_hz"], rel=0.002)
    assert b["pitch_start_hz"] == pytest.approx(a["pitch_start_hz"], rel=0.01)
    assert b["harmonics_db"][3] == pytest.approx(a["harmonics_db"][3], abs=0.5)


def test_envelope_cut_short_is_none_with_a_reason():
    # 100 ms of a 50 ms time constant only falls 17.4 dB before the file ends
    r = oneshot.profile_array(hit(100, 0.050, seconds=0.1), SR)
    assert r["fall_10_ms"] is not None
    assert r["fall_20_ms"] is None and r["fall_40_ms"] is None
    assert any("20 dB" in n for n in r["notes"]) and any("40 dB" in n for n in r["notes"])


def test_envelope_silent_or_too_short_is_refused():
    with pytest.raises(ValueError, match="silent"):
        oneshot.profile_array(np.zeros((SR, 1)), SR)
    with pytest.raises(ValueError, match="shorter"):
        oneshot.profile_array(hit(100, 0.05, seconds=0.002), SR)


# ---------------------------------------------------------------- pitch

def test_pitch_sweep_from_200_to_100_hz():
    r = oneshot.profile_array(hit(sweep, 0.15), SR)
    assert r["pitched"] is True
    assert r["pitch_start_hz"] == pytest.approx(200, abs=6)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=2)
    assert r["pitch_drop_st"] == pytest.approx(12, abs=0.5)
    # the glide is within 0.5 st of 100 Hz once 150 + 50 cos(pi (t - 10 ms) / 30 ms)
    # <= 100 * 2^(0.5/12); a 3-period (30 ms) frame centred on t sees up to 15 ms later
    settle = 10 + 30 * math.acos((100 * 2 ** (0.5 / 12) - 150) / 50) / math.pi    # 36.7 ms
    assert settle <= r["pitch_settle_ms"] <= settle + 15
    track = r["pitch_track_hz"]
    assert track[0] == pytest.approx(200, abs=6)
    assert track[20] == pytest.approx(150 + 50 * math.cos(math.pi / 3), abs=6)     # 175 Hz at 20 ms
    assert track[80] == pytest.approx(100, abs=1) and track[150] == pytest.approx(100, abs=1)


def test_pitch_fast_glide_reads_the_pitch_where_it_was_measured():
    # 100 + 100 e^(-t / 15 ms): its first cycle is shorter than any pitch frame, so the
    # start is the pitch at pitch_start_at_ms, about 1.5 periods in
    def f(t):
        return 100 + 100 * np.exp(-np.asarray(t) / 0.015)
    r = oneshot.profile_array(hit(f, 0.15), SR)
    assert r["pitch_start_at_ms"] <= 1.6 * 1000 / r["pitch_start_hz"]
    assert r["pitch_start_hz"] == pytest.approx(float(f(r["pitch_start_at_ms"] / 1000)), rel=0.03)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=2)
    # within 0.5 st of 100 Hz once 100 e^(-t/15) <= 100 (2^(0.5/12) - 1)
    assert r["pitch_settle_ms"] == pytest.approx(-15 * math.log(2 ** (0.5 / 12) - 1), abs=5)   # 52.9


@pytest.mark.parametrize("f0", [55.0, 100.0, 300.0])
def test_pitch_is_the_fundamental_not_the_loudest_partial(f0):
    # 3rd harmonic at twice the fundamental's amplitude: 6 dB above it
    r = oneshot.profile_array(hit(f0, 0.2, partials={1: 1.0, 3: 2.0}), SR)
    assert r["pitched"] is True
    assert r["pitch_body_hz"] == pytest.approx(f0, rel=0.02)
    assert r["pitch_start_hz"] == pytest.approx(f0, rel=0.02)
    assert all(v == pytest.approx(f0, rel=0.02) for v in r["pitch_track_hz"].values() if v is not None)
    assert r["harmonics_db"][3] == pytest.approx(20 * math.log10(2), abs=1)


def test_pitch_steady_tone_has_no_drop():
    r = oneshot.profile_array(hit(100, 0.1), SR)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=0.5)
    assert r["pitch_drop_st"] == pytest.approx(0, abs=0.1)
    assert r["pitch_settle_ms"] == 0.0


# ---------------------------------------------------------------- tone

def test_harmonic_levels_of_odd_partials_at_1_over_n():
    # a square-like tone: odd harmonics at 1/n, so the nth is 20 log10(1/n) dB
    r = oneshot.profile_array(hit(100, 0.15, partials={n: 1.0 / n for n in range(1, 16, 2)}), SR)
    h = r["harmonics_db"]
    assert h[3] == pytest.approx(20 * math.log10(1 / 3), abs=1)      # -9.5
    assert h[5] == pytest.approx(20 * math.log10(1 / 5), abs=1)      # -14.0
    assert h[7] == pytest.approx(20 * math.log10(1 / 7), abs=1)      # -16.9
    assert h[2] < -40 and h[4] < -40
    assert r["partials"] == []


def test_harmonic_series_leaves_out_an_inharmonic_partial():
    t = np.arange(int(0.6 * SR)) / SR
    x = (np.sin(2 * np.pi * 100 * t) + 10 ** (-10 / 20) * np.sin(2 * np.pi * 237 * t)) * np.exp(-t / 0.15)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.01)
    assert len(r["partials"]) == 1
    assert r["partials"][0]["hz"] == pytest.approx(237, abs=1)
    assert r["partials"][0]["db"] == pytest.approx(-10, abs=1)
    assert r["harmonics_db"][2] < -40


def test_noise_share_of_a_sine_with_white_noise_20db_down():
    # both share one envelope, so the noise is 1/101 of the power: -20.04 dB
    t = np.arange(int(0.6 * SR)) / SR
    noise = np.random.default_rng(3).standard_normal(len(t)) * math.sqrt(0.5 / 100)
    x = (np.sin(2 * np.pi * 150 * t) + noise) * np.exp(-t / 0.1)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is True
    assert r["noise_db"]["body"] == pytest.approx(-20.04, abs=1.5)
    assert r["noise_db"]["attack"] == pytest.approx(-20.04, abs=1.5)


def test_noise_alone_is_unpitched_with_no_harmonic_values():
    t = np.arange(int(0.6 * SR)) / SR
    x = np.random.default_rng(4).standard_normal(len(t)) * np.exp(-t / 0.1) * 0.3
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitched"] is False
    assert r["harmonics_db"] is None
    for k in ("pitch_start_hz", "pitch_body_hz", "pitch_drop_st", "pitch_settle_ms", "pitch_track_hz"):
        assert r[k] is None, k
    assert r["noise_db"]["body"] > -1.0
    # a white-noise periodogram is exponentially distributed: geometric / arithmetic
    # mean = e^(-Euler's gamma) = 0.561
    assert r["flatness"]["body"] == pytest.approx(math.exp(-0.5772156649), abs=0.05)
    assert r["notes"]


def test_noise_flatness_of_a_pure_tone_is_near_zero():
    r = oneshot.profile_array(hit(200, 0.1), SR)
    assert r["flatness"]["attack"] < 0.01 and r["flatness"]["body"] < 0.01
    assert r["noise_db"]["body"] < -25


def test_click_level_of_a_01ms_and_a_2ms_attack():
    # a tone starting at its crest: 0.1 ms to full level is a step, 2 ms is a ramp
    sharp = oneshot.profile_array(hit(100, 0.15, phase=math.pi / 2, attack=0.0001), SR)
    soft = oneshot.profile_array(hit(100, 0.15, phase=math.pi / 2, attack=0.002), SR)
    assert sharp["click_db"] >= soft["click_db"] + 10


def test_bands_of_a_1khz_hit_are_all_mid():
    r = oneshot.profile_array(hit(1000, 0.1), SR)
    assert r["bands_db"]["mid"] == pytest.approx(0.0, abs=0.1)
    assert all(v < -30 for k, v in r["bands_db"].items() if k != "mid")


def test_width_of_mono_identical_and_antiphase_channels():
    x = hit(200, 0.1)
    assert oneshot.profile_array(x, SR)["width"] == 0.0
    assert oneshot.profile_array(np.hstack([x, x]), SR)["width"] == pytest.approx(0.0, abs=1e-3)
    assert oneshot.profile_array(np.hstack([x, -x]), SR)["width"] == pytest.approx(1.0, abs=1e-3)


def test_profile_reads_a_file(tmp_path):
    p = tmp_path / "tom.wav"
    sf.write(str(p), hit(100, 0.05), SR, subtype="FLOAT")
    r = oneshot.profile(p)
    assert r["file"] == str(p) and r["sample_rate"] == SR and r["channels"] == 1
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)
