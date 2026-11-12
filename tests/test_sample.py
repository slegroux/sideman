"""sideman.oneshot on synthetic hits whose answers are known without the code.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_sample.py -q
Skipped (not failed) when numpy or soundfile is missing.
"""
import json
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


# 200 Hz held for its first 10 ms, then a raised-cosine glide down to 100 Hz by
# 40 ms: the shared test hit for the invariance and compare tests. The plan's own
# sweep case (no hold) is test_pitch_sweep_from_200_to_100_hz.
def sweep(t):
    return np.where(t < 0.010, 200.0,
                    np.where(t < 0.040, 150.0 + 50.0 * np.cos(np.pi * (t - 0.010) / 0.030), 100.0))


def glide(t):
    """200 Hz falling linearly to 100 Hz over the first 30 ms, from the onset on."""
    return np.where(t < 0.030, 200.0 - 100.0 * t / 0.030, 100.0)


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
    for k in ("onset_ms", "onsets_ms"):
        a.pop(k), b.pop(k)
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
    for k in ("attack_ms", "fall_10_ms", "fall_20_ms", "fall_40_ms", "duration_ms", "pitch_settle_ms"):
        assert b[k] == pytest.approx(a[k], abs=2), k
    assert b["pitch_body_hz"] == pytest.approx(a["pitch_body_hz"], rel=0.002)
    assert b["pitch_start_hz"] == pytest.approx(a["pitch_start_hz"], rel=0.01)
    assert b["pitch_drop_st"] == pytest.approx(a["pitch_drop_st"], abs=0.1)
    assert b["harmonics_db"][3] == pytest.approx(a["harmonics_db"][3], abs=0.5)
    assert a["harmonics_db"][2] < -40 and b["harmonics_db"][2] < -40      # none there: both read the floor
    for part in ("attack", "body"):
        assert b["noise_db"][part] == pytest.approx(a["noise_db"][part], abs=1.5), part
    assert b["click_db"] == pytest.approx(a["click_db"], abs=1.5)
    for band in ("sub", "bass", "low_mid", "mid"):
        assert b["bands_db"][band] == pytest.approx(a["bands_db"][band], abs=0.5), band
    assert b["pitch_track_hz"] == pytest.approx(a["pitch_track_hz"], rel=0.02)


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
    r = oneshot.profile_array(hit(glide, 0.15), SR)
    assert r["pitched"] is True
    assert r["pitch_start_hz"] == pytest.approx(200, abs=6)
    assert r["pitch_body_hz"] == pytest.approx(100, abs=2)
    assert r["pitch_drop_st"] == pytest.approx(12, abs=0.5)
    # within 0.5 st of 100 Hz once 200 - 100 t / 30 ms <= 100 * 2^(0.5/12); a 3-period
    # (30 ms) frame centred on t sees up to 15 ms later
    settle = 30 * (2 - 2 ** (0.5 / 12))                                    # 29.1 ms
    assert settle <= r["pitch_settle_ms"] <= settle + 15
    track = r["pitch_track_hz"]
    for ms in (0, 5, 10, 20):                                               # 200, 183.3, 166.7, 133.3 Hz
        assert track[ms] == pytest.approx(200 - 100 * ms / 30, abs=6), ms
    assert track[80] == pytest.approx(100, abs=1) and track[150] == pytest.approx(100, abs=1)


def test_pitch_kick_glide_from_160_to_45_hz():
    # 45 + 115 e^(-t / 12 ms): 160 Hz at the onset, 120.8 at 5 ms, 95.0 at 10 ms,
    # a drop of 12 log2(160 / 45) = 21.96 st to its 45 Hz body
    def f(t):
        return 45 + 115 * np.exp(-np.asarray(t) / 0.012)
    r = oneshot.profile_array(hit(f, 0.12), SR)
    assert r["pitch_start_hz"] == pytest.approx(160, rel=0.10)
    assert r["pitch_drop_st"] == pytest.approx(12 * math.log2(160 / 45), abs=2)
    assert r["pitch_track_hz"][5] == pytest.approx(float(f(0.005)), rel=0.05)
    assert r["pitch_track_hz"][10] == pytest.approx(float(f(0.010)), rel=0.05)


def test_pitch_start_of_a_steady_sawtooth():
    # harmonics at 1/k: zero crossings still come twice a period, and none of them
    # read anything but 80 Hz
    r = oneshot.profile_array(hit(80, 0.2, partials={k: 1.0 / k for k in range(1, 60)}), SR)
    assert r["pitch_start_hz"] == pytest.approx(80, rel=0.02)
    assert all(v == pytest.approx(80, rel=0.02) for v in r["pitch_track_hz"].values() if v is not None)


def test_pitch_track_before_the_first_frame_is_none_not_a_copy():
    # sin + 2 sin(3x) crosses zero six times a period, so the crossings give no pitch:
    # the points before the first frame (15 ms) are None with a note, not its value
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 1.0, 3: 2.0}), SR)
    assert r["pitch_start_at_ms"] == pytest.approx(15, abs=1)
    assert [r["pitch_track_hz"][ms] for ms in (0, 5, 10)] == [None, None, None]
    assert any("no value before 15 ms" in n for n in r["notes"])


def test_pitch_that_never_holds_still_has_no_settle_time():
    # 300 e^(-t / 100 ms) falls 0.5 st in 2.9 ms: it never stays within 0.5 st of its
    # body pitch for 10 ms
    r = oneshot.profile_array(hit(lambda t: 300 * np.exp(-t / 0.1), 0.15), SR)
    assert r["pitch_settle_ms"] is None
    assert any("never settles" in n or "not followed" in n for n in r["notes"])


def test_pitch_fast_glide_reads_the_pitch_where_it_was_measured():
    # 100 + 100 e^(-t / 15 ms): its first cycle is shorter than any pitch frame; the
    # zero crossings read it, and the start is the pitch at pitch_start_at_ms
    def f(t):
        return 100 + 100 * np.exp(-np.asarray(t) / 0.015)
    r = oneshot.profile_array(hit(f, 0.15), SR)
    assert r["pitch_start_at_ms"] <= 0.5 * 1000 / r["pitch_start_hz"]      # within the first half-period
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
    assert any("no tonal peak in the body" in n for n in r["notes"])


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


# ---------------------------------------------------------------- compare
# The base hit uses the sweep above (200 Hz held 10 ms, then a glide to 100 Hz),
# so pitch drop and settle are compared too.

BASE = {1: 1.0, 2: 0.3, 3: 0.2}


def up2(t):
    return sweep(t) * 2 ** (2 / 12)


def lowpassed_noise(n, hz, seed):
    spec = np.fft.rfft(np.random.default_rng(seed).standard_normal(n))
    spec[np.fft.rfftfreq(n, 1 / SR) >= hz] = 0
    v = np.fft.irfft(spec, n)
    return v / np.sqrt((v ** 2).mean())


def with_noise(x, noise):
    # the same envelope as the tone; the tone's power is 0.5 * (1 + 0.09 + 0.04),
    # so noise at 1/100 of it sits 20 dB down: a noise share of 1/101 = -20.04 dB
    t = np.arange(len(x)) / SR
    return x + (noise * np.exp(-t / 0.1) * math.sqrt(0.5 * 1.13 / 100))[:, None]


def test_compare_level_silence_and_sample_rate_give_no_reading():
    x = hit(sweep, 0.1, partials=BASE)
    ref = oneshot.profile_array(x, SR)
    quieter = oneshot.compare(oneshot.profile_array(x * 10 ** (-6 / 20), SR), ref)
    assert quieter["readings"] == []
    assert quieter["diff"]["peak_db"] == pytest.approx(-6.0, abs=0.01)
    late = oneshot.profile_array(np.concatenate([np.zeros((int(0.030 * SR), 1)), x]), SR)
    assert oneshot.compare(late, ref)["readings"] == []
    # the same hit synthesised at 44.1 kHz, not resampled
    at44 = oneshot.profile_array(hit(sweep, 0.1, partials=BASE, sr=44100), 44100)
    assert oneshot.compare(at44, ref)["readings"] == []


def test_compare_two_semitones_up_reads_only_the_pitch():
    a = oneshot.profile_array(hit(sweep, 0.1, partials=BASE), SR)
    b = oneshot.profile_array(hit(up2, 0.1, partials=BASE), SR)
    c = oneshot.compare(a, b)
    assert c["diff"]["pitch_body_st"] == pytest.approx(-2.0, abs=0.1)
    assert len(c["readings"]) == 1 and c["readings"][0].startswith("body pitch 2.0 st below the reference")
    # a profile read back from JSON (string keys) compares the same
    assert oneshot.compare(json.loads(json.dumps(a)), json.loads(json.dumps(b)))["readings"] == c["readings"]


def test_compare_half_the_decay_reads_only_decay():
    a = oneshot.profile_array(hit(sweep, 0.05, partials=BASE), SR)
    b = oneshot.profile_array(hit(sweep, 0.1, partials=BASE), SR)
    c = oneshot.compare(a, b)
    assert c["readings"] and all(r.startswith(("falls", "lasts")) for r in c["readings"]), c["readings"]
    assert any(r.startswith("falls 20 dB") and "shorter" in r for r in c["readings"])
    # an exponential's fall times scale with its time constant: 50 / 100 ms
    for k in ("fall_10_ms", "fall_20_ms", "fall_40_ms"):
        assert c["diff"][k]["ratio"] == pytest.approx(0.5, abs=0.05), k


def test_compare_noise_added_reads_only_the_noise():
    # noise kept under 2 kHz: white noise also brightens the attack, which
    # click_db rightly reads (next test), so this isolates the noise share
    x = hit(sweep, 0.1, partials=BASE)
    noisy = oneshot.profile_array(with_noise(x, lowpassed_noise(len(x), 2000, 7)), SR)
    c = oneshot.compare(noisy, oneshot.profile_array(x, SR))
    assert noisy["noise_db"]["body"] == pytest.approx(-20.04, abs=1.5)
    assert len(c["readings"]) == 1, c["readings"]
    assert c["readings"][0].startswith("noise") and "more than the reference in the body" in c["readings"][0]


def test_compare_white_noise_reads_noise_and_a_brighter_attack():
    x = hit(sweep, 0.1, partials=BASE)
    white = np.random.default_rng(8).standard_normal(len(x))
    c = oneshot.compare(oneshot.profile_array(with_noise(x, white), SR), oneshot.profile_array(x, SR))
    kinds = sorted(r.split()[0] for r in c["readings"])
    assert kinds == ["click", "noise"], c["readings"]       # no harmonic reading from the noise floor


def test_compare_pitched_against_unpitched_is_a_reading():
    t = np.arange(int(0.6 * SR)) / SR
    noise = (np.random.default_rng(9).standard_normal(len(t)) * np.exp(-t / 0.1) * 0.3)[:, None]
    c = oneshot.compare(oneshot.profile_array(hit(100, 0.1), SR), oneshot.profile_array(noise, SR))
    assert c["readings"][0].startswith("pitched (100 Hz), the reference is not")
    assert c["diff"]["pitch_body_st"] is None and c["diff"]["harmonics_db"] is None


def test_compare_floors_none_and_order():
    a = {"pitched": True, "pitch_body_hz": 100.0 * 2 ** (0.6 / 12), "attack_ms": 0.4, "fall_20_ms": 100.0,
         "fall_40_ms": None, "pitch_settle_ms": 0.0, "harmonics_db": {2: -55.0, 3: -10.0},
         "harmonics_floor_db": {2: -90.0, 3: -90.0}, "noise_db": {"attack": -30.0, "body": None},
         "click_db": -45.0, "width": 0.30}
    b = {"pitched": True, "pitch_body_hz": 100.0, "attack_ms": 0.3, "fall_20_ms": 200.0,
         "fall_40_ms": 300.0, "pitch_settle_ms": 1.5, "harmonics_db": {2: -60.0, 3: -10.0},
         "harmonics_floor_db": {2: -90.0, 3: -90.0}, "noise_db": {"attack": -40.0, "body": -20.0},
         "click_db": -60.0, "width": 0.15}
    c = oneshot.compare(a, b)
    # body pitch 0.6 st = 2x its threshold, fall-20 50 % shorter = 3.3x, width 0.15 = 1.5x;
    # attack 0.1 ms and settle 1.5 ms are under the 2 ms floor, the 2nd harmonic is under
    # -50 dB, attack noise under -25 dB and click under -40 dB on both sides
    assert [r.split()[0] for r in c["readings"]] == ["falls", "body", "width"], c["readings"]
    assert c["diff"]["fall_40_ms"] is None and c["diff"]["noise_db"]["body"] is None
    assert c["diff"]["attack_ms"] == {"ms": pytest.approx(0.1), "ratio": pytest.approx(4 / 3, abs=1e-3)}


# ---------------------------------------------------------------- hits

def test_hits_four_at_500ms_and_the_third_alone():
    one = hit(100, 0.050, seconds=0.5, partials={n: 1.0 / n for n in range(1, 16, 2)})
    take = np.concatenate([one] * 4)
    r = oneshot.profile_array(take, SR, hit=2)
    assert r["hits"] == 4 and r["hit"] == 2 and r["truncated"] is False
    assert r["onsets_ms"] == pytest.approx([0, 500, 1000, 1500], abs=0.5)
    assert r["onset_ms"] == pytest.approx(1000, abs=0.5)
    # the single hit's numbers, as items 1 and 2 derive them
    assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.02)
    assert r["harmonics_db"][3] == pytest.approx(20 * math.log10(1 / 3), abs=1)
    assert r["harmonics_db"][5] == pytest.approx(20 * math.log10(1 / 5), abs=1)
    assert oneshot.profile_array(take, SR)["hit"] == 0


def test_hits_file_ending_60ms_into_a_300ms_decay_is_truncated():
    # fall-40 of e^(-t/tau) is tau ln(100): 300 ms for tau = 65.1 ms
    r = oneshot.profile_array(hit(100, 0.300 / math.log(100), seconds=0.060), SR)
    assert r["hits"] == 1 and r["truncated"] is True and r["fall_40_ms"] is None
    assert any("file ends before the hit falls 40 dB" in n for n in r["notes"])


def test_hits_next_hit_cuts_the_one_before():
    # two hits 200 ms apart, each falling 40 dB only after 100 ln(100) = 460 ms
    take = np.concatenate([hit(100, 0.1, seconds=0.2), hit(100, 0.1, seconds=0.8)])
    first = oneshot.profile_array(take, SR, hit=0)
    assert first["hits"] == 2 and first["truncated"] is True and first["fall_40_ms"] is None
    assert any("next hit starts before the hit falls 40 dB" in n for n in first["notes"])
    assert first["fall_10_ms"] == pytest.approx(100 * math.log(10 ** 0.5), abs=6)    # 115 ms, before the cut
    default = oneshot.profile_array(take, SR)
    assert default["hit"] == 1 and default["truncated"] is False


def test_hits_fast_roll_is_split_at_120ms():
    # each hit is 21 dB down (8.686 * 120 / 50) when the next starts
    take = np.concatenate([hit(100, 0.05, seconds=0.12)] * 5 + [hit(100, 0.05, seconds=0.5)])
    r = oneshot.profile_array(take, SR)
    assert r["hits"] == 6
    assert r["onsets_ms"] == pytest.approx([0, 120, 240, 360, 480, 600], abs=1)


@pytest.mark.parametrize("kind", ["tremolo 6 Hz", "tremolo 20 Hz", "beating 4 Hz", "beating 15 Hz"])
def test_hits_tremolo_or_beating_is_not_split(kind):
    t = np.arange(int(1.0 * SR)) / SR
    rate = float(kind.split()[1])
    if kind.startswith("tremolo"):         # 99 % depth: 40 dB troughs
        x = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.3) * (1 - 0.99 * (0.5 - 0.5 * np.cos(2 * np.pi * rate * t)))
    else:                                  # two equal modes: true nulls `rate` times a second
        x = (np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * (100 + rate) * t)) * np.exp(-t / 0.4)
    assert oneshot.profile_array(x[:, None], SR)["hits"] == 1


def test_hits_out_of_range_is_refused(tmp_path):
    take = np.concatenate([hit(100, 0.05, seconds=0.5)] * 2)
    with pytest.raises(ValueError, match="hit 2 is out of range: the take has 2 hits"):
        oneshot.profile_array(take, SR, hit=2)
    p = tmp_path / "take.wav"
    sf.write(str(p), take, SR, subtype="FLOAT")
    import mcp_server.server as S
    with pytest.raises(S.LiveError, match="out of range"):
        S.sample_analyze(str(p), hit=5)
    assert json.loads(S.sample_analyze(str(p), hit=1))["onset_ms"] == pytest.approx(500, abs=0.5)
    c = json.loads(S.sample_compare(str(p), str(p), hit_a=0, hit_b=1))
    assert c["readings"] == []


# ---------------------------------------------------------------- review fixes

def test_envelope_dc_offset_does_not_change_the_falls():
    # a DC offset is no part of the decay: + 0.05 (-26 dB) must read as the plain hit
    plain = oneshot.profile_array(hit(100, 0.050), SR)
    for dc in (0.05, -0.2, 0.005):
        r = oneshot.profile_array(hit(100, 0.050) + dc, SR)
        assert r["fall_20_ms"] == pytest.approx(50 * math.log(10), abs=6), dc
        assert r["fall_40_ms"] == pytest.approx(50 * math.log(100), abs=6), dc
        assert r["bands_db"] == pytest.approx(plain["bands_db"], abs=0.2), dc     # DC adds no sub


def test_envelope_onset_ignores_a_lone_spike_before_the_hit():
    # a single sample at -38 dB, 8 ms before a hit that starts at 10 ms: its 1 ms frame
    # is 38 + 10 log10(48) = 55 dB down, so the onset is the hit's
    x = np.concatenate([np.zeros((480, 1)), hit(100, 0.050)])
    x[96] = 10 ** (-38 / 20)
    assert oneshot.profile_array(x, SR)["onset_ms"] == pytest.approx(10.0, abs=0.05)


def test_envelope_nan_or_inf_is_refused():
    x = hit(100, 0.05, seconds=1.0)
    for bad in (np.nan, np.inf):
        y = x.copy()
        y[100] = bad
        with pytest.raises(ValueError, match="NaN or infinite"):
            oneshot.profile_array(y, SR)


def test_envelope_5ms_file_is_none_with_reasons_not_numbers():
    r = oneshot.profile_array(hit(200, 0.05, seconds=0.005), SR)
    assert r["fall_20_ms"] is None and r["fall_40_ms"] is None and r["truncated"] is True
    assert r["pitch_body_hz"] is None and r["harmonics_db"] is None
    assert any("too short" in n for n in r["notes"])


def test_envelope_attack_ignores_start_phase_and_clipping():
    # both start phases of an instant 40 Hz hit, and a hit clipped flat at full scale,
    # peak at the onset; a 10 ms linear ramp first comes within 1 dB of its top at
    # 10 * 10^(-1/20) = 8.9 ms
    for x in (hit(40, 0.2), hit(40, 0.2, phase=math.pi / 2), np.clip(4 * hit(100, 0.1), -1, 1)):
        assert oneshot.profile_array(x, SR)["attack_ms"] <= 1.0
    r = oneshot.profile_array(hit(200, 0.15, attack=0.010), SR)
    assert r["attack_ms"] == pytest.approx(10 * 10 ** (-1 / 20), abs=1.0)


def test_pitch_weak_fundamental_under_strong_overtones():
    # 100 Hz at -26 dB under 200, 300, 400 Hz: 300 Hz belongs to 100's series and
    # not to 200's, so 100 Hz is the fundamental
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 0.05, 2: 1.0, 3: 0.5, 4: 0.3}), SR)
    assert r["pitch_body_hz"] == pytest.approx(100, rel=0.02)
    assert r["harmonics_db"][2] == pytest.approx(20 * math.log10(1 / 0.05), abs=1)    # +26 dB


def test_pitch_lone_peak_under_the_pitch_is_named_not_taken():
    # a 150 Hz tom (and its 2nd harmonic) over a 50 Hz rumble 14 dB down: nothing but the
    # rumble itself is in 50 Hz's series and not 150 Hz's, so the pitch stays 150 Hz
    t = np.arange(int(0.6 * SR)) / SR
    x = (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)
         + 0.2 * np.sin(2 * np.pi * 50 * t)) * np.exp(-t / 0.15)
    r = oneshot.profile_array(x[:, None], SR)
    assert r["pitch_body_hz"] == pytest.approx(150, rel=0.01)
    assert any(n.startswith("50 Hz") and "1/3 of the pitch" in n for n in r["notes"])
    # the same shape with the low peak further down (100 Hz at -22 dB under 300 and 600 Hz)
    # cannot be told from a rumble either: it is named, not dropped
    r = oneshot.profile_array(hit(100, 0.2, partials={1: 0.08, 3: 1.0, 6: 0.3}), SR)
    assert any(n.startswith("100 Hz") and "1/3 of the pitch" in n for n in r["notes"])


@pytest.mark.parametrize("snr", [0, -3, -6])
def test_noise_tone_buried_in_noise_stays_pitched(snr):
    # a 200 Hz tone with white noise at the tone's power + |snr| dB: per bin the tone
    # still stands far above the noise, so it is pitched at 200 Hz
    t = np.arange(int(0.5 * SR)) / SR
    noise = np.random.default_rng(1).standard_normal(len(t)) * math.sqrt(0.5) * 10 ** (-snr / 20)
    r = oneshot.profile_array(((np.sin(2 * np.pi * 200 * t) + noise) * np.exp(-t / 0.08))[:, None], SR)
    assert r["pitched"] is True and r["pitch_body_hz"] == pytest.approx(200, rel=0.01)


@pytest.mark.parametrize("f0, tau", [(100, 0.012), (300, 0.003), (60, 0.040), (1000, 0.004), (1000, 0.002)])
def test_harmonic_short_decay_shows_no_fake_harmonic_or_noise(f0, tau):
    # a pure tone has no 2nd harmonic and no noise, however fast it decays: either the
    # body is too short to say (None, with a note) or the reading says so
    r = oneshot.profile_array(hit(f0, tau, seconds=0.3), SR)
    assert r["pitch_body_hz"] == pytest.approx(f0, rel=0.02)
    if r["harmonics_db"] is None:
        assert r["noise_db"]["body"] is None
        assert any("under 4 periods" in n for n in r["notes"])
    else:
        assert r["harmonics_db"][2] < -40
        assert r["noise_db"]["body"] < -30
