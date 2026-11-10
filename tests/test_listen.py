"""sideman.listen on synthetic signals whose answers are known without the code.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_listen.py -q
Skipped (not failed) when numpy or soundfile is missing.
"""
import math
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import listen  # noqa: E402

SR = 48000


def sine(freq, seconds, amp=1.0, phase=0.0, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


def stereo(left, right=None):
    return np.stack([left, left if right is None else right], axis=1)


def pink(seconds, seed=0, sr=SR):
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    spec[1:] /= np.sqrt(f[1:])
    spec[0] = 0
    x = np.fft.irfft(spec, n)
    return 0.25 * x / np.abs(x).max()


def lowpass(x, hz, sr=SR):
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(len(x), 1 / sr) >= hz] = 0
    return np.fft.irfft(spec, len(x))


def write(tmp_path, name, x, sr=SR):
    p = tmp_path / name
    sf.write(str(p), x, sr, subtype="FLOAT")
    return p


# ---------------------------------------------------------------- loudness

def test_k_weighting_matches_the_bs1770_48k_table():
    (b1, a1), (b2, a2) = listen.k_weighting(48000)
    assert np.allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-6)
    assert np.allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-6)
    assert np.allclose(b2, [1.0, -2.0, 1.0], atol=1e-9)
    assert np.allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-6)


def test_stereo_997hz_sine_at_minus_20_dbfs_reads_minus_20_lufs():
    # BS.1770: a 0 dBFS 997 Hz sine on one channel reads -3.01 LKFS, so on both
    # channels 0.0, and at amplitude 0.1 (-20 dBFS) -20.0.
    r = listen.analyze_array(stereo(sine(997, 10, amp=0.1)), SR)
    assert r["integrated_lufs"] == pytest.approx(-20.0, abs=0.2)


def test_gating_ignores_silence():
    x = np.concatenate([sine(997, 10, amp=0.1), np.zeros(10 * SR)])
    r = listen.analyze_array(stereo(x), SR)
    assert r["integrated_lufs"] == pytest.approx(-20.0, abs=0.2)   # ungated would be -23


def test_loudness_works_at_44100():
    r = listen.analyze_array(stereo(sine(997, 10, amp=0.1, sr=44100)), 44100)
    assert r["integrated_lufs"] == pytest.approx(-20.0, abs=0.2)


# BS.1770-4 Table 1/2 coefficients at 48 kHz, typed from the standard, and a
# plain sample-by-sample filter: an implementation the code under test does not use.
SHELF = ([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585])
HIPASS = ([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621])


def reference_lufs(x):
    def biquad(b, a, v):
        y = np.zeros_like(v)
        x1 = x2 = y1 = y2 = 0.0
        for i, s in enumerate(v):
            o = b[0] * s + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
            x2, x1, y2, y1 = x1, s, y1, o
            y[i] = o
        return y
    # BS.1770-4 section 2.8: 400 ms blocks every 100 ms, absolute then relative gate
    y = [biquad(*HIPASS, biquad(*SHELF, x[:, c])) for c in range(x.shape[1])]
    size, hop = int(0.4 * SR), int(0.1 * SR)
    z = np.array([sum(np.mean(ch[j:j + size] ** 2) for ch in y)
                  for j in range(0, len(y[0]) - size + 1, hop)])
    loud = -0.691 + 10 * np.log10(z)
    z = z[loud > -70]
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    z = z[-0.691 + 10 * np.log10(z) > rel]
    return -0.691 + 10 * np.log10(z.mean())


@pytest.mark.parametrize("signal", ["sub_noise", "sine45", "kick"])
def test_low_end_loudness_matches_a_time_domain_reference(signal):
    n = 4 * SR
    if signal == "sub_noise":
        x = lowpass(np.random.default_rng(1).standard_normal(n), 80) * 0.5
    elif signal == "sine45":
        x = sine(45, 4, amp=0.5)
    else:
        t = np.arange(SR // 2) / SR
        hit = np.sin(2 * np.pi * (50 + 120 * np.exp(-t * 30)) * t) * np.exp(-t * 8)
        x = np.tile(hit, 8)[:n] * 0.8
    r = listen.analyze_array(stereo(x)[SR // 2:], SR)       # skip the filters' start-up
    assert r["integrated_lufs"] == pytest.approx(reference_lufs(stereo(x)[SR // 2:]), abs=0.02)


def test_mono_file_is_one_channel_not_dual_mono():
    # BS.1770: 0 dBFS 997 Hz on a single channel reads -3.01 LKFS
    r = listen.analyze_array(sine(997, 10)[:, None], SR)
    assert r["integrated_lufs"] == pytest.approx(-3.01, abs=0.2)


def test_more_than_two_channels_is_refused(tmp_path):
    p = write(tmp_path, "six.wav", np.zeros((SR, 6)))
    with pytest.raises(ValueError, match="6 channels"):
        listen.analyze(p)
    with pytest.raises(ValueError, match="6 channels"):
        listen.analyze_array(np.zeros((SR, 6)), SR)


def test_unmeasurable_loudness_is_none_not_a_number():
    short = listen.analyze_array(stereo(sine(997, 0.35, amp=0.1)), SR)
    assert short["integrated_lufs"] is None and short["momentary_max_lufs"] is None
    half = listen.analyze_array(stereo(sine(997, 0.45, amp=0.1)), SR)
    assert half["integrated_lufs"] is not None and half["short_term_max_lufs"] is None
    silent = listen.analyze_array(np.zeros((5 * SR, 2)), SR)
    assert silent["integrated_lufs"] is None and silent["short_term_max_lufs"] is None
    c = listen.compare(short, listen.analyze_array(stereo(sine(997, 4, amp=0.1)), SR))
    assert "integrated_lufs" not in c["diff"]


# ---------------------------------------------------------------- peaks

def test_intersample_peak_fs_over_4_at_45_degrees():
    # samples land at sin(45 deg + k*90 deg) = +-0.7071: sample peak -3.01 dB,
    # while the waveform between them reaches 1.0.
    r = listen.analyze_array(stereo(sine(SR / 4, 2, phase=math.pi / 4)), SR)
    assert r["sample_peak_db"] == pytest.approx(-3.01, abs=0.1)
    assert r["true_peak_dbtp"] == pytest.approx(0.0, abs=0.3)


def test_odd_length_file_does_not_wrap_around():
    # a lone full-scale step at the end of an odd-length file: the interpolated
    # overshoot (Gibbs) stays near 1.18 (+1.4 dBTP), never the wrap-around blow-up
    x = np.zeros(4001)
    x[-1000:] = 1.0
    r = listen.analyze_array(stereo(x), SR)
    assert 0.0 <= r["true_peak_dbtp"] <= 2.0


def test_nyquist_bin_is_counted_once():
    # a periodic sine at fs/2 sampled at its peaks: with no padding (margin 0)
    # each even-length chunk is exactly periodic, so the ideal interpolation is
    # the samples themselves, 0.5 = -6.02 dB. Counting the Nyquist bin twice reads 0 dB.
    x = np.tile([0.5, -0.5], SR)[:, None]
    assert 20 * np.log10(listen.true_peak(x, margin=0)) == pytest.approx(-6.02, abs=0.05)


def test_sine_crest_centroid_and_band_level():
    r = listen.analyze_array(stereo(sine(1000, 4)), SR)
    assert r["crest_db"] == pytest.approx(3.01, abs=0.1)
    assert r["centroid_hz"] == pytest.approx(1000, abs=20)
    assert r["bands_db"]["mid"] == pytest.approx(-3.01, abs=0.1)    # full-scale sine power 0.5
    assert r["bands_db"]["bass"] < -60


# ---------------------------------------------------------------- stereo

def test_identical_channels_are_mono():
    r = listen.analyze_array(stereo(pink(4)), SR)
    assert r["width"] == pytest.approx(0.0, abs=1e-3)
    assert r["low_end_correlation"] == pytest.approx(1.0, abs=1e-3)


def test_polarity_flipped_low_end_reads_minus_one():
    low = lowpass(pink(4), 100)
    r = listen.analyze_array(stereo(low, -low), SR)
    assert r["low_end_correlation"] == pytest.approx(-1.0, abs=1e-3)
    assert r["width"] == pytest.approx(1.0, abs=1e-3)


def test_dc_offset_does_not_count_as_low_end_decorrelation():
    tone = sine(50, 4, amp=0.3)
    r = listen.analyze_array(stereo(tone + 0.2, tone - 0.2), SR)
    assert r["low_end_correlation"] == pytest.approx(1.0, abs=0.02)


def test_mono_file_is_treated_as_centred(tmp_path):
    p = write(tmp_path, "mono.wav", sine(1000, 2, amp=0.5))
    r = listen.analyze(p)
    assert r["channels"] == 1 and r["width"] == pytest.approx(0.0, abs=1e-3)
    assert r["file"] == str(p)


# ---------------------------------------------------------------- compare

def test_compare_flags_only_the_bands_above_a_2k_lowpass(tmp_path):
    x = pink(8)
    a = listen.analyze(write(tmp_path, "full.wav", stereo(x)))
    b = listen.analyze(write(tmp_path, "dark.wav", stereo(lowpass(x, 2000))))
    c = listen.compare(b, a)
    assert {f["band"] for f in c["flagged_bands"]} == {"high_mid", "high"}
    assert all(f["diff_db"] < 0 for f in c["flagged_bands"])


def test_compare_ignores_a_pure_level_difference(tmp_path):
    x = stereo(pink(8))
    a = listen.analyze(write(tmp_path, "a.wav", x))
    b = listen.analyze(write(tmp_path, "b.wav", x * 10 ** (-6 / 20)))
    c = listen.compare(b, a)
    assert c["flagged_bands"] == []
    assert c["level_offset_db"] == pytest.approx(-6.0, abs=0.1)
    assert c["diff"]["integrated_lufs"] == pytest.approx(-6.0, abs=0.1)


# ---------------------------------------------------------------- mix report

TEMPO = 120.0          # 1 bar = 2 s


def bars(n):
    return int(n * 2 * SR)


def pulses(freq, n_bars, amp=0.5):
    """A kick-like 4-on-the-floor of `freq` bursts (120 ms each beat)."""
    x = np.zeros(bars(n_bars))
    hit = sine(freq, 0.12, amp) * np.hanning(int(0.12 * SR))
    for b in range(n_bars * 4):
        s = int(b * 0.5 * SR)
        x[s:s + len(hit)] += hit
    return x


def capture_files(tmp_path):
    """3 sections x 8 bars. Kick 45 Hz bursts and bass 110 Hz, except in
    section B where both hold a sustained 60 Hz sine. Hats change level every
    4 bars in A and C and are silent in B."""
    n = 24
    kick = np.concatenate([pulses(45, 8), sine(60, 16, 0.4)[:bars(8)], pulses(45, 8)])
    bass = np.concatenate([sine(110, 16, 0.3), sine(60, 16, 0.3)[:bars(8)], sine(110, 16, 0.3)])
    rng = np.random.default_rng(3)
    hats = np.zeros(bars(n))
    for blk in range(0, n, 4):
        level = 0.0 if 8 <= blk < 16 else (0.02 if (blk // 4) % 2 else 0.08)
        hats[bars(blk):bars(blk + 4)] = lowpass(rng.standard_normal(bars(4)), 20000) * level
    master = kick + bass + hats
    paths = {"master": write(tmp_path, "master.wav", stereo(master)),
             "kick": write(tmp_path, "kick.wav", stereo(kick)),
             "bass": write(tmp_path, "bass.wav", stereo(bass))}
    return {"tempo": TEMPO, "beats_per_bar": 4, "offset_s": 0.0,
            "sources": {k: str(v) for k, v in paths.items()},
            "sections": [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8},
                         {"name": "C", "start_bar": 16, "bars": 8}]}


def test_masking_flags_only_the_section_where_kick_and_bass_share_60hz(tmp_path):
    r = listen.mix_report(capture_files(tmp_path))
    flags = {s["name"]: s["masking_flag"] for s in r["sections"]}
    assert flags == {"A": False, "B": True, "C": False}, [s["kick_bass_overlap"] for s in r["sections"]]


def test_masking_is_zero_when_one_ducks_under_the_other():
    kick = pulses(60, 4)
    bass = sine(60, 8, 0.3) * (np.abs(kick) < 1e-9)    # perfect sidechain: silent under each hit
    ov = listen.masking(stereo(kick), stereo(bass), SR)
    assert ov is not None and ov["overlap"] < 0.2


def test_change_rate_flags_one_static_run(tmp_path):
    m = capture_files(tmp_path)
    loop = pulses(45, 4) + sine(110, 8, 0.3)
    master = np.concatenate([np.tile(loop, 4), pulses(45, 4) + sine(220, 8, 0.3)])   # 16 static bars, then a change
    m["sources"] = {"master": str(write(tmp_path, "loop.wav", stereo(master)))}
    m["sections"] = [{"name": "LOOP", "start_bar": 0, "bars": 16}, {"name": "NEXT", "start_bar": 16, "bars": 4}]
    r = listen.mix_report(m)
    runs = r["change_rate"]["static_runs"]
    assert len(runs) == 1 and runs[0]["start_bar"] == 0 and runs[0]["end_bar"] == 16
    assert runs[0]["sections"] == ["LOOP"]


def test_an_8_bar_static_span_is_not_flagged(tmp_path):
    m = capture_files(tmp_path)
    loop = pulses(45, 4) + sine(110, 8, 0.3)
    master = np.concatenate([np.tile(loop, 2), pulses(45, 4) + sine(220, 8, 0.3)])
    m["sources"] = {"master": str(write(tmp_path, "eight.wav", stereo(master)))}
    m["sections"] = [{"name": "X", "start_bar": 0, "bars": 12}]
    assert listen.mix_report(m)["change_rate"]["static_runs"] == []


def test_reference_equal_to_master_gives_zero_band_differences(tmp_path):
    m = capture_files(tmp_path)
    m["sections"] = [{"name": "ALL", "start_bar": 0, "bars": 24}]
    r = listen.mix_report(m, reference=m["sources"]["master"])
    bal = r["sections"][0]["vs_reference"]["band_balance_db"]
    assert all(abs(v) <= 0.1 for v in bal.values()), bal


def test_report_files_are_written(tmp_path):
    m = capture_files(tmp_path)
    r = listen.mix_report(m, out_dir=tmp_path / "out")
    md = pathlib.Path(r["md_path"]).read_text()
    assert pathlib.Path(r["json_path"]).is_file()
    assert "| A | 1–8 |" in md and "| B | 9–16 |" in md and "Kick/bass overlap" in md


def test_manifest_errors_are_value_errors(tmp_path):
    with pytest.raises(ValueError, match="master"):
        listen.mix_report({"tempo": 120, "sources": {}, "sections": []})
    with pytest.raises(ValueError, match="tempo"):
        listen.mix_report({"sources": {}, "sections": []})


def test_an_inaudible_bass_under_the_kick_is_not_flagged(tmp_path):
    m = capture_files(tmp_path)
    kick, bass = sine(60, 16, 0.4), sine(60, 16, 0.4 * 10 ** (-40 / 20))   # same cells, bass 40 dB down
    m["sources"] = {"master": str(write(tmp_path, "m.wav", stereo(kick + bass))),
                    "kick": str(write(tmp_path, "k.wav", stereo(kick))),
                    "bass": str(write(tmp_path, "b.wav", stereo(bass)))}
    m["sections"] = [{"name": "Q", "start_bar": 0, "bars": 8}]
    row = listen.mix_report(m)["sections"][0]
    assert row["kick_bass_overlap"] == pytest.approx(1.0, abs=0.01)
    assert row["kick_bass_quieter_db"] == pytest.approx(-40.0, abs=0.5)
    assert row["masking_flag"] is False


def test_sections_outside_the_files_are_refused(tmp_path):
    m = capture_files(tmp_path)                       # 24 bars long
    m["sections"] = [{"name": "PAST", "start_bar": 20, "bars": 8}]
    with pytest.raises(ValueError, match="sections run to bar 28"):
        listen.mix_report(m)
    m["sections"] = [{"name": "NEG", "start_bar": -4, "bars": 8}]
    with pytest.raises(ValueError, match="start_bar >= 0"):
        listen.mix_report(m)


def test_a_source_shorter_than_the_sections_is_refused(tmp_path):
    m = capture_files(tmp_path)
    m["sources"]["kick"] = str(write(tmp_path, "short.wav", stereo(pulses(45, 4))))
    with pytest.raises(ValueError, match="kick is 8.00 s long"):
        listen.mix_report(m)


def test_a_10_bar_static_run_is_flagged(tmp_path):
    m = capture_files(tmp_path)
    loop = pulses(45, 1) + sine(110, 2, 0.3)          # a 1-bar loop
    m["sources"] = {"master": str(write(tmp_path, "ten.wav", stereo(np.tile(loop, 10))))}
    m["sections"] = [{"name": "TEN", "start_bar": 0, "bars": 10}]
    runs = listen.mix_report(m)["change_rate"]["static_runs"]
    assert runs == [{"start_bar": 0, "end_bar": 10, "bars": 10, "sections": ["TEN"]}]


def test_an_element_far_below_the_mix_is_not_a_change_but_a_hat_is(tmp_path):
    m = capture_files(tmp_path)
    loop = np.tile(pulses(45, 1, 0.8) + sine(110, 2, 0.5), 8)
    rng = np.random.default_rng(5)
    hiss = np.zeros(len(loop))
    hiss[bars(4):] = rng.standard_normal(bars(4)) * 10 ** (-60 / 20)   # -60 dBFS noise enters at bar 5
    hats = np.zeros(len(loop))
    hats[bars(4):] = rng.standard_normal(bars(4)) * 0.05               # a hat-level noise enters at bar 5
    for name, extra, want_change in (("hiss", hiss, False), ("hats", hats, True)):
        m["sources"] = {"master": str(write(tmp_path, name + ".wav", stereo(loop + extra)))}
        m["sections"] = [{"name": "S", "start_bar": 0, "bars": 8}]
        step = listen.mix_report(m)["change_rate"]["steps"][0]["change_db"]
        assert (step >= listen.CHANGE_DB) is want_change, (name, step)


def test_silence_is_never_a_static_run(tmp_path):
    m = capture_files(tmp_path)
    m["sources"] = {"master": str(write(tmp_path, "quiet.wav", np.zeros((bars(16), 2))))}
    m["sections"] = [{"name": "SILENCE", "start_bar": 0, "bars": 16}]
    r = listen.mix_report(m)
    assert r["change_rate"]["static_runs"] == [] and all(s["silent"] for s in r["change_rate"]["steps"])


def test_manifest_validation_and_relative_sources(tmp_path):
    import json
    for bad, msg in (({"tempo": 0}, "tempo"), ({"tempo": float("nan")}, "tempo"), ({"sections": []}, "no sections"),
                     ({"sections": [4]}, "must be an object"),
                     ({"sources": {"master": "master.wav"}}, "needs absolute paths"),
                     ({"sections": [{"name": "X", "start_bar": 2.5, "bars": 4}]}, "whole-number")):
        m = capture_files(tmp_path)
        m.update(bad)
        with pytest.raises(ValueError, match=msg):
            listen.mix_report(m)
    m = capture_files(tmp_path)
    m["sources"] = {k: pathlib.Path(v).name for k, v in m["sources"].items()}   # relative to the manifest
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(m))
    assert len(listen.mix_report(str(mp))["sections"]) == 3
    assert len(listen.mix_report(json.dumps({**m, "sources": {k: str(tmp_path / v) for k, v in m["sources"].items()}}))["sections"]) == 3
    with pytest.raises(ValueError, match="no such manifest"):
        listen.mix_report(str(tmp_path / "nope.json"))
