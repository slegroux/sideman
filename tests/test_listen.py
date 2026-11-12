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


# --------------------------------------------------- reference profile / elements
#
# Every stem is a sustained sine, so its level is known without the code:
# amplitude A reads 20*log10(A / sqrt(2)) dBFS, and a role's level against the
# kick is the dB difference of the amplitudes. The files start T0 seconds
# before bar 0, so a profile that ignored t0 would read the first bar low.

T0 = 0.5
KICK_AMP = 0.4
DRUM_FILE_DB = -16.0      # each of hh and snare, in dB against the kick's amplitude
DRUMS_VS_KICK = DRUM_FILE_DB + 10 * math.log10(2)    # two equal files: twice the power of one
ROLE_HZ = {"kick": 50, "bass": 110, "synth": 1500, "hh": 8000, "snare": 3000}


def gain(db):
    return 10 ** (db / 20)


def level_sine(freq, db_by_bar):
    """A sine at KICK_AMP * gain(db) per bar (None = silent), T0 of silence in front."""
    amp = np.repeat([0.0 if d is None else KICK_AMP * gain(d) for d in db_by_bar], bars(1))
    t = np.arange(len(amp)) / SR
    return np.concatenate([np.zeros(int(T0 * SR)), amp * np.sin(2 * np.pi * freq * t)])


def stem_set(tmp_path, folder, bass_db=-6.0, bars_n=24):
    """kick, bass (`bass_db` against the kick, a number or one per bar), synth (12 dB under the
    kick, silent in bars 8-15), hh + snare (DRUMS_VS_KICK together) and their mix."""
    d = tmp_path / folder
    d.mkdir()
    per_bar = {"kick": [0.0] * bars_n,
               "bass": list(bass_db) if isinstance(bass_db, (list, tuple)) else [bass_db] * bars_n,
               "synth": [None if 8 <= b < 16 else -12.0 for b in range(bars_n)],
               "hh": [DRUM_FILE_DB] * bars_n, "snare": [DRUM_FILE_DB] * bars_n}
    sig = {k: level_sine(ROLE_HZ[k], v) for k, v in per_bar.items()}
    paths = {k: write(d, k + ".wav", stereo(v)) for k, v in sig.items()}
    paths["drums"] = write(d, "drums.wav", stereo(sig["hh"] + sig["snare"]))     # for a capture: one file
    paths["mix"] = write(d, "mix.wav", stereo(sum(sig.values())))
    return paths


SECTIONS = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "B", "start_bar": 8, "bars": 8},
            {"name": "C", "start_bar": 16, "bars": 8}]


def profile_of(tmp_path, ref, **kw):
    return listen.reference_profile(
        ref["mix"], {"kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"],
                     "drums": [ref["hh"], ref["snare"]]},
        TEMPO, T0, SECTIONS, out=tmp_path / "ref.profile.json", **kw)


def test_profile_ratios_to_the_kick_match_the_gains_in_every_section(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    assert [s["name"] for s in p["sections"]] == ["A", "B", "C"]
    for s in p["sections"]:
        r = s["roles"]
        assert r["kick"]["vs_kick_db"] == 0.0
        assert abs(r["bass"]["vs_kick_db"] - (-6.0)) <= 0.1, s["name"]
        assert abs(r["drums"]["vs_kick_db"] - DRUMS_VS_KICK) <= 0.1, s["name"]
        assert abs(r["kick"]["level_db"] - 20 * math.log10(KICK_AMP / math.sqrt(2))) <= 0.05
    for name in ("A", "C"):
        assert abs(p["sections"][0 if name == "A" else 2]["roles"]["synth"]["vs_kick_db"] - (-12.0)) <= 0.1


def test_profile_a_silent_role_is_absent_not_minus_120(tmp_path):
    p = profile_of(tmp_path, stem_set(tmp_path, "ref"))
    b = p["sections"][1]["roles"]["synth"]
    assert b == {"absent": True}
    assert all(not s["roles"][r]["absent"] for s in p["sections"] for r in ("kick", "bass", "drums"))
    assert not p["sections"][0]["roles"]["synth"]["absent"]


def test_profile_per_bar_curve_follows_t0_and_the_bar_grid(tmp_path):
    ref = stem_set(tmp_path, "ref", bass_db=[-6.0] * 4 + [-2.0] * 4 + [-6.0] * 16)
    p = profile_of(tmp_path, ref)
    curve = p["sections"][0]["roles"]["bass"]["per_bar_db"]
    want = [20 * math.log10(KICK_AMP * gain(d) / math.sqrt(2)) for d in [-6.0] * 4 + [-2.0] * 4]
    assert len(curve) == 8 and all(abs(c - w) <= 0.05 for c, w in zip(curve, want)), (curve, want)


def test_profile_bands_width_and_mix_are_measured(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    s = p["sections"][0]
    assert s["roles"]["bass"]["bands_db"]["bass"] > s["roles"]["bass"]["bands_db"]["mid"] + 40   # 110 Hz is in 60-250
    assert s["roles"]["synth"]["bands_db"]["mid"] > s["roles"]["synth"]["bands_db"]["bass"] + 40   # 1.5 kHz is in 500-2k
    assert s["roles"]["kick"]["width"] == 0.0 and s["mix"]["width"] == 0.0
    want = 10 * math.log10(sum((KICK_AMP * gain(d)) ** 2 / 2 for d in (0, -6, -12, DRUM_FILE_DB, DRUM_FILE_DB)))
    assert abs(s["mix"]["level_db"] - want) <= 0.1
    assert s["mix"]["integrated_lufs"] is not None and len(s["mix"]["per_bar_db"]) == 8
    wide = write(tmp_path, "wide.wav", np.stack([sine(1000, 4, 0.3), -sine(1000, 4, 0.3)], axis=1))
    q = listen.reference_profile(wide, {"kick": wide}, TEMPO, 0.0, [{"name": "W", "start_bar": 0, "bars": 2}])
    assert abs(q["sections"][0]["roles"]["kick"]["width"] - 1.0) <= 0.01      # opposite polarity is all side


def test_profile_is_written_with_its_inputs_and_goes_stale_when_one_changes(tmp_path):
    import json
    import os
    ref = stem_set(tmp_path, "ref")
    p = profile_of(tmp_path, ref)
    dest = pathlib.Path(p["json_path"])
    assert dest == (tmp_path / "ref.profile.json").resolve()
    on_disk = json.loads(dest.read_text())
    assert on_disk["tempo"] == TEMPO and on_disk["t0"] == T0 and on_disk["sections"][1]["name"] == "B"
    assert [i["path"] for i in on_disk["inputs"]["stems"]["drums"]] == [str(ref["hh"].resolve()), str(ref["snare"].resolve())]
    assert on_disk["inputs"]["mix"][0]["mtime"] == ref["mix"].stat().st_mtime
    assert listen.profile_stale(p) == [] and listen.profile_stale(on_disk) == []
    st = ref["bass"].stat()
    os.utime(ref["bass"], (st.st_atime, st.st_mtime + 5))
    assert [(s["role"], s["reason"]) for s in listen.profile_stale(on_disk)] == [("bass", "changed")]
    ref["snare"].unlink()
    assert {(s["role"], s["reason"]) for s in listen.profile_stale(on_disk)} == {("bass", "changed"), ("drums", "missing")}


def test_profile_default_path_is_next_to_the_mix(tmp_path):
    ref = stem_set(tmp_path, "ref")
    p = listen.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, SECTIONS[:1])
    assert pathlib.Path(p["json_path"]) == ref["mix"].resolve().with_name("mix.profile.json")
    assert pathlib.Path(p["json_path"]).is_file()


def test_profile_refuses_mixed_sample_rates_and_trims_different_lengths(tmp_path):
    ref = stem_set(tmp_path, "ref")
    odd = write(tmp_path, "odd.wav", stereo(sine(110, 2, 0.1, sr=44100, ), None), sr=44100)
    with pytest.raises(ValueError, match="sample rate"):
        listen.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": odd}, TEMPO, T0, SECTIONS[:1])
    short = write(tmp_path, "short.wav", stereo(level_sine(50, [0.0] * 10)))     # 10 bars + T0, the others 24
    p = listen.reference_profile(ref["mix"], {"kick": short, "bass": ref["bass"]}, TEMPO, T0,
                                 [{"name": "A", "start_bar": 0, "bars": 8}])
    assert any("trimmed" in n and "short.wav" in n for n in p["notes"]), p["notes"]
    with pytest.raises(ValueError, match="sections run to"):
        listen.reference_profile(ref["mix"], {"kick": short}, TEMPO, T0, SECTIONS)


def test_profile_refuses_bad_input(tmp_path):
    ref = stem_set(tmp_path, "ref")
    with pytest.raises(ValueError, match="unique"):
        listen.reference_profile(ref["mix"], {"kick": ref["kick"]}, TEMPO, T0, [SECTIONS[0], SECTIONS[0]])
    with pytest.raises(ValueError, match="no such audio file"):
        listen.reference_profile(ref["mix"], {"kick": tmp_path / "nope.wav"}, TEMPO, T0, SECTIONS[:1])
    with pytest.raises(ValueError, match="role"):
        listen.reference_profile(ref["mix"], {"mix": ref["kick"]}, TEMPO, T0, SECTIONS[:1])
    with pytest.raises(ValueError, match="tempo"):
        listen.reference_profile(ref["mix"], {"kick": ref["kick"]}, 0, T0, SECTIONS[:1])
def capture_of(ref, **over):
    """A manifest whose sources are the files of a stem_set (drums as the one summed file)."""
    src = {"master": ref["mix"], "kick": ref["kick"], "bass": ref["bass"], "synth": ref["synth"],
           "drums": ref["drums"]}
    src.update(over)
    return {"tempo": TEMPO, "beats_per_bar": 4, "offset_s": T0,
            "sources": {k: str(v) for k, v in src.items() if v is not None},
            "sections": [dict(s) for s in SECTIONS]}


def test_elements_bass_raised_4_db_is_the_only_gap(tmp_path):
    prof = profile_of(tmp_path, stem_set(tmp_path, "ref"))
    ours = stem_set(tmp_path, "ours", bass_db=-6.0 + 4.0)
    r = listen.mix_report(capture_of(ours), reference=prof, out_dir=tmp_path / "out")
    assert r["reference_kind"] == "profile" and r["reference_stale"] == []
    for s in r["sections"]:
        e = s["elements"]
        assert e["paired"] and not e["capture_is_reference"], s["name"]
        assert [g["role"] for g in e["gaps"]] == ["bass"], (s["name"], e["gaps"])
        assert abs(e["gaps"][0]["difference_db"] - 4.0) <= 0.2
        row = {t["role"]: t for t in e["table"]}
        assert abs(row["bass"]["reference_db"] - (-6.0)) <= 0.1 and abs(row["bass"]["ours_db"] - (-2.0)) <= 0.1
        assert abs(row["drums"]["difference_db"]) <= 0.1
    b = r["sections"][1]["elements"]
    assert "synth" not in {t["role"] for t in b["table"]} and b["silent_in_both"] == ["synth"]
    assert "master" not in {t["role"] for s in r["sections"] for t in s["elements"]["table"]}
    md = pathlib.Path(r["md_path"]).read_text()
    assert "## Elements against the kick" in md and "bass +4.0 dB" in md
    assert "bass" in pathlib.Path(r["json_path"]).read_text()


def test_elements_the_capture_that_is_the_reference_is_flagged(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    r = listen.mix_report(capture_of(ref), reference=prof["json_path"], out_dir=tmp_path / "out")   # a profile path
    for s in r["sections"]:
        e = s["elements"]
        assert e["capture_is_reference"] and "the capture is the reference: check solos" in e["notes"]
        assert e["gaps"] == []
        assert s["vs_reference"]["flagged_bands"] == []
    assert "the capture is the reference: check solos" in pathlib.Path(r["md_path"]).read_text()
    ours = stem_set(tmp_path, "ours", bass_db=-6.0 + 4.0)           # a different master: no flag
    assert not any(s["elements"]["capture_is_reference"] for s in listen.mix_report(capture_of(ours), reference=prof)["sections"])


def test_elements_without_a_kick_source_leaves_the_table_out_but_keeps_the_mix_layer(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    r = listen.mix_report(capture_of(ref, kick=None), reference=prof, out_dir=tmp_path / "out")
    for s in r["sections"]:
        e = s["elements"]
        assert e["table"] is None and e["gaps"] == [] and e["arc"] == {}
        assert any("no kick source" in n for n in e["notes"]), e["notes"]
        assert s["integrated_lufs"] is not None and "bands_db" in s and "vs_reference" in s
    assert "no kick source" in pathlib.Path(r["md_path"]).read_text()


def test_elements_roles_in_one_side_only_and_sections_that_do_not_pair(tmp_path):
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    ours = stem_set(tmp_path, "ours")
    m = capture_of(ours, drums=None, vox=ours["hh"])
    m["sections"] = [{"name": "A", "start_bar": 0, "bars": 8}, {"name": "Z", "start_bar": 8, "bars": 8},
                     {"name": "C", "start_bar": 16, "bars": 4}]
    r = listen.mix_report(m, reference=prof)
    a, z, c = (s["elements"] for s in r["sections"])
    assert [x["role"] for x in a["reference_only"]] == ["drums"] and a["reference_only"][0]["reason"] == "not in the manifest"
    assert abs(a["reference_only"][0]["vs_kick_db"] - DRUMS_VS_KICK) <= 0.1
    assert [x["role"] for x in a["ours_only"]] == ["vox"] and a["ours_only"][0]["reason"] == "not in the profile"
    assert z == {"paired": False, "notes": ["no section named 'Z' in the profile: mix layer only"]}
    assert not c["paired"] and "4 bars here and 8 in the profile" in c["notes"][0]
    assert "vs_reference" not in r["sections"][1] and r["sections"][1]["integrated_lufs"] is not None


def test_elements_arc_follows_the_ratio_per_four_bars(tmp_path):
    one = [{"name": "A", "start_bar": 0, "bars": 8}]
    ref = stem_set(tmp_path, "ref", bass_db=-6.0)
    prof = listen.reference_profile(ref["mix"], {"kick": ref["kick"], "bass": ref["bass"]}, TEMPO, T0, one)
    ours = stem_set(tmp_path, "ours", bass_db=[-6.0] * 4 + [-2.0] * 4 + [-6.0] * 16)
    m = capture_of(ours)
    m["sections"] = one
    arc = listen.mix_report(m, reference=prof)["sections"][0]["elements"]["arc"]["bass"]
    assert all(abs(v - w) <= 0.2 for v, w in zip(arc["reference"], [-6.0, -6.0])), arc
    assert all(abs(v - w) <= 0.2 for v, w in zip(arc["ours"], [-6.0, -2.0])), arc


def test_elements_a_stale_profile_is_reported_and_a_plain_audio_reference_is_unchanged(tmp_path):
    import os
    ref = stem_set(tmp_path, "ref")
    prof = profile_of(tmp_path, ref)
    st = ref["kick"].stat()
    os.utime(ref["kick"], (st.st_atime, st.st_mtime + 5))
    r = listen.mix_report(capture_of(stem_set(tmp_path, "ours")), reference=prof, out_dir=tmp_path / "o1")
    assert [s["role"] for s in r["reference_stale"]] == ["kick"]
    assert "changed since the profile was built" in pathlib.Path(r["md_path"]).read_text()
    r = listen.mix_report(capture_of(ref), reference=str(ref["mix"]), out_dir=tmp_path / "o2")
    assert "reference_kind" not in r and r["reference"] == str(ref["mix"])
    for s in r["sections"]:
        assert "elements" not in s and set(s["vs_reference"]) == {"band_balance_db", "flagged_bands", "lufs_diff"}
    assert "Elements against the kick" not in pathlib.Path(r["md_path"]).read_text()
    with pytest.raises(ValueError, match="not a reference profile"):
        listen.mix_report(capture_of(ref), reference={"sections": []})


