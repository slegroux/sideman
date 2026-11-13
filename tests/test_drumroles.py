"""sideman.drumroles.hit_groups on synthetic drum stems at 124 BPM whose answers follow from how they are built.

Needs the `listen` extra:  uv run --extra listen pytest tests/test_drumroles.py -q
Skipped (not failed) when numpy or soundfile is missing.

Conventions used below: bar 0 starts at T0; a 16th is 60 / 124 / 4 s; step 0 is the downbeat, 4 / 8 / 12 are the other
beats, and the offbeats ("and" of each beat) are steps 2, 6, 10, 14, so `s % 4 == 2`. Every hit has its own noise seed
and a gain within +-2 dB.
"""
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import drumroles, oneshot  # noqa: E402

SR = 48000
BPM = 124.0
T0 = 0.25
STEP = 60.0 / BPM / 4.0
OFFBEAT = lambda s: s % 4 == 2  # noqa: E731


def noise_hit(seed, tau, db=0.0, seconds=0.6):
    """High-passed white noise (a hat) decaying with time constant `tau`, `db` dB from nominal."""
    t = np.arange(int(seconds * SR)) / SR
    return np.diff(np.random.default_rng(seed).standard_normal(len(t) + 1)) * np.exp(-t / tau) * 10 ** (db / 20)


def tom_hit(db=0.0, f=150.0, tau=0.08, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    return (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t / tau) * 10 ** (db / 20)


def snare_hit(seed, db=0.0, seconds=0.6):
    t = np.arange(int(seconds * SR)) / SR
    w = np.random.default_rng(seed).standard_normal(len(t))
    return (w * np.exp(-t / 0.08) * 0.4 + np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)) * 10 ** (db / 20)


def clap_hit(seed, db=0.0, seconds=0.6):
    """Three noise bursts 10 ms apart (each a little softer than the one before, so the hit is found at the first),
    then a tail, all band-limited by a 4-sample mean (duller than a snare's noise)."""
    t = np.arange(int(seconds * SR)) / SR
    w = np.convolve(np.random.default_rng(seed).standard_normal(len(t)), np.ones(4) / 4, mode="same")
    env = sum(a * np.exp(-np.maximum(t - b, 0) / 0.004) * (t >= b) for a, b in ((1.0, 0.0), (0.7, 0.010), (0.5, 0.020)))
    env = env + 0.3 * np.exp(-np.maximum(t - 0.02, 0) / 0.1) * (t >= 0.02)
    return w * env * 10 ** (db / 20)


def stem(events, bars, bed_db=None):
    """Mono stem [n]: `events` is [(time s, audio)]; the stem runs a second past the last bar. With `bed_db` a white
    bed `bed_db` dB under the loudest millisecond of a nominal closed hat runs under the whole stem."""
    x = np.zeros(int((T0 + bars * 16 * STEP + 1.0) * SR))
    for t, h in events:
        o = int(round(t * SR))
        m = min(len(h), len(x) - o)
        x[o:o + m] += h[:m]
    if bed_db is not None:
        nominal = oneshot._loudest_ms(noise_hit(1, 0.005)[:, None], SR)
        x += np.random.default_rng(5).standard_normal(len(x)) * nominal * 10 ** (-bed_db / 20)
    return x


def at(index, ms=0.0):
    """Time of the 16th `index` counted from bar 0 (bar * 16 + step), `ms` late."""
    return T0 + index * STEP + ms / 1000.0


def hat_events(bars, open_tau=0.025, closed_tau=0.005, rng_seed=7):
    """Open hats on the offbeats (steps 2, 6, 10, 14), closed hats on every other 16th. At the next 16th (121 ms) an
    open hat of tau 25 ms is 42 dB down: the closed hat after it starts out of a tail well under the -30 dB the
    grouping accepts (the +-2 dB gains and the 10 ms window the tail is read over cost it about 5 dB)."""
    rng = np.random.default_rng(rng_seed)
    return [(at(i), noise_hit(i, open_tau if OFFBEAT(i % 16) else closed_tau, rng.uniform(-2, 2)))
            for i in range(bars * 16)]


def run(x, **kw):
    return drumroles.hit_groups(x, BPM, T0, sr=SR, **kw)


def by_count(r):
    return sorted(r["groups"], key=lambda g: -g["count"])


def truth_label(h):
    return "open" if OFFBEAT(h["step"]) else "closed"


# ---------------------------------------------------------------- the hat stem

@pytest.fixture(scope="module")
def hat_stem():
    return stem(hat_events(8), 8)


@pytest.fixture(scope="module")
def hats(hat_stem):
    return run(hat_stem)


def test_hats_open_on_the_offbeats_and_closed_on_the_rest_are_two_groups(hats):
    r = hats
    assert len(r["hits"]) == 8 * 16 and all(h["status"] == "grouped" for h in r["hits"]), r["unmeasured"]
    closed, open_ = r["groups"]                      # ordered by count: 96 closed, 32 open
    assert (closed["count"], open_["count"]) == (96, 32)
    assert closed["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert open_["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    assert closed["bars_with"] == open_["bars_with"] == 8 and closed["off_grid"] == open_["off_grid"] == 0
    for h in r["hits"]:
        assert (h["bar"], h["step"]) == divmod(h["id"], 16) and abs(h["offset_ms"]) < 1, h
        assert h["group"] == (open_ if OFFBEAT(h["step"]) else closed)["id"]
    assert r["unmeasured"] == [] and r["linkage"] == "average"


def test_hats_group_spread_is_the_noise_realisation_and_under_the_threshold(hats, hat_stem):
    # members are within the threshold of their medoid by construction of the grouping; two takes of one noise hat read
    # 0.5-0.8 (the reviews), so the spread is neither zero nor near the threshold
    for g in hats["groups"]:
        assert 0.2 < g["spread"]["median"] < g["spread"]["max"] <= drumroles.THRESHOLD, g["spread"]
    closed, open_ = (hats["hits"][i] for i in (0, 2))                  # step 0 (closed) and step 2 (open) of bar 0
    d = oneshot.spectral_distance(hat_stem[closed["start"]:closed["start"] + int(STEP * SR) - 100, None],
                                  hat_stem[open_["start"]:open_["start"] + int(STEP * SR) - 100, None], SR)
    assert d > 1.5 * drumroles.THRESHOLD, d           # closed against open: well apart (the reviews: 2.5 at the least)


def test_hats_levels_and_the_clean_hit(hats, hat_stem):
    closed, open_ = hats["groups"]
    # levels were built within +-2 dB of one another; the median of a group sits inside that band
    assert abs(closed["level_db"] - open_["level_db"]) < 4.0
    # a closed hat falls 40 dB (tau 5 ms: in 23 ms) inside its 121 ms: it has a clean hit, and it is a closed hat, cut as reported
    c = closed["clean_hit"]
    assert c is not None and c["hit"] in closed["members"]
    assert len(c["audio"]) == c["end"] - c["start"] <= int(drumroles.MAX_CUT_S * SR)
    assert np.array_equal(c["audio"][:, 0], hat_stem[c["start"]:c["end"]])
    assert abs(c["start"] / SR - hats["hits"][c["hit"]]["onset_s"]) < 1e-9
    assert open_["clean_hit"] is None or open_["clean_hit"]["hit"] in open_["members"]


def test_open_hats_with_a_rest_after_them_have_a_clean_hit_that_falls_40_db():
    # a hat of tau 100 ms falls 40 dB in 460 ms; the next hit is 4 sixteenths (484 ms) away
    r = run(stem([(at(i), noise_hit(i, 0.1)) for i in range(2, 64, 4)], 4))
    g, = r["groups"]
    c = g["clean_hit"]
    assert g["count"] == 16 and c is not None and c["hit"] in g["members"] and g["clean_note"] is None
    assert 0.46 * SR <= len(c["audio"]) <= drumroles.MAX_CUT_S * SR


# ---------------------------------------------------------------- one sound, and where it falls

def test_one_repeated_tom_is_one_group():
    rng = np.random.default_rng(3)
    x = stem([(at(i), tom_hit(rng.uniform(-2, 2))) for i in range(0, 4 * 16, 4)], 4)
    r = run(x)
    assert len(r["hits"]) == 16 and len(r["groups"]) == 1 and r["groups"][0]["count"] == 16
    assert r["groups"][0]["pattern"] == [1.0 if s % 4 == 0 else 0.0 for s in range(16)]
    assert r["groups"][0]["clean_hit"] is not None       # a tom falls 40 dB (tau 150 ms) inside its 484 ms to the next


def test_a_hit_8_ms_early_and_one_12_ms_late_keep_their_16th_and_say_how_far_off_they_are():
    events = [(at(i), tom_hit()) for i in range(0, 64, 4)]
    events[6] = (at(24, -8.0), tom_hit())                  # bar 1, beat 3 (step 8), 8 ms early
    events[10] = (at(40, 12.0), tom_hit())                 # bar 2, beat 3, 12 ms late
    events[14] = (at(56, 30.0), tom_hit())                 # bar 3, beat 3, 30 ms late: past the 20 ms the grid allows
    r = run(stem(events, 4))
    h = r["hits"]
    assert [(x["bar"], x["step"]) for x in h] == [(i // 4, 4 * (i % 4)) for i in range(16)]
    assert h[6]["offset_ms"] == pytest.approx(-8.0, abs=1.0) and h[10]["offset_ms"] == pytest.approx(12.0, abs=1.0)
    assert h[14]["offset_ms"] == pytest.approx(30.0, abs=1.0)
    assert h[6]["on_grid"] and h[10]["on_grid"] and not h[14]["on_grid"]
    assert all(abs(x["offset_ms"]) < 1.0 for i, x in enumerate(h) if i not in (6, 10, 14))
    # one sound in one group; the 30 ms hit is in it but counts in no pattern (3 of 4 bars at step 8)
    g, = r["groups"]
    assert g["count"] == 16 and g["off_grid"] == 1
    assert g["pattern"][8] == pytest.approx(0.75) and g["pattern"][0] == 1.0


def test_clap_and_snare_on_beats_2_and_4_are_two_groups():
    events = []
    for b in range(8):
        events += [(at(16 * b + 4), clap_hit(100 + b, (b % 3 - 1) * 1.5)), (at(16 * b + 12), snare_hit(200 + b, (b % 3 - 1) * 1.5))]
    r = run(stem(events, 8))
    assert len(r["groups"]) == 2 and all(g["count"] == 8 for g in r["groups"])
    by_step = {tuple(i for i, p in enumerate(g["pattern"]) if p): g for g in r["groups"]}
    assert set(by_step) == {(4,), (12,)}                  # clap on beat 2, snare on beat 4, nothing else
    assert all(g["pattern"][s] == 1.0 for (s,), g in by_step.items())
    assert all(g["clean_hit"] is not None for g in r["groups"])
    assert by_step[(4,)]["id"] != by_step[(12,)]["id"]


# ---------------------------------------------------------------- a bleed bed

def test_a_bed_40_db_under_the_hits_does_not_change_the_groups(hats):
    r = run(stem(hat_events(4), 4, bed_db=40.0))
    assert len(r["hits"]) == 64 and all(h["status"] == "grouped" for h in r["hits"]), r["unmeasured"]
    closed, open_ = r["groups"]
    assert (closed["count"], open_["count"]) == (48, 16)
    assert closed["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert open_["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    assert all(h["group"] == (open_ if OFFBEAT(h["step"]) else closed)["id"] for h in r["hits"])
    assert all(abs(h["offset_ms"]) < 1 for h in r["hits"])      # the first onset is not pulled onto the bed


def test_a_bed_30_db_under_the_hits_gives_fewer_groups_never_wrong_ones():
    # the hits start out of a bed only 30 dB under them (a lead-in of -30 dB or louder is "overlapped"): the hits that
    # are measured may still be grouped, and each one must be in a group of its own kind; the rest are reported
    r = run(stem(hat_events(4), 4, bed_db=30.0))
    assert {h["status"] for h in r["hits"]} <= {"grouped", "overlapped", "faint", "no_lead"}
    assert sum(h["status"] == "grouped" for h in r["hits"]) + len(r["unmeasured"]) == len(r["hits"])
    for g in r["groups"]:
        assert len({truth_label(r["hits"][i]) for i in g["members"]}) == 1, g["pattern"]
    assert len(r["groups"]) <= 2
    assert all(u["reason"] and u["status"] in ("overlapped", "faint", "no_lead") for u in r["unmeasured"])


def test_a_bed_20_db_under_the_hits_measures_nothing_and_says_why():
    r = run(stem(hat_events(4), 4, bed_db=20.0))
    assert r["groups"] == [] and all(h["group"] is None for h in r["hits"])
    assert {h["status"] for h in r["hits"]} <= {"overlapped", "faint", "no_lead"}
    assert sum(h["status"] == "overlapped" for h in r["hits"]) >= 60
    assert all("dB" in h["reason"] or "start of the stem" in h["reason"] for h in r["hits"])


# ---------------------------------------------------------------- order, sections, cost

def test_the_order_the_hits_come_in_does_not_change_the_groups():
    ev = hat_events(2)
    cuts = [h[:int(STEP * SR)][:, None] for _, h in ev]      # each hat as far as the next 16th
    keys = [t for t, _ in ev]

    def partition(order, sample):
        groups, rest = drumroles.group_cuts([cuts[i] for i in order], [keys[i] for i in order], SR, sample=sample)
        assert rest == []
        return sorted(sorted(keys[order[i]] for i in g["members"]) for g in groups)
    base = partition(list(range(len(cuts))), 6)           # 32 hits against a sample of 6: sample, assign, repeat
    assert [len(g) for g in sorted(base, key=len)] == [8, 24]
    for seed in (1, 2, 3):
        order = list(np.random.default_rng(seed).permutation(len(cuts)))
        assert partition(order, 6) == base
    assert partition(list(range(len(cuts))), 32) == base  # and the sample does not matter here


def test_a_sound_the_sample_misses_still_gets_its_group():
    # 63 closed hats and one open hat among them (bar 2, step 6); a sample of 8 evenly spaced hits does not include the
    # open one, which is further than the threshold from the closed medoid and so goes round again as a group of its own
    ev = [(at(i), noise_hit(i, 0.025 if i == 38 else 0.005)) for i in range(64)]
    r = run(stem(ev, 4), sample=8)
    assert [g["count"] for g in r["groups"]] == [63, 1]
    assert r["groups"][1]["members"] == [38] and r["groups"][1]["pattern"][38 % 16] == 0.25


def plain(groups):
    """Groups without their audio (arrays do not compare)."""
    return [{k: v for k, v in g.items() if k != "clean_hit"} | {"clean": g["clean_hit"] and g["clean_hit"]["hit"]}
            for g in groups]


def test_sections_are_the_same_groups_restricted_and_their_order_does_not_matter():
    a, b = {"name": "a", "start_bar": 0, "bars": 2}, {"name": "b", "start_bar": 2, "bars": 2}
    # section a: closed hats only; section b: the full pattern
    ev = [(at(i), noise_hit(i, 0.005)) for i in range(32)] + hat_events(4)[32:]
    x = stem(ev, 4)
    r1, r2 = run(x, sections=[a, b]), run(x, sections=[b, a])
    s1, s2 = ({s["name"]: s for s in r["sections"]} for r in (r1, r2))
    assert plain(r1["groups"]) == plain(r2["groups"])
    for name in "ab":
        assert plain(s1[name]["groups"]) == plain(s2[name]["groups"]) and s1[name]["unmeasured"] == s2[name]["unmeasured"]
    assert [g["count"] for g in s1["a"]["groups"]] == [32]            # 2 bars of closed hats: one group
    assert s1["a"]["groups"][0]["pattern"] == [1.0] * 16
    assert [g["count"] for g in s1["b"]["groups"]] == [24, 8]
    assert s1["b"]["groups"][0]["pattern"] == [0.0 if OFFBEAT(s) else 1.0 for s in range(16)]
    assert s1["b"]["groups"][1]["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)]
    ids = {g["id"] for g in r1["groups"]}
    assert ids == {0, 1} and {g["id"] for sec in r1["sections"] for g in sec["groups"]} == ids   # one id per sound in every section


# ---------------------------------------------------------------- what cannot be measured

def test_16th_hats_under_an_open_hats_tail_are_marked_overlapped_or_not_found_and_the_same_every_bar():
    # the open hat (tau 120 ms) rings through the closed 16ths after it. Its tail is, at k 16ths after it,
    # exp(-k * 0.121 / 0.12) = -8.8 dB (k=1), -17.5 (k=2), -26 (k=3), -35 (k=4, the next open hat)
    r = run(stem(hat_events(8, open_tau=0.12, closed_tau=0.012), 8))
    h = r["hits"]
    assert len(h) < 8 * 16                                 # hits that do not rise out of the tail are not hits to the detector...
    assert any("tail" in n for n in r["notes"])            # ...and the report says so
    assert all(x["through"] >= 1 for x in h if OFFBEAT(x["step"]))     # an open hat's tail sounds through the next line
    for x in h:
        if x["bar"] >= 1 and x["step"] % 4 in (3, 0):      # 1 and 2 sixteenths after an open hat: a tail 9 and 17 dB down
            assert x["status"] == "overlapped" and "dB" in x["reason"] and x["group"] is None, x
    open_ = [x for x in h if OFFBEAT(x["step"]) and x["bar"] >= 1]
    assert open_ and all(x["status"] == "grouped" for x in open_)      # the open hats follow a tail of -35 dB: measured
    g, = [g for g in r["groups"] if g["pattern"][2] > 0]
    assert all(OFFBEAT(h[i]["step"]) for i in g["members"]) and all(g["pattern"][s] == 0 for s in range(16) if not OFFBEAT(s))
    # the same bar after bar: the status of each step is one value from bar 1 on (not 3 sixteenths after an open hat, where
    # the tail is -26 dB, within the gains' +-2 dB and the noise of the -30 dB the grouping accepts)
    for s in (s for s in range(16) if s % 4 != 1):
        assert len({x["status"] for x in h if x["step"] == s and x["bar"] >= 1}) <= 1, s
    assert all(u["status"] != "grouped" for u in r["unmeasured"])


def test_a_grace_note_55_ms_before_a_beat_is_too_short_and_the_hit_after_it_starts_out_of_its_tail():
    events = [(at(i), tom_hit()) for i in range(0, 32, 4)]
    events.append((at(16) - 0.055, tom_hit(-12)))          # a grace note 55 ms before the beat at bar 1
    r = run(stem(events, 2))
    grace, main = (x for x in r["hits"] if abs(x["onset_s"] - at(16)) < 0.1)
    assert abs(grace["onset_s"] - (at(16) - 0.055)) < 0.001 and grace["status"] == "too_short" and "55 ms" in grace["reason"]
    assert grace["group"] is None and grace["bar"] == 1 and grace["step"] == 0 and grace["offset_ms"] == pytest.approx(-55.0, abs=1)
    # the hit after it starts out of the grace note's tail (-12 dB, 55 ms later -6 dB more): not measured, and says why
    assert main["status"] == "overlapped" and "dB" in main["reason"]
    assert r["groups"][0]["count"] == 7 and not {grace["id"], main["id"]} & set(r["groups"][0]["members"])
    assert r["unmeasured_pattern"] == [0.5 if i == 0 else 0.0 for i in range(16)]       # the main hit, in 1 bar of 2


def test_a_stem_read_from_a_file_and_the_inputs_that_are_refused(tmp_path):
    x = stem([(at(i), tom_hit()) for i in range(0, 32, 4)], 2)
    sf.write(str(tmp_path / "toms.wav"), x, SR, subtype="FLOAT")
    from_file = drumroles.hit_groups(tmp_path / "toms.wav", BPM, T0)
    from_array = run(x[:, None])
    assert from_file["sample_rate"] == SR and len(from_file["hits"]) == 8
    assert [g["pattern"] for g in from_file["groups"]] == [g["pattern"] for g in from_array["groups"]]
    assert drumroles.hit_groups(np.zeros(SR), BPM, T0, sr=SR)["hits"] == []
    for kw in ({"tempo": 0}, {"t0": float("nan")}, {"sections": [{"start_bar": 0}]}):
        with pytest.raises(ValueError):
            drumroles.hit_groups(x, **{"tempo": BPM, "t0": T0, "sr": SR, **kw})
    with pytest.raises(ValueError, match="sample rate"):
        drumroles.hit_groups(x, BPM, T0)


def test_average_linkage_does_not_chain():
    # points 0, 1.4 and 2.8 on a line: single linkage would chain all three at a threshold of 1.5; average linkage
    # merges the first two (1.4) and then finds the third 2.1 away
    p = np.array([0.0, 1.4, 2.8, 10.0, 10.5])
    clusters = drumroles._agglomerate(np.abs(p[:, None] - p[None, :]), 1.5)
    assert sorted(sorted(c) for c in clusters) == [[0, 1], [2], [3, 4]]
