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
    """A longer hat (`open_tau`) on the offbeats (steps 2, 6, 10, 14), a shorter one (`closed_tau`) on every other
    16th. With the defaults the longer hat is not an open hat but a short one: at the next 16th (121 ms) it is 42 dB
    down, so the hat after it starts out of a tail well under the -30 dB the grouping accepts (the +-2 dB gains and the
    10 ms window the tail is read over cost it about 5 dB). A real open hat (tau 120 ms and more) is the H1 tests."""
    rng = np.random.default_rng(rng_seed)
    return [(at(i), noise_hit(i, open_tau if OFFBEAT(i % 16) else closed_tau, rng.uniform(-2, 2)))
            for i in range(bars * 16)]


def run(x, **kw):
    kw.setdefault("stability_runs", 0)                       # the regroupings triple the time; the stability tests ask for them
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


def test_a_short_hat_on_the_offbeats_and_a_shorter_one_on_the_rest_are_two_groups(hats):
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
    # its tail (-10.5, -21, -31.5 dB at the next three 16th lines, -42 at the fourth, where the next hit is) sounds through 3 lines
    assert all(r["hits"][i]["through"] == 3 for i in g["members"])
    # so the lines 1-3 after each hat (steps 3-5, 7-9, 11-13, 15-1) are inside a tail with no hit; the hats' own lines are not;
    # the steps 0 and 1 of bar 0 have no earlier hat (bars 1-3 of 4: 0.75)
    assert r["hidden_pattern"] == [0.75, 0.75, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]


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
    assert {h["status"] for h in r["hits"]} <= {"grouped", "assigned", "overlapped", "faint", "no_lead"}
    assert sum(h["status"] in ("grouped", "assigned") for h in r["hits"]) + len(r["unmeasured"]) == len(r["hits"])
    for g in r["groups"]:
        assert len({truth_label(r["hits"][i]) for i in g["members"] + g["assigned_members"]}) == 1, g["pattern"]
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

def test_a_grace_note_55_ms_before_a_beat_is_too_short_and_the_hit_after_it_is_assigned():
    events = [(at(i), tom_hit()) for i in range(0, 32, 4)]
    events.append((at(16) - 0.055, tom_hit(-12)))          # a grace note 55 ms before the beat at bar 1
    r = run(stem(events, 2))
    grace, main = (x for x in r["hits"] if abs(x["onset_s"] - at(16)) < 0.1)
    assert abs(grace["onset_s"] - (at(16) - 0.055)) < 0.001 and grace["status"] == "too_short" and "55 ms" in grace["reason"]
    assert grace["group"] is None and grace["bar"] == 1 and grace["step"] == 0 and grace["offset_ms"] == pytest.approx(-55.0, abs=1)
    # the hit after it starts out of the grace note's tail (-12 dB, 55 ms later -6 dB more, so -18 dB under it: assigned over
    # its first 40 ms to the tom group, flagged as assigned, and not one of its measured members)
    assert main["status"] == "assigned" and "overlapped" in main["reason"] and main["group"] == r["groups"][0]["id"]
    g = r["groups"][0]
    assert g["count"] == 7 and g["assigned"] == 1 and main["id"] in g["assigned_members"] and main["id"] not in g["members"]
    assert g["assigned_pattern"] == [0.5 if i == 0 else 0.0 for i in range(16)] and g["pattern"][0] == 1.0
    assert r["unmeasured_pattern"] == [0.0] * 16           # the grace note is off the grid, the main hit has a group


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


# ---------------------------------------------------------------- H1: 16th hats under a real open hat

def kinds_of(r, played, tol=0.015):
    """({hit id: kind of the played hat it is}, hits that are none of them) for `played` = [(time s, kind)]."""
    on = np.array([h["onset_s"] for h in r["hits"]])
    out = {}
    for t, kind in played:
        k = int(np.argmin(np.abs(on - t)))
        if abs(on[k] - t) <= tol:
            out[k] = kind
    return out, len(r["hits"]) - len(out)


@pytest.mark.parametrize("tau,grid,least_closed", [(0.12, False, 45), (0.12, True, 61), (0.2, False, 14), (0.2, True, 29)])
def test_16th_hats_under_a_real_open_hat_are_placed_in_the_hat_group_or_reported(tau, grid, least_closed):
    # Open hats (tau 120 and 200 ms) on the offbeats, closed hats (tau 12 ms) on the 96 other 16ths, its own seed and +-2 dB
    # each. The closed hat after an open one starts out of a tail 9 dB (tau 120) or 14 dB (200) under it, which the detector
    # does not see as a hit (`grid_onsets` looks for it at the 16th line), and where it is found its tail makes it
    # "overlapped": it is assigned to the nearest existing group, over a window that shrinks with the tail. The review
    # measured, with both: 61-63 closed hats placed (tau 120) and 29-33 (tau 200), where there were 2 before; the numbers
    # asserted for A off are what this code placed on these stems (51 and 16), less a hat or two. No hat in the wrong group.
    ev = hat_events(8, open_tau=tau, closed_tau=0.012)
    played = [(t, "open" if OFFBEAT(i % 16) else "closed") for i, (t, _) in enumerate(ev)]
    r = run(stem(ev, 8), grid_onsets=grid)
    kinds, extra = kinds_of(r, played)
    placed = {k: sum(r["hits"][i]["status"] in ("grouped", "assigned") for i, kd in kinds.items() if kd == k)
              for k in ("open", "closed")}
    assert placed["open"] == 32 and placed["closed"] >= least_closed, placed
    assert extra == 0                                       # no hit that was not played, with `grid_onsets` or without
    for g in r["groups"]:                                   # nothing in a group of the other kind
        assert len({kinds[i] for i in g["members"] + g["assigned_members"]}) == 1, g
    g, = [g for g in r["groups"] if kinds[(g["members"] + g["assigned_members"])[0]] == "open"]
    assert g["pattern"] == [1.0 if OFFBEAT(s) else 0.0 for s in range(16)] and g["bars_with"] == 8
    assert g["count"] + g["assigned"] == 32
    assert all(h["status"] == "assigned" and h["distance"] <= drumroles.THRESHOLD and "overlapped" in h["reason"]
               for h in r["hits"] if h["group"] is not None and h["status"] != "grouped")
    assert g["clean_hit"] is None or g["clean_hit"]["hit"] in g["members"]       # an assigned hit is never the clean hit
    # the lines inside a tail where no hit was found: the open hats' own lines never; the line right after an open hat
    # (steps 3, 7, 11, 15: its tail is 9 dB down there) always without the search, less so with it
    assert [r["hidden_pattern"][s] for s in (2, 6, 10, 14)] == [0.0] * 4
    after = [r["hidden_pattern"][s] for s in (3, 7, 11, 15)]
    assert (max(after) < 1.0) if grid else (after == [1.0] * 4), after
    assert r["grid_hits"] == sum(h["source"] == "grid" for h in r["hits"]) == (len(r["hits"]) - sum(
        h["source"] == "segment" for h in r["hits"]))
    assert (r["grid_hits"] > 0) == grid


# ---------------------------------------------------------------- the grid-line detector on stems that are not hats

def bleed_stem(kind):
    """2 bars of a tom (every beat) or a kick (every beat), with a hat (every 16th) or a snare-like burst (the offbeats)
    bleeding in 45 dB under it: under the 40 dB `_segments` reads, over what the grid detector's 6 dB rise asks."""
    if kind == "tom":
        main = [(at(i), tom_hit(f=150.0, tau=0.08)) for i in range(0, 32, 4)]
        bleed = [(at(i), noise_hit(500 + i, 0.01, -45.0, 0.1)) for i in range(32)]
        lines = [i for i in range(32) if i % 4 != 0]             # 16th lines with no tom within 50 ms
    else:
        main = [(at(i), tom_hit(f=45.0, tau=0.10)) for i in range(0, 32, 4)]
        bleed = [(at(i), noise_hit(500 + i, 0.05, -45.0, 0.3)) for i in range(2, 32, 4)]
        lines = [i for i in range(32) if i % 4 == 2]
    return stem(main + bleed, 2), lines


@pytest.mark.parametrize("kind", ["tom", "kick"])
def test_grid_onsets_on_a_tom_or_kick_stem_count_the_bleed_as_hits_and_say_so(kind):
    x, lines = bleed_stem(kind)
    off, on = run(x), run(x, grid_onsets=True)
    assert len(off["hits"]) == 8 and off["grid_hits"] == 0 and not off["grid_onsets"]       # the beats of the drum
    # with the search every bleed hit at a free line is a hit: 24 hat bleeds (12 of the 16 lines a bar) on the tom stem,
    # 8 burst bleeds (the offbeats) on the kick stem. The kick stem gives 8 more: the line after each kick, where
    # the kick's own 45 Hz tail is found as a hit, because the first-difference power of a low tone swings with its
    # phase by more than the 6 dB the search asks for (its 22 ms period against the 3 and 5 ms windows).
    grid = [h for h in on["hits"] if h["source"] == "grid"]
    expected = lines if kind == "tom" else sorted(lines + [i - 1 for i in lines])
    assert on["grid_onsets"] and on["grid_hits"] == len(grid) == len(expected) == len(on["hits"]) - 8, (on["grid_hits"], len(expected))
    assert sorted(h["bar"] * 16 + h["step"] for h in grid) == expected and all(abs(h["offset_ms"]) < 20 for h in grid)
    # and the stem is not hat-like, which is what the option is for: nearly all of its energy lies under 4 kHz
    assert off["hat_like"] < 0.01 and on["hat_like"] == off["hat_like"]
    assert any("meant for hat-like stems" in n and "hat_like 0.00" in n for n in on["notes"])
    assert not any("meant for hat-like stems" in n for n in off["notes"])


def test_grid_onsets_on_the_hat_stem_find_only_hats_and_the_stem_reads_hat_like(hat_stem):
    # the same stem as the first test, the search on: nothing is added (every 16th has a hit), and the stem is hat-like
    r = run(hat_stem, grid_onsets=True)
    assert r["grid_hits"] == 0 and len(r["hits"]) == 128 and r["hat_like"] > 0.9
    assert not any("meant for hat-like stems" in n for n in r["notes"])


# ---------------------------------------------------------------- pitch: toms an octave or a fifth apart

@pytest.mark.parametrize("semitones", [12.0, 7.0])
def test_two_toms_an_octave_or_a_fifth_apart_are_two_groups(semitones):
    # beats 1 and 3 a tom at 150 Hz, beats 2 and 4 the same drum `semitones` up: the spectral distance alone puts toms an
    # octave apart in one group (review: 0.65 / 1.06 / 1.55 for 7 / 12 / 19 semitones on a 100 Hz tom); their body pitches
    # are clusters 7 and 12 semitones apart
    hi = 150.0 * 2 ** (semitones / 12)
    rng = np.random.default_rng(4)
    ev = [(at(i), tom_hit(rng.uniform(-2, 2), f=150.0 if (i // 4) % 2 == 0 else hi)) for i in range(0, 48, 4)]
    r = run(stem(ev, 3))
    lo_g, hi_g = sorted(r["groups"], key=lambda g: g["pattern"].index(1.0))
    assert len(r["groups"]) == 2 and lo_g["count"] == hi_g["count"] == 6
    assert [s for s, p in enumerate(lo_g["pattern"]) if p] == [0, 8] and [s for s, p in enumerate(hi_g["pattern"]) if p] == [4, 12]
    assert lo_g["pattern"][0] == lo_g["pattern"][8] == hi_g["pattern"][4] == hi_g["pattern"][12] == 1.0


def test_one_tom_with_a_third_of_a_semitone_of_jitter_stays_one_group():
    rng = np.random.default_rng(5)
    ev = [(at(i), tom_hit(rng.uniform(-2, 2), f=150.0 * 2 ** (rng.uniform(-0.3, 0.3) / 12))) for i in range(0, 48, 4)]
    r = run(stem(ev, 3))
    assert len(r["groups"]) == 1 and r["groups"][0]["count"] == 12


# ---------------------------------------------------------------- stability, tentative groups

def test_well_separated_sounds_are_stable_and_a_decay_continuum_is_not():
    r = run(stem(hat_events(4), 4), stability_runs=2)
    assert [g["stability"] for g in r["groups"]] == [1.0, 1.0] and not any(g["tentative"] for g in r["groups"])
    # hats of tau 6 to 30 ms in 2 ms steps in a shuffled order: distances run on from 0.5 to 3 with no gap, so the
    # cut lies where the sample puts it and a regrouping with the sample moved does not find the same groups
    taus = np.random.default_rng(6).permutation(np.tile(np.arange(6, 31, 2) * 1e-3, 5))[:64]
    ev = [(at(i), noise_hit(i, float(tau))) for i, tau in enumerate(taus)]
    c = run(stem(ev, 4), stability_runs=2)
    measured = [g for g in c["groups"] if not g["tentative"]]
    assert measured and min(g["stability"] for g in measured) < 0.8, [(g["count"], g["stability"]) for g in c["groups"]]
    assert all(0.0 <= g["stability"] <= 1.0 for g in measured)
    assert all(g["stability"] is None for g in c["groups"] if g["tentative"])            # a tentative group has none
    # separation: the mean distance to the nearest other group's members. Separated sounds lie well over the threshold
    # from each other (closed against open 2.9 here, each the other's nearest); a continuum has groups that touch
    assert [g["nearest_group"] for g in r["groups"]] == [1, 0]
    assert all(g["separation"] > 1.4 * drumroles.THRESHOLD for g in r["groups"])
    assert min(g["separation"] for g in measured) < 1.4 * drumroles.THRESHOLD, [(g["count"], g["separation"]) for g in measured]
    assert all(g["nearest_group"] != g["id"] for g in c["groups"] if g["nearest_group"] is not None)
    # with no more measured hits than the sample the sample cannot move: no figure, a note, and no regrouping
    short = run(stem([(at(i), noise_hit(i, float(tau))) for i, tau in enumerate(taus[:32])], 2), stability_runs=2)
    assert all(g["stability"] is None for g in short["groups"]) and any("cannot move" in n for n in short["notes"])
    assert run(stem(hat_events(2), 2))["groups"][0]["stability"] is None             # not measured when not asked for


def test_a_group_of_under_4_hits_is_tentative_and_has_no_stability():
    # 30 short hats and 2 longer ones, a sample of 8 (it can move, 32 hits against 8): the pair is tentative, with no figure
    ev = [(at(i), noise_hit(i, 0.025 if i in (10, 26) else 0.005)) for i in range(32)]
    r = run(stem(ev, 2), sample=8, stability_runs=2)
    assert [(g["count"], g["tentative"]) for g in r["groups"]] == [(30, False), (2, True)]
    assert r["groups"][0]["stability"] is not None and r["groups"][1]["stability"] is None


# ---------------------------------------------------------------- review round 1: tests that pin the clustering

def test_a_decay_continuum_shuffled_three_times_gives_one_partition():
    # hats of tau 6 to 30 ms (distances 0.5 to 3, no gap): the groups depend on where the sample falls, so they only agree
    # when the hits are put in time order first and the same hits are sampled whatever order they came in
    taus = np.random.default_rng(8).permutation(np.tile(np.arange(6, 31, 2) * 1e-3, 3))[:32]
    cuts = [noise_hit(i, float(tau))[:int(STEP * SR), None] for i, tau in enumerate(taus)]

    def partition(order):
        groups, rest = drumroles.group_cuts([cuts[i] for i in order], [order[i] for i in range(len(order))], SR, sample=8)
        return sorted(sorted(order[i] for i in g["members"]) for g in groups), sorted(order[i] for i in rest)
    base = partition(list(range(32)))
    assert 1 < len(base[0]) < 8                              # the continuum is cut into a few groups, not one and not all
    for seed in (1, 2, 3):
        order = [int(i) for i in np.random.default_rng(seed).permutation(32)]
        assert partition(order) == base


def test_the_medoid_is_the_most_central_hit_not_the_first_and_the_outlier_is_the_furthest():
    # nine hats of one kind in one group, the earliest of them a little different (tau 20 ms against 12: about 1 away,
    # inside the threshold): the medoid is one of the others and the outlier is the member furthest from it
    cuts = [noise_hit(i, 0.02 if i == 0 else 0.012)[:int(STEP * SR), None] for i in range(9)]
    groups, rest = drumroles.group_cuts(cuts, list(range(9)), SR)
    g, = groups
    assert rest == [] and sorted(g["members"]) == list(range(9))
    assert g["medoid"] != 0 and g["distance"][0] == max(g["distance"].values()) and g["distance"][0] > 0.8


def test_a_hit_25_db_under_its_neighbours_is_faint_and_is_counted_where_it_fell():
    # a ghost note on a hat stem: 31 hats and one 25 dB under them (bar 0, step 13). It is not grouped (it may be a
    # bed or bleed), it is not dropped: the stem says where it fell (`unmeasured_pattern`) and why
    ev = [(at(i), noise_hit(i, 0.005, -25.0 if i == 13 else 0.0)) for i in range(32)]
    r = run(stem(ev, 2))
    ghost = r["hits"][13]
    assert ghost["status"] == "faint" and ghost["group"] is None and "ghost" in ghost["reason"]
    assert all(h["status"] == "grouped" for h in r["hits"] if h["id"] != 13) and r["groups"][0]["count"] == 31
    assert r["unmeasured_pattern"] == [0.5 if s == 13 else 0.0 for s in range(16)]
    assert r["unmeasured"] == [{"hit": 13, "status": "faint", "reason": ghost["reason"]}]
    assert r["groups"][0]["pattern"][13] == 0.5


def test_more_sounds_than_max_groups_leave_the_rest_ungrouped_and_say_so():
    # a hat on beats 1 and 3, a tom on beat 2, a snare on beat 4, 3 bars: three sounds, two groups allowed
    ev = []
    for b in range(3):
        ev += [(at(16 * b), noise_hit(b, 0.005)), (at(16 * b + 4), tom_hit()), (at(16 * b + 8), noise_hit(10 + b, 0.005)),
               (at(16 * b + 12), snare_hit(20 + b))]
    x = stem(ev, 3)
    r3 = run(x)
    assert len(r3["groups"]) == 3
    # separation: the tom (beat 2) and the snare (beat 4) are each other's nearest group (1.7 apart), the hats are 5.7 from
    # the tom: a group's nearest is the group with the nearest medoid, not the farthest
    by_step = {[st for st, p in enumerate(g["pattern"]) if p][0]: g for g in r3["groups"]}
    assert by_step[4]["nearest_group"] == by_step[12]["id"] and by_step[12]["nearest_group"] == by_step[4]["id"]
    assert by_step[0]["nearest_group"] in (by_step[4]["id"], by_step[12]["id"]) and by_step[0]["separation"] > 3 * by_step[4]["separation"]
    r = run(x, max_groups=2)
    left = [h for h in r["hits"] if h["status"] == "ungrouped"]
    assert len(r["groups"]) == 2 and left and "not forced" in left[0]["reason"]
    assert len({h["step"] for h in left}) <= 2 and {h["id"] for h in left}.isdisjoint(i for g in r["groups"] for i in g["members"])
    assert sum(g["count"] for g in r["groups"]) + len(left) == len(r["hits"]) == 12


def test_bars_0_to_3_alone_and_inside_16_bars_give_the_same_groups_on_separated_sounds():
    big = run(stem(hat_events(16), 16), sections=[{"name": "a", "start_bar": 0, "bars": 4}])
    small = run(stem(hat_events(4), 4))
    keep = lambda groups: [(g["count"], g["pattern"], g["bars_with"], g["off_grid"], g["tentative"]) for g in groups]  # noqa: E731
    assert keep(big["sections"][0]["groups"]) == keep(small["groups"])
    assert [g["count"] for g in big["groups"]] == [192, 64] and [g["count"] for g in small["groups"]] == [48, 16]


def test_an_overlapped_hit_between_two_alike_groups_is_not_assigned_and_one_next_to_a_group_is():
    # _assign_overlapped on hand-made hits: group A's medoid a hat of tau 5 ms, group B's of tau 7 ms (about 0.9 apart), the
    # overlapped hit one of tau 6 ms: it is within the threshold of both and not 15 % nearer to either, so it stays
    # overlapped, with the numbers in its reason; against a group of tau 40 ms (far) it goes to the group of tau 5 ms
    def hat(i, tau):
        return noise_hit(i, tau)[:int(STEP * SR)]
    parts = [hat(1, 0.005), hat(2, 0.007), hat(3, 0.006), hat(4, 0.04)]
    x = np.concatenate([np.concatenate([p, np.zeros(SR // 10)]) for p in parts])[:, None]
    starts = np.cumsum([0] + [len(p) + SR // 10 for p in parts])[:4]
    hits = [{"id": i, "start": int(a), "end": int(a) + len(p), "status": "grouped", "reason": None, "lead_db": None,
             "group": None, "distance": None} for i, (a, p) in enumerate(zip(starts, parts))]
    hits[2].update(status="overlapped", reason="it starts out of a tail or bed only 20 dB under its own loudest ms (it needs 30)",
                   lead_db=-20.0)

    def assign(medoids):
        hits[2].update(group=None, distance=None)
        hits[2]["status"], hits[2]["reason"] = "overlapped", "it starts out of a tail or bed only 20 dB under its own loudest ms (it needs 30)"
        found = [(gid, {"medoid": m, "members": [m] * drumroles.TENTATIVE_N, "assigned": []}) for gid, m in enumerate(medoids)]
        drumroles._assign_overlapped(x, SR, hits, found, drumroles.THRESHOLD)
        return hits[2]["status"], hits[2]["reason"], found
    status, reason, _ = assign([0, 1])
    assert status == "overlapped" and "not assigned" in reason and "groups 0 and 1" in reason, reason
    status, reason, found = assign([0, 3])
    assert status == "assigned" and hits[2]["group"] == 0 and found[0][1]["assigned"] == [2] and "first 40 ms" in reason, reason


def test_a_lone_hit_on_a_tail_does_not_attract_the_hits_of_a_real_group():
    # a real group (6 clean hats, medoid 0, rose out of silence) and a lone hit heard over a noise tail 36 dB under it;
    # two overlapped hats over the same kind of tail, 29 dB under them. Their tail makes them nearer the lone hit than
    # the clean group, which is the tail's doing: a group of fewer than 4 hits takes assigned hits only when its medoid
    # rose out of near silence (-40 dB or under), so the real group gets them; with a clean medoid the lone one takes them
    n = int(STEP * SR)
    tail = noise_hit(77, 0.12)[:n]

    def over(seed, db):
        h = noise_hit(seed, 0.012)[:n]
        g = oneshot._loudest_ms(h[:, None], SR) * 10 ** (db / 20) / np.sqrt((tail[:int(0.01 * SR)] ** 2).mean())
        return h + g * tail
    parts = [noise_hit(i, 0.012)[:n] for i in range(1, 7)] + [over(10, -36.0), over(11, -29.0), over(12, -29.0)]
    x = np.concatenate([np.concatenate([p, np.zeros(SR // 10)]) for p in parts])[:, None]
    starts = np.cumsum([0] + [len(p) + SR // 10 for p in parts])[:len(parts)]
    d_lone = drumroles._pair_distance(x[starts[7]:starts[7] + n], x[starts[6]:starts[6] + n], SR)
    d_real = drumroles._pair_distance(x[starts[7]:starts[7] + n], x[starts[0]:starts[0] + n], SR)
    assert d_lone < d_real < drumroles.THRESHOLD, (d_lone, d_real)          # the premise: nearer the tail, still within reach

    def assign(lone_lead):
        hits = [{"id": i, "start": int(a), "end": int(a) + n, "status": "grouped", "reason": None, "lead_db": -90.0,
                 "group": None, "distance": None} for i, a in enumerate(starts)]
        hits[6]["lead_db"] = lone_lead
        for i in (7, 8):
            hits[i].update(status="overlapped", lead_db=-29.0,
                           reason="it starts out of a tail or bed only 29 dB under its own loudest ms (it needs 30)")
        found = [(0, {"members": list(range(6)), "medoid": 0, "assigned": []}), (1, {"members": [6], "medoid": 6, "assigned": []})]
        drumroles._assign_overlapped(x, SR, hits, found, drumroles.THRESHOLD)
        return [g["assigned"] for _, g in found], [hits[i]["status"] for i in (7, 8)]
    assert assign(-36.0) == ([[7, 8], []], ["assigned", "assigned"])        # the lone hit sits on a tail: the real group
    assert assign(-45.0) == ([[], [7, 8]], ["assigned", "assigned"])        # a lone hit from near silence is a candidate


def test_a_hat_of_tau_60_ms_with_a_rest_sounds_through_two_lines():
    # -17.5 and -35 dB at the next two 16th lines, -52 at the third: two lines inside 40 dB (a tau 30 ms hat has one,
    # -35 then -70)
    r = run(stem([(at(i), noise_hit(i, 0.06)) for i in range(2, 32, 4)], 2))
    assert [h["through"] for h in r["hits"]] == [2] * 8
    r = run(stem([(at(i), noise_hit(i, 0.03)) for i in range(2, 32, 4)], 2))
    assert [h["through"] for h in r["hits"]] == [1] * 8
