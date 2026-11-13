"""sideman.patch's Live renderer (LiveRenderer, OperatorRenderer, repeat_check, patch_match) against a FAKE client.

No socket and no Live: `FakeLive` records every request it receives and, when asked for the recording's file_path,
writes a WAV that it synthesises itself from what it was sent: the notes of the MIDI clip, the wall time the clip
was fired at and the parameter writes, each note played by the stand-in synth of tests/test_patch.py with the
parameters last written before that note. So the tests prove the renderer's bookkeeping (what is written when, which
slot, which solo state, how a take is cut into hits and what a failure returns), not anything about Live: its timing,
its launch quantization, whether `loop_end` and `end_marker` can be set on a session clip, or how Operator sounds.

Needs the `listen` extra.
"""
import functools
import itertools
import math
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sideman import oneshot, patch  # noqa: E402
from test_patch import SR, TARGET_SEED, TRUE, profile_of, standin, standin_from_live  # noqa: E402

DEV, CLIP, REC = "live_set tracks 1 devices 0", "live_set tracks 1 clip_slots 0 clip", "live_set tracks 2"
HIT_S = 0.25                       # the stand-in's hits are SECONDS = 0.3 s long
UNITS = patch.operator_units()
NAMES = list(patch.operator_parameters())


class Clock:
    """A fake wall clock: sleeping moves it, nothing else does (the fake client adds a latency per request)."""

    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += max(0.0, s)


class FakeLive:
    """Every request costs `latency` +- `jitter` seconds of the fake clock (Live: 0.4 to 0.7 s, whatever the request
    holds), or uniform(lo, hi) when those are given, `latency_in_take` instead while the clip is playing (the engine is
    busier), a spike of `spike` seconds with probability `p_spike`, and `poll_spike` once on the first sync poll that
    finds the clip playing. Its effect lands `handle_frac` of the way through the request (0.5: the middle), except a
    parameter write, which lands `write_frac` of the way (1.0: only when the request returns, the worst case the
    renderer plans for). Clips start on the next multiple of `quantize_beats` (launch quantization), the record slot and
    the clip alike (the clip `clip_late_beats` later when asked: the two fires straddling a grid step).
    Tracks: `names` (index order; the scratch tracks are 1 "OP" with the Operator and 2 "REC"), per-track solo and arm
    (`exclusive_arm`: arming one disarms the others), `insert_track(at)` (a track inserted by the user, which moves
    every path below it), `insert_on_fire` (do that when the clip is fired), `record_mode`, `flip_playing_at_state`
    (the Set starts playing between the first guard and the state batch), `fail_setup`, `fail_restore`,
    `clip_length_cap` (loop_end and end_marker cannot lengthen the clip past it)."""

    def __init__(self, tmp_path, clock, *, tempo=120.0, ntracks=4, nslots=8, filled=(), playing=False, solo=(), armed=True,
                 fresh_noise=False, latency=0.6, jitter=0.15, lo=None, hi=None, p_spike=0.0, spike=1.5, poll_spike=None,
                 latency_in_take=None, handle_frac=0.5, write_frac=1.0, quantize_beats=4.0, clip_late_beats=0.0,
                 never_plays=False, drop_notes=(), cut_last_ms=None, fail_write=(), raise_on_file=False, first_seed=1000,
                 seed=1, names=None, armed_tracks=(), exclusive_arm=False, record_mode=False, insert_on_fire=None,
                 flip_playing_at_state=False, fail_setup=False, fail_restore=False, clip_length_cap=None):
        self.tmp, self.clock, self.tempo, self.latency, self.jitter = tmp_path, clock, tempo, latency, jitter
        self.lo, self.hi, self.p_spike, self.spike, self.poll_spike, self._next = lo, hi, p_spike, spike, poll_spike, None
        self.latency_in_take, self.handle_frac, self.write_frac = latency_in_take, handle_frac, write_frac
        self.never_plays = never_plays
        self.quantize_beats, self.clip_late_beats, self.rng = quantize_beats, clip_late_beats, np.random.default_rng(seed)
        self.ntracks, self.nslots, self.filled, self.playing = ntracks, nslots, set(filled), playing
        self.names = list(names) if names is not None else ["T%d" % i for i in range(ntracks)]
        if names is None:
            self.names[1], self.names[2] = "OP", "REC"
        self.solo = [i in solo for i in range(ntracks)]
        self.arms = [i in armed_tracks for i in range(ntracks)]
        self.arms[2] = bool(armed)
        self.exclusive_arm, self.record_mode, self.insert_on_fire = exclusive_arm, record_mode, insert_on_fire
        self.flip_playing_at_state, self.fail_setup, self.fail_restore = flip_playing_at_state, fail_setup, fail_restore
        self.clip_length_cap = clip_length_cap
        self.fresh_noise, self.drop_notes, self.cut_last_ms = fresh_noise, set(drop_notes), cut_last_ms
        self.fail_write, self.raise_on_file, self.seeds = set(fail_write), raise_on_file, itertools.count(first_seed)
        self.calls, self.writes, self.notes = [], [], []        # writes: (time it lands, {param index: Live value})
        self.clip_t0 = self.clip_t0_used = self.rec_t0 = self.rec_t1 = None
        self.clip_length, self.recorded, self.files, self.in_take = 4.0, {}, 0, False
        self.deleted, self.transactions = [], 0

    @property
    def armed(self):
        return self.arms[2]

    def insert_track(self, at):
        self.names.insert(at, "NEW")
        self.solo.insert(at, False)
        self.arms.insert(at, False)
        self.ntracks += 1

    # ---- the wire
    def _tick(self, *what):
        """Advance the clock by this request's duration; return the time its effect lands."""
        base = self.latency_in_take if (self.in_take and self.latency_in_take is not None) else self.latency
        if self.lo is not None:
            lat = self.rng.uniform(self.lo, self.hi)
        else:
            lat = max(0.02, base + (self.rng.uniform(-self.jitter, self.jitter) if self.jitter else 0.0))
        if self.p_spike and self.rng.uniform() < self.p_spike:
            lat = self.spike
        if self._next is not None:
            lat, self._next = self._next, None
        t0 = self.clock.now
        self.clock.now = t0 + lat
        self._write_time = t0 + self.write_frac * lat
        handled = t0 + self.handle_frac * lat
        self.calls.append((handled,) + what)
        return handled

    def _boundary(self, t):
        q = self.quantize_beats * 60.0 / self.tempo
        return math.ceil(t / q) * q if q else t

    def _quant_enum(self):
        return min(range(len(patch.QUANT_BEATS)), key=lambda k: abs(patch.QUANT_BEATS[k] - self.quantize_beats))

    def get(self, path, prop, **kw):
        self._tick("get", path, prop)
        if (path, prop) == ("live_set", "is_playing"):
            return {"value": self.playing}
        if (path, prop) == ("live_set", "tempo"):
            return {"value": self.tempo}
        if (path, prop) == ("live_set", "tracks"):
            return {"value": {"count": self.ntracks}}
        if (path, prop) == (REC, "clip_slots"):
            return {"value": {"count": self.nslots}}
        if prop == "file_path" and path.startswith(REC + " clip_slots "):
            if self.raise_on_file:
                raise RuntimeError("no file")
            return {"value": self._file(int(path.split()[-2]))}
        raise AssertionError("unexpected get %r %r" % (path, prop))

    def get_batch(self, specs):
        before = self.clock.now
        if self.flip_playing_at_state and any(s["property"] == "record_mode" for s in specs):
            self.playing, self.flip_playing_at_state = True, False
        at = self._tick("get_batch", tuple((s["path"], s["property"]) for s in specs))
        if self.poll_spike is not None and specs and specs[0].get("path") == CLIP and self.clip_t0 is not None and at >= self.clip_t0 \
                and self.playing and not self.never_plays:
            self.clock.now, self.poll_spike = before + self.poll_spike, None      # the first poll that finds the clip playing is slow
        out = []
        for s in specs:
            path, prop = s["path"], s["property"]
            ok = lambda v: {"ok": True, "value": v}  # noqa: E731
            if prop == "name" and path.startswith("live_set tracks ") and " clip_slots " not in path:
                out.append(ok(self.names[int(path.split()[-1])]))
            elif prop == "name":
                out.append(ok("take %s" % path.split()[-2]))
            elif prop == "class_name":
                out.append(ok("Operator" if self.names[int(path.split()[2])] == "OP" else "Other"))
            elif prop == "solo":
                out.append(ok(self.solo[int(path.split()[-1])]))
            elif prop == "arm":
                out.append(ok(self.arms[int(path.split()[-1])]))
            elif (path, prop) == ("live_set", "is_playing"):
                out.append(ok(self.playing))
            elif (path, prop) == ("live_set", "record_mode"):
                out.append(ok(self.record_mode))
            elif (path, prop) == ("live_set", "clip_trigger_quantization"):
                out.append(ok(self._quant_enum()))
            elif prop == "has_clip":
                k = int(path.split()[-1])
                out.append(ok(k in self.filled or k in self.recorded))
            elif path == CLIP and prop == "is_playing":
                out.append(ok(self.clip_t0 is not None and self.playing and at >= self.clip_t0 and not self.never_plays))
            elif path == CLIP and prop == "playing_position":
                started = self.clip_t0 is not None and at >= self.clip_t0
                out.append(ok((at - self.clip_t0) / (60.0 / self.tempo) if started else 0.0))
            elif (path, prop) == (CLIP, "length"):
                out.append(ok(self.clip_length))
            else:
                raise AssertionError("unexpected batch read %r %r" % (path, prop))
        return {"results": out}

    def _apply(self, path, prop, value, lands):
        if path.startswith(DEV + " parameters "):
            return int(path.split()[-1]), value
        if prop == "solo":
            self.solo[int(path.split()[-1])] = bool(value)
        elif prop == "arm":
            i = int(path.split()[-1])
            if value and self.exclusive_arm:
                self.arms = [False] * len(self.arms)
            self.arms[i] = bool(value)
        elif path == CLIP and prop in ("loop_end", "end_marker"):
            self.clip_length = float(value) if self.clip_length_cap is None else min(float(value), self.clip_length_cap)
        elif not (path == CLIP and prop == "looping"):
            raise AssertionError("unexpected set %r %r" % (path, prop))
        return None

    def call(self, path, function, args=None, confirm=False, **kw):
        at = self._tick("call", path, function)
        if function == "fire" and path.startswith(REC + " clip_slots "):
            self.rec_t0, self.rec_t1, self.playing = self._boundary(at), None, True
            self.rec_slot = int(path.split()[-1])
        elif function == "fire" and path == "live_set tracks 1 clip_slots 0":
            self.clip_t0 = self.clip_t0_used = self._boundary(at) + self.clip_late_beats * 60.0 / self.tempo
            self.in_take = True
            if self.insert_on_fire is not None:
                self.insert_track(self.insert_on_fire)
                self.insert_on_fire = None
        elif function == "stop" and path.startswith(REC + " clip_slots "):
            self.rec_t1 = at
            self.recorded[self.rec_slot] = True
        elif function == "stop_playing":
            self.playing, self.clip_t0, self.in_take = False, None, False
        elif function == "delete_clip":
            self.deleted.append((path, kw.get("expect"), confirm))
            self.filled.discard(int(path.split()[-1]))
            self.recorded.pop(int(path.split()[-1]), None)
        else:
            raise AssertionError("unexpected call %r %r" % (path, function))
        return {}

    def transaction(self, ops, **kw):
        self._tick("transaction", tuple((o["path"], o["property"]) for o in ops))
        self.transactions += 1
        params, results = {}, []
        is_restore = any(o["property"] in ("solo", "arm") for o in ops) and not any(o["path"] == CLIP for o in ops)
        if self.fail_restore and is_restore:
            return {"results": [{"ok": False, "error": "refused"} for _ in ops]}                       # nothing applied
        for o in ops:
            got = self._apply(o["path"], o["property"], o["value"], self._write_time)
            if got:
                params[got[0]] = got[1]
            results.append({"ok": True})
        if self.fail_setup and any(o["path"] == CLIP for o in ops):
            results[0] = {"ok": False, "error": "refused"}
        if params:
            k = len(self.writes)
            self.writes.append((self._write_time, params))
            if k in self.fail_write:
                first = len(ops) - len(params)
                results[first] = {"ok": False, "error": "refused"}
        return {"results": results}

    def notes_remove(self, path, **w):
        self._tick("notes_remove", path)
        self.notes = []

    def notes_add(self, path, notes):
        self._tick("notes_add", path, len(notes))
        self.notes = list(notes)

    # ---- the recording: the stand-in, played with the parameters in force at each note
    def _device_at(self, wall):
        state = {}
        for t, w in self.writes:
            if t <= wall:
                state.update(w)
        return state

    def _file(self, slot):
        spb = 60.0 / self.tempo
        end = self.rec_t1 if self.rec_t1 is not None else self.clock.now
        last = max(self.clip_t0_used + n["start_time"] * spb for n in self.notes) if self.notes else end
        if self.cut_last_ms is not None:
            end = min(end, last + self.cut_last_ms / 1000.0)
        x = np.zeros(max(0, int(round((end - self.rec_t0) * SR))))
        for k, note in enumerate(self.notes):
            if k in self.drop_notes:
                continue
            wall = self.clip_t0_used + note["start_time"] * spb
            dev = self._device_at(wall)
            live = {NAMES[i]: v for i, v in dev.items()}
            phys = patch.from_live(live, UNITS)
            p = standin_from_live({"params": phys, "note": note["pitch"]})
            seed = next(self.seeds) if self.fresh_noise else 5
            hit = standin(p, seed)
            a = int(round((wall - self.rec_t0) * SR))
            b = min(len(x), a + len(hit))
            if b > a:
                x[a:b] += hit[:b - a]
        self.files += 1
        path = self.tmp / ("take_%d.wav" % self.files)
        sf.write(str(path), x, SR, subtype="FLOAT")
        return str(path)


def make(tmp_path, **kw):
    tmp_path.mkdir(exist_ok=True)
    clock = Clock()
    fake = FakeLive(tmp_path, clock, **kw)
    return fake, clock


def guess_of(kind="kick"):
    return patch.operator_guess(profile_of(TRUE[kind], 1))


def candidates(g, n=6):
    """n patches that differ in the tone decay (10 % apart) and the carrier: distinct hits, distinct notes."""
    return [dict(g["patch"], tone_decay_ms=g["patch"]["tone_decay_ms"] * (1.0 + 0.1 * k),
                 carrier_hz=g["patch"]["carrier_hz"] * 2 ** (k / 12.0)) for k in range(n)]


def renderer(fake, clock, g, **kw):
    kw.setdefault("hit_s", HIT_S)
    return patch.OperatorRenderer(fake, DEV, CLIP, REC, g["structure"], sleep=clock.sleep, clock=clock.time, **kw)


def dist(a, b):
    return oneshot.spectral_distance(a, b, SR)


# ---- batch mode: one take, parameters in time, notes, order, equal to single mode

def test_batch_mode_is_one_take_with_the_parameters_written_in_time_and_the_notes_placed(tmp_path):
    g = guess_of()
    ps = candidates(g)
    fake, clock = make(tmp_path)
    r = renderer(fake, clock, g)
    hits = r.render(ps)
    assert len(hits) == len(ps) and all(len(h) > int(0.9 * HIT_S * SR) for h in hits)
    fires = [c for c in fake.calls if c[1] == "call" and c[3] == "fire"]
    assert [c[2] for c in fires] == ["%s clip_slots 0" % REC, "live_set tracks 1 clip_slots 0"] and len(r.takes) == 1
    take, spb = r.takes[0], 0.5
    live = [patch.operator_live(p, g["structure"]) for p in ps]
    # the notes: the right pitches, the first at start_beat, then the take's spacing apart; the spacing leaves room for
    # the longest hit and the margin (rule: (max(hit_s, longest note) + margin) rounded up to 1/16 beat)
    assert [n["pitch"] for n in fake.notes] == [L["note"] for L in live]
    starts = [n["start_time"] for n in fake.notes]
    assert starts[0] == take["start_beat"] and np.allclose(np.diff(starts), take["spacing_beats"])
    longest = max(L["note_length_ms"] for L in live) / 1000.0
    need = max(longest + take["lead_s"] + take["margin_s"], HIT_S + take["margin_s"])          # the spacing rule
    assert need <= take["spacing_beats"] * spb < need + spb / 16 + 1e-9
    assert all(abs(n["duration"] * spb - L["note_length_ms"] / 1000.0) < 1e-6 for n, L in zip(fake.notes, live))
    # the parameters: write k holds candidate k's values, landing (at the return of its request, the worst case) at least
    # min_lead_s before its note and after the previous note ended
    assert len(fake.writes) == len(ps)
    for k, (t, w) in enumerate(fake.writes):
        assert w[NAMES.index("Ae Decay")] == pytest.approx(patch.to_live(live[k]["params"], UNITS)["Ae Decay"])
        note_wall = fake.clip_t0_used + starts[k] * spb
        if k:
            assert note_wall - t >= r.min_lead_s
            assert t >= fake.clip_t0_used + starts[k - 1] * spb + live[k - 1]["note_length_ms"] / 1000.0
        assert take["candidates"][k]["ok"] and take["candidates"][k]["lead_ms"] >= 1000 * r.min_lead_s
        assert abs(take["candidates"][k]["offset_ms"]) < 5.0                    # the stand-in's onset is the note's
    assert take["sample_rate"] == SR and r.sample_rate == SR and take["slot"] == 0 and abs(take["offset_median_ms"]) < 5.0 and abs(take["offset_trend_ms"]) < 5.0


def test_hits_come_back_in_candidate_order_and_equal_the_single_mode_hits(tmp_path):
    g = guess_of()
    ps = candidates(g, 5)
    fake, clock = make(tmp_path)
    batch = renderer(fake, clock, g).render(ps)
    fake2, clock2 = make(tmp_path / "single")
    single_r = renderer(fake2, clock2, g, batch=False)
    single = single_r.render(ps)
    assert len(single_r.takes) == len(ps)                    # one take per candidate
    assert [t["slot"] for t in single_r.takes] == list(range(len(ps)))             # each in the next empty slot
    for i, (a, b) in enumerate(zip(batch, single)):
        assert dist(a, b) < 0.01, i                          # the same hit...
        for j in range(len(ps)):
            if j != i:
                assert dist(a, single[j]) > 10 * max(dist(a, b), 1e-4), (i, j)       # ...and not a neighbour's


# ---- what fails, and only that candidate

def test_a_missing_hit_fails_only_its_candidate_with_a_reason(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, drop_notes={2})
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 5))
    assert [len(h) > 0 for h in hits] == [True, True, False, True, True]
    c = r.takes[0]["candidates"][2]
    assert not c["ok"] and "no hit near" in c["reason"] and r.takes[0]["failures"][0][0] == 2


def test_a_take_cut_short_fails_the_hits_it_cut_and_no_other(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, cut_last_ms=100.0)         # the file ends 100 ms into the last hit; HIT_S = 250 ms is needed
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 4))
    assert [len(h) > 0 for h in hits] == [True, True, True, False]
    assert "take ends" in r.takes[0]["candidates"][3]["reason"]


def test_a_failed_parameter_write_fails_only_its_candidate_and_never_returns_the_audio_it_got(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, fail_write={2})
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 5))
    assert [len(h) > 0 for h in hits] == [True, True, False, True, True]
    assert "parameter writes failed" in r.takes[0]["candidates"][2]["reason"]
    assert r.takes[0]["candidates"][3]["ok"]                 # the next candidate wrote its own parameters over the failure


def test_requests_at_0_6_s_with_jitter_give_every_candidate_of_a_batch_of_16_a_positive_lead_above_the_minimum(tmp_path):
    """Every request costs 0.6 +- 0.15 s of the fake clock, as in Live, and a write lands only when its request
    returns. During the take the renderer makes ONE request per candidate (the write), plus the sync polls."""
    g = guess_of()
    fake, clock = make(tmp_path)
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 16))
    take = r.takes[0]
    assert all(len(h) for h in hits) and not take["failures"], take["failures"]
    leads = [c["lead_ms"] for c in take["candidates"][1:]]
    assert min(leads) >= 1000 * r.min_lead_s and min(leads) > 100.0, leads        # in practice 0.2 to 0.5 s before the note
    assert 0.5 < take["request_s"] < 0.76 and take["lead_s"] == pytest.approx(max(1.5 * take["request_s"], take["request_s"] + 0.5))
    assert take["lead_raised_s"] == 0.0 and r.lead_extra_s == 0.0
    # the requests between the clip's fire and the stop of the record slot: sync polls and one transaction per later candidate
    kinds = [c[1] for c in fake.calls]
    a = max(k for k, c in enumerate(fake.calls) if c[1] == "call" and c[3] == "fire")
    b = min(k for k, c in enumerate(fake.calls) if c[1] == "call" and c[3] == "stop")
    inside = kinds[a + 1:b]
    assert inside.count("transaction") == 15 and set(inside) == {"transaction", "get_batch"}          # + the sync polls and the re-read
    assert inside.count("get_batch") <= 8
    assert not [c for c in fake.calls[a + 1:b] if c[1] == "get" ]
    assert [t[0] for t in fake.writes][0] < fake.clip_t0_used                      # candidate 0 was written before the clip started


@pytest.mark.parametrize("handle_frac", [0.0, 0.5, 1.0])
def test_the_hits_are_the_candidates_own_wherever_in_its_request_live_reads_the_clip_position(tmp_path, handle_frac):
    """The clip position is read in a request that lasts 0.6 s: Live took it at the start, the middle or the end of it.
    Whichever, each hit must be its candidate's (equal to the same candidate's single-mode hit). The sync takes the
    send time of the request (the earliest reading): the middle failed at frac 0.0 with candidate 1 playing the
    previous candidate's parameters."""
    g = guess_of()
    ps = candidates(g, 8)
    fake, clock = make(tmp_path, handle_frac=handle_frac)
    r = renderer(fake, clock, g)
    batch = r.render(ps)
    ref_fake, ref_clock = make(tmp_path / "ref", latency=0.01, jitter=0.0)
    ref = renderer(ref_fake, ref_clock, g, batch=False).render(ps)
    assert not r.takes[0]["failures"]
    for i, (a, b) in enumerate(zip(batch, ref)):
        assert dist(a, b) < 0.01, (i, dist(a, b), r.takes[0]["candidates"][i])


def test_requests_that_slow_down_during_the_take_fail_the_first_batch_on_timing_and_the_next_batch_adapts(tmp_path):
    """Requests cost 0.6 s while the guards measure them and 1.5 s once the clip plays (a busier engine): the lead of
    1.1 s is shorter than one write, and the sync poll (1.5 s) is longer than the margin (0.75 s). Nearly the whole
    first batch fails with a reason (late writes, and writes that could have landed in the previous hit), no wrong
    audio comes back; the renderer raises its lead by the shortfall, its margin to the poll it saw, and says so; the
    next batch is rendered."""
    g = guess_of()
    fake, clock = make(tmp_path, latency=0.6, latency_in_take=1.5, jitter=0.0)
    r = renderer(fake, clock, g)
    first = r.render(candidates(g, 8))
    t0 = r.takes[0]
    assert sum(len(h) > 0 for h in first) == 0 and all(c["reason"] for c in t0["candidates"])
    assert any("ms before the note" in c["reason"] for c in t0["candidates"]) and any("could have ended" in c["reason"] for c in t0["candidates"])
    assert t0["sync_poll_s"] == pytest.approx(1.5) and t0["sync_repolled"] is True
    assert t0["lead_raised_s"] == pytest.approx(r.min_lead_s + (1.5 - t0["lead_s"]) + 0.1, abs=0.01)
    assert r.lead_extra_s == t0["lead_raised_s"]
    second = r.render(candidates(g, 8))
    t1 = r.takes[1]
    assert all(len(h) for h in second) and not t1["failures"] and t1["lead_extra_s"] == t0["lead_raised_s"]
    assert t1["lead_s"] > t0["lead_s"] + 0.4 and t1["margin_s"] >= 1.5 and t1["spacing_beats"] > t0["spacing_beats"]
    assert t1["lead_raised_s"] == 0.0 and r.lead_extra_s == pytest.approx(t0["lead_raised_s"] / 2)      # a clean take: the extra halves


def test_requests_at_1_5_s_throughout_are_measured_and_the_lead_follows_them(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, latency=1.5, jitter=0.1)
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 8))
    take = r.takes[0]
    assert all(len(h) for h in hits) and take["request_s"] > 1.3 and take["lead_s"] == pytest.approx(1.5 * take["request_s"])
    assert min(c["lead_ms"] for c in take["candidates"][1:]) > 300.0


def test_a_clip_that_starts_a_grid_step_after_the_recording_is_found_by_the_shift_of_its_first_onset(tmp_path):
    """The two fires are two requests apart, so launch quantization can start the record slot one grid step (4 beats,
    2 s) before the clip: the file has 2 s extra lead. The hits are cut from where the notes sound, and are the same
    hits as without the shift."""
    g = guess_of()
    ps = candidates(g, 5)
    fake, clock = make(tmp_path, clip_late_beats=4.0)
    r = renderer(fake, clock, g)
    hits = r.render(ps)
    take = r.takes[0]
    assert take["file_shift_s"] >= 1.95 and take["file_shift_s"] % 2.0 == pytest.approx(0.0, abs=0.05) and not take["failures"]
    straight_fake, straight_clock = make(tmp_path / "s")
    straight = renderer(straight_fake, straight_clock, g).render(ps)
    for a, b in zip(hits, straight):
        assert dist(a, b) < 0.01


def test_a_clip_that_never_plays_raises_with_solo_arm_and_transport_restored(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={3}, armed=False, never_plays=True)
    with pytest.raises(RuntimeError, match="did not start playing"):
        renderer(fake, clock, g).render(candidates(g, 3))
    assert fake.solo == [False, False, False, True] and fake.armed is False and fake.playing is False


def test_a_candidate_outside_the_units_table_fails_without_a_note(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    r = renderer(fake, clock, g)
    ps = candidates(g, 3)
    ps[1] = dict(ps[1], tone_decay_ms=1e9)                   # far outside the table's range
    hits = r.render(ps)
    assert [len(h) > 0 for h in hits] == [True, False, True]
    assert "does not convert" in r.takes[0]["candidates"][1]["reason"] and len(fake.notes) == 2


# ---- guards

def test_a_playing_set_is_refused_before_anything_is_touched(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, playing=True, solo={0})
    with pytest.raises(RuntimeError, match="playing"):
        renderer(fake, clock, g).render(candidates(g, 2))
    assert not [c for c in fake.calls if c[1] in ("set", "call", "notes_add", "transaction")]


def test_solo_and_arm_are_restored_and_the_transport_stopped_after_a_take(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={0, 3}, armed=False)
    renderer(fake, clock, g).render(candidates(g, 3))
    assert fake.solo == [True, False, False, True] and fake.armed is False and fake.playing is False
    solos = [(p, prop) for c in fake.calls if c[1] == "transaction" for p, prop in c[2] if prop == "solo"]
    assert {p for p, _ in solos} == {REC}                  # only the record track: the two tracks that were soloed are not written again
    assert [c for c in fake.calls if c[1] == "call"][-1][3] == "stop_playing"          # the transport is the last thing stopped
    touched = {c[2] for c in fake.calls if c[1] in ("call", "notes_add", "notes_remove")}
    assert all(t.startswith(("live_set tracks 1 ", REC)) or t in ("live_set", "live_set tracks 1") for t in touched), touched
    assert {p for c in fake.calls if c[1] == "transaction" for p, _ in c[2] if not p.startswith(DEV)} <= {REC, CLIP}


def test_solo_and_arm_are_restored_and_the_transport_stopped_when_a_take_fails(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={2}, armed=False, raise_on_file=True)
    with pytest.raises(RuntimeError, match="no recording"):
        renderer(fake, clock, g).render(candidates(g, 3))
    assert fake.solo == [False, False, True, False] and fake.armed is False and fake.playing is False


def test_the_solo_of_another_track_is_restored_if_live_unsoloes_it(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={0})
    original = fake._apply

    def exclusive(path, prop, value, lands):                 # Live's exclusive solo: soloing one track clears the others
        if prop == "solo" and value and path == REC:
            fake.solo = [False] * fake.ntracks
        return original(path, prop, value, lands)
    fake._apply = exclusive
    renderer(fake, clock, g).render(candidates(g, 2))
    assert fake.solo == [True, False, False, False]


def test_the_next_empty_slot_is_used_and_running_out_raises_before_recording(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, filled={0, 1})
    r = renderer(fake, clock, g)
    r.render(candidates(g, 2))
    assert r.takes[0]["slot"] == 2
    r.render(candidates(g, 2))
    assert r.takes[1]["slot"] == 3
    full, clock2 = make(tmp_path / "full")
    full.filled = set(range(full.nslots))
    with pytest.raises(RuntimeError, match="no empty clip slot"):
        renderer(full, clock2, g).render(candidates(g, 2))
    assert not [c for c in full.calls if c[1] in ("set", "call", "notes_add", "transaction")]
    rr = renderer(full, clock2, g)
    with pytest.raises(ValueError, match="confirm"):
        rr.clear_takes()
    assert rr.clear_takes(confirm=True) == full.nslots and not full.filled


def test_the_constructor_and_plan_refuse_what_cannot_be_scheduled(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    with pytest.raises(ValueError, match="lead_factor"):
        renderer(fake, clock, g, lead_factor=0.5)
    r = renderer(fake, clock, g, spacing_beats=1.0)
    with pytest.raises(ValueError, match="this take needs"):
        r.plan([0.8], 0.6, 120.0)
    plan = renderer(fake, clock, g).plan([0.847], 0.6, 124.0)                # a kick, hit_s = 0.25 here: lead 1.1 s, margin 0.75 s
    assert plan["lead_s"] == pytest.approx(1.1) and plan["margin_s"] == pytest.approx(0.75)
    assert plan["spacing_s"] >= 0.847 + 1.1 + 0.75 and plan["spacing_s"] < 0.847 + 1.1 + 0.75 + 60 / 124 / 16 + 1e-9


# ---- review round 1: identity, the file's shift, the early side of the schedule, aborts, restores

def mark(fake):
    return len(fake.calls)


def writes_since(fake, k):
    """The requests after call number k that change anything."""
    return [c for c in fake.calls[k:] if c[1] in ("call", "notes_add", "notes_remove", "transaction")]


def raising_once(fake, method, when, exc):
    """Make fake.<method> raise `exc` the first time `when(*args, **kwargs)` holds (before the request does anything)."""
    original, state = getattr(fake, method), {"done": False}

    def wrapper(*a, **k):
        if not state["done"] and when(*a, **k):
            state["done"] = True
            raise exc
        return original(*a, **k)
    setattr(fake, method, wrapper)


def test_a_track_inserted_above_the_scratch_tracks_between_takes_is_refused_with_no_write(tmp_path):
    """Paths are positions: after an insertion `live_set tracks 2` is another track. The take must be refused from the
    names and the device class read in its state batch, before any note, parameter, solo or arm is written."""
    g = guess_of()
    fake, clock = make(tmp_path, solo={0})
    r = renderer(fake, clock, g)
    r.render(candidates(g, 2))
    fake.insert_track(0)
    k = mark(fake)
    with pytest.raises(RuntimeError, match="not where they were"):
        r.render(candidates(g, 2))
    assert not writes_since(fake, k) and fake.solo == [False, True, False, False, False]
    # a rename counts too, and the identity can be given instead of captured
    fake2, clock2 = make(tmp_path / "b")
    r2 = patch.OperatorRenderer(fake2, DEV, CLIP, REC, guess_of()["structure"], sleep=clock2.sleep, clock=clock2.time, hit_s=HIT_S,
                                expect={"record": "REC", "midi": "OP", "device": "Operator"})
    assert len(r2.render(candidates(g, 2))[0])
    fake2.names[2] = "REC (old)"
    k = mark(fake2)
    with pytest.raises(RuntimeError, match="not where they were"):
        r2.render(candidates(g, 2))
    assert not writes_since(fake2, k)


def test_a_track_inserted_during_a_take_restores_nothing_on_the_wrong_track_and_raises_after_cleanup(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={3}, insert_on_fire=0)
    r = renderer(fake, clock, g)
    with pytest.raises(RuntimeError, match="moved during the take"):
        r.render(candidates(g, 3))
    shifted = max(k for k, c in enumerate(fake.calls) if c[1] == "call" and c[3] == "fire")
    after = fake.calls[shifted + 1:]
    assert fake.playing is False and [c for c in after if c[1] == "call"][-1][3] == "stop_playing"      # the transport is stopped
    assert not [c for c in after if c[1] == "call" and c[3] == "stop"]                                # no stop on the slot of the wrong track
    assert not [c for c in after if c[1] == "transaction" and any(prop in ("solo", "arm") for _, prop in c[2])]   # no restore
    assert fake.solo[2] is False and fake.solo[3] is True                        # the old record track (now 3) stays soloed: the message says so
    assert "may still be soloed" in r.takes[0]["cleanup_errors"][0]


def test_the_constructor_checks_that_the_device_and_the_clip_share_a_track_and_the_record_track_is_another(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    kw = dict(sleep=clock.sleep, clock=clock.time, capture=False)
    with pytest.raises(ValueError, match="different tracks"):
        patch.OperatorRenderer(fake, DEV, "live_set tracks 3 clip_slots 0 clip", REC, g["structure"], **kw)
    with pytest.raises(ValueError, match="two tracks"):
        patch.OperatorRenderer(fake, DEV, CLIP, "live_set tracks 1", g["structure"], **kw)
    with pytest.raises(ValueError, match="live_set tracks N"):
        patch.OperatorRenderer(fake, DEV, CLIP, "live_set tracks 2 devices 0", g["structure"], **kw)
    with pytest.raises(ValueError, match="live_set tracks N"):
        patch.OperatorRenderer(fake, "devices 0", CLIP, REC, g["structure"], **kw)


def test_clear_takes_is_refused_when_a_track_moved_and_deletes_each_clip_with_its_name(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, filled={0, 2})
    r = renderer(fake, clock, g)
    assert r.clear_takes(confirm=True) == 2
    assert fake.deleted == [("%s clip_slots 0" % REC, "take 0", True), ("%s clip_slots 2" % REC, "take 2", True)]
    fake.insert_track(0)
    fake.filled = {1}
    with pytest.raises(RuntimeError, match="not where they were"):
        r.clear_takes(confirm=True)
    assert len(fake.deleted) == 2


def test_a_missing_note_0_is_rejected_and_never_shifts_the_cuts_onto_the_next_candidates(tmp_path):
    """Note 0 absent from the file: the first onset is note 1's, one spacing late. That is not a grid step and not within
    the time the two fires took, so the whole take is rejected with a reason; no candidate gets a neighbour's hit."""
    g = guess_of()
    fake, clock = make(tmp_path, drop_notes={0}, lo=0.6, hi=0.6, jitter=0.0)
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 5))
    take = r.takes[0]
    assert not any(len(h) for h in hits) and "may be missing" in take["rejected"]
    assert all(c["reason"] for c in take["candidates"]) and take["file_shift_s"] > take["spacing_s"] - 0.2


def test_the_first_onset_may_be_late_by_the_time_between_the_fires_when_nothing_is_quantized(tmp_path):
    g = guess_of()
    ps = candidates(g, 4)
    fake, clock = make(tmp_path, quantize_beats=0.0, lo=0.6, hi=0.6, jitter=0.0)
    r = renderer(fake, clock, g)
    hits = r.render(ps)
    take = r.takes[0]
    assert 0.3 < take["file_shift_s"] <= take["fire_window_s"] and take["quantization_beats"] == 0.0 and not take["failures"]
    ref_fake, ref_clock = make(tmp_path / "r", latency=0.01, jitter=0.0, nslots=8)
    for a, b in zip(hits, renderer(ref_fake, ref_clock, g, batch=False).render(ps)):
        assert dist(a, b) < 0.01


def test_a_shift_as_long_as_a_spacing_needs_every_onset_there_and_a_shift_off_the_grid_is_refused(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    r = renderer(fake, clock, g)
    hit = standin(standin_from_live(patch.operator_live(g["patch"], g["structure"])), 5)
    sr, spacing, step = SR, 2.0, 1.0          # a spacing of two grid steps: a missing note 0 looks like a grid shift

    def file(n_hits, first_at):
        x = np.zeros(int((first_at + spacing * n_hits + 0.5) * sr))
        for k in range(n_hits):
            a = int((first_at + k * spacing) * sr)
            x[a:a + len(hit)] += hit
        return x[:, None]
    take = {"tempo": 60.0, "fire_window_s": 0.2}                   # 60 BPM: one beat is one second, the grid is 1 beat
    assert r._file_shift(dict(take), file(5, 2.0), sr, 0.0, 5, spacing, 1.0) == pytest.approx(2.0, abs=0.01)      # all 5 onsets: a shift of one spacing
    t = dict(take)
    assert r._file_shift(t, file(4, 2.0), sr, 0.0, 5, spacing, 1.0) is None and "may be missing" in t["rejected"]  # 4 onsets for 5 notes
    t = dict(take)
    assert r._file_shift(t, file(5, 1.5), sr, 0.0, 5, spacing, 1.0) is None and "not a whole number" in t["rejected"]
    assert r._file_shift(dict(take), file(5, 1.0), sr, 0.0, 5, spacing, 1.0) == pytest.approx(1.0, abs=0.01)      # one step, under a spacing
    assert step == 1.0


def test_the_early_guard_fails_the_previous_candidate_when_the_sync_poll_is_longer_than_the_margin(tmp_path):
    """A sync poll of 2 s (and no re-poll, a short margin) leaves the true clock up to 2 s later than believed: a write can
    land while the previous voice still sounds. Those candidates are failed with a reason; whatever comes back is right."""
    g = guess_of()
    ps = candidates(g, 6)
    ref_fake, ref_clock = make(tmp_path / "r", latency=0.01, jitter=0.0, nslots=8)
    ref = renderer(ref_fake, ref_clock, g, batch=False).render(ps)
    fake, clock = make(tmp_path / "a", poll_spike=2.0, lo=0.6, hi=0.6, jitter=0.0)
    r = renderer(fake, clock, g, repoll_ratio=100.0, margin_s=0.2)
    hits = r.render(ps)
    take = r.takes[0]
    assert take["sync_poll_s"] == pytest.approx(2.0) and take["sync_repolled"] is False
    flagged = [c for c in take["candidates"] if c["reason"] and "could have ended" in c["reason"]]
    assert flagged and all(not len(hits[c["index"]]) for c in flagged)
    for i, h in enumerate(hits):
        assert not len(h) or dist(h, ref[i]) < 0.01


def test_a_slow_sync_poll_is_repeated_once_and_the_shorter_one_is_used(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, poll_spike=2.0, lo=0.6, hi=0.6, jitter=0.0)
    r = renderer(fake, clock, g)
    hits = r.render(candidates(g, 5))
    take = r.takes[0]
    assert take["sync_repolled"] is True and take["sync_poll_s"] == pytest.approx(0.6) and all(len(h) for h in hits)


@functools.lru_cache(maxsize=None)
def _reference(kind, n):
    g = guess_of()
    ps = adverse_candidates(kind, n)
    fake, clock = make(pathlib.Path(__import__("tempfile").mkdtemp()), latency=0.01, jitter=0.0, nslots=n + 2)
    return tuple(renderer(fake, clock, g, batch=False).render(ps))


def adverse_candidates(kind, n):
    g = guess_of()
    if kind == "kick":
        return candidates(g, n)
    out = []
    for k in range(n):                                     # short notes: the margin is the room there is
        p = dict(g["patch"], tone_decay_ms=110.0 * (1 + 0.1 * k), carrier_hz=g["patch"]["carrier_hz"] * 2 ** (k / 12.0))
        if "noise_decay_ms" in p:
            p["noise_decay_ms"] = 60.0
        out.append(p)
    return out


@pytest.mark.parametrize("kind, kw", [
    ("short", dict(lo=0.45, hi=0.75, handle_frac=0.5, write_frac=1.0)),
    ("short", dict(lo=0.4, hi=0.7, handle_frac=1.0, write_frac=0.0)),
    ("short", dict(lo=0.4, hi=0.7, handle_frac=1.0, write_frac=1.0, poll_spike=1.5, quantize_beats=0.0)),
    ("short", dict(lo=0.4, hi=0.7, handle_frac=0.0, write_frac=0.5, poll_spike=1.5, quantize_beats=0.0)),
    ("kick", dict(lo=0.4, hi=0.7, handle_frac=1.0, write_frac=0.0)),
    ("kick", dict(lo=0.4, hi=0.7, handle_frac=0.5, write_frac=1.0, p_spike=0.08, spike=1.5)),
])
def test_no_hit_is_returned_with_a_neighbours_sound_under_adverse_latency(tmp_path, kind, kw):
    """The reviewer's adverse cases: short notes, Live reading the position at the start or the end of the poll, writes
    landing at the start or the end of their request, a slow sync poll, spikes. Every hit that comes back is its own
    candidate's (nearest to its own single-mode reference, and equal to it); a candidate that could not be trusted is failed."""
    n, bad, ok = 8, 0, 0
    ref = _reference(kind, n)
    g = guess_of()
    for seed in range(3):
        fake, clock = make(tmp_path / ("s%d" % seed), seed=seed, jitter=0.0, **kw)
        r = renderer(fake, clock, g)
        hits = r.render(adverse_candidates(kind, n))
        for i, h in enumerate(hits):
            if not len(h):
                continue
            ok += 1
            d = [dist(h, rr) for rr in ref]
            bad += int(np.argmin(d)) != i or d[i] > 0.01
    assert bad == 0 and ok > 0, (bad, ok)
    if not kw.get("p_spike"):                              # without spikes nothing needs to be failed: the schedule leaves the room
        assert ok == 3 * n, (ok, 3 * n)


@pytest.mark.parametrize("which", ["stop_playing", "stop", "restore", "write"])
def test_a_ctrl_c_in_the_cleanup_skips_no_other_step_and_the_first_exception_is_raised(tmp_path, which):
    g = guess_of()
    fake, clock = make(tmp_path, solo={0}, armed=False)
    boom = KeyboardInterrupt("injected")
    if which == "stop_playing":
        raising_once(fake, "call", lambda path, function, *a, **k: function == "stop_playing", boom)
    elif which == "stop":
        raising_once(fake, "call", lambda path, function, *a, **k: function == "stop", boom)
    elif which == "restore":
        raising_once(fake, "transaction", lambda ops, **k: any(o["property"] == "arm" and o["value"] is False for o in ops), boom)
    else:                                                  # a Ctrl-C in the 3rd write, and a second one when the slot is stopped
        raising_once(fake, "transaction", lambda ops, **k: len(fake.writes) == 2 and any(o["path"].startswith(DEV) for o in ops), boom)
        raising_once(fake, "call", lambda path, function, *a, **k: function == "stop", KeyboardInterrupt("second"))
    with pytest.raises(KeyboardInterrupt) as e:
        renderer(fake, clock, g).render(candidates(g, 6))
    assert "injected" in str(e.value)
    assert fake.playing is False and fake.solo == [True, False, False, False] and fake.armed is False


def test_a_failed_restore_raises_after_the_cleanup_and_the_next_take_restores_to_the_state_before_it(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={0}, armed=False, fail_restore=True)
    r = renderer(fake, clock, g)
    with pytest.raises(RuntimeError, match="restoring solo/arm failed"):
        r.render(candidates(g, 2))
    assert fake.playing is False and fake.solo[2] is True                    # the failed restore left the record track soloed
    fake.fail_restore = False
    r.render(candidates(g, 2))
    assert fake.solo == [True, False, False, False] and fake.armed is False


def test_arrangement_record_and_a_set_that_starts_playing_after_the_first_check_are_refused(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, record_mode=True)
    with pytest.raises(RuntimeError, match="Arrangement Record"):
        renderer(fake, clock, g).render(candidates(g, 2))
    assert not writes_since(fake, 0)
    late, clock2 = make(tmp_path / "b", flip_playing_at_state=True)
    with pytest.raises(RuntimeError, match="started after the first check"):
        renderer(late, clock2, g).render(candidates(g, 2))
    assert not writes_since(late, 0)


def test_exclusive_arm_that_disarms_another_track_is_undone(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, armed=False, armed_tracks={1}, exclusive_arm=True)
    renderer(fake, clock, g).render(candidates(g, 2))
    assert fake.arms == [False, True, False, False]


def test_a_failing_setup_and_a_clip_that_cannot_be_lengthened_abort_the_take_after_cleanup(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, fail_setup=True, solo={3})
    with pytest.raises(RuntimeError, match="setting up the take failed"):
        renderer(fake, clock, g).render(candidates(g, 2))
    assert not [c for c in fake.calls if c[1] == "call" and c[3] == "fire"] and fake.playing is False and fake.solo == [False, False, False, True]
    short, clock2 = make(tmp_path / "b", clip_length_cap=3.0)
    with pytest.raises(RuntimeError, match="MIDI clip is"):
        renderer(short, clock2, g).render(candidates(g, 4))
    assert short.playing is False and short.solo == [False] * 4


# ---- repeat_check

def test_repeat_check_is_zero_on_a_deterministic_renderer_and_the_standins_spread_with_fresh_noise(tmp_path):
    g = guess_of("tom")
    p = g["patch"]
    fake, clock = make(tmp_path)
    flat = patch.repeat_check(renderer(fake, clock, g), p, n=6)
    assert flat["ok"] == 6 and flat["pairwise"]["max"] < 1e-3 and flat["readings"]["fall_20_ms"]["std"] < 0.5
    fresh, clock2 = make(tmp_path / "fresh")
    fresh.fresh_noise = True
    got = patch.repeat_check(renderer(fresh, clock2, g), p, n=6)
    play = standin_from_live(patch.operator_live(p, g["structure"]))
    direct = [standin(play, 1000 + k) for k in range(6)]
    want = np.array([dist(a, b) for i, a in enumerate(direct) for b in direct[i + 1:]])
    assert got["pairwise"]["count"] == 15 and want.max() > 0.01                      # fresh noise does make them differ
    assert got["pairwise"]["median"] == pytest.approx(float(np.median(want)), rel=0.2, abs=0.01)
    assert got["pairwise"]["max"] == pytest.approx(float(want.max()), rel=0.2, abs=0.01)
    target = standin(dict(play, noise_db=-80.0), TARGET_SEED)
    steady = renderer(*make(tmp_path / "t"), g)
    assert patch.repeat_check(steady, p, n=4, target=target)["repeat_spread"] == pytest.approx(0.0, abs=1e-3)


# ---- patch_match

def _sample(tmp_path, kind="kick"):
    x = np.concatenate([np.zeros(int(0.05 * SR)), standin(TRUE[kind], TARGET_SEED), np.zeros(int(0.05 * SR))])
    path = tmp_path / "sample.wav"
    sf.write(str(path), x, SR, subtype="FLOAT")
    return path


def test_noise_floor_is_read_from_the_tail_the_lead_or_not_at_all():
    sr = 24000
    t = np.arange(int(1.4 * sr)) / sr
    rng = np.random.default_rng(3)
    hit = np.sin(2 * np.pi * 200 * t) * np.exp(-t / 0.02) * 0.5
    bed = rng.standard_normal(len(t)) * 0.5 / math.sqrt(2) * 10 ** (-50 / 20)       # white, 50 dB under the hit's peak RMS
    x = (hit + bed)[:, None]
    f, how = patch.noise_floor_db(x, sr, oneshot.profile_array(x, sr, single=True))
    assert 44.0 <= f <= 50.0 and "last 0.1 s" in how, (f, how)          # depth 50 dB (a sine's 1 ms RMS is a bit under its peak): 47 +- 3
    lead = np.concatenate([bed[:int(0.2 * sr), None], (hit[:int(0.8 * sr)] + 0 * bed[:int(0.8 * sr)])[:, None]])
    p = {"onset_ms": 200.0, "fall_40_ms": None}
    f2, how2 = patch.noise_floor_db(lead, sr, p)
    assert 44.0 <= f2 <= 50.0 and "before the onset" in how2, (f2, how2)
    f3, how3 = patch.noise_floor_db(x[:int(0.3 * sr)], sr, {"onset_ms": 0.0, "fall_40_ms": None})
    assert f3 == 60.0 and "no bed" in how3
    quiet = patch.noise_floor_db(np.concatenate([bed[:int(0.2 * sr), None] * 1e3, hit[:int(0.8 * sr), None]]), sr, p)
    assert quiet[0] == patch.MIN_FLOOR_DB or quiet[0] >= patch.MIN_FLOOR_DB     # never under the minimum


def test_patch_match_on_the_fake_improves_on_its_start_and_reports_its_takes(tmp_path):
    sample = _sample(tmp_path, "tom")
    g = patch.operator_guess(oneshot.profile(str(sample)))
    fake, clock = make(tmp_path / "x")
    r = renderer(fake, clock, g)
    res = patch.patch_match(str(sample), r, rounds=3, population=8)
    assert r.hit_s == HIT_S                                  # the run's own hit length is not left on the renderer
    assert res["distance"] < res["start_distance"] and all(a >= b - 1e-12 for a, b in zip(res["per_round"], res["per_round"][1:]))
    assert res["rounds"] == 3 and res["renders"] == 24 and res["error"] is None
    assert res["takes"] == res["rounds"] + 2 and len([c for c in fake.calls if c[1] == "call" and c[3] == "fire"]) == 2 * res["takes"]   # + the probe + the verification
    assert res["start_distance_round0"] == res["start_distance"] and res["distance_search"] == res["distance"]
    assert res["start_distance_verified"] > 0 and res["distance_verified"] > 0 and res["start_distance_verified_std"] is not None
    assert res["verification_error"] is None
    assert set(res["live"]) == set(patch.operator_live(res["best"], res["guess"]["structure"])["params"])
    assert 20.0 <= res["floor_db"] <= 60.0 and res["floor_reading"] and isinstance(res["readings"], list)
    assert len(res["best_render"]) and res["search"].failures == []
    assert fake.solo == [False] * 4 and fake.playing is False


def test_patch_match_returns_the_error_that_ended_the_run_early(tmp_path):
    sample = _sample(tmp_path, "tom")
    g = patch.operator_guess(oneshot.profile(str(sample)))
    fake, clock = make(tmp_path / "x", nslots=3)                 # the probe take and two rounds, then no empty slot
    res = patch.patch_match(str(sample), renderer(fake, clock, g), rounds=6, population=8)
    assert res["error"] and "no empty clip slot" in res["error"] and res["rounds"] < 6
    assert res["verification_error"] and fake.solo == [False] * 4 and fake.playing is False


def test_patch_match_takes_any_renderer(tmp_path):
    sample = _sample(tmp_path, "kick")
    g = patch.operator_guess(oneshot.profile(str(sample)))
    seeds = itertools.count(7 * 10 ** 6)

    def render(patches):
        return [standin(standin_from_live(patch.operator_live(p, g["structure"])), next(seeds))[:, None] for p in patches]
    res = patch.patch_match(str(sample), render, rounds=2, population=6, sr=SR)
    assert math.isfinite(res["distance"]) and res["takes"] == 0 and res["renders"] == 12      # fresh noise per render: no ordering to assert
    assert set(res["best"]) == set(g["patch"]) and len(res["best_render"])
    with pytest.raises(ValueError, match="sample rate"):
        patch.patch_match(str(sample), render, rounds=1, population=4)
