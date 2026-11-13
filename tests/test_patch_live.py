"""sideman.patch's Live renderer (LiveRenderer, OperatorRenderer, repeat_check, patch_match) against a FAKE client.

No socket and no Live: `FakeLive` records every request it receives and, when asked for the recording's file_path,
writes a WAV that it synthesises itself from what it was sent: the notes of the MIDI clip, the wall time the clip
was fired at and the parameter writes, each note played by the stand-in synth of tests/test_patch.py with the
parameters last written before that note. So the tests prove the renderer's bookkeeping (what is written when, which
slot, which solo state, how a take is cut into hits and what a failure returns), not anything about Live: its timing,
its launch quantization, whether `loop_end` and `end_marker` can be set on a session clip, or how Operator sounds.

Needs the `listen` extra.
"""
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
    def __init__(self, tmp_path, clock, *, tempo=120.0, ntracks=4, nslots=8, filled=(), playing=False, solo=(), armed=True,
                 fresh_noise=False, latency=0.004, drop_notes=(), cut_last_ms=None, fail_write=(), raise_on_file=False,
                 first_seed=1000):
        self.tmp, self.clock, self.tempo, self.latency = tmp_path, clock, tempo, latency
        self.ntracks, self.nslots, self.filled, self.playing = ntracks, nslots, set(filled), playing
        self.solo, self.armed = [i in solo for i in range(ntracks)], armed
        self.fresh_noise, self.drop_notes, self.cut_last_ms = fresh_noise, set(drop_notes), cut_last_ms
        self.fail_write, self.raise_on_file, self.seeds = set(fail_write), raise_on_file, itertools.count(first_seed)
        self.calls, self.writes, self.notes = [], [], []        # writes: (wall time, {param index: Live value})
        self.clip_t0 = self.clip_t0_used = self.rec_t0 = self.rec_t1 = None
        self.clip_length, self.recorded, self.files = 4.0, {}, 0

    # ---- the wire
    def _tick(self, *what):
        self.clock.now += self.latency
        self.calls.append((self.clock.now,) + what)

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
        if (path, prop) == (REC, "arm"):
            return {"value": self.armed}
        if (path, prop) == (CLIP, "length"):
            return {"value": self.clip_length}
        if prop == "file_path" and path.startswith(REC + " clip_slots "):
            if self.raise_on_file:
                raise RuntimeError("no file")
            return {"value": self._file(int(path.split()[-2]))}
        raise AssertionError("unexpected get %r %r" % (path, prop))

    def get_batch(self, specs):
        self._tick("get_batch", tuple((s["path"], s["property"]) for s in specs))
        out = []
        for s in specs:
            path, prop = s["path"], s["property"]
            if prop == "solo":
                out.append({"value": self.solo[int(path.split()[-1])]})
            elif prop == "has_clip":
                k = int(path.split()[-1])
                out.append({"value": k in self.filled or k in self.recorded})
            elif path == CLIP and prop == "is_playing":
                out.append({"value": self.clip_t0 is not None and self.playing})
            elif path == CLIP and prop == "playing_position":
                out.append({"value": (self.clock.now - self.clip_t0) / (60.0 / self.tempo) if self.clip_t0 is not None else 0.0})
            else:
                raise AssertionError("unexpected batch read %r %r" % (path, prop))
        return {"results": out}

    def set(self, path, prop, value):
        self._tick("set", path, prop, value)
        if prop == "solo":
            self.solo[int(path.split()[-1])] = bool(value)
        elif (path, prop) == (REC, "arm"):
            self.armed = bool(value)
        elif path == CLIP and prop in ("loop_end", "end_marker"):
            self.clip_length = float(value)
        elif not (path == CLIP and prop == "looping"):
            raise AssertionError("unexpected set %r %r" % (path, prop))
        return value

    def call(self, path, function, args=None, confirm=False, **kw):
        self._tick("call", path, function)
        if function == "fire" and path.startswith(REC + " clip_slots "):
            self.rec_t0, self.rec_t1, self.playing = self.clock.now, None, True
            self.rec_slot = int(path.split()[-1])
        elif function == "fire" and path == "live_set tracks 1 clip_slots 0":
            self.clip_t0 = self.clip_t0_used = self.clock.now
        elif function == "stop" and path.startswith(REC + " clip_slots "):
            self.rec_t1 = self.clock.now
            self.recorded[self.rec_slot] = True
        elif function == "stop" and path == "live_set tracks 1 clip_slots 0":
            self.clip_t0 = None
        elif function == "stop_playing":
            self.playing, self.clip_t0 = False, None
        elif function == "delete_clip":
            self.filled.discard(int(path.split()[-1]))
            self.recorded.pop(int(path.split()[-1]), None)
        else:
            raise AssertionError("unexpected call %r %r" % (path, function))
        return {}

    def transaction(self, ops, **kw):
        self._tick("transaction", len(ops))
        n = len(self.writes)
        self.writes.append((self.clock.now, {int(o["path"].split()[-1]): o["value"] for o in ops}))
        if n in self.fail_write:
            return {"results": [{"ok": False, "error": "refused"}] + [{"ok": True}] * (len(ops) - 1)}
        return {"results": [{"ok": True}] * len(ops)}

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
        x = np.zeros(int(round((end - self.rec_t0) * SR)))
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
    assert starts[0] == r.start_beat and np.allclose(np.diff(starts), take["spacing_beats"])
    longest = max(L["note_length_ms"] for L in live) / 1000.0
    need = max(HIT_S, longest) + r.margin_s
    assert need <= take["spacing_beats"] * spb < need + spb / 16 + 1e-9
    assert all(abs(n["duration"] * spb - L["note_length_ms"] / 1000.0) < 1e-6 for n, L in zip(fake.notes, live))
    # the parameters: write k holds candidate k's values, written at least lead_s - the request latency before its note
    # and after the previous note ended
    assert len(fake.writes) == len(ps)
    for k, (t, w) in enumerate(fake.writes):
        assert w[NAMES.index("Ae Decay")] == pytest.approx(patch.to_live(live[k]["params"], UNITS)["Ae Decay"])
        note_wall = fake.clip_t0_used + starts[k] * spb
        if k:
            assert note_wall - t >= r.lead_s - 0.02 and note_wall - t >= r.min_lead_s
            assert t >= fake.clip_t0_used + starts[k - 1] * spb + live[k - 1]["note_length_ms"] / 1000.0
        assert take["candidates"][k]["ok"] and take["candidates"][k]["lead_ms"] >= 1000 * r.min_lead_s
        assert abs(take["candidates"][k]["offset_ms"]) < 5.0                    # the stand-in's onset is the note's
    assert take["sample_rate"] == SR and r.sample_rate == SR and take["slot"] == 0 and take["drift_ms"] < 1.0


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


def test_a_write_that_ends_too_close_to_its_note_fails_the_candidate(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    r = renderer(fake, clock, g, min_lead_s=0.1, lead_s=0.12)
    original = fake.transaction

    def slow(ops, **kw):                                     # a write that takes 60 ms: it ends 60 ms before its note
        out = original(ops, **kw)
        clock.now += 0.06
        return out
    fake.transaction = slow
    hits = r.render(candidates(g, 3))
    assert [len(h) > 0 for h in hits] == [True, False, False]
    assert "ms before the note" in r.takes[0]["candidates"][1]["reason"]


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
    sets = [c for c in fake.calls if c[1] == "set" and c[3] == "solo"]
    assert {c[2] for c in sets} == {REC}                                    # only the record track's solo was written
    assert [c for c in fake.calls if c[1] == "call"][-1][3] == "stop_playing"          # the transport is the last thing stopped
    touched = {c[2] for c in fake.calls if c[1] in ("set", "call", "notes_add", "notes_remove")}
    assert all(t.startswith(("live_set tracks 1 ", REC)) or t in ("live_set", "live_set tracks 1") for t in touched), touched


def test_solo_and_arm_are_restored_and_the_transport_stopped_when_a_take_fails(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={2}, armed=False, raise_on_file=True)
    with pytest.raises(RuntimeError, match="no recording"):
        renderer(fake, clock, g).render(candidates(g, 3))
    assert fake.solo == [False, False, True, False] and fake.armed is False and fake.playing is False


def test_the_solo_of_another_track_is_restored_if_live_unsoloes_it(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path, solo={0})
    original = fake.set

    def exclusive(path, prop, value):                        # Live's exclusive solo: soloing one track clears the others
        if prop == "solo" and value:
            fake.solo = [False] * fake.ntracks
        return original(path, prop, value)
    fake.set = exclusive
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


def test_the_constructor_refuses_a_margin_smaller_than_the_lead(tmp_path):
    g = guess_of()
    fake, clock = make(tmp_path)
    with pytest.raises(ValueError, match="margin_s"):
        renderer(fake, clock, g, margin_s=0.05, lead_s=0.15)
