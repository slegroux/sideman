"""Profile a drum one-shot in the terms a synth is programmed in: envelope,
pitch and pitch drop, harmonics, noise, click, band balance and width.

Every measure is relative to the hit itself (its onset, its peak, its own
fundamental), so two files at different levels, sample rates or with
different leading silence profile alike. Local only; needs the `listen` extra.

Choices that are not obvious from the code:

- The onset is the first 1 ms frame within 40 dB of the hit's loudest 1 ms
  (and in it, the first sample at that level); everything before it is
  dropped and every time is counted from it. A DC offset (the mean of the
  quietest 10 ms blocks, where the hit has decayed or not begun) above
  -60 dB of the peak is removed first.
- Decay uses a power envelope smoothed over whole periods of the body pitch
  (5 ms for an unpitched hit), so partials beating against each other do not
  read as decay. A fall time is the last moment the envelope is still above
  the threshold, measured from the envelope's peak. The attack time is where
  a 1 ms-smoothed power envelope first comes within 1 dB of its peak: a
  crest, so it is good to about half a period of a low tone (and reads a
  linear ramp about 11 % short). (An analytic
  envelope was tried: it dips 6 dB or spikes 5 dB at an abrupt onset.)
- Pitch frames are Hann windows on the analytic signal (no negative-frequency
  image), three periods long, zero-padded 8x, with parabolic interpolation on
  the dB spectrum. A spectral peak must be the maximum within one bin of the
  unpadded FFT, so window sidelobes never count. The fundamental is the
  lowest peak within 20 dB of the strongest, not the loudest one.
- The track is chained: frames 1 ms apart, from the body window's centre
  back to the onset and on to 300 ms, each sized by and held within 2 st of
  its neighbour's pitch, so it cannot jump to a harmonic or to a thump below.
  A frame is centred on its time; one that would start before the onset
  starts there, so the early points (and pitch_start_hz) are the pitch about
  1.5 periods in: a glide that moves within its first three periods reads
  lower than its very first cycle, and pitch_start_at_ms says when it was read.
- The body pitch comes from one long window, 40-150 ms (shortened for a hit
  that is over sooner), since a 25 ms window resolves only about 40 Hz.
- A hit is pitched when the body has a peak at least 15 dB above its local
  spectral floor and the fundamental's harmonic series holds at least half of
  the energy of all such peaks. Otherwise (noise, or an inharmonic bell) it
  reports its partials and no pitch.
- Noise share is the energy outside the main lobes of the tonal peaks (every
  peak 15 dB above its local floor, harmonic or not; a lobe runs down to its
  first minimum, at most 4 bins out), against the total, from 20 Hz up. The
  attack window is a plain Hann over 0-30 ms, so it weighs the middle of the
  attack most; the click right at the onset is click_db's job.
  A glide smears its fundamental, so a sweeping attack reads noisier than
  the same tone held still.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .listen import BANDS, FLOOR_DB, _check_channels, _db, load

ONSET_DB = 40.0          # the onset is the first 1 ms frame within this of the loudest
FALLS_DB = (10, 20, 40)
END_DB = 60.0            # duration runs to the last moment within this of the peak
GATE_DB = 50.0           # pitch frames must end before the hit falls this far
TRACK_MS = (0, 5, 10, 20, 40, 80, 150, 300)
ATTACK_S = 0.030
ATTACK_DB = 1.0          # the attack ends within this of the envelope peak
BODY_S = (0.040, 0.150)
PERIODS = 3.0            # pitch frame length, in periods of the frame's pitch
FRAME_S = (0.004, 0.040)
STRONG_DB = 20.0         # a partial this far under the strongest still counts as the fundamental
PROMINENT_DB = 15.0      # a peak must stand this far above its local floor to be tonal
HARM_TOL = 0.03          # a partial within 3 % of k * f0 is harmonic k
SETTLE_ST = 0.5
SETTLE_HOLD_S = 0.010    # settled = within SETTLE_ST and still there 10 ms later
STEP_ST = 2.0            # a track frame may move this far from the one 1 ms before it
NOISY_DB = -3.0          # a body whose noise share is above this is unpitched
PARTIALS_DB = 40.0       # inharmonic partials are listed down to this under the strongest peak
MAX_PARTIALS = 5
CLICK_HZ = 2000.0
DC_REL = 1e-3            # a DC offset under -60 dB of the peak is left alone
MAX_S = 10.0             # a one-shot longer than this is cut, with a note
RISE_S = 0.010           # hits: a new hit rises RISE_DB within this...
RISE_DB = 15.0
FAST_DB = 6.0            # ...reaching within this of its peak (the max over PEAK_S)...
PEAK_S = 0.050
JUMP_DB = 6.0            # ...and tops the CEILING_S before it by this
CEILING_S = 0.050
MIN_GAP_S = 0.050
# compare(): a difference past one of these is a reading
PITCH_ST = 0.3           # body pitch
DROP_ST = 1.0            # pitch drop
TIME_RATIO = 0.15        # attack, falls, duration, settle: 15 %...
TIME_FLOOR_MS = 2.0      # ...and at least this, so 0.3 vs 0.4 ms is not a reading
LEVEL_DB = 3.0           # harmonics, noise share
CLICK_STEP_DB = 6.0
WIDTH_STEP = 0.1
# compare(): below these on both sides a difference is the method's floor, not the sound
HARM_FLOOR_DB = -50.0    # a harmonic must be above this...
HARM_CLEAR_DB = PROMINENT_DB  # ...and stand clear of its own spectral floor like a tonal peak
NOISE_FLOOR_DB = -25.0   # a pure decaying tone reads -26 to -42 from window leakage
CLICK_FLOOR_DB = -40.0


def _boxcar(p: np.ndarray, length: int) -> np.ndarray:
    """Centred moving mean of p over `length` samples, shrinking at the ends."""
    c = np.concatenate([[0.0], np.cumsum(p)])
    i = np.arange(len(p))
    lo = np.clip(i - length // 2, 0, len(p))
    hi = np.clip(i - length // 2 + length, 0, len(p))
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def _analytic(y: np.ndarray) -> np.ndarray:
    n = y.shape[0]
    nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    h = np.zeros(nfft)
    h[0] = h[nfft // 2] = 1.0
    h[1:nfft // 2] = 2.0
    return np.fft.ifft(np.fft.fft(y, nfft, axis=0) * h[:, None], axis=0)[:n]


def _spectrum(z: np.ndarray, a: int, b: int, sr: int) -> dict:
    """Channel-mean power spectrum (dB) of the analytic signal over [a, b),
    Hann-windowed, zero-padded at least 8x."""
    m = b - a
    win = np.hanning(m)
    nfft = 1 << int(math.ceil(math.log2(8 * m)))
    p = (np.abs(np.fft.fft(z[a:b] * win[:, None], nfft, axis=0)[:nfft // 2]) ** 2).mean(axis=1)
    return {"p": p, "d": 10.0 * np.log10(p + 1e-30), "hz": np.arange(nfft // 2) * sr / nfft,
            "bin": max(1, int(round(nfft / m))), "sr": sr, "nfft": nfft}


def _peaks(s: dict, hi_hz: float) -> np.ndarray:
    """Bins that are the maximum within one unpadded bin either side (so a
    window sidelobe never qualifies), from 20 Hz to hi_hz, ascending."""
    d, h = s["d"], s["bin"]
    top = sliding_window_view(np.pad(d, h, constant_values=-np.inf), 2 * h + 1).max(axis=1)
    return np.nonzero((d >= top) & (s["hz"] >= 20.0) & (s["hz"] <= hi_hz))[0]


def _interp(s: dict, i: int) -> tuple[float, float]:
    """(Hz, dB) of the peak at bin i, parabolic on the dB spectrum."""
    d = s["d"]
    off, level = 0.0, float(d[i])
    if 0 < i < len(d) - 1:
        y0, y1, y2 = d[i - 1], d[i], d[i + 1]
        den = y0 - 2.0 * y1 + y2
        if den < 0:
            off = 0.5 * (y0 - y2) / den
            level = float(y1 - 0.25 * (y0 - y2) * off)
    return float((i + off) * s["sr"] / s["nfft"]), level


def _floor(s: dict, i: int) -> float | None:
    """Local spectral floor at bin i: the median (dB) of a ring 2-10 bins away,
    outside the main lobe of a peak at i."""
    d, h = s["d"], s["bin"]
    ring = np.r_[d[max(0, i - 10 * h):max(0, i - 2 * h)], d[i + 2 * h + 1:i + 10 * h + 1]]
    return float(np.median(ring)) if len(ring) else None


def _tonal(s: dict, hi_hz: float) -> list[tuple[float, float, int]]:
    """Peaks standing PROMINENT_DB above their local floor, as (Hz, dB, bin),
    strongest first."""
    out = []
    for i in _peaks(s, hi_hz):
        floor = _floor(s, i)
        if floor is not None and s["d"][i] - floor >= PROMINENT_DB:
            out.append(_interp(s, i) + (int(i),))
    return sorted(out, key=lambda t: -t[1])


def _frame_pitch(z: np.ndarray, t: float, sr: int, guess: float, end: int) -> tuple:
    """(Hz, centre in s, clamped) of a frame centred on t, PERIODS periods of
    `guess` long. A frame that would start before the onset starts there
    (clamped); one that would run past `end` is not read. The pitch is the
    lowest peak within STRONG_DB of the frame's strongest that is also within
    STEP_ST of `guess`, so the track cannot jump to a harmonic or to rumble."""
    w = int(round(min(max(PERIODS / guess, FRAME_S[0]), FRAME_S[1]) * sr))
    a = int(round(t * sr)) - w // 2
    clamped, a = a <= 0, max(0, a)
    if a + w > end:
        return None, None, clamped
    s = _spectrum(z, a, a + w, sr)
    idx = _peaks(s, min(20000.0, 0.45 * sr))
    if len(idx):
        d = s["d"][idx]
        idx = idx[(d >= d.max() - STRONG_DB) & (np.abs(12 * np.log2(s["hz"][idx] / guess)) <= STEP_ST)]
    if not len(idx):
        return None, None, clamped
    return _interp(s, int(idx[0]))[0], (a + w / 2) / sr, clamped


def _track(z: np.ndarray, sr: int, f0: float, mid: float, end: int) -> tuple[list, bool]:
    """Dense pitch track, frames 1 ms apart, chained outward from `mid` (the
    body window's centre, where f0 holds) back to the onset and on to TRACK_MS's
    last point or `end`. ([(centre s, Hz)] in time order, whether it reached the onset)."""
    back, guess, reached = [], f0, False
    for t in np.arange(mid, -1e-9, -0.001):
        f, c, clamped = _frame_pitch(z, float(t), sr, guess, end)
        if f is None:
            break
        back.append((c, f))
        guess = f
        if clamped:
            reached = True
            break
    fwd, guess = [], f0
    for t in np.arange(mid + 0.001, TRACK_MS[-1] / 1000.0 + 0.001, 0.001):
        f, c, _ = _frame_pitch(z, float(t), sr, guess, end)
        if f is None:
            break
        fwd.append((c, f))
        guess = f
    return back[::-1] + fwd, reached


def _last_above(env: np.ndarray, start: int, level: float) -> float | None:
    """Fractional index where env falls below `level` for the last time after
    `start`, or None if it is still above at the end of the file."""
    above = np.nonzero(env[start:] >= level)[0]
    if not len(above):
        return float(start)
    k = start + int(above[-1])
    if k >= len(env) - 1:
        return None
    a, b = 10 * math.log10(env[k]), 10 * math.log10(max(env[k + 1], 1e-30))
    lv = 10 * math.log10(level)
    return k + ((a - lv) / (a - b) if a > b else 0.0)


def _noise_flatness(s: dict, peaks: list, hi_hz: float) -> tuple[float, float]:
    """(noise share in dB, spectral flatness 0-1) over 20 Hz-hi_hz."""
    band = (s["hz"] >= 20.0) & (s["hz"] <= hi_hz)
    d, reach = s["d"], 4 * s["bin"]
    tonal = np.zeros(len(d), bool)
    for _, _, i in peaks:            # the main lobe: down to its first minimum, at most 4 bins out
        lo = i
        while lo > max(0, i - reach) and d[lo - 1] < d[lo]:
            lo -= 1
        hi = i
        while hi < min(len(d) - 1, i + reach) and d[hi + 1] < d[hi]:
            hi += 1
        tonal[lo:hi + 1] = True
    p = s["p"][band]
    total = float(p.sum())
    noise = _db(float(s["p"][band & ~tonal].sum()) / total) if total > 0 else None
    flat = float(np.exp(np.mean(np.log(p + 1e-30))) / np.mean(p)) if total > 0 else None
    return noise, flat


def _harmonic(hz: float, f0: float) -> bool:
    """Whether hz is within HARM_TOL of a whole multiple (1, 2, ...) of f0."""
    k = round(hz / f0)
    return k >= 1 and abs(hz - k * f0) <= HARM_TOL * k * f0


def _ms(samples: float, sr: int) -> float:
    return round(1000.0 * samples / sr, 1)


def _segments(x: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """[start, end) sample ranges of the hits in a take. A new hit is a rise of
    RISE_DB that comes within FAST_DB of its peak inside RISE_S and tops the
    CEILING_S before it by JUMP_DB, at least MIN_GAP_S after the last one, so a
    tremolo, a tom's beating or a slow swell does not split a hit. A hit that
    starts before the last one has fallen 15 dB (a fast roll) is not split
    off either."""
    hop = max(1, int(round(0.001 * sr)))                     # the envelope on a 1 ms grid
    env = _boxcar((x ** 2).mean(axis=1), max(1, int(0.005 * sr)))[::hop]
    e = 10.0 * np.log10(env / env.max() + 1e-30)
    rise, gap, ahead, ceil = (int(round(v * 1000)) for v in (RISE_S, MIN_GAP_S, PEAK_S, CEILING_S))
    first = int(np.argmax(e >= -ONSET_DB))
    starts, last = [0], first
    for j in range(first + 1, len(e)):
        if j - last < gap or e[j] < -ONSET_DB:
            continue
        jm = j - rise + int(np.argmin(e[j - rise:j])) if j >= rise else int(np.argmin(e[:j]))
        if e[j] - e[jm] < RISE_DB:
            continue
        top = e[jm:jm + ahead].max()
        if e[jm:jm + rise + 1].max() < top - FAST_DB or top < e[max(0, jm - ceil):jm].max() + JUMP_DB:
            continue
        starts.append(jm * hop)
        last = jm
    return list(zip(starts, starts[1:] + [x.shape[0]]))


def _dc(x: np.ndarray, sr: int) -> np.ndarray:
    """Per-channel DC offset: the mean of the quietest tenth of the 10 ms
    blocks (lowest variance, which DC does not add to), where the hit has
    decayed or not yet begun. A decaying hit's own mean is not DC."""
    n = max(1, int(round(0.010 * sr)))
    m = x.shape[0] // n
    if m < 1:
        return x.mean(axis=0) if len(x) else np.zeros(x.shape[1])
    blocks = x[:m * n].reshape(m, n, x.shape[1])
    quiet = np.argsort(blocks.var(axis=1).sum(axis=1))[:max(1, m // 10)]
    return blocks[quiet].mean(axis=(0, 1))


def _onset(x: np.ndarray, mag: np.ndarray, sr: int, after_hit: bool) -> int:
    """Index of a hit's onset in x: the first 1 ms frame whose mean power is
    within ONSET_DB of the hit's loudest 1 ms, and in that frame the first
    sample at that level, so a lone click or dither spike does not count.
    After an earlier hit, the level must also be twice its tail's."""
    frame = max(1, int(round(0.001 * sr)))
    c = np.concatenate([[0.0], np.cumsum((x ** 2).mean(axis=1))])
    fwd = (c[frame:] - c[:-frame]) / frame if len(x) > frame else c[-1:] / max(1, len(x))
    floor = fwd.max() * 10 ** (-ONSET_DB / 10)
    level = math.sqrt(floor)
    if after_hit:              # rise out of the tail: four times its power, twice its peak
        floor = max(floor, 4.0 * float(fwd[0]))
        level = max(level, 2.0 * float(mag[:frame].max()))
    first = int(np.argmax(fwd >= floor))
    inside = np.nonzero(mag[first:first + frame] >= level)[0]
    return first + (int(inside[0]) if len(inside) else 0)


def profile_array(x: np.ndarray, sr: int, hit: int | None = None) -> dict:
    """The one-shot profile of one signal [n, channels]: of hit `hit` (from 0)
    when it holds several, by default the first that is not cut off. See profile()."""
    _check_channels(x)
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("the file holds NaN or infinite samples")
    dc = _dc(x, sr)            # a DC offset would read as level that never decays
    if len(x) and np.abs(dc).max() > DC_REL * np.abs(x).max():
        x = x - dc
    mag = np.abs(x).max(axis=1)
    if not len(mag) or mag.max() <= 0:
        raise ValueError("the file is silent")
    segs = _segments(x, sr)
    if hit is not None and not 0 <= hit < len(segs):
        raise ValueError("hit %r is out of range: the take has %d hit%s (0-%d)"
                         % (hit, len(segs), "" if len(segs) == 1 else "s", len(segs) - 1))
    onsets = [_onset(x[s:e], mag[s:e], sr, s > 0) + s for s, e in segs]
    picked, extra = None, []
    for k in ([hit] if hit is not None else range(len(segs))):
        cut = "the file ends" if k == len(segs) - 1 else "the next hit starts"
        try:
            got = _measure(x[onsets[k]:segs[k][1]], sr, cut)
        except ValueError:
            if hit is not None or k == len(segs) - 1:
                raise
            continue
        if picked is None:
            picked = (k, got)
        if hit is not None or not got["truncated"]:
            picked = (k, got)
            break
    else:
        extra.append("every hit is cut off before it falls 40 dB: this is the first")
    k, got = picked
    out = {"sample_rate": sr, "channels": x.shape[1], "hits": len(segs),
           "onsets_ms": [_ms(o, sr) for o in onsets], "hit": k, "onset_ms": _ms(onsets[k], sr)}
    out.update(got)
    out["notes"] = extra + out["notes"]
    return out


def _measure(y: np.ndarray, sr: int, cut: str) -> dict:
    """The profile of one hit y [n, channels], which starts at its onset and
    ends where `cut` says ("the file ends" or "the next hit starts")."""
    notes = []
    peak = float(np.abs(y).max())
    if len(y) < FRAME_S[0] * sr:
        raise ValueError("the hit is shorter than %g ms" % (1000 * FRAME_S[0]))
    if len(y) > MAX_S * sr:
        y = y[:int(MAX_S * sr)]
        notes.append("only the first %g s after the onset are measured" % MAX_S)
    n, nyq_hi = len(y), min(20000.0, 0.45 * sr)
    power = (y ** 2).mean(axis=1)
    out = {"peak_db": round(_db(peak * peak), 2)}

    # a first, pitch-blind envelope places the body window and the pitch gate
    rough = _boxcar(power, max(1, int(0.005 * sr)))
    rp = int(np.argmax(rough))
    t40 = _last_above(rough, rp, rough[rp] * 10 ** (-40 / 10))
    gate = _last_above(rough, rp, rough[rp] * 10 ** (-GATE_DB / 10))
    gate = n if gate is None else int(gate)
    t1 = min(BODY_S[1] * sr, n if t40 is None else t40)
    t0 = min(BODY_S[0] * sr, t1 / 2)
    b0, b1 = int(t0), int(t1)
    out["body_window_ms"] = [_ms(b0, sr), _ms(b1, sr)]

    z = _analytic(y)
    f0, tonal, body = None, [], None
    if b1 - b0 >= int(FRAME_S[0] * sr):
        body = _spectrum(z, b0, b1, sr)
        tonal = _tonal(body, nyq_hi)
    if tonal:
        top = tonal[0][1]
        f0, l1 = min((t for t in tonal if t[1] >= top - STRONG_DB), key=lambda t: t[0])[:2]
        series = sum(10 ** (t[1] / 10) for t in tonal if _harmonic(t[0], f0))
        if series < 0.5 * sum(10 ** (t[1] / 10) for t in tonal):
            f0 = None
            notes.append("no harmonic series: the partials are listed, no pitch is given")
        elif _noise_flatness(body, tonal, nyq_hi)[0] > NOISY_DB:
            f0 = None
            notes.append("the body is mostly noise: the partials are listed, no pitch is given")
    elif body is None:
        notes.append("the hit is too short for a body window")
    else:
        notes.append("no tonal peak in the body: unpitched")
    pitched = f0 is not None
    out["pitched"] = pitched

    # envelope: the attack ends where a 1 ms envelope first comes within
    # ATTACK_DB of its peak (a crest, so within half a period of a low tone;
    # the argmax itself wanders between near-equal crests when noise rides on
    # them), falls on whole periods of the body pitch
    fine = _boxcar(power, max(1, int(0.001 * sr)))
    out["attack_ms"] = _ms(int(np.argmax(fine >= fine.max() * 10 ** (-ATTACK_DB / 10))), sr)
    if pitched:
        period = sr / f0
        length = int(round(period * math.ceil(0.0025 * f0)))
    else:
        length = int(0.005 * sr)
    env = _boxcar(power, max(1, length))
    ip = int(np.argmax(env))
    for drop in FALLS_DB:
        k = _last_above(env, ip, env[ip] * 10 ** (-drop / 10))
        out["fall_%d_ms" % drop] = None if k is None else _ms(k - ip, sr)
        if k is None:
            notes.append("%s before the hit falls %d dB" % (cut, drop))
    out["truncated"] = out["fall_40_ms"] is None
    k = _last_above(env, ip, env[ip] * 10 ** (-END_DB / 10))
    out["duration_ms"] = _ms(n if k is None else k, sr)
    if k is None:
        notes.append("%s before the hit falls %d dB: duration runs to there" % (cut, END_DB))
    end = n if k is None else max(1, int(k))

    # pitch: start, body, drop, settle, track
    for key in ("pitch_start_hz", "pitch_start_at_ms", "pitch_body_hz", "pitch_drop_st",
                "pitch_settle_ms", "pitch_track_hz"):
        out[key] = None
    if pitched:
        out["pitch_body_hz"] = round(f0, 2)
        dense, reached = _track(z, sr, f0, round((b0 + b1) / 2 / sr, 3), gate)
        if dense:
            c, fs = dense[0]
            out["pitch_start_hz"], out["pitch_start_at_ms"] = round(fs, 1), round(1000 * c, 1)
            out["pitch_drop_st"] = round(12 * math.log2(fs / f0), 2)
            if not reached:
                notes.append("the pitch cannot be followed back to the onset: the start is %.0f ms in"
                             % (1000 * c))
            near = [abs(12 * math.log2(f / f0)) <= SETTLE_ST for _, f in dense]
            for i, (c, _) in enumerate(dense):
                hold = [ok for (cc, _), ok in zip(dense[i:], near[i:]) if cc - c <= SETTLE_HOLD_S]
                if all(hold):
                    out["pitch_settle_ms"] = 0.0 if i == 0 and reached else round(1000 * c, 1)
                    break
            else:
                notes.append("the pitch never settles within %.1f st of the body pitch" % SETTLE_ST)
        else:
            notes.append("no pitch track: the body pitch cannot be followed frame by frame")
        track = {}
        for ms in TRACK_MS:
            t = ms / 1000.0
            if dense and t <= dense[0][0]:
                track[ms] = round(dense[0][1], 1) if reached else None
            else:
                got = [f for c, f in dense if abs(c - t) < 0.0006]   # frames sit on whole ms
                track[ms] = round(got[0], 1) if got else None
        out["pitch_track_hz"] = track
    else:
        notes.append("unpitched: no pitch track and no harmonics")

    # tone: harmonics, inharmonic partials, noise, flatness, click
    out["harmonics_db"] = out["harmonics_floor_db"] = None
    if pitched:
        h, fl = {}, {}
        for k in range(2, 9):
            target = k * f0
            if target > nyq_hi:
                h[k] = fl[k] = None
                continue
            sel = np.nonzero(np.abs(body["hz"] - target) <= max(HARM_TOL * target, body["hz"][1]))[0]
            i = int(sel[np.argmax(body["d"][sel])])
            h[k] = round(max(FLOOR_DB, _interp(body, i)[1] - l1), 1)
            floor = _floor(body, i)
            fl[k] = None if floor is None else round(max(FLOOR_DB, floor - l1), 1)
        out["harmonics_db"], out["harmonics_floor_db"] = h, fl
    ref = l1 if pitched else (tonal[0][1] if tonal else None)
    partials = [t for t in tonal if t[1] >= tonal[0][1] - PARTIALS_DB and not (pitched and _harmonic(t[0], f0))]
    out["partials"] = [{"hz": round(t[0], 1), "db": round(t[1] - ref, 1)} for t in partials[:MAX_PARTIALS]]
    a1 = min(n, int(ATTACK_S * sr))
    att = _spectrum(z, 0, a1, sr)
    out["noise_db"], out["flatness"] = {}, {}
    for name, s in (("attack", att), ("body", body)):
        if s is None:
            out["noise_db"][name] = out["flatness"][name] = None
            continue
        nz, fl = _noise_flatness(s, _tonal(s, nyq_hi) if s is att else tonal, nyq_hi)
        out["noise_db"][name] = None if nz is None else round(nz, 1)
        out["flatness"][name] = None if fl is None else round(fl, 3)
    # click: the attack's power above 2 kHz against the attack's own power, so
    # the decay and the body's level do not move it
    nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
    spec = np.fft.rfft(y, nfft, axis=0)
    spec[np.fft.rfftfreq(nfft, 1.0 / sr) < CLICK_HZ] = 0
    hf = np.fft.irfft(spec, nfft, axis=0)[:a1]
    pa = float(power[:a1].mean())
    out["click_db"] = round(_db(float((hf ** 2).mean()) / pa), 1) if pa > 0 else None

    # balance and width over the whole hit
    hit = y[:end]
    spec = (np.abs(np.fft.rfft(hit, axis=0)) ** 2).mean(axis=1)
    hz = np.fft.rfftfreq(len(hit), 1.0 / sr)
    bands = {name: float(spec[(hz >= lo) & (hz < hi)].sum()) for name, lo, hi in BANDS}
    total = sum(bands.values())
    out["bands_db"] = {k: round(_db(v / total), 1) for k, v in bands.items()} if total > 0 else None
    if hit.shape[1] == 2:
        pm = float((((hit[:, 0] + hit[:, 1]) / 2) ** 2).mean())
        ps = float((((hit[:, 0] - hit[:, 1]) / 2) ** 2).mean())
        out["width"] = round(math.sqrt(ps) / (math.sqrt(pm) + math.sqrt(ps)), 3) if pm + ps > 0 else None
    else:
        out["width"] = 0.0
    out["notes"] = notes
    return out


def profile(path: str | pathlib.Path, hit: int | None = None) -> dict:
    """Envelope, pitch and tone of a one-shot file, relative to the hit itself.

    hits, onsets_ms the hits found in the file and where each starts
    hit             which one this profile is (from 0): `hit`, by default the first
                    that is not cut off
    onset_ms        where the hit starts in the file (first 1 ms within 40 dB of its loudest)
    truncated       the next hit or the end of the file cuts it before it falls 40 dB
    peak_db         sample peak, dBFS (the only absolute level here)
    attack_ms       onset to the envelope peak (first within 1 dB of it)
    fall_10/20/40_ms  envelope peak to the last moment above peak - 10/20/40 dB
    duration_ms     onset to the last moment above peak - 60 dB
    pitched         false for noise or an inharmonic sound: no pitch, no harmonics
    pitch_start_hz  pitch of the first frame (about 1.5 periods in: pitch_start_at_ms)
    pitch_body_hz   pitch over body_window_ms (40-150 ms, shorter for a short hit)
    pitch_drop_st   start minus body, semitones
    pitch_settle_ms when the pitch comes within 0.5 st of the body pitch and stays 10 ms
    pitch_track_hz  pitch at 0, 5, 10, 20, 40, 80, 150, 300 ms (None once the hit is 50 dB down)
    harmonics_db    harmonics 2-8 against the fundamental, on the body (None past 20 kHz)
    harmonics_floor_db  the spectral floor around each of those, so a value near it
                    is the floor, not a harmonic
    partials        strongest peaks that are not harmonics: Hz and dB against the
                    fundamental (or the strongest peak when unpitched)
    noise_db        energy outside the tonal peaks against the total, attack (0-30 ms) and body
    flatness        spectral flatness, attack and body: 0 pure tones .. ~0.56 white noise
    click_db        attack-window (0-30 ms) power above 2 kHz against the attack's
                    total: a click, or any bright attack (a broadband noise layer too)
    bands_db        six-band balance against the hit's 20 Hz-20 kHz total
    width           0 = mono, 0.5 = equal mid and side, 1 = all side
    notes           why a value is None
    """
    x, sr = load(path)
    out = {"file": str(path)}
    out.update(profile_array(x, sr, hit))
    return out


TIMES = (("attack_ms", "attack %s ms, reference %s ms"),
         ("fall_10_ms", "falls 10 dB in %s ms, reference %s ms"),
         ("fall_20_ms", "falls 20 dB in %s ms, reference %s ms"),
         ("fall_40_ms", "falls 40 dB in %s ms, reference %s ms"),
         ("duration_ms", "lasts %s ms, reference %s ms"),
         ("pitch_settle_ms", "pitch settles in %s ms, reference %s ms"))


def _key(d: dict, k: int):
    """d[k] for a profile fresh from profile() (int keys) or read back from JSON (str keys)."""
    return d.get(k, d.get(str(k)))


def _num(v: float) -> str:
    return "%.1f" % v if abs(v) < 10 else "%.0f" % v


def _ordinal(k: int) -> str:
    return {2: "2nd", 3: "3rd"}.get(k, "%dth" % k)


def compare(a: dict, b: dict) -> dict:
    """How profile `a` (the candidate, e.g. a synth take) differs from profile
    `b` (the reference). `diff` holds a - b for every measure both sides have:
    pitch in semitones, times as {"ms", "ratio"} (a / b), the rest in their
    own units. `readings` is one line per difference past its threshold,
    largest first relative to that threshold. A measure that is None on
    either side gives no reading, and levels below the method's floor on both
    sides (harmonics under -50 dB or near the noise, noise share under -25 dB,
    click under -40 dB) are not read."""
    diff, found = {}, []

    def add(score, text):
        found.append((score, text))

    if a.get("pitched") != b.get("pitched"):
        add(math.inf, "pitched (%s Hz), the reference is not" % _num(a["pitch_body_hz"]) if a.get("pitched")
            else "unpitched, the reference is pitched (%s Hz)" % _num(b["pitch_body_hz"]))
    for key, name in (("pitch_body_hz", "pitch_body_st"), ("pitch_start_hz", "pitch_start_st")):
        va, vb = a.get(key), b.get(key)
        diff[name] = round(12 * math.log2(va / vb), 2) if va and vb else None
    d = diff["pitch_body_st"]
    if d is not None and abs(d) > PITCH_ST:
        add(abs(d) / PITCH_ST, "body pitch %.1f st %s the reference (%s Hz, reference %s Hz)"
            % (abs(d), "above" if d > 0 else "below", _num(a["pitch_body_hz"]), _num(b["pitch_body_hz"])))
    va, vb = a.get("pitch_drop_st"), b.get("pitch_drop_st")
    diff["pitch_drop_st"] = round(va - vb, 2) if va is not None and vb is not None else None
    d = diff["pitch_drop_st"]
    if d is not None and abs(d) > DROP_ST:
        add(abs(d) / DROP_ST, "pitch drops %.1f st, reference %.1f st" % (va, vb))

    for key, text in TIMES:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            diff[key] = None
            continue
        d, ratio = va - vb, (va / vb if vb > 0 else None)
        diff[key] = {"ms": round(d, 1), "ratio": None if ratio is None else round(ratio, 3)}
        if abs(d) > TIME_FLOOR_MS and (ratio is None or abs(ratio - 1) > TIME_RATIO):
            line = text % (_num(va), _num(vb))
            if ratio is not None:
                line += " (%d %% %s)" % (round(abs(ratio - 1) * 100), "longer" if d > 0 else "shorter")
            add(abs(ratio - 1) / TIME_RATIO if ratio is not None else abs(d) / TIME_FLOOR_MS, line)

    ha, hb = a.get("harmonics_db"), b.get("harmonics_db")
    diff["harmonics_db"] = None
    if ha and hb:
        diff["harmonics_db"] = {}

        def heard(p, h, k):
            v, floor = _key(h, k), _key(p.get("harmonics_floor_db") or {}, k)
            return v is not None and v > HARM_FLOOR_DB and (floor is None or v - floor >= HARM_CLEAR_DB)
        for k in range(2, 9):
            va, vb = _key(ha, k), _key(hb, k)
            diff["harmonics_db"][k] = None if va is None or vb is None else round(va - vb, 1)
            d = diff["harmonics_db"][k]
            if d is not None and abs(d) > LEVEL_DB and (heard(a, ha, k) or heard(b, hb, k)):
                add(abs(d) / LEVEL_DB, "%s harmonic %.1f dB %s (%s dB, reference %s dB)"
                    % (_ordinal(k), abs(d), "high" if d > 0 else "low", _num(va), _num(vb)))

    for key, floor, step, word in (("noise_db", NOISE_FLOOR_DB, LEVEL_DB, "noise"),
                                   ("flatness", None, None, None)):
        diff[key] = {}
        for part in ("attack", "body"):
            va, vb = (a.get(key) or {}).get(part), (b.get(key) or {}).get(part)
            diff[key][part] = None if va is None or vb is None else round(va - vb, 3 if word is None else 1)
            d = diff[key][part]
            if word and d is not None and abs(d) > step and max(va, vb) > floor:
                add(abs(d) / step, "%s %.1f dB %s than the reference in the %s (%s dB of the total, reference %s dB)"
                    % (word, abs(d), "more" if d > 0 else "less", part, _num(va), _num(vb)))
    va, vb = a.get("click_db"), b.get("click_db")
    diff["click_db"] = round(va - vb, 1) if va is not None and vb is not None else None
    d = diff["click_db"]
    if d is not None and abs(d) > CLICK_STEP_DB and max(va, vb) > CLICK_FLOOR_DB:
        add(abs(d) / CLICK_STEP_DB, "click %.1f dB %s than the reference" % (abs(d), "stronger" if d > 0 else "weaker"))
    va, vb = a.get("width"), b.get("width")
    diff["width"] = round(va - vb, 3) if va is not None and vb is not None else None
    d = diff["width"]
    if d is not None and abs(d) > WIDTH_STEP:
        add(abs(d) / WIDTH_STEP, "width %.2f, reference %.2f (%s)" % (va, vb, "wider" if d > 0 else "narrower"))
    ba, bb = a.get("bands_db"), b.get("bands_db")
    diff["bands_db"] = {k: round(ba[k] - bb[k], 1) for k in ba if k in bb} if ba and bb else None
    va, vb = a.get("peak_db"), b.get("peak_db")
    diff["peak_db"] = round(va - vb, 2) if va is not None and vb is not None else None
    return {"a": a, "b": b, "diff": diff, "readings": [t for _, t in sorted(found, key=lambda f: -f[0])]}
