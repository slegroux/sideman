"""Profile a drum one-shot in the terms a synth is programmed in: envelope,
pitch and pitch drop, harmonics, noise, click, band balance and width.

Every measure is relative to the hit itself (its onset, its peak, its own
fundamental), so two files at different levels, sample rates or with
different leading silence profile alike. Local only; needs the `listen` extra.

Choices that are not obvious from the code:

- The onset is the first sample within 40 dB of the file's peak; everything
  before it is dropped and every time is counted from it.
- Decay uses a power envelope smoothed over whole periods of the body pitch
  (5 ms for an unpitched hit), so partials beating against each other do not
  read as decay. A fall time is the last moment the envelope is still above
  the threshold, measured from the envelope's peak. The attack time is the
  loudest moment of a 1 ms-smoothed power envelope instead: its largest
  crest, so it is good to about half a period of a low tone. (An analytic
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

ONSET_DB = 40.0          # the onset is the first sample within this of the peak
FALLS_DB = (10, 20, 40)
END_DB = 60.0            # duration runs to the last moment within this of the peak
GATE_DB = 50.0           # pitch frames must end before the hit falls this far
TRACK_MS = (0, 5, 10, 20, 40, 80, 150, 300)
ATTACK_S = 0.030
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
MAX_S = 10.0             # a one-shot longer than this is cut, with a note


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


def _tonal(s: dict, hi_hz: float) -> list[tuple[float, float, int]]:
    """Peaks standing PROMINENT_DB above the median of a ring 2-10 bins away
    (outside their own main lobe), as (Hz, dB, bin), strongest first."""
    d, h = s["d"], s["bin"]
    out = []
    for i in _peaks(s, hi_hz):
        ring = np.r_[d[max(0, i - 10 * h):max(0, i - 2 * h)], d[i + 2 * h + 1:i + 10 * h + 1]]
        if len(ring) and d[i] - np.median(ring) >= PROMINENT_DB:
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


def profile_array(x: np.ndarray, sr: int) -> dict:
    """The one-shot profile of one signal [n, channels]. See profile()."""
    _check_channels(x)
    x = np.asarray(x, dtype=np.float64)
    notes = []
    mag = np.abs(x).max(axis=1)
    peak = float(mag.max()) if len(mag) else 0.0
    if peak <= 0:
        raise ValueError("the file is silent")
    onset = int(np.argmax(mag >= peak * 10 ** (-ONSET_DB / 20)))
    y = x[onset:]
    if len(y) < FRAME_S[0] * sr:
        raise ValueError("the hit is shorter than %g ms" % (1000 * FRAME_S[0]))
    if len(y) > MAX_S * sr:
        y = y[:int(MAX_S * sr)]
        notes.append("only the first %g s after the onset are measured" % MAX_S)
    n, nyq_hi = len(y), min(20000.0, 0.45 * sr)
    power = (y ** 2).mean(axis=1)
    out = {"sample_rate": sr, "channels": x.shape[1], "onset_ms": _ms(onset, sr),
           "peak_db": round(_db(peak * peak), 2)}

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

    # envelope: attack at the loudest 1 ms (its largest crest, so within half
    # a period of a low tone), falls on whole periods of the body pitch
    fine = _boxcar(power, max(1, int(0.001 * sr)))
    out["attack_ms"] = _ms(int(np.argmax(fine)), sr)
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
            notes.append("the file ends before the hit falls %d dB" % drop)
    k = _last_above(env, ip, env[ip] * 10 ** (-END_DB / 10))
    out["duration_ms"] = _ms(n if k is None else k, sr)
    if k is None:
        notes.append("duration is the file's length: the hit does not fall %d dB before it ends" % END_DB)
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
    out["harmonics_db"] = None
    if pitched:
        h = {}
        for k in range(2, 9):
            target = k * f0
            if target > nyq_hi:
                h[k] = None
                continue
            sel = np.nonzero(np.abs(body["hz"] - target) <= max(HARM_TOL * target, body["hz"][1]))[0]
            i = int(sel[np.argmax(body["d"][sel])])
            h[k] = round(max(FLOOR_DB, _interp(body, i)[1] - l1), 1)
        out["harmonics_db"] = h
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
    out["click_db"] = None
    if body is not None:
        nfft = 1 << int(math.ceil(math.log2(max(2, 2 * n))))
        spec = np.fft.rfft(y, nfft, axis=0)
        spec[np.fft.rfftfreq(nfft, 1.0 / sr) < CLICK_HZ] = 0
        hf = np.fft.irfft(spec, nfft, axis=0)[:a1]
        pb = float(power[b0:b1].mean())
        if pb > 0:
            out["click_db"] = round(_db(float((hf ** 2).mean()) / pb), 1)

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


def profile(path: str | pathlib.Path) -> dict:
    """Envelope, pitch and tone of a one-shot file, relative to the hit itself.

    onset_ms        where the hit starts in the file (first sample within 40 dB of the peak)
    peak_db         sample peak, dBFS (the only absolute level here)
    attack_ms       onset to the envelope peak
    fall_10/20/40_ms  envelope peak to the last moment above peak - 10/20/40 dB
    duration_ms     onset to the last moment above peak - 60 dB
    pitched         false for noise or an inharmonic sound: no pitch, no harmonics
    pitch_start_hz  pitch of the first frame (about 1.5 periods in: pitch_start_at_ms)
    pitch_body_hz   pitch over body_window_ms (40-150 ms, shorter for a short hit)
    pitch_drop_st   start minus body, semitones
    pitch_settle_ms when the pitch comes within 0.5 st of the body pitch and stays 10 ms
    pitch_track_hz  pitch at 0, 5, 10, 20, 40, 80, 150, 300 ms (None once the hit is 50 dB down)
    harmonics_db    harmonics 2-8 against the fundamental, on the body (None past 20 kHz)
    partials        strongest peaks that are not harmonics: Hz and dB against the
                    fundamental (or the strongest peak when unpitched)
    noise_db        energy outside the tonal peaks against the total, attack (0-30 ms) and body
    flatness        spectral flatness, attack and body: 0 pure tones .. ~0.56 white noise
    click_db        attack-window power above 2 kHz against the body's mean power
    bands_db        six-band balance against the hit's 20 Hz-20 kHz total
    width           0 = mono, 0.5 = equal mid and side, 1 = all side
    notes           why a value is None
    """
    x, sr = load(path)
    out = {"file": str(path)}
    out.update(profile_array(x, sr))
    return out
