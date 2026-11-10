"""Measure what an audio file sounds like, locally, in numbers an agent can act on.

Everything here runs in the MCP server process on files: Live's Python API has
no audio tap, so the Remote Script never sees samples. Nothing leaves the
machine and nothing here opens a network connection.

Needs the optional `listen` extra (numpy, soundfile). Importing this module
without them raises ImportError, which the server turns into an install hint.

Choices that are not obvious from the code:

- LUFS follows ITU-R BS.1770-4 (K-weighting, 400 ms blocks at 75 % overlap,
  absolute gate -70 LUFS, relative gate -10 LU, channel weight 1.0 for L, R
  or a mono file's single channel). numpy has no lfilter and a Python IIR loop
  over a 7-minute file takes minutes, so the two biquads run once on an
  impulse and the signal is FFT-convolved with that 8192-tap response
  (overlap-add, so filter state carries across blocks). Filtering each 100 ms
  block separately in the frequency domain was tried first and read up to
  1 LU off on sub-heavy material, from the block edges.
- True peak is 4x band-limited (FFT) oversampling in overlapping chunks, the
  ideal interpolator BS.1770's polyphase FIR approximates.
- Band levels are mean power in dBFS (a full-scale sine reads -3.01 dB); the
  six bands cover 20 Hz-20 kHz, so they sum to the file's power within that range. Spectral comparisons remove the
  median band offset first, so a level difference does not read as a tonal one.
"""
from __future__ import annotations

import math
import pathlib

import numpy as np
import soundfile as sf

BANDS = (("sub", 20.0, 60.0), ("bass", 60.0, 250.0), ("low_mid", 250.0, 500.0),
         ("mid", 500.0, 2000.0), ("high_mid", 2000.0, 6000.0), ("high", 6000.0, 20000.0))
LOW_END_HZ = 120.0
FLOOR_DB = -120.0
FRAME = 8192            # spectrum frame; hop is half
BATCH = 128             # frames per FFT batch, bounds memory on long files
FLAG_DB = 3.0           # audio_compare flags a band that differs by more than this
K_TAPS = 8192           # K-weighting impulse response length; the high-pass is below -170 dB by then


def _db(power: float) -> float:
    return max(FLOOR_DB, 10.0 * math.log10(power)) if power > 0 else FLOOR_DB


def _db_amp(amp: float) -> float:
    return _db(amp * amp)


def load(path: str | pathlib.Path) -> tuple[np.ndarray, int]:
    """(samples as float32 [n, channels], sample rate). Mono stays one channel.
    More than two channels is refused rather than silently cut to L/R."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    _check_channels(x)
    return x, sr


def _check_channels(x: np.ndarray) -> None:
    if x.ndim != 2 or not 1 <= x.shape[1] <= 2:
        raise ValueError("%s channels: only mono and stereo are measured"
                         % (x.shape[1] if x.ndim == 2 else "?"))


def _stereo(x: np.ndarray) -> np.ndarray:
    if x.shape[1] == 1:
        return np.repeat(x, 2, axis=1)
    return x[:, :2]


# --------------------------------------------------------------- loudness

def k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The two BS.1770 biquads (b, a) for sample rate `sr`: a high shelf (+4 dB
    above ~1.7 kHz) then a high-pass (~38 Hz). Derived from the analog prototype
    so any rate works; at 48 kHz they reproduce the standard's table."""
    fc, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * fc / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = (np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0,
                       (vh - vb * k / q + k * k) / a0]),
             np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]))
    fc, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * fc / sr)
    a0 = 1.0 + k / q + k * k
    hp = (np.array([1.0, -2.0, 1.0]),
          np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]))
    return [shelf, hp]


def _biquad(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Direct form I, sample by sample. Only ever run on the short impulse."""
    y = np.zeros_like(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        out = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, v, y1, out
        y[i] = out
    return y


def k_impulse(sr: int, taps: int = K_TAPS) -> np.ndarray:
    h = np.zeros(taps)
    h[0] = 1.0
    for b, a in k_weighting(sr):
        h = _biquad(b, a, h)
    return h


def _subblock_powers(x: np.ndarray, sr: int) -> np.ndarray:
    """K-weighted mean square per 100 ms sub-block, summed over channels.
    Shape [n_subblocks]; a trailing partial sub-block is dropped."""
    n = int(round(sr * 0.1))
    m = x.shape[0] // n
    if m == 0:
        return np.zeros(0)
    h = k_impulse(sr)
    per = 16                                  # sub-blocks per convolution chunk
    L = n * per
    nfft = 1 << int(math.ceil(math.log2(L + len(h) - 1)))
    H = np.fft.rfft(h, nfft)
    out = np.zeros(m)
    for c in range(x.shape[1]):
        tail = np.zeros(len(h) - 1)
        for s in range(0, m * n, L):
            chunk = x[s:min(s + L, m * n), c].astype(np.float64)
            y = np.fft.irfft(np.fft.rfft(chunk, nfft) * H, nfft)[:len(chunk) + len(h) - 1]
            y[:len(tail)] += tail
            emit, tail = y[:len(chunk)], y[len(chunk):].copy()
            k = len(chunk) // n
            out[s // n:s // n + k] += (emit[:k * n] ** 2).reshape(k, n).mean(axis=1)
            if len(tail) < len(h) - 1:
                tail = np.pad(tail, (0, len(h) - 1 - len(tail)))
    return out


def _lufs(power: np.ndarray | float) -> np.ndarray | float:
    return -0.691 + 10.0 * np.log10(np.maximum(power, 1e-30))


def loudness(x: np.ndarray, sr: int) -> dict:
    """Integrated LUFS (gated), short-term max (3 s) and momentary max (400 ms).
    A value that cannot be measured (file too short, or all of it below the
    -70 LUFS gate) is None, never a placeholder number."""
    sub = _subblock_powers(x, sr)
    out = {"integrated_lufs": None, "short_term_max_lufs": None, "momentary_max_lufs": None}
    if len(sub) < 4:
        return out
    c = np.concatenate([[0.0], np.cumsum(sub)])
    block = (c[4:] - c[:-4]) / 4.0                       # 400 ms, 100 ms hop
    lb = _lufs(block)
    out["momentary_max_lufs"] = round(float(lb.max()), 2) if lb.max() > -70.0 else None
    gated = block[lb > -70.0]
    if len(gated):
        rel = _lufs(gated.mean()) - 10.0
        gated = block[(lb > -70.0) & (lb > rel)]
        if len(gated):
            out["integrated_lufs"] = round(float(_lufs(gated.mean())), 2)
    if len(sub) >= 30:
        st = float(_lufs(((c[30:] - c[:-30]) / 30.0).max()))
        out["short_term_max_lufs"] = round(st, 2) if st > -70.0 else None
    return out


def true_peak(x: np.ndarray, factor: int = 4, chunk: int = 1 << 16, margin: int = 1024) -> float:
    """Highest absolute value of the band-limited signal, as linear amplitude."""
    peak = 0.0
    n = x.shape[0]
    for c in range(x.shape[1]):
        ch = np.pad(x[:, c], margin)          # silence around the file, so its ends do not wrap
        n = len(ch)
        for s in range(0, n, chunk):
            lo, hi = max(0, s - margin), min(n, s + chunk + margin)
            seg = ch[lo:hi].astype(np.float64)
            spec = np.fft.rfft(seg)
            if len(seg) % 2 == 0:
                spec[-1] *= 0.5     # the Nyquist bin splits between +fs/2 and -fs/2 once padded
            up = np.fft.irfft(spec, n=len(seg) * factor) * factor
            core = up[(s - lo) * factor:(s - lo + min(chunk, n - s)) * factor]
            if len(core):
                peak = max(peak, float(np.abs(core).max()))
    return max(peak, float(np.abs(x).max()) if x.size else 0.0)


def _peak_rms(x: np.ndarray, chunk: int = 1 << 20) -> tuple[float, float]:
    peak, ss = 0.0, 0.0
    for s in range(0, x.shape[0], chunk):
        c = x[s:s + chunk].astype(np.float64)
        peak = max(peak, float(np.abs(c).max()))
        ss += float((c * c).sum())
    return peak, (math.sqrt(ss / x.size) if x.size else 0.0)


# --------------------------------------------------------------- spectrum

def _spectra(x: np.ndarray, sr: int) -> dict:
    """Frame-averaged one-sided power spectra of L, R, mid and side, plus the
    low-end cross spectrum. Powers are scaled so that sum over bins = mean square."""
    n = FRAME if x.shape[0] >= FRAME else max(256, 1 << int(math.log2(max(2, x.shape[0]))))
    hop = n // 2
    if x.shape[0] < n:
        x = np.pad(x, ((0, n - x.shape[0]), (0, 0)))
    win = np.hanning(n)
    scale = 1.0 / (n * (win ** 2).sum())
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    onesided = np.full(len(freqs), 2.0)
    onesided[0] = 1.0
    if n % 2 == 0:
        onesided[-1] = 1.0
    starts = np.arange(0, x.shape[0] - n + 1, hop)
    acc = {k: np.zeros(len(freqs)) for k in ("l", "r", "m", "s")}
    cross = np.zeros(len(freqs))
    for b in range(0, len(starts), BATCH):
        idx = starts[b:b + BATCH, None] + np.arange(n)[None, :]
        L = np.fft.rfft(x[idx, 0].astype(np.float64) * win, axis=1)
        R = np.fft.rfft(x[idx, 1].astype(np.float64) * win, axis=1)
        M, S = (L + R) / 2.0, (L - R) / 2.0
        for k, X in (("l", L), ("r", R), ("m", M), ("s", S)):
            acc[k] += (np.abs(X) ** 2).sum(axis=0)
        cross += np.real(L * np.conj(R)).sum(axis=0)
    f = len(starts)
    out = {k: v * scale * onesided / f for k, v in acc.items()}
    out["cross"] = cross * scale * onesided / f
    out["freqs"] = freqs
    return out


def _band(p: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    return float(p[(freqs >= lo) & (freqs < hi)].sum())


def analyze_array(x: np.ndarray, sr: int) -> dict:
    """The full profile of one signal [n, channels]. See analyze()."""
    _check_channels(x)
    st = _stereo(x)
    sp = _spectra(st, sr)
    freqs = sp["freqs"]
    lr = (sp["l"] + sp["r"]) / 2.0                     # mean channel power
    total = float(lr.sum())
    peak, rms = _peak_rms(st)
    bands, side_mid = {}, {}
    for name, lo, hi in BANDS:
        bands[name] = round(_db(_band(lr, freqs, lo, hi)), 2)
        pm, ps = _band(sp["m"], freqs, lo, hi), _band(sp["s"], freqs, lo, hi)
        side_mid[name] = round(_db(ps) - _db(pm), 2) if pm > 0 else None
    low = (freqs >= BANDS[0][1]) & (freqs < LOW_END_HZ)   # 20-120 Hz: no DC or infrasound
    pl, pr = float(sp["l"][low].sum()), float(sp["r"][low].sum())
    corr = float(sp["cross"][low].sum() / math.sqrt(pl * pr)) if pl > 0 and pr > 0 else None
    pm, ps = float(sp["m"].sum()), float(sp["s"].sum())
    out = {"duration_s": round(x.shape[0] / sr, 3), "sample_rate": sr, "channels": x.shape[1],
           "sample_peak_db": round(_db_amp(peak), 2),
           "true_peak_dbtp": round(_db_amp(true_peak(st)), 2),
           "rms_db": round(_db_amp(rms), 2),
           "crest_db": round(_db_amp(peak) - _db_amp(rms), 2) if rms > 0 else None,
           "centroid_hz": round(float((freqs * lr).sum() / total), 1) if total > 0 else None,
           "width": round(math.sqrt(ps) / (math.sqrt(pm) + math.sqrt(ps)), 3) if pm + ps > 0 else None,
           "low_end_correlation": round(corr, 3) if corr is not None else None,
           "bands_db": bands, "side_minus_mid_db": side_mid}
    out.update(loudness(x, sr))     # BS.1770 weights the real channels: mono is not dual mono
    return out


def analyze(path: str | pathlib.Path) -> dict:
    """Loudness, peak, dynamics, spectrum and stereo profile of an audio file.

    integrated_lufs / short_term_max_lufs / momentary_max_lufs  BS.1770-4
    true_peak_dbtp, sample_peak_db, rms_db, crest_db (peak minus RMS)
    bands_db         mean power per band in dBFS: sub 20-60, bass 60-250,
                     low_mid 250-500, mid 500-2k, high_mid 2k-6k, high 6k-20k
    side_minus_mid_db  per band; above 0 means more side than centre
    width            0 = mono, 0.5 = equal mid and side, 1 = all side
    low_end_correlation  L/R correlation below 120 Hz: 1 mono-safe, <0 cancels in mono
    centroid_hz      spectral centre of mass
    """
    x, sr = load(path)
    out = analyze_array(x, sr)
    out["file"] = str(path)
    return out


def compare(a: dict, b: dict) -> dict:
    """Differences a - b between two analyze() results (a = the Set, b = the
    reference). Band balance is compared after removing the median band offset,
    so a quieter master does not read as a darker one; `flagged_bands` names the
    bands that still differ by more than FLAG_DB."""
    names = [n for n, _, _ in BANDS]
    raw = {n: a["bands_db"][n] - b["bands_db"][n] for n in names}
    offset = float(np.median(list(raw.values())))
    balance = {n: round(raw[n] - offset, 2) for n in names}
    flagged = [{"band": n, "diff_db": balance[n],
                "reading": "more than the reference" if balance[n] > 0 else "less than the reference"}
               for n in names if abs(balance[n]) > FLAG_DB]
    diff = {}
    for k in ("integrated_lufs", "short_term_max_lufs", "true_peak_dbtp", "crest_db",
              "rms_db", "width", "low_end_correlation", "centroid_hz"):
        if a.get(k) is not None and b.get(k) is not None:
            diff[k] = round(a[k] - b[k], 3)
    sm = {}
    for n in names:
        va, vb = a["side_minus_mid_db"].get(n), b["side_minus_mid_db"].get(n)
        if va is not None and vb is not None:
            sm[n] = round(va - vb, 2)
    return {"a": a.get("file"), "b": b.get("file"), "level_offset_db": round(offset, 2),
            "band_balance_db": balance, "flagged_bands": flagged,
            "side_minus_mid_diff_db": sm, "diff": diff}
