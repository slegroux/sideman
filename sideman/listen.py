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


# --------------------------------------------------------------- mix report
#
# A capture manifest (written by `capture`, or by hand) places a few
# time-aligned files on the Set's bar grid:
#
#   {"tempo": 122.0, "beats_per_bar": 4, "offset_s": 0.0,
#    "sources": {"master": "/abs/master.wav", "kick": "...", "bass": "..."},
#    "sections": [{"name": "INTRO A", "start_bar": 0, "bars": 8}, ...]}
#
# start_bar counts from the files' bar 0, which sits at offset_s seconds.
# Every source shares that timeline. "master" is required; masking needs a
# "kick" (or "drums") and a "bass" source.

MASK_LO, MASK_HI = 40.0, 120.0
MASK_FRAME = 4096        # ~85 ms at 48 kHz: short enough to see a sidechain duck
MASK_FLAG = 0.5          # overlap above this is flagged...
MASK_AUDIBLE_DB = -20.0  # ...unless the quieter part is this far below the louder in 40-120 Hz
CHANGE_BLOCK_BARS = 4    # change rate is measured between consecutive 4-bar blocks
CHANGE_DB = 1.0          # a block differing from the last by less than this is "no change"
CHANGE_RANGE_DB = 40.0   # bands more than this below a block's loudest band count as that floor
STATIC_MAX_BARS = 8      # flag a run longer than this without a change (the 4/8-bar rule)
SILENT_DB = -70.0        # a block whose loudest third-octave band is under this is silence


def _check_sections(sections: list) -> None:
    for sec in sections:
        if not isinstance(sec, dict):
            raise ValueError("each section must be an object with start_bar and bars, got %r" % (sec,))
        sb, nb = sec.get("start_bar"), sec.get("bars")
        if not (isinstance(sb, int) and isinstance(nb, int)) or sb < 0 or nb < 1:
            raise ValueError("section %r needs whole-number start_bar >= 0 and bars >= 1"
                             % sec.get("name", sec))


def _load_manifest(manifest: dict | str | pathlib.Path) -> tuple[dict, pathlib.Path | None]:
    """The manifest as a dict, with every source resolved to an existing
    absolute path (relative paths are taken from the manifest's folder) and
    tempo, sections and bar numbers checked. Raises ValueError saying what is wrong."""
    import json
    mpath = None
    if isinstance(manifest, dict):
        m = dict(manifest)
    elif isinstance(manifest, str) and manifest.lstrip().startswith("{"):
        m = json.loads(manifest)
    else:
        mpath = pathlib.Path(manifest).expanduser().resolve()
        if not mpath.is_file():
            raise ValueError("no such manifest: %s" % mpath)
        m = json.loads(mpath.read_text())
    for key in ("tempo", "sources", "sections"):
        if key not in m:
            raise ValueError("manifest has no %r" % key)
    if not isinstance(m["tempo"], (int, float)) or not math.isfinite(m["tempo"]) or m["tempo"] <= 0:
        raise ValueError("tempo must be a number above 0, got %r" % (m["tempo"],))
    if not isinstance(m.get("beats_per_bar", 4), (int, float)) or m.get("beats_per_bar", 4) <= 0:
        raise ValueError("beats_per_bar must be above 0")
    if "master" not in m["sources"]:
        raise ValueError("manifest sources need a 'master'")
    if not m["sections"]:
        raise ValueError("manifest has no sections")
    _check_sections(m["sections"])
    base = mpath.parent if mpath else None
    srcs = {}
    for role, path in m["sources"].items():
        p = pathlib.Path(path).expanduser()
        if not p.is_absolute() and mpath is None:
            raise ValueError("source %r: %s is relative; a manifest given inline needs absolute paths"
                             % (role, path))
        p = (p if p.is_absolute() else base / p).resolve()
        if not p.is_file():
            raise ValueError("source %r: no such audio file: %s" % (role, p))
        srcs[role] = str(p)
    m["sources"] = srcs
    return m, mpath


def _bar_seconds(m: dict) -> float:
    return float(m.get("beats_per_bar", 4)) * 60.0 / float(m["tempo"])


def _slice(x: np.ndarray, sr: int, t0: float, t1: float) -> np.ndarray:
    return x[max(0, int(round(t0 * sr))):max(0, int(round(t1 * sr)))]


def masking(kick: np.ndarray, bass: np.ndarray, sr: int) -> float | None:
    """How kick and bass share 40-120 Hz, over ~85 ms frames. Returns
    {"overlap", "quieter_db"} or None when either is silent there.

    overlap: how much of the quieter part's energy sits in the same
    time-frequency cells as the other's, sum(min) / min(sum kick, sum bass).
    0 = they never coincide (separate notes, or a sidechain that ducks the
    bass under each hit); 1 = the quieter one is entirely under the louder.
    quieter_db: the quieter part's 40-120 Hz energy relative to the louder's.
    mix_report flags masking only when both are high, since a part 40 dB down
    is inaudible whatever it overlaps."""
    k, b = _stereo(kick).mean(axis=1), _stereo(bass).mean(axis=1)
    n = min(len(k), len(b))
    if n < MASK_FRAME:
        return None
    win = np.hanning(MASK_FRAME)
    freqs = np.fft.rfftfreq(MASK_FRAME, 1.0 / sr)
    sel = (freqs >= MASK_LO) & (freqs < MASK_HI)
    starts = np.arange(0, n - MASK_FRAME + 1, MASK_FRAME // 2)
    lo_sum = k_sum = b_sum = 0.0
    for s in range(0, len(starts), BATCH):
        idx = starts[s:s + BATCH, None] + np.arange(MASK_FRAME)[None, :]
        pk = np.abs(np.fft.rfft(k[idx].astype(np.float64) * win, axis=1)[:, sel]) ** 2
        pb = np.abs(np.fft.rfft(b[idx].astype(np.float64) * win, axis=1)[:, sel]) ** 2
        lo_sum += float(np.minimum(pk, pb).sum())
        k_sum += float(pk.sum())
        b_sum += float(pb.sum())
    if k_sum == 0 or b_sum == 0:
        return None
    return {"overlap": round(lo_sum / min(k_sum, b_sum), 3),
            "quieter_db": round(_db(min(k_sum, b_sum) / max(k_sum, b_sum)), 1)}


def _third_octave_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Mean power per third-octave band, 25 Hz-16 kHz (below Nyquist), in dB."""
    sp = _spectra(_stereo(x), sr)
    lr = (sp["l"] + sp["r"]) / 2.0
    centres = 1000.0 * 2.0 ** (np.arange(-16, 13) / 3.0)
    centres = centres[centres * 2 ** (1 / 6) <= sr / 2.0]
    return np.array([_db(_band(lr, sp["freqs"], c / 2 ** (1 / 6), c * 2 ** (1 / 6))) for c in centres])


def change_rate(x: np.ndarray, sr: int, bar_s: float, offset_s: float, first_bar: int,
                last_bar: int) -> dict:
    """Spectral change between consecutive 4-bar blocks on the bar grid: the mean
    absolute third-octave difference (dB) over the bands sounding in either
    block, each block's bands floored at 40 dB under its loudest, so an
    element far below the mix entering does not read as a 70 dB jump. Runs longer than 8 bars
    with every step under CHANGE_DB are flagged. Silent blocks break a run and
    are never flagged. Each block is compared with the one before it, so very
    slow automation (under 1 dB per 4 bars) also reads as no change."""
    blocks, vecs, floors, steps = [], [], [], []
    for b0 in range(first_bar, last_bar, CHANGE_BLOCK_BARS):
        b1 = min(b0 + CHANGE_BLOCK_BARS, last_bar)
        seg = _slice(x, sr, offset_s + b0 * bar_s, offset_s + b1 * bar_s)
        if len(seg) < sr // 10:
            continue
        v = _third_octave_db(seg, sr)
        silent = bool(v.max() < SILENT_DB)
        floor = v.max() - CHANGE_RANGE_DB
        v = np.maximum(v, floor)
        blocks.append((b0, b1, silent))
        if vecs:
            p = vecs[-1]
            live = (v > floor) | (p > floors[-1])
            d = float(np.abs(v - p)[live].mean()) if live.any() else 0.0
            steps.append({"at_bar": b0, "change_db": round(d, 2), "silent": silent})
        vecs.append(v)
        floors.append(floor)
    flags, run_start = [], 0
    for i in range(1, len(blocks) + 1):
        if i == len(blocks) or steps[i - 1]["change_db"] >= CHANGE_DB or blocks[i][2] != blocks[i - 1][2]:
            b0, b1, silent = blocks[run_start][0], blocks[i - 1][1], blocks[run_start][2]
            if b1 - b0 > STATIC_MAX_BARS and not silent:
                flags.append({"start_bar": b0, "end_bar": b1, "bars": b1 - b0})
            run_start = i
    return {"block_bars": CHANGE_BLOCK_BARS,
            "steps": steps, "static_runs": flags}


# ------------------------------------------------------- reference profile
#
# A reference track that comes with stems, measured once per section so a
# capture of the Set can be compared with it later (mix_report's `reference`).
# Levels are mean power in dBFS over stereo (mono is centred); a role's
# "vs_kick_db" is its level minus the kick's in the same section, which cancels
# a gain the stems and the mix share.

PROFILE_KIND = "reference_profile"
ABSENT_DB = 60.0           # a role this far under the mix (or exactly silent) is absent, not -120 dB
ELEMENT_GAP_DB = 1.0       # an element's ratio to the kick differing by more than this is a gap
CAPTURE_IS_REF_DB = 0.05   # a master capture within this level...
CAPTURE_IS_REF_WIDTH = 1e-3  # ...and this width of the reference mix is the reference itself
KICK_OUT_DB = 30.0         # the kick is out where it sits this far under the loudest other role
DRUM_ROLES = ("kick", "toms", "snare", "hh", "clap", "rim")   # roles the drum layer reads


def _level_db(x: np.ndarray) -> float:
    """Mean power of a stereo segment in dBFS; FLOOR_DB when it is silent or empty."""
    return _db_amp(_peak_rms(x)[1])


def _bar_levels(x: np.ndarray, sr: int, t0: float, bar_s: float, first_bar: int, n: int) -> list:
    """Level in dBFS of each of `n` bars from `first_bar` (bar 0 at t0 s); None for a silent bar."""
    out = []
    for i in range(n):
        lv = _level_db(_slice(x, sr, t0 + (first_bar + i) * bar_s, t0 + (first_bar + i + 1) * bar_s))
        out.append(None if lv <= FLOOR_DB else round(lv, 2))
    return out


def _bands_width(x: np.ndarray, sr: int) -> tuple[dict, float | None]:
    """Six-band mean power (dBFS) and width of a stereo segment, as analyze_array reads them."""
    sp = _spectra(x, sr)
    lr = (sp["l"] + sp["r"]) / 2.0
    bands = {name: round(_db(_band(lr, sp["freqs"], lo, hi)), 2) for name, lo, hi in BANDS}
    pm, ps = float(sp["m"].sum()), float(sp["s"].sum())
    return bands, (round(math.sqrt(ps) / (math.sqrt(pm) + math.sqrt(ps)), 3) if pm + ps > 0 else None)


def _mean_db(vals: list) -> float | None:
    """Power mean of per-bar dB values (None counts as silence); None when all are silent."""
    p = float(np.mean([0.0 if v is None else 10.0 ** (v / 10.0) for v in vals])) if vals else 0.0
    return 10.0 * math.log10(p) if p > 0 else None


def _arc(role_bars: list, kick_bars: list) -> list:
    """The role's level against the kick for each CHANGE_BLOCK_BARS block of bars (None where
    either is silent)."""
    out = []
    for i in range(0, len(role_bars), CHANGE_BLOCK_BARS):
        r, k = _mean_db(role_bars[i:i + CHANGE_BLOCK_BARS]), _mean_db(kick_bars[i:i + CHANGE_BLOCK_BARS])
        out.append(None if r is None or k is None else round(r - k, 2))
    return out


def _input(p: pathlib.Path) -> dict:
    st = p.stat()
    return {"path": str(p), "mtime": st.st_mtime, "size": st.st_size}


def reference_profile(mix: str | pathlib.Path, stems: dict, tempo: float, t0: float, sections: list,
                      beats_per_bar: int = 4, out: str | pathlib.Path | None = None) -> dict:
    """Profile of a reference track and its stems, per section; written as JSON.

    `stems` maps a role to a file path or a list of paths summed into that role
    (`drums` = hh + snare + toms). Roles are free-form; `kick` is the yardstick
    for every level against the kick, and "mix" is reserved. `sections` is a list
    of {"name", "start_bar", "bars"} as in a capture manifest, names unique;
    `t0` is where bar 0 sits in the files, in seconds.

    Per section: the mix's level (mean power, dBFS), integrated LUFS, six bands,
    width and per-bar levels; and for each role its level, its level against the
    kick (`vs_kick_db`), per-bar levels, six bands and width. A role that is
    silent, or more than ABSENT_DB under the mix, in a section is
    {"absent": true} there with no numbers. Files must share one sample rate
    (refused otherwise); of different lengths they are trimmed to the shortest
    and `notes` says so.

    Writes <mix stem>.profile.json next to the mix (`out`: a file path ending
    in .json, or a folder) and returns the profile with its `json_path`. The
    profile records each input's path, mtime and size under `inputs`;
    profile_stale() says which have changed since."""
    import json
    if not isinstance(tempo, (int, float)) or not math.isfinite(tempo) or tempo <= 0:
        raise ValueError("tempo must be a number above 0, got %r" % (tempo,))
    if not isinstance(t0, (int, float)) or not math.isfinite(t0) or t0 < 0:
        raise ValueError("t0 must be a number of seconds >= 0, got %r" % (t0,))
    if not isinstance(beats_per_bar, (int, float)) or beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be above 0")
    if not sections:
        raise ValueError("no sections")
    _check_sections(sections)
    names = [s.get("name", "bar %d" % s["start_bar"]) for s in sections]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError("section names must be unique (mix_report pairs sections by name): %s" % ", ".join(dup))
    if not stems:
        raise ValueError("no stems")

    def existing(label: str, path) -> pathlib.Path:
        p = pathlib.Path(path).expanduser().resolve()
        if not p.is_file():
            raise ValueError("%s: no such audio file: %s" % (label, p))
        return p

    mixp = existing("mix", mix)
    files: dict[str, list[pathlib.Path]] = {}
    for role, v in stems.items():
        if not isinstance(role, str) or not role or role == "mix":
            raise ValueError("stem role must be a non-empty name other than 'mix', got %r" % (role,))
        paths = [v] if isinstance(v, (str, pathlib.Path)) else list(v)
        if not paths:
            raise ValueError("stem %r has no files" % role)
        files[role] = [existing("stem %r" % role, p) for p in paths]
    infos = {p: sf.info(str(p)) for p in [mixp] + [p for ps in files.values() for p in ps]}
    rates = {p: i.samplerate for p, i in infos.items()}
    if len(set(rates.values())) > 1:
        raise ValueError("files must share one sample rate: " + ", ".join(
            "%s %d Hz" % (p.name, r) for p, r in rates.items()))
    sr = next(iter(rates.values()))
    frames = {p: i.frames for p, i in infos.items()}
    n = min(frames.values())
    notes = []
    if len(set(frames.values())) > 1:
        shortest = min(frames, key=frames.get)
        notes.append("files differ in length (%.2f to %.2f s); all trimmed to the shortest, %s (%.2f s)"
                     % (min(frames.values()) / sr, max(frames.values()) / sr, shortest.name, n / sr))
    bar_s = float(beats_per_bar) * 60.0 / float(tempo)
    end_s = t0 + max(s["start_bar"] + s["bars"] for s in sections) * bar_s
    if n / sr + 0.01 < end_s:
        raise ValueError("the files are %.2f s long but the sections run to %.2f s" % (n / sr, end_s))
    if "kick" not in files:
        notes.append("no kick stem: levels against the kick are left out")

    def span(sec):
        a = t0 + sec["start_bar"] * bar_s
        return a, a + sec["bars"] * bar_s

    x, _ = load(mixp)
    x = x[:n]
    mix_rows, mix_levels = [], []
    for sec in sections:
        a = analyze_array(_slice(x, sr, *span(sec)), sr)
        mix_levels.append(a["rms_db"])
        mix_rows.append({"level_db": a["rms_db"], "integrated_lufs": a["integrated_lufs"],
                         "width": a["width"], "bands_db": a["bands_db"],
                         "per_bar_db": _bar_levels(_stereo(x), sr, t0, bar_s, sec["start_bar"], sec["bars"])})
    role_rows: dict[str, list] = {}
    raw: dict[str, list] = {}
    for role, paths in files.items():
        y = None
        for p in paths:
            z = _stereo(load(p)[0])[:n]
            y = z.copy() if y is None else y + z
        role_rows[role], raw[role] = [], []
        for sec, mlv in zip(sections, mix_levels):
            seg = _slice(y, sr, *span(sec))
            lv = _level_db(seg)
            if lv <= FLOOR_DB or lv < mlv - ABSENT_DB:
                role_rows[role].append({"absent": True})
                raw[role].append(None)
                continue
            bands, width = _bands_width(seg, sr)
            role_rows[role].append({"absent": False, "level_db": round(lv, 2), "vs_kick_db": None,
                                    "per_bar_db": _bar_levels(y, sr, t0, bar_s, sec["start_bar"], sec["bars"]),
                                    "bands_db": bands, "width": width})
            raw[role].append(lv)
    if "kick" in files:
        for i in range(len(sections)):
            for role in files:
                if raw[role][i] is not None and raw["kick"][i] is not None:
                    role_rows[role][i]["vs_kick_db"] = round(raw[role][i] - raw["kick"][i], 2)
    profile = {
        "kind": PROFILE_KIND, "version": 1, "mix": str(mixp), "tempo": tempo,
        "beats_per_bar": beats_per_bar, "t0": t0, "sample_rate": sr, "notes": notes,
        "inputs": {"mix": [_input(mixp)], "stems": {r: [_input(p) for p in ps] for r, ps in files.items()}},
        "sections": [{"name": nm, "start_bar": sec["start_bar"], "bars": sec["bars"], "mix": mix_rows[i],
                      "roles": {role: role_rows[role][i] for role in files}}
                     for i, (nm, sec) in enumerate(zip(names, sections))]}
    dest = pathlib.Path(out).expanduser() if out else mixp.parent
    if dest.is_dir() or not str(dest).endswith(".json"):
        dest = dest / (mixp.stem + ".profile.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    profile["json_path"] = str(dest.resolve())
    dest.write_text(json.dumps(profile, indent=1))
    return profile


def profile_stale(profile: dict) -> list:
    """The inputs of a reference profile that have changed since it was built, as
    {"role", "path", "reason"} ("changed": mtime or size differs; "missing"). Empty when
    the profile still matches its files."""
    out = []
    groups = [("mix", profile["inputs"]["mix"])] + list(profile["inputs"]["stems"].items())
    for role, items in groups:
        for it in items:
            p = pathlib.Path(it["path"])
            if not p.is_file():
                out.append({"role": role, "path": it["path"], "reason": "missing"})
                continue
            st = p.stat()
            if st.st_mtime != it["mtime"] or st.st_size != it["size"]:
                out.append({"role": role, "path": it["path"], "reason": "changed"})
    return out


def _load_profile(reference) -> dict | None:
    """The reference as a profile dict when it is one (a dict, or a path to a .json file);
    None for an audio file path or no reference."""
    import json
    if isinstance(reference, dict):
        prof = reference
    elif isinstance(reference, (str, pathlib.Path)) and str(reference).lower().endswith(".json"):
        p = pathlib.Path(reference).expanduser()
        if not p.is_file():
            raise ValueError("no such profile: %s" % p)
        prof = json.loads(p.read_text())
        prof.setdefault("json_path", str(p.resolve()))
    else:
        return None
    if prof.get("kind") != PROFILE_KIND or "sections" not in prof or "inputs" not in prof:
        raise ValueError("not a reference profile (build one with reference_profile)")
    return prof


def _pair(prof: dict, sec: dict, name: str) -> tuple[dict | None, int, list]:
    """(profile section, first bar inside it, notes) for a manifest section. By default the
    profile section of the same name from its first bar; the manifest section's
    "reference": {"section": <name>, "from_bar": N} (both optional) picks another section or
    starts N bars in. The range must lie inside the profile section: a shorter manifest
    section pairs with its first bars (noted), a longer one is not paired."""
    ref = sec.get("reference")
    if ref is not None:
        ok = (isinstance(ref, dict) and set(ref) <= {"section", "from_bar"}
              and isinstance(ref.get("section", ""), str)
              and isinstance(ref.get("from_bar", 0), int) and ref.get("from_bar", 0) >= 0)
        if not ok:
            raise ValueError('section %r: "reference" must be {"section": name, "from_bar": whole number >= 0}, '
                             "got %r" % (name, ref))
    ref = ref or {}
    pname, fb = ref.get("section", name), ref.get("from_bar", 0)
    psec = next((s for s in prof["sections"] if s["name"] == pname), None)
    if psec is None:
        return None, 0, ["no section named %r in the profile: mix layer only" % pname]
    if fb + sec["bars"] > psec["bars"]:
        return None, 0, ["section %r is %d bars here; from bar %d of the profile's %r (%d bars) only %d fit: "
                         "mix layer only" % (name, sec["bars"], fb + 1, pname, psec["bars"],
                                             max(0, psec["bars"] - fb))]
    notes = []
    if fb or sec["bars"] != psec["bars"]:
        notes.append("paired with bars %d-%d of the profile's %r (%d bars); the reference's bands, width "
                     "and loudness are for the whole section, so the capture-is-the-reference check is "
                     "not made" % (fb + 1, fb + sec["bars"], pname, psec["bars"]))
    return psec, fb, notes


def _element_layer(prof: dict, psec: dict, fb: int, notes: list, sec: dict, a: dict, audio: dict, sr: int,
                   t0: float, bar_s: float) -> tuple[dict, dict]:
    """(vs_reference, elements) for one manifest section paired with `psec` of the profile
    (starting `fb` bars in). Levels are ratios to the kick, or, where the kick is out on either
    side (absent, or KICK_OUT_DB under the loudest other role), to the loudest role present on
    both sides: `yardstick` says which."""
    n = sec["bars"]
    partial = bool(fb) or n != psec["bars"]
    pm = psec["mix"]
    c = compare(a, {"bands_db": pm["bands_db"], "side_minus_mid_db": {},
                    "integrated_lufs": pm["integrated_lufs"], "width": pm["width"]})
    vs = {"band_balance_db": c["band_balance_db"], "flagged_bands": [f["band"] for f in c["flagged_bands"]],
          "lufs_diff": c["diff"].get("integrated_lufs"), "width_diff": c["diff"].get("width")}
    notes = list(notes)
    is_ref = (not partial and a["width"] is not None and pm["width"] is not None
              and abs(a["rms_db"] - pm["level_db"]) <= CAPTURE_IS_REF_DB
              and abs(a["width"] - pm["width"]) <= CAPTURE_IS_REF_WIDTH)
    if is_ref:
        notes.append("the capture is the reference: check solos")
    rng = slice(fb, fb + n)
    mix_lv = _mean_db(pm["per_bar_db"][rng]) if partial else pm["level_db"]

    def ref_level(role):
        e = psec["roles"][role]
        if e["absent"]:
            return None
        if not partial:
            return e["level_db"]
        lv = _mean_db(e["per_bar_db"][rng])
        return None if lv is None or (mix_lv is not None and lv < mix_lv - ABSENT_DB) else lv

    ref_lv = {r: ref_level(r) for r in psec["roles"] if r != "master"}
    ours, ours_lv = {}, {}
    for role in audio:
        if role == "master":                     # sits after the master chain: never ratioed against elements
            continue
        seg = _stereo(_slice(audio[role][0], sr, t0, t0 + n * bar_s))
        lv = _level_db(seg)
        ours_lv[role] = None if lv <= FLOOR_DB or lv < a["rms_db"] - ABSENT_DB else lv
        ours[role] = _bar_levels(seg, sr, 0.0, bar_s, 0, n)

    def kick_out(lv):
        """Whether the kick is absent or far under the loudest other role in `lv`."""
        others = [v for r, v in lv.items() if r != "kick" and v is not None]
        return lv["kick"] is None or bool(others and lv["kick"] < max(others) - KICK_OUT_DB)

    yard, kick_is_out = None, False
    if "kick" not in ours_lv:
        notes.append("no kick source in the manifest: elements against the kick left out")
    elif "kick" not in ref_lv:
        notes.append("no kick in the profile: elements against the kick left out")
    elif not (kick_out(ref_lv) or kick_out(ours_lv)):
        yard = "kick"
    else:
        kick_is_out = True
        where = " and ".join(w for w, lv in (("the reference", ref_lv), ("ours", ours_lv)) if kick_out(lv))
        both = sorted(r for r in ref_lv if r != "kick" and ref_lv[r] is not None and ours_lv.get(r) is not None)
        if both:
            yard = max(both, key=lambda r: ref_lv[r])
            notes.append("the kick is out in %s (absent, or more than %g dB under the loudest other role): "
                         "elements against the %s, the loudest role in both" % (where, KICK_OUT_DB, yard))
        else:
            notes.append("the kick is out in %s and no other element sounds in both: elements left out" % where)
    table, ref_only, ours_only, arc, silent_both = [], [], [], {}, []
    for role in sorted((set(ref_lv) | set(ours_lv)) - {yard}):
        r_lv, o_lv = ref_lv.get(role), ours_lv.get(role)
        r_ratio = round(r_lv - ref_lv[yard], 2) if yard and r_lv is not None else None
        o_ratio = round(o_lv - ours_lv[yard], 2) if yard and o_lv is not None else None
        if r_lv is not None and o_lv is not None:
            if yard:
                table.append({"role": role, "reference_db": r_ratio, "ours_db": o_ratio,
                              "difference_db": round(o_ratio - r_ratio, 2)})
                if n > CHANGE_BLOCK_BARS:
                    arc[role] = {"reference": _arc(psec["roles"][role]["per_bar_db"][rng],
                                                   psec["roles"][yard]["per_bar_db"][rng]),
                                 "ours": _arc(ours[role], ours[yard])}
        elif r_lv is not None:
            ref_only.append({"role": role, "vs_yardstick_db": r_ratio,
                             "reason": "not in the manifest" if role not in ours_lv else "silent in ours"})
        elif o_lv is not None:
            ours_only.append({"role": role, "vs_yardstick_db": o_ratio,
                              "reason": "not in the profile" if role not in ref_lv else "silent in the reference"})
        else:
            silent_both.append(role)
    gaps = sorted(({"role": t["role"], "difference_db": t["difference_db"]} for t in table
                   if abs(t["difference_db"]) > ELEMENT_GAP_DB), key=lambda g: -abs(g["difference_db"]))
    el = {"paired": True, "paired_with": {"section": psec["name"], "from_bar": fb, "bars": n},
          "notes": notes, "capture_is_reference": bool(is_ref), "yardstick": yard,
          "kick_out": kick_is_out,
          "table": table if yard else None, "gaps": gaps if yard else [],
          "reference_only": ref_only, "ours_only": ours_only, "silent_in_both": silent_both,
          "arc": arc if yard else {}}
    return vs, el


def _drum_layer(prof: dict, psec: dict, fb: int, sec: dict, audio: dict, sr: int, t0: float,
                bar_s: float, stale: set, cache: dict) -> dict:
    """For each drum role in both the manifest's sources and the profile's stems: the readings
    of oneshot.compare(ours, reference) on the clean hit of each side over the section's bars.
    The profile stores paths only, so the reference stems are read again (summed per role, kept
    in `cache`); a role whose stems changed or are missing since the profile is skipped, with a note."""
    from . import oneshot
    n = sec["bars"]
    pbar = float(prof["beats_per_bar"]) * 60.0 / float(prof["tempo"])
    r0 = prof["t0"] + (psec["start_bar"] + fb) * pbar
    out = []
    for role in DRUM_ROLES:
        if role not in audio or role not in prof["inputs"]["stems"]:
            continue
        entry = {"role": role, "ours_hit": None, "reference_hit": None, "readings": [], "notes": []}
        out.append(entry)
        if role in stale:
            entry["notes"].append("a stem of this role changed or is missing since the profile was built: skipped")
            continue
        if role not in cache:
            ys = [_stereo(load(it["path"])[0]) for it in prof["inputs"]["stems"][role]]
            m = min(len(y) for y in ys)
            cache[role] = sum(y[:m] for y in ys)
        profs = {}
        for side, x, rate, start in (("ours", audio[role][0], sr, t0), ("the reference", cache[role],
                                                                      prof["sample_rate"], r0)):
            try:
                profs[side] = oneshot.profile_array(_slice(x, rate, start, start + n * (bar_s if side == "ours" else pbar)),
                                                    rate, hit="clean")
            except ValueError as e:
                entry["notes"].append("%s: %s" % (side, e))
        if len(profs) == 2:
            entry["readings"] = oneshot.compare(profs["ours"], profs["the reference"])["readings"]
            entry["ours_hit"], entry["reference_hit"] = profs["ours"]["hit"], profs["the reference"]["hit"]
    return {"roles": out}


def mix_report(manifest: dict | str | pathlib.Path,
               reference: dict | str | pathlib.Path | None = None,
               out_dir: str | pathlib.Path | None = None) -> dict:
    """Per-section measurements of a capture, plus change rate and masking.
    Writes mix_report.json and mix_report.md into out_dir (default: the
    master file's folder), replacing earlier ones there, and returns the
    report with their paths. Every section must lie inside every source;
    `whole` and the change rate span first to last bar, gaps included.

    `reference` is an audio file (each section's bands are compared with the
    whole file's), or a reference_profile: the profile dict or the path of its
    .profile.json. With a profile, sections pair BY NAME (the manifest's
    section "A" with the profile's section "A"); a manifest section may carry
    "reference": {"section": <profile section name>, "from_bar": N} (both
    optional) to pair with another section, or N bars into it. The range must
    fit inside the profile's section: a shorter manifest section pairs with its
    first bars (noted; the reference's bands, width and loudness are then those of
    the whole section, and the capture-is-the-reference check is not made); a
    section with no match, or a longer one, reports the mix layer only, with a note.
    A paired section gets the mix compared with the profile's section (bands,
    width, loudness), an `elements` layer and a `drums` layer.
    `elements`: each source in both the manifest and the profile as a ratio to the
    kick (reference, ours, difference), roles in one side only, the ratio per 4 bars,
    and `gaps` (ratios differing by more than ELEMENT_GAP_DB, largest first). Where
    the kick is out on either side (absent, or KICK_OUT_DB under the loudest other
    role) the yardstick is the loudest role present on both sides, named in
    `yardstick` (none: the table is left out with a note). The `master` source is never
    ratioed against the elements: it sits after the master chain. A master capture
    whose level and width equal the reference mix's is flagged ("the capture is the
    reference: check solos"). Without a `kick` source (or kick stem) the table is left
    out with a note; the mix layer still reports.
    `drums`: for each of DRUM_ROLES in both the manifest's sources and the profile's
    stems, the `oneshot.compare` readings between the clean hit of our audio and of the
    reference stem over the section (the stems are read again from the profile's paths;
    a role whose stem changed since the profile is skipped, with a note); a side with
    no clean hit gives a note and no readings. `reference_stale` lists profile inputs
    that changed since it was built."""
    import json
    m, mpath = _load_manifest(manifest)
    audio = {}
    for role, path in m["sources"].items():
        audio[role] = load(path)
    sr = audio["master"][1]
    for role, (_, r) in audio.items():
        if r != sr:
            raise ValueError("%s is %d Hz but master is %d Hz" % (role, r, sr))
    bar_s, off = _bar_seconds(m), float(m.get("offset_s", 0.0))
    if off < 0:
        raise ValueError("offset_s must be >= 0")
    first = min(s["start_bar"] for s in m["sections"])
    last = max(s["start_bar"] + s["bars"] for s in m["sections"])
    end_s = off + last * bar_s
    for role, (x, _) in audio.items():
        have = x.shape[0] / sr
        if have + 0.01 < end_s:
            raise ValueError("%s is %.2f s long but the sections run to bar %d = %.2f s"
                             % (role, have, last, end_s))
    prof = _load_profile(reference)
    stale = profile_stale(prof) if prof is not None else []
    stale_roles, ref_cache = {x["role"] for x in stale}, {}
    ref = analyze(reference) if reference and prof is None else None
    kick_role = "kick" if "kick" in audio else ("drums" if "drums" in audio else None)
    rows = []
    for sec in m["sections"]:
        t0 = off + sec["start_bar"] * bar_s
        t1 = t0 + sec["bars"] * bar_s
        a = analyze_array(_slice(audio["master"][0], sr, t0, t1), sr)
        row = {"name": sec.get("name", "bar %d" % sec["start_bar"]), "start_bar": sec["start_bar"],
               "bars": sec["bars"]}
        for k in ("integrated_lufs", "short_term_max_lufs", "true_peak_dbtp", "crest_db", "width",
                  "low_end_correlation", "centroid_hz", "bands_db", "side_minus_mid_db"):
            row[k] = a[k]
        if ref is not None:
            c = compare(a, ref)
            row["vs_reference"] = {"band_balance_db": c["band_balance_db"],
                                   "flagged_bands": [f["band"] for f in c["flagged_bands"]],
                                   "lufs_diff": c["diff"].get("integrated_lufs")}
        if prof is not None:
            psec, fb, pnotes = _pair(prof, sec, row["name"])
            if psec is None:
                row["elements"] = {"paired": False, "notes": pnotes}
            else:
                row["vs_reference"], row["elements"] = _element_layer(prof, psec, fb, pnotes, sec, a, audio,
                                                                      sr, t0, bar_s)
                row["drums"] = _drum_layer(prof, psec, fb, sec, audio, sr, t0, bar_s, stale_roles, ref_cache)
        if kick_role and "bass" in audio:
            mk = masking(_slice(audio[kick_role][0], sr, t0, t1), _slice(audio["bass"][0], sr, t0, t1), sr)
            row["kick_bass_overlap"] = mk and mk["overlap"]
            row["kick_bass_quieter_db"] = mk and mk["quieter_db"]
            row["masking_flag"] = bool(mk and mk["overlap"] > MASK_FLAG
                                       and mk["quieter_db"] >= MASK_AUDIBLE_DB)
        rows.append(row)
    cr = change_rate(audio["master"][0], sr, bar_s, off, first, last)
    names = sorted((s["start_bar"], s["start_bar"] + s["bars"], s.get("name", "")) for s in m["sections"])
    for f in cr["static_runs"]:
        f["sections"] = [n for a0, a1, n in names if a0 < f["end_bar"] and a1 > f["start_bar"]]
    report = {"manifest": str(mpath) if mpath else None,
              "reference": (prof.get("json_path") or prof["mix"]) if prof is not None
              else (str(reference) if reference else None),
              "tempo": m["tempo"], "sections": rows, "change_rate": cr,
              "whole": {k: v for k, v in analyze_array(
                  _slice(audio["master"][0], sr, off + first * bar_s, off + last * bar_s), sr).items()
                  if k in ("integrated_lufs", "short_term_max_lufs", "true_peak_dbtp", "crest_db", "width")}}
    if prof is not None:
        report["reference_kind"] = "profile"
        report["reference_stale"] = stale
    out = pathlib.Path(out_dir) if out_dir else pathlib.Path(m["sources"]["master"]).parent
    out.mkdir(parents=True, exist_ok=True)
    jp, mdp = out / "mix_report.json", out / "mix_report.md"
    report["json_path"], report["md_path"] = str(jp), str(mdp)
    jp.write_text(json.dumps(report, indent=2))
    mdp.write_text(_markdown(report))
    return report


def _fmt(v, nd=1):
    return "—" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def _elements_markdown(r: dict) -> list:
    lines = ["", "## Elements against the kick", "",
             "Each element's mean power minus the kick's over the same bars (dB; negative = quieter than the "
             "kick). A gap is a difference of more than %.1f dB. Where the kick is out (absent, or more than "
             "%g dB under the loudest other role, on either side) the section says which element is the "
             "yardstick instead. The master capture is never ratioed against the elements: it sits after "
             "the master chain." % (ELEMENT_GAP_DB, KICK_OUT_DB), ""]
    for s in r["sections"]:
        e = s["elements"]
        lines += ["### %s" % s["name"], ""]
        if e["paired"] and e["kick_out"] and e["yardstick"]:
            lines += ["**Elements against the %s (kick out)**" % e["yardstick"], ""]
        lines += ["- ⚠ %s" % n if "check solos" in n else "- %s" % n for n in e["notes"]]
        if not e["paired"]:
            lines.append("")
            continue
        if e["table"] is not None:
            lines += ["| Element | Reference | Ours | Diff |", "|---|---|---|---|"]
            lines += ["| %s | %s | %s | %s |" % (t["role"], _fmt(t["reference_db"]), _fmt(t["ours_db"]),
                                                 "%+.1f" % t["difference_db"]) for t in e["table"]]
            lines.append("")
            lines.append("Gaps, largest first: " + ("; ".join("%s %+.1f dB" % (g["role"], g["difference_db"])
                                                             for g in e["gaps"]) or "none") + ".")
        for key, label in (("reference_only", "In the reference only"), ("ours_only", "In ours only")):
            if e[key]:
                lines.append("%s: " % label + "; ".join(
                    "%s (%s%s)" % (x["role"], x["reason"], "" if x["vs_yardstick_db"] is None
                                   else ", %+.1f dB vs %s" % (x["vs_yardstick_db"], e["yardstick"]))
                    for x in e[key]) + ".")
        if e["arc"]:
            lines += ["", "Per 4 bars (reference / ours):"]
            for role, a in e["arc"].items():
                lines.append("- %s: " % role + " | ".join("%s / %s" % (_fmt(x), _fmt(y))
                                                         for x, y in zip(a["reference"], a["ours"])))
        lines.append("")
    return lines


def _drums_markdown(r: dict) -> list:
    lines = ["", "## Drum sounds", "",
             "Our clean hit of each drum role against the reference's, over the section's bars "
             "(oneshot readings: a is ours, b the reference; none = nothing past the thresholds).", ""]
    for s in r["sections"]:
        if "drums" not in s:
            continue
        lines += ["### %s" % s["name"], ""]
        if not s["drums"]["roles"]:
            lines += ["- no drum role is in both the manifest and the profile", ""]
        for d in s["drums"]["roles"]:
            body = "; ".join(d["readings"]) or ("none" if not d["notes"] else "")
            lines.append("- %s: %s" % (d["role"], "; ".join(x for x in [body] + d["notes"] if x)))
        lines.append("")
    return lines


def _markdown(r: dict) -> str:
    w = r["whole"]
    lines = ["# Mix report", "",
             "Tempo %s BPM. Whole capture: %s LUFS integrated, short-term max %s, true peak %s dBTP, "
             "crest %s dB, width %s." % (_fmt(r["tempo"]), _fmt(w["integrated_lufs"]),
                                         _fmt(w["short_term_max_lufs"]), _fmt(w["true_peak_dbtp"]),
                                         _fmt(w["crest_db"]), _fmt(w["width"], 2)),
             ""]
    if r.get("reference_kind") == "profile":
        lines += ["Reference profile: `%s`. Sections are paired with the profile's by name." % r["reference"], ""]
        lines += ["- ⚠ %s `%s` %s since the profile was built" % (s["role"], s["path"], s["reason"])
                  for s in r["reference_stale"]]
        lines += [""] if r["reference_stale"] else []
    if r["reference"]:
        lines += ["Reference: `%s`. Band balance is relative to it after removing the overall level "
                  "offset; bands more than 3 dB away are listed." % r["reference"], ""]
    head = ["Section", "Bars", "LUFS", "ST max", "TP", "Crest", "Width", "Low corr"]
    has_ref = any("vs_reference" in s for s in r["sections"])
    has_mask = any("kick_bass_overlap" in s for s in r["sections"])
    head += (["vs ref"] if has_ref else []) + (["Kick/bass overlap"] if has_mask else [])
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s in r["sections"]:
        cells = [s["name"], "%d–%d" % (s["start_bar"] + 1, s["start_bar"] + s["bars"]),
                 _fmt(s["integrated_lufs"]), _fmt(s["short_term_max_lufs"]), _fmt(s["true_peak_dbtp"]),
                 _fmt(s["crest_db"]), _fmt(s["width"], 2), _fmt(s["low_end_correlation"], 2)]
        if has_ref:
            v = s.get("vs_reference") or {}
            cells.append(", ".join("%s %+.1f" % (b, v["band_balance_db"][b]) for b in v.get("flagged_bands", []))
                         or "ok")
        if has_mask:
            ov, q = s.get("kick_bass_overlap"), s.get("kick_bass_quieter_db")
            cells.append(_fmt(ov, 2) + ("" if q is None else " (quieter %+.0f dB)" % q)
                         + (" ⚠" if s.get("masking_flag") else ""))
        lines.append("| " + " | ".join(cells) + " |")
    if r.get("reference_kind") == "profile":
        lines += _elements_markdown(r) + _drums_markdown(r)
    cr = r["change_rate"]
    lines += ["", "## Change rate", "",
              "Spectral change between consecutive %d-bar blocks (mean third-octave difference, dB). "
              "Under %.1f dB counts as no change." % (cr["block_bars"], CHANGE_DB), ""]
    if cr["static_runs"]:
        for f in cr["static_runs"]:
            lines.append("- ⚠ bars %d–%d (%d bars, %s): no change for more than 8 bars"
                         % (f["start_bar"] + 1, f["end_bar"], f["bars"], ", ".join(f["sections"]) or "—"))
    else:
        lines.append("- Something changes at least every 8 bars.")
    lines += ["", "Steps: " + ", ".join("bar %d %.1f%s" % (s["at_bar"] + 1, s["change_db"], " (silent)" if s["silent"] else "")
                                        for s in cr["steps"]), ""]
    return "\n".join(lines)
