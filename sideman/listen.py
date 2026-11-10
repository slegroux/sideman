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
from collections.abc import Iterable
from dataclasses import dataclass

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
SUBBLOCK_S = 0.1        # BS.1770 loudness is built from 100 ms sub-blocks...
MOMENTARY_SUBBLOCKS = 4     # ...400 ms momentary window, 100 ms hop
SHORT_TERM_SUBBLOCKS = 30   # ...3 s short-term window
ABS_GATE_LUFS = -70.0   # absolute gate, and the floor under which a max is "not measurable"
REL_GATE_LU = -10.0     # relative gate, below the absolute-gated mean


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
    n = int(round(sr * SUBBLOCK_S))
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
    return loudness_from_subblocks(_subblock_powers(x, sr))


def loudness_from_subblocks(sub: np.ndarray) -> dict:
    """`loudness` from K-weighted 100 ms sub-block powers summed over channels:
    the file path computes them from samples, the live-listen device sends them
    (sideman.stream)."""
    out = {"integrated_lufs": None, "short_term_max_lufs": None, "momentary_max_lufs": None}
    if len(sub) < MOMENTARY_SUBBLOCKS:
        return out
    c = np.concatenate([[0.0], np.cumsum(sub)])
    mom = MOMENTARY_SUBBLOCKS
    block = (c[mom:] - c[:-mom]) / mom                   # 400 ms, 100 ms hop
    lb = _lufs(block)
    out["momentary_max_lufs"] = round(float(lb.max()), 2) if lb.max() > ABS_GATE_LUFS else None
    gated = block[lb > ABS_GATE_LUFS]
    if len(gated):
        rel = _lufs(gated.mean()) + REL_GATE_LU
        gated = block[(lb > ABS_GATE_LUFS) & (lb > rel)]
        if len(gated):
            out["integrated_lufs"] = round(float(_lufs(gated.mean())), 2)
    if len(sub) >= SHORT_TERM_SUBBLOCKS:
        sh = SHORT_TERM_SUBBLOCKS
        st = float(_lufs(((c[sh:] - c[:-sh]) / sh).max()))
        out["short_term_max_lufs"] = round(st, 2) if st > ABS_GATE_LUFS else None
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
    for sec in m["sections"]:
        if not isinstance(sec, dict):
            raise ValueError("each section must be an object with start_bar and bars, got %r" % (sec,))
        sb, nb = sec.get("start_bar"), sec.get("bars")
        if not (isinstance(sb, int) and isinstance(nb, int)) or sb < 0 or nb < 1:
            raise ValueError("section %r needs whole-number start_bar >= 0 and bars >= 1"
                             % sec.get("name", sec))
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


def _mask_cells(mono: np.ndarray, sr: int) -> np.ndarray:
    """Power in the 40-120 Hz bins of each ~85 ms frame (50 % overlap) of a mono
    signal: [frames, bins]. The FFT runs BATCH frames at a time and only the
    ~7 low bins per frame are kept, so a 7-minute file holds under a megabyte of
    cells however long it is. Fewer samples than one frame gives zero frames."""
    freqs = np.fft.rfftfreq(MASK_FRAME, 1.0 / sr)
    sel = (freqs >= MASK_LO) & (freqs < MASK_HI)
    win = np.hanning(MASK_FRAME)
    starts = np.arange(0, len(mono) - MASK_FRAME + 1, MASK_FRAME // 2)
    cells = [np.abs(np.fft.rfft(mono[starts[s:s + BATCH, None] + np.arange(MASK_FRAME)[None, :]]
                                .astype(np.float64) * win, axis=1)[:, sel]) ** 2
             for s in range(0, len(starts), BATCH)]
    return np.concatenate(cells) if cells else np.zeros((0, int(sel.sum())))


def masking_cells(kick_cells: np.ndarray, bass_cells: np.ndarray) -> dict | None:
    """The kick/bass overlap rule on two [frames, bins] grids of low-band power.
    Both grids must have the same shape (the same frames and bins): a mismatch
    means the two parts are not aligned, and is refused rather than cut.
    Returns {"overlap", "quieter_db"} or None when either grid is silent.

    overlap: how much of the quieter part's energy sits in the same
    time-frequency cells as the other's, sum(min) / min(sum kick, sum bass).
    quieter_db: the quieter part's energy relative to the louder's."""
    pk, pb = np.asarray(kick_cells, dtype=float), np.asarray(bass_cells, dtype=float)
    if pk.shape != pb.shape or pk.ndim != 2:
        raise ValueError("masking cells must be two [frames, bins] grids of one shape; "
                         "got kick %s and bass %s" % (pk.shape, pb.shape))
    lo_sum, k_sum, b_sum = float(np.minimum(pk, pb).sum()), float(pk.sum()), float(pb.sum())
    if k_sum == 0 or b_sum == 0:
        return None
    return {"overlap": round(lo_sum / min(k_sum, b_sum), 3),
            "quieter_db": round(_db(min(k_sum, b_sum) / max(k_sum, b_sum)), 1)}


def masking(kick: np.ndarray, bass: np.ndarray, sr: int) -> dict | None:
    """How kick and bass share 40-120 Hz, over ~85 ms frames. Returns
    {"overlap", "quieter_db"} or None when either is silent there.

    overlap: how much of the quieter part's energy sits in the same
    time-frequency cells as the other's, sum(min) / min(sum kick, sum bass).
    0 = they never coincide (separate notes, or a sidechain that ducks the
    bass under each hit); 1 = the quieter one is entirely under the louder.
    quieter_db: the quieter part's 40-120 Hz energy relative to the louder's.
    mix_report flags masking only when both are high, since a part 40 dB down
    is inaudible whatever it overlaps. Builds the FFT cells (_mask_cells), then
    applies masking_cells."""
    n = min(len(kick), len(bass))
    return masking_cells(_mono_cells(kick[:n], sr), _mono_cells(bass[:n], sr))


def _mono_cells(x: np.ndarray, sr: int) -> np.ndarray:
    """_mask_cells of [n, channels] samples, on the mean of the channels."""
    return _mask_cells(_stereo(x).mean(axis=1), sr)


def _third_octave_db(x: np.ndarray, sr: int) -> np.ndarray:
    """Mean power per third-octave band, 25 Hz-16 kHz (below Nyquist), in dB."""
    sp = _spectra(_stereo(x), sr)
    lr = (sp["l"] + sp["r"]) / 2.0
    centres = 1000.0 * 2.0 ** (np.arange(-16, 13) / 3.0)
    centres = centres[centres * 2 ** (1 / 6) <= sr / 2.0]
    return np.array([_db(_band(lr, sp["freqs"], c / 2 ** (1 / 6), c * 2 ** (1 / 6))) for c in centres])


def change_steps(blocks: Iterable[tuple[int, int, np.ndarray]], *,
                 block_bars: int = CHANGE_BLOCK_BARS) -> dict:
    """The step and static-run logic of change_rate on any source of blocks.
    `blocks` is a sequence of (b0, b1, v): the bar range of a block and its
    third-octave levels in dB (a silent block is all FLOOR_DB, as
    _third_octave_db gives). Returns {"block_bars", "steps", "static_runs"};
    a static run carries start_bar, end_bar and bars, and change_rate adds the
    section names. `block_bars` is only reported (the markdown header)."""
    spans, vecs, floors, steps = [], [], [], []
    for b0, b1, v in blocks:
        v = np.asarray(v, dtype=float)
        silent = bool(v.max() < SILENT_DB)
        floor = v.max() - CHANGE_RANGE_DB
        v = np.maximum(v, floor)
        spans.append((b0, b1, silent))
        if vecs:
            p = vecs[-1]
            live = (v > floor) | (p > floors[-1])
            d = float(np.abs(v - p)[live].mean()) if live.any() else 0.0
            steps.append({"at_bar": b0, "change_db": round(d, 2), "silent": silent})
        vecs.append(v)
        floors.append(floor)
    flags, run_start = [], 0
    for i in range(1, len(spans) + 1):
        if i == len(spans) or steps[i - 1]["change_db"] >= CHANGE_DB or spans[i][2] != spans[i - 1][2]:
            b0, b1, silent = spans[run_start][0], spans[i - 1][1], spans[run_start][2]
            if b1 - b0 > STATIC_MAX_BARS and not silent:
                flags.append({"start_bar": b0, "end_bar": b1, "bars": b1 - b0})
            run_start = i
    return {"block_bars": block_bars,
            "steps": steps, "static_runs": flags}


def change_rate(x: np.ndarray, sr: int, bar_s: float, offset_s: float, first_bar: int,
                last_bar: int) -> dict:
    """Spectral change between consecutive 4-bar blocks on the bar grid: the mean
    absolute third-octave difference (dB) over the bands sounding in either
    block, each block's bands floored at 40 dB under its loudest, so an
    element far below the mix entering does not read as a 70 dB jump. Runs longer than 8 bars
    with every step under CHANGE_DB are flagged. Silent blocks break a run and
    are never flagged. Each block is compared with the one before it, so very
    slow automation (under 1 dB per 4 bars) also reads as no change.
    Measures each block from `x`, then applies change_steps."""
    def blocks():
        for b0 in range(first_bar, last_bar, CHANGE_BLOCK_BARS):
            b1 = min(b0 + CHANGE_BLOCK_BARS, last_bar)
            seg = _slice(x, sr, offset_s + b0 * bar_s, offset_s + b1 * bar_s)
            if len(seg) >= sr // 10:
                yield b0, b1, _third_octave_db(seg, sr)
    return change_steps(blocks())


# One section of a report, from audio today and from a device's per-block rows
# later: `stats` is the analyze_array() dict for the master over the section;
# the optional kick/bass cells are masking_cells() inputs. The live-listen path
# (plan 19 item 7) will add a constructor that fills the same fields from a
# device's precomputed stats and cell matrices, with no audio.

@dataclass(eq=False)
class SectionFeatures:
    stats: dict
    kick_cells: np.ndarray | None = None
    bass_cells: np.ndarray | None = None

    @classmethod
    def from_arrays(cls, master_x: np.ndarray, sr: int, kick_x: np.ndarray | None = None,
                    bass_x: np.ndarray | None = None) -> SectionFeatures:
        """Features of one section's samples [n, channels]. Masking cells exist
        only when both kick and bass are given; a part shorter than one
        MASK_FRAME has no cells, so masking reads as unmeasurable."""
        cells = {}
        if kick_x is not None and bass_x is not None:
            n = min(len(kick_x), len(bass_x))
            cells = {"kick_cells": _mono_cells(kick_x[:n], sr),
                     "bass_cells": _mono_cells(bass_x[:n], sr)}
        return cls(analyze_array(master_x, sr), **cells)


def section_stats(f: SectionFeatures, ref: dict | None) -> dict:
    """The measurements of one report row: loudness, peak, width, centroid,
    bands, and `vs_reference` against an analyze()-shaped `ref` when given."""
    a = f.stats
    row = {k: a[k] for k in ("integrated_lufs", "short_term_max_lufs", "true_peak_dbtp", "crest_db", "width",
                             "low_end_correlation", "centroid_hz", "bands_db", "side_minus_mid_db")}
    if ref is not None:
        c = compare(a, ref)
        row["vs_reference"] = {"band_balance_db": c["band_balance_db"],
                               "flagged_bands": [fb["band"] for fb in c["flagged_bands"]],
                               "lufs_diff": c["diff"].get("integrated_lufs")}
    return row


def section_flags(mk: dict | None) -> list[str]:
    """Flags raised on one section from its masking_cells() result: "masking"
    when kick and bass overlap above MASK_FLAG and the quieter is no further
    below the louder than MASK_AUDIBLE_DB in 40-120 Hz."""
    return ["masking"] if mk and mk["overlap"] > MASK_FLAG and mk["quieter_db"] >= MASK_AUDIBLE_DB else []


def section_row(name: str, start_bar: int, bars: int, f: SectionFeatures, ref: dict | None) -> dict:
    """A mix_report section: name and bars, section_stats, then the kick/bass
    masking fields when the features carry cells."""
    row = {"name": name, "start_bar": start_bar, "bars": bars, **section_stats(f, ref)}
    if f.kick_cells is not None and f.bass_cells is not None:
        mk = masking_cells(f.kick_cells, f.bass_cells)
        row["kick_bass_overlap"] = mk and mk["overlap"]
        row["kick_bass_quieter_db"] = mk and mk["quieter_db"]
        row["masking_flag"] = "masking" in section_flags(mk)
    return row


def mix_report(manifest: dict | str | pathlib.Path, reference: str | pathlib.Path | None = None,
               out_dir: str | pathlib.Path | None = None) -> dict:
    """Per-section measurements of a capture, plus change rate and masking.
    Writes mix_report.json and mix_report.md into out_dir (default: the
    master file's folder), replacing earlier ones there, and returns the
    report with their paths. Every section must lie inside every source;
    `whole` and the change rate span first to last bar, gaps included."""
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
    ref = analyze(reference) if reference else None
    kick_role = "kick" if "kick" in audio else ("drums" if "drums" in audio else None)
    rows = []
    mask = bool(kick_role and "bass" in audio)
    for sec in m["sections"]:
        t0 = off + sec["start_bar"] * bar_s
        t1 = t0 + sec["bars"] * bar_s
        feat = SectionFeatures.from_arrays(
            _slice(audio["master"][0], sr, t0, t1), sr,
            _slice(audio[kick_role][0], sr, t0, t1) if mask else None,
            _slice(audio["bass"][0], sr, t0, t1) if mask else None)
        rows.append(section_row(sec.get("name", "bar %d" % sec["start_bar"]), sec["start_bar"], sec["bars"],
                                feat, ref))
    cr = change_rate(audio["master"][0], sr, bar_s, off, first, last)
    names = sorted((s["start_bar"], s["start_bar"] + s["bars"], s.get("name", "")) for s in m["sections"])
    for f in cr["static_runs"]:
        f["sections"] = [n for a0, a1, n in names if a0 < f["end_bar"] and a1 > f["start_bar"]]
    report = {"manifest": str(mpath) if mpath else None, "reference": str(reference) if reference else None,
              "tempo": m["tempo"], "sections": rows, "change_rate": cr,
              "whole": {k: v for k, v in analyze_array(
                  _slice(audio["master"][0], sr, off + first * bar_s, off + last * bar_s), sr).items()
                  if k in ("integrated_lufs", "short_term_max_lufs", "true_peak_dbtp", "crest_db", "width")}}
    out = pathlib.Path(out_dir) if out_dir else pathlib.Path(m["sources"]["master"]).parent
    out.mkdir(parents=True, exist_ok=True)
    jp, mdp = out / "mix_report.json", out / "mix_report.md"
    report["json_path"], report["md_path"] = str(jp), str(mdp)
    jp.write_text(json.dumps(report, indent=2))
    mdp.write_text(_markdown(report))
    return report


def _fmt(v, nd=1):
    return "—" if v is None else ("%.*f" % (nd, v) if isinstance(v, float) else str(v))


def _markdown(r: dict) -> str:
    w = r["whole"]
    lines = ["# Mix report", "",
             "Tempo %s BPM. Whole capture: %s LUFS integrated, short-term max %s, true peak %s dBTP, "
             "crest %s dB, width %s." % (_fmt(r["tempo"]), _fmt(w["integrated_lufs"]),
                                         _fmt(w["short_term_max_lufs"]), _fmt(w["true_peak_dbtp"]),
                                         _fmt(w["crest_db"]), _fmt(w["width"], 2)),
             ""]
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
