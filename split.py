import numpy as np
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, savgol_filter
import pywt

from pattern_utils import SegmentType, string_to_pattern


def _short_time_motion_envelope(
    x: np.ndarray,
    fs: float,
    win_sec: float = 0.02,
    hop_sec: float = 0.01,
    method: Literal['diff_rms', 'mag_rms', 'phase_std'] = 'diff_rms',
    eps: float = 1e-12,
):
    """
    Compute a motion-sensitive short-time envelope from a complex baseband signal.

    Parameters
    - x: complex ndarray, shape (N,)
    - fs: sampling rate in Hz
    - win_sec: window length (seconds)
    - hop_sec: hop length (seconds)
    - method: envelope method
        'diff_rms'  -> RMS of first difference magnitude (robust to DC)
        'mag_rms'   -> RMS magnitude (good if vibration modulates amplitude)
        'phase_std' -> Std of unwrapped phase (good for micro-motions)

    Returns
    - env: float ndarray of length num_frames
    - frame_times: float ndarray (seconds) for each frame center
    - win_len: int window length in samples
    - hop_len: int hop length in samples
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('x must be 1-D complex array')

    N = x.shape[0]
    win_len = max(1, int(round(win_sec * fs)))
    hop_len = max(1, int(round(hop_sec * fs)))
    if win_len > N:
        win_len = N

    # Prepare feature per-sample
    if method == 'diff_rms':
        d = np.diff(x, prepend=x[0])
        feat = np.abs(d)
    elif method == 'mag_rms':
        feat = np.abs(x)
    elif method == 'phase_std':
        phase = np.unwrap(np.angle(x))
        # Convert to per-sample std via a moving window later; here feat is phase
        feat = phase
    else:
        raise ValueError("Unknown method: %s" % method)

    # Framing with sliding windows
    frames = []
    idx = []
    for start in range(0, N - win_len + 1, hop_len):
        seg = feat[start:start + win_len]
        if method == 'phase_std':
            val = np.std(seg)
        else:
            val = np.sqrt(np.mean(seg.astype(np.complex128) * np.conjugate(seg.astype(np.complex128))).real + eps) if np.iscomplexobj(seg) else float(np.sqrt(np.mean(seg ** 2) + eps))
        frames.append(float(val))
        idx.append(start + win_len / 2.0)

    env = np.asarray(frames, dtype=float)
    frame_times = np.asarray(idx, dtype=float) / float(fs)

    # Normalize envelope to zero-mean unit-variance to stabilize matching
    if env.size > 0:
        env = (env - env.mean()) / (env.std() + eps)

    return env, frame_times, win_len, hop_len


def _build_pattern_template(
    segments: List[Tuple[SegmentType, float]],
    fs: float,
    hop_sec: float,
    vib_value: float = 1.0,
    still_value: float = 0.0,
):
    """
    Build a frame-domain template corresponding to a sequence of vibration/stillness durations.

    Parameters
    - segments: list like [('v', 0.3), ('s', 0.2), ...] with durations in seconds
    - fs: sampling rate in Hz (used only to be consistent with hop_sec)
    - hop_sec: envelope hop in seconds (determines template resolution)
    - vib_value: value to assign to vibration frames
    - still_value: value to assign to stillness frames

    Returns
    - template: float ndarray of length M (number of frames)
    - frames_per_segment: list of ints for each segment length in frames
    """
    hop_len = max(1, int(round(hop_sec * fs)))
    if hop_len <= 0:
        raise ValueError('Invalid hop_len derived from hop_sec and fs')

    tpl = []
    frames_per_segment: List[int] = []
    for kind, dur_sec in segments:
        n_frames = max(1, int(round(dur_sec / hop_sec)))
        frames_per_segment.append(n_frames)
        val = vib_value if kind == 'v' else still_value
        tpl.extend([val] * n_frames)

    template = np.asarray(tpl, dtype=float)
    # Zero-mean and unit-norm to create a matched filter kernel
    template = template - template.mean()
    norm = np.linalg.norm(template)
    if norm > 0:
        template = template / norm
    return template, frames_per_segment


def _valid_normalized_correlation(env: np.ndarray, template: np.ndarray, eps: float = 1e-12):
    """
    Compute valid-mode normalized cross-correlation between a 1-D envelope and a template.

    Returns
    - corr: ndarray of length len(env) - len(template) + 1
    """
    E = env.astype(float)
    T = template.astype(float)
    M = len(T)
    N = len(E)
    if M > N:
        return np.asarray([], dtype=float)

    # Precompute sliding mean/std for env windows for normalization
    # Use cumulative sums for efficiency
    csum = np.cumsum(np.concatenate(([0.0], E)))
    csum2 = np.cumsum(np.concatenate(([0.0], E * E)))
    win_sum = csum[M:] - csum[:-M]
    win_sum2 = csum2[M:] - csum2[:-M]
    win_mean = win_sum / M
    win_var = np.maximum(win_sum2 / M - win_mean * win_mean, 0.0)
    win_std = np.sqrt(win_var) + eps

    # Correlate (equivalent to convolution with time-reversed template)
    # Since T is already zero-mean and unit-norm, we can compute dot products efficiently
    corr = np.empty(N - M + 1, dtype=float)
    # Compute sliding dot products using convolution-like cumulative method
    # Vectorized approach via FFT would require scipy; stick to direct for clarity
    # For moderate M this is fine.
    for k in range(N - M + 1):
        seg = E[k:k + M]
        # Normalize segment to zero-mean, unit-std on the fly using precomputed stats
        z = (seg - win_mean[k]) / win_std[k]
        corr[k] = float(np.dot(z, T))
    return corr


def _candidate_peak_indices(corr: np.ndarray, rel_height: float = 0.6) -> np.ndarray:
    """
    Pick correlation peak candidates that reach at least `rel_height` * max(corr).
    Returns sorted, unique indices. Falls back to global argmax when no peak passes.
    """
    if corr.size == 0:
        return np.asarray([], dtype=int)
    rel_height = float(rel_height)
    if not np.isfinite(rel_height):
        rel_height = 0.6
    rel_height = min(max(rel_height, 0.0), 1.0)
    max_val = float(np.max(corr))
    if not np.isfinite(max_val):
        return np.asarray([], dtype=int)
    threshold = max_val * rel_height
    peaks: list[int] = []
    for k in range(corr.size):
        left = corr[k - 1] if k > 0 else -np.inf
        right = corr[k + 1] if k < corr.size - 1 else -np.inf
        if corr[k] >= threshold and corr[k] >= left and corr[k] >= right:
            peaks.append(int(k))
    if not peaks:
        peaks.append(int(np.argmax(corr)))
    # Ensure deterministic ordering and uniqueness
    return np.asarray(sorted(set(peaks)), dtype=int)


def _select_corr_match_index(
    corr: np.ndarray,
    env_raw: np.ndarray,
    candidate_idxs: np.ndarray,
    pattern: List[Tuple[SegmentType, float]],
    frames_per_segment: List[int],
    prefer_earlier: float,
    contrast_weight: float,
    onset_weight: float,
    min_onset_z: float,
    min_contrast_z: float,
    eps: float = 1e-12,
) -> int:
    """
    Pick the best candidate index by re-scoring peaks using envelope contrast heuristics.
    Falls back to the global maximum if no candidate satisfies the heuristics.
    """
    if corr.size == 0:
        return 0

    env_std = float(np.std(env_raw)) if env_raw.size > 0 else 0.0
    if not np.isfinite(env_std) or env_std < eps:
        env_std = 1.0

    seg_types = [seg[0] for seg in pattern]
    total_frames = int(sum(frames_per_segment))
    prefer_earlier = float(prefer_earlier)
    contrast_weight = float(contrast_weight)
    onset_weight = float(onset_weight)
    min_onset_z = float(min_onset_z)
    min_contrast_z = float(min_contrast_z)

    best_idx: int | None = None
    best_score = -np.inf
    if candidate_idxs.size == 0:
        candidate_idxs = np.asarray([int(np.argmax(corr))], dtype=int)

    for idx in map(int, candidate_idxs):
        if idx < 0 or idx >= corr.size:
            continue

        if total_frames > 0:
            if idx + total_frames > env_raw.size:
                continue
            raw_seg = env_raw[idx:idx + total_frames]
        else:
            raw_seg = env_raw[idx:idx + 1]

        if raw_seg.size == 0:
            continue

        pos = 0
        vib_means: list[float] = []
        still_means: list[float] = []
        valid = True
        for seg_type, seg_frames in zip(seg_types, frames_per_segment):
            seg_frames = int(seg_frames)
            if seg_frames <= 0 or pos + seg_frames > raw_seg.size:
                valid = False
                break
            seg_vals = raw_seg[pos:pos + seg_frames]
            seg_mean = float(seg_vals.mean()) if seg_vals.size > 0 else 0.0
            if seg_type == 'v':
                vib_means.append(seg_mean)
            else:
                still_means.append(seg_mean)
            pos += seg_frames
        if not valid:
            continue

        v_mean = float(np.mean(vib_means)) if vib_means else float(raw_seg.mean())
        if still_means:
            s_mean = float(np.mean(still_means))
        elif idx > 0:
            s_mean = float(env_raw[:idx].mean())
        else:
            s_mean = v_mean

        contrast = (v_mean - s_mean) / env_std if env_std > 0 else 0.0
        if not np.isfinite(contrast):
            contrast = 0.0

        onset_jump = contrast
        if frames_per_segment and seg_types and seg_types[0] == 'v':
            first_len = int(frames_per_segment[0])
            if first_len > 0:
                first_seg = raw_seg[:first_len]
                first_mean = float(first_seg.mean()) if first_seg.size > 0 else v_mean
                pre_slice = env_raw[max(0, idx - first_len):idx]
                if pre_slice.size > 0:
                    pre_mean = float(pre_slice.mean())
                elif still_means:
                    pre_mean = float(np.mean(still_means))
                else:
                    pre_mean = s_mean
                onset_jump = (first_mean - pre_mean) / env_std if env_std > 0 else 0.0
                if not np.isfinite(onset_jump):
                    onset_jump = 0.0

        if onset_jump < min_onset_z or contrast < min_contrast_z:
            continue

        time_norm = idx / max(1, corr.size - 1)
        total_score = (
            float(corr[idx])
            + contrast_weight * contrast
            + onset_weight * onset_jump
            - prefer_earlier * time_norm
        )

        if (
            total_score > best_score
            or (np.isclose(total_score, best_score) and (best_idx is None or idx < best_idx))
        ):
            best_score = total_score
            best_idx = idx

    if best_idx is None:
        best_idx = int(np.argmax(corr))
    return int(best_idx)


def find_pattern_start(
    x: np.ndarray,
    fs: float,
    pattern: List[Tuple[SegmentType, float]],
    win_sec: float = 0.02,
    hop_sec: float = 0.01,
    method: Literal['diff_rms', 'mag_rms', 'phase_std'] = 'diff_rms',
    vib_value: float = 1.0,
    still_value: float = 0.0,
) -> dict:
    """
    Detect the start of a vibration–stillness repeating pattern via matched filtering
    on a motion envelope derived from a complex radar signal.

    Parameters
    - x: complex ndarray of shape (N,)
    - fs: sampling rate in Hz
    - pattern: list of (segment_type, duration_sec), e.g.,
        [('v', 0.3), ('s', 0.2), ('v', 0.3), ('s', 0.2), ('v', 0.3), ('s', 0.2)]
    - win_sec: window size for envelope (seconds)
    - hop_sec: hop size for envelope (seconds)
    - method: envelope method ('diff_rms' recommended by default)
    - vib_value, still_value: values in template; default 1 and 0

    Returns dict with keys:
    - start_frame: int, frame index of detected start
    - start_sample: int, sample index (approximate, aligned to frame hop)
    - start_time: float seconds
    - score: float, correlation peak
    - corr: ndarray, correlation curve
    - env: ndarray, computed envelope (z-scored)
    - template: ndarray, template used (zero-mean, unit-norm)
    - frame_times: ndarray of frame centers (seconds)
    - meta: dict with metadata (fs, win_sec, hop_sec, method)
    """
    env, frame_times, win_len, hop_len = _short_time_motion_envelope(
        x=x, fs=fs, win_sec=win_sec, hop_sec=hop_sec, method=method
    )
    template, _ = _build_pattern_template(
        segments=pattern, fs=fs, hop_sec=hop_sec, vib_value=vib_value, still_value=still_value
    )

    corr = _valid_normalized_correlation(env, template)
    if corr.size == 0:
        return {
            'start_frame': None,
            'start_sample': None,
            'start_time': None,
            'score': None,
            'corr': corr,
            'env': env,
            'template': template,
            'frame_times': frame_times,
            'meta': {
                'fs': fs, 'win_sec': win_sec, 'hop_sec': hop_sec, 'method': method,
                'win_len': win_len, 'hop_len': hop_len,
            },
        }

    k = int(np.argmax(corr))
    start_sample = int(round(k * hop_len + win_len / 2.0))
    start_time = start_sample / float(fs)
    return {
        'start_frame': k,
        'start_sample': start_sample,
        'start_time': start_time,
        'score': float(corr[k]),
        'corr': corr,
        'env': env,
        'template': template,
        'frame_times': frame_times,
        'meta': {
            'fs': fs, 'win_sec': win_sec, 'hop_sec': hop_sec, 'method': method,
            'win_len': win_len, 'hop_len': hop_len,
        },
    }

def _try_load_mat(mat_path: str):
    """
    Load a MAT file, supporting both v7 (scipy.io.loadmat) and v7.3 (HDF5 via h5py).

    Returns a dict-like object mapping variable names to numpy arrays or h5py datasets.
    """
    # Try scipy first
    try:
        from scipy.io import loadmat  # type: ignore
        md = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
        # Remove meta keys that scipy adds
        return {k: v for k, v in md.items() if not k.startswith('__')}
    except Exception as e_scipy:
        # Fall back to h5py for v7.3
        try:
            import h5py  # type: ignore
            f = h5py.File(mat_path, 'r')
            return f
        except Exception as e_h5:
            raise RuntimeError(
                f"Failed to load MAT file with scipy and h5py. scipy error: {e_scipy}; h5py error: {e_h5}"
            )


def _as_numpy(arr):
    """
    Convert scipy or h5py-stored arrays to numpy arrays with native dtype.
    """
    try:
        import numpy as _np
        if 'h5py' in type(arr).__module__:
            # h5py datasets need [:] to realize
            return _np.array(arr[...])
        return _np.array(arr)
    except Exception:
        return np.array(arr)


def _load_signal_and_dims(store, var_name: str = 'signal_raw'):
    """
    Extract `signal_raw` and optionally `numLoops` / `numFrames` if present.
    Returns tuple (signal_raw_np, numLoops, numFrames) where latter may be inferred.
    """
    # Access strategy differs for scipy dict vs h5py group
    is_h5 = hasattr(store, 'keys') and not isinstance(store, dict)
    if isinstance(store, dict):
        if var_name not in store:
            raise KeyError(f"Variable {var_name} not found in MAT file")
        sig = _as_numpy(store[var_name])
        return sig
    else:
        # h5py File or Group
        if var_name not in store:
            # Try common nesting patterns
            candidates = [k for k in store.keys() if var_name in k]
            if candidates:
                key = candidates[0]
            else:
                raise KeyError(f"Variable {var_name} not found in MAT file (h5)")
        else:
            key = var_name
        sig = _as_numpy(store[key])
        return sig

def prepare_complex_series_from_mat(
    mat_path: str,
    antenna_idx: int = 0,
    sample_idx: int = 19,
    loop_start: int = 1,
    angle_idx: int = 0,
    var_name: str = 'signal_raw',
):
    """
    Mirror MATLAB:
        temp_data = squeeze(signal_raw(1,20,2:end,1,:));
        data = reshape(temp_data, [], (numLoops-1)*numFrames);

    In Python (0-based): antenna_idx=0, sample_idx=19, loops 1:, angle_idx=0, frames :
    Returns: 1-D complex ndarray of length (numLoops-1)*numFrames (row-major equivalent).
    """
    store = _try_load_mat(mat_path)
    sig = _load_signal_and_dims(store, var_name=var_name)

    numFrames, numAngles, numLoops, numSamples, numAntennas = sig.shape

    # Slice per MATLAB instruction
    sel = sig[:, 0, loop_start:, sample_idx, antenna_idx]

    # Ensure complex dtype if stored as real-imag parts along last dim; otherwise cast
    sel_np = np.asarray(sel)
    mat2d = np.squeeze(sel_np['real'] + 1j * sel_np['imag'])

    Fnum, Lnum = mat2d.shape

    # Reshape to 1 x ((numLoops-1)*numFrames) in MATLAB (column-major).
    # In numpy (row-major), flatten in column-major to match MATLAB behavior.
    vec = np.reshape(mat2d, (Lnum * Fnum,), order='C')
    return vec.astype(np.complex128), int(numLoops), int(numFrames)


def find_end_by_zero(x: np.ndarray, atol: float = 0.0) -> int:
    """
    Find first index where amplitude is exactly zero (or within atol), meaning acquisition ended.
    Returns the usable length (exclusive end index). If none found, returns len(x).
    """
    mag = np.abs(x)
    if atol <= 0.0:
        zero_mask = (mag == 0.0)
    else:
        zero_mask = (mag <= float(atol))
    idx = np.nonzero(zero_mask)[0]
    return int(idx[0]) if idx.size > 0 else int(len(x))


# def detrend_amplitude(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     """
#     对复数信号的幅度做去趋势：估计线性幅度趋势并归一化幅度，保持相位。

#     注意：仅当 `x` 为复数包络/基带时使用此函数。如果输入是相位序列
#     （实数），请改用 `detrend_phase`。
#     """
#     x = np.asarray(x)
#     n = x.size
#     if n == 0:
#         return x
#     # 仅幅度归一化，不改变相位（实数正比例因子）
#     t = np.arange(n, dtype=float)
#     mag = np.abs(x) + eps
#     a, b = np.polyfit(t, mag, 1)  # mag ~ a*t + b
#     trend = a * t + b
#     trend = np.maximum(trend, eps)
#     y = x / trend * np.mean(trend)
#     return y.astype(x.dtype, copy=False)


def detrend_phase(phi: np.ndarray,
                  window: int | None = None,
                  polyorder: int = 2) -> np.ndarray:
    """
    去除缓慢变化的相位趋势（实数输入 -> 实数输出）。

    方法：Savitzky–Golay 平滑得到“趋势”，然后做相减。
    - `window` 未指定时，自动选取为信号长度的 ~5% 且为奇数，至少 11。
    - `polyorder` 默认为 2，可根据曲率适当增减。
    结果会保留原始均值，便于对比。
    """
    phi = np.asarray(phi, dtype=float)
    n = phi.size
    if n == 0:
        return phi

    # 自动窗口：约 5% 的长度，且为奇数，至少 11，且不超过 n-1
    if window is None:
        w = max(11, int(round(n * 0.05)))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        window = max(5, w)

    # 估计趋势并相减，同时把均值加回以保证尺度感知一致
    trend = savgol_filter(phi, window_length=window, polyorder=min(polyorder, window - 1), mode='interp')
    detrended = phi - trend + float(np.mean(trend))
    return detrended


def run_find_start_on_mat(
    mat_path: str,
    fs: float,
    pattern: List[Tuple[SegmentType, float]],
    win_sec: float = 0.02,
    hop_sec: float = 0.01,
    method: Literal['diff_rms', 'mag_rms', 'phase_std'] = 'diff_rms',
    antenna_idx: int = 0,
    sample_idx: int = 19,
    loop_start: int = 1,
    angle_idx: int = 0,
    zero_atol: float = 0.0,
    var_name: str = 'signal_raw',
    detector: Literal['envelope', 'cwt'] = 'envelope',
    detector_kwargs: dict | None = None,
):
    """
    Full pipeline for the provided MAT file and MATLAB-like slicing.
    1) Load and slice per MATLAB code
    2) Truncate at first zero-amplitude point
    3) Detrend amplitude
    4) Run the selected pattern detector (short-time envelope or CWT)
    """
    x_raw, numLoops, numFrames = prepare_complex_series_from_mat(
        mat_path=mat_path,
        antenna_idx=antenna_idx,
        sample_idx=sample_idx,
        loop_start=loop_start,
        angle_idx=angle_idx,
        var_name=var_name,
    )

    end_idx = find_end_by_zero(x_raw, atol=zero_atol)
    x_used = x_raw[:end_idx]
    phase_unwrap = np.unwrap(np.angle(x_used))
    phase_dt = detrend_phase(phase_unwrap)
    phase_filt = bandpass(phase_dt, fs, 10, 200, 3)
    
    # Plot
    t_used = t_axis(x_used, fs)
    plt.figure(figsize=(10, 4))
    plt.plot(t_used, phase_unwrap, label='unwrap', linewidth=1)
    plt.plot(t_used, phase_dt, label='dt', linewidth=1)
    plt.plot(t_used, phase_filt, label='filt', linewidth=1)

    detector_key = detector.lower()
    if detector_key not in ('envelope', 'cwt'):
        raise ValueError(f"detector must be 'envelope' or 'cwt', got {detector}")

    base_kwargs = dict(detector_kwargs or {})
    if detector_key == 'cwt':
        kwargs = dict(base_kwargs)
        win_used = kwargs.pop('win_sec', win_sec)
        hop_used = kwargs.pop('hop_sec', hop_sec)
        res = find_pattern_start_cwt(
            x=phase_filt,
            fs=fs,
            pattern=pattern,
            win_sec=win_used,
            hop_sec=hop_used,
            **kwargs,
        )
        detector_params = {'win_sec': win_used, 'hop_sec': hop_used, **kwargs}
    else:
        kwargs = dict(base_kwargs)
        win_used = kwargs.pop('win_sec', win_sec)
        hop_used = kwargs.pop('hop_sec', hop_sec)
        method_used = kwargs.pop('method', method)
        res = find_pattern_start(
            x=phase_filt,
            fs=fs,
            pattern=pattern,
            win_sec=win_used,
            hop_sec=hop_used,
            method=method_used,
            **kwargs,
        )
        detector_params = {'win_sec': win_used, 'hop_sec': hop_used, 'method': method_used, **kwargs}

    res_meta = {**res.get('meta', {})}
    res_meta.update({
        'end_idx': end_idx,
        'numLoops': numLoops,
        'numFrames': numFrames,
        'detector': detector_key,
        'detector_params': detector_params,
    })
    res['meta'] = res_meta
    return res

def bandpass(x, fs, fmin, fmax, order):
    wn = [2* fmin / fs, 2*fmax / fs]
    b, a = butter(order, wn, btype='band')
    return filtfilt(b, a, x) 

def t_axis(x, fs):
    x = np.asarray(x)
    return np.arange(x.shape[0]) / fs


def compute_cwt_power(
    x: np.ndarray,
    fs: float,
    totalscal: int,
    wavename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the CWT power spectrum for a 1-D real-valued signal.

    Returns
    -------
    power : ndarray of shape (totalscal, N)
        Squared magnitude of the wavelet coefficients.
    freqs : ndarray
        Center frequency for each scale (Hz).
    """
    # Heuristic parameters (candidate_rel_height, prefer_earlier, contrast_weight,
    # onset_weight, min_onset_z, min_contrast_z) tune the post-peak selection.
    x = np.asarray(x, dtype=float)
    if x.size == 0 or totalscal <= 0:
        return np.empty((0, x.size), dtype=float), np.asarray([], dtype=float)

    dt = 1.0 / float(fs) if fs > 1.0 else float(fs)
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)

    coeff, freqs = pywt.cwt(x, scales, wavename, dt)
    amp = np.abs(coeff)
    power = amp ** 2
    
    return amp, power, freqs


def find_pattern_start_cwt(
    x: np.ndarray,
    fs: float,
    pattern: List[Tuple[SegmentType, float]],
    totalscal: int = 256,
    wavename: str = 'morl',
    fmin: float | None = 10.0,
    fmax: float | None = 200.0,
    win_sec: float = 0.02,
    hop_sec: float = 0.01,
    smooth_sec: float = 0.02,
    candidate_rel_height: float = 0.6,
    prefer_earlier: float = 0.1,
    contrast_weight: float = 0.4,
    onset_weight: float = 0.6,
    min_onset_z: float = 0.0,
    min_contrast_z: float = -np.inf,
) -> dict:
    """
    使用连续小波变换(CWT)的时频能量包络做模板匹配，检测给定振动/静止模式的起点。

    Parameters
    - x: 1-D 实数序列（建议传相位滤波后的 `phase_filt`）
    - fs: 采样率 Hz
    - pattern: 与 `find_pattern_start` 相同的 [(kind, duration_sec), ...]
    - totalscal, wavename: CWT 参数（与 cwt.py 一致）
    - fmin, fmax: 统计能量时选取的频率范围；None 表示不限制
    - win_sec, hop_sec: 将包络下采样成帧用于匹配的窗口与步长
    - smooth_sec: 对时间包络作移动平均的平滑时长

    Returns: dict 与 `find_pattern_start` 类似，包含 start_time、score、corr、env 等。
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N == 0:
        return {
            'start_frame': None, 'start_sample': None, 'start_time': None,
            'score': None, 'corr': np.asarray([]), 'env': np.asarray([]),
            'template': np.asarray([]), 'frame_times': np.asarray([]),
            'candidate_indices': [],
            'meta': {
                'fs': fs,
                'candidate_rel_height': candidate_rel_height,
                'prefer_earlier': prefer_earlier,
                'contrast_weight': contrast_weight,
                'onset_weight': onset_weight,
                'min_onset_z': min_onset_z,
                'min_contrast_z': min_contrast_z,
            }
        }

    amp, power, freqs = compute_cwt_power(x, fs, totalscal, wavename)
    if amp.size == 0:
        return {
            'start_frame': None, 'start_sample': None, 'start_time': None,
            'score': None, 'corr': np.asarray([]), 'env': np.asarray([]),
            'template': np.asarray([]), 'frame_times': np.asarray([]),
            'candidate_indices': [],
            'meta': {
                'fs': fs, 'method': 'cwt_env', 'totalscal': totalscal, 'wavename': wavename,
                'fmin': fmin, 'fmax': fmax, 'win_sec': win_sec, 'hop_sec': hop_sec,
                'candidate_rel_height': candidate_rel_height,
                'prefer_earlier': prefer_earlier,
                'contrast_weight': contrast_weight,
                'onset_weight': onset_weight,
                'min_onset_z': min_onset_z,
                'min_contrast_z': min_contrast_z,
            },
        }

    # 频带选择
    fi = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        fi &= (freqs >= float(fmin))
    if fmax is not None:
        fi &= (freqs <= float(fmax))
    if not np.any(fi):
        fi[:] = True  # 若范围无效，则使用全频带\
            
    # 绘制频谱
    t = t_axis(x, fs)
    plt.figure(figsize=(10, 4))
    plt.contourf(t, freqs[fi], amp[fi, :], cmap='jet')

    band_amp = amp[fi, :].mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.plot(t, band_amp)
    # 时间平滑
    k = max(1, int(round(smooth_sec * fs)))
    if k > 1:
        kernel = np.ones(k, dtype=float) / k
        band_amp = np.convolve(band_amp, kernel, mode='same')

    # 帧化（与 find_pattern_start 的 envelope 输入保持一致的时间分辨率）
    win_len = max(1, int(round(win_sec * fs)))
    hop_len = max(1, int(round(hop_sec * fs)))
    frames = []
    idx = []
    for start in range(0, N - win_len + 1, hop_len):
        seg = band_amp[start:start + win_len]
        frames.append(float(np.mean(seg)))
        idx.append(start + win_len / 2.0)
    env = np.asarray(frames, dtype=float)
    env_raw = env.copy()
    frame_times = np.asarray(idx, dtype=float) / float(fs)

    # 标准化
    eps = 1e-12
    if env.size > 0:
        env = (env - env.mean()) / (env.std() + eps)

    # 模板
    template, frames_per_segment = _build_pattern_template(pattern, fs=fs, hop_sec=hop_sec)

    # 匹配
    corr = _valid_normalized_correlation(env, template)
    if corr.size == 0:
        return {
            'start_frame': None, 'start_sample': None, 'start_time': None,
            'score': None, 'corr': corr, 'env': env, 'template': template,
            'frame_times': frame_times,
            'candidate_indices': [],
            'meta': {
                'fs': fs, 'method': 'cwt_env', 'totalscal': totalscal, 'wavename': wavename,
                'fmin': fmin, 'fmax': fmax, 'win_sec': win_sec, 'hop_sec': hop_sec,
                'candidate_rel_height': candidate_rel_height,
                'prefer_earlier': prefer_earlier,
                'contrast_weight': contrast_weight,
                'onset_weight': onset_weight,
                'min_onset_z': min_onset_z,
                'min_contrast_z': min_contrast_z,
            },
        }

    candidate_idxs = _candidate_peak_indices(corr, rel_height=candidate_rel_height)
    # Re-score candidate peaks to prefer the earliest valid onset and strong vibration/still contrast
    kmax = _select_corr_match_index(
        corr=corr,
        env_raw=env_raw,
        candidate_idxs=candidate_idxs,
        pattern=pattern,
        frames_per_segment=frames_per_segment,
        prefer_earlier=prefer_earlier,
        contrast_weight=contrast_weight,
        onset_weight=onset_weight,
        min_onset_z=min_onset_z,
        min_contrast_z=min_contrast_z,
    )
    start_sample = int(round(kmax * hop_len + win_len / 2.0))
    start_time = start_sample / float(fs)
    return {
        'start_frame': kmax,
        'start_sample': start_sample,
        'start_time': start_time,
        'score': float(corr[kmax]),
        'corr': corr,
        'env': env,
        'template': template,
        'frame_times': frame_times,
        'candidate_indices': candidate_idxs.tolist(),
        'meta': {
            'fs': fs, 'method': 'cwt_env', 'totalscal': totalscal, 'wavename': wavename,
            'fmin': fmin, 'fmax': fmax, 'win_sec': win_sec, 'hop_sec': hop_sec,
            'candidate_rel_height': candidate_rel_height,
            'prefer_earlier': prefer_earlier,
            'contrast_weight': contrast_weight,
            'onset_weight': onset_weight,
            'min_onset_z': min_onset_z,
            'min_contrast_z': min_contrast_z,
        },
    }

if __name__ == '__main__':
    mat_path = "E:/high_DiskSpy/1028/saveData/25_1028_exp3_disk_8_75_25_v28.mat"
    time_vib = 0.075
    time_still = 0.025
    pattern = string_to_pattern("100001000010000", vib_sec=time_vib, still_sec=time_still)
    fs = 255 / 16.8769 * 1000
    win_sec = 0.01
    hop_sec = 0.001
    # 'diff_rms'  -> RMS of first difference magnitude (robust to DC)
    # 'mag_rms'   -> RMS magnitude (good if vibration modulates amplitude)
    # 'phase_std' -> Std of unwrapped phase (good for micro-motions)
    method = 'mag_rms'
    antenna_idx = 0
    sample_idx = 19
    angle_idx = 0
    loop_start = 0
    zero_atol = 0.0
    var_name = "signal_raw"
    detector = 'cwt'  # change to 'cwt' to use the wavelet-based detector
    detector_kwargs = {'totalscal': 256, 'wavename': 'morl','fmin': 10,'fmax': 200} if detector == 'cwt' else {}

    res = run_find_start_on_mat(
        mat_path=mat_path,
        fs=fs,
        pattern=pattern,
        win_sec=win_sec,
        hop_sec=hop_sec,
        method=method,
        antenna_idx=antenna_idx,
        sample_idx=sample_idx,
        angle_idx=angle_idx,
        loop_start=loop_start,
        zero_atol=zero_atol,
        var_name=var_name,
        detector=detector,
        detector_kwargs=detector_kwargs,
    )

    print(f"[{detector}] start_frame: {res['start_frame']}")
    print(f"[{detector}] start_sample: {res['start_sample']}")
    print(f"[{detector}] start_time: {res['start_time']:.6f}s")

    corr = res.get('corr', np.asarray([]))
    frame_times = res.get('frame_times', np.asarray([]))
    if corr.size > 0 and frame_times.size > 0:
        corr_times = frame_times[:corr.size]
        plt.figure(figsize=(10, 4))
        plt.plot(corr_times, corr, label='correlation')
        if res.get('start_time') is not None:
            plt.axvline(res['start_time'], color='r', linestyle='--', label='detected start')
        plt.title(f"Pattern match correlation ({detector})")
        plt.xlabel('Time (s)')
        plt.ylabel('Correlation')
        plt.legend()

    plt.show()
