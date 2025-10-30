import numpy as np
from typing import List, Tuple, Literal
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


SegmentType = Literal['v', 's']  # 'v' for vibration, 's' for stillness


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


def _demo_synthetic():
    """
    Minimal synthetic example to sanity-check the detector.
    Generates a complex tone whose phase is wobbled during 'v' segments and steady during 's'.
    """
    fs = 2000.0
    T = 5.0
    t = np.arange(int(T * fs)) / fs

    # Ground truth pattern: v-s-v-s-v-s
    pattern = [('v', 0.4), ('s', 0.3), ('v', 0.4), ('s', 0.3), ('v', 0.4), ('s', 0.3)]

    # Build segment-wise phase perturbation
    vib_f = 12.0  # vibration frequency (Hz)
    vib_amp = 0.2  # phase deviation (radians)

    phase = 2 * np.pi * 10.0 * t  # base carrier
    cur = 0
    for kind, dur in pattern:
        n = int(round(dur * fs))
        idx = slice(cur, cur + n)
        if kind == 'v':
            phase[idx] += vib_amp * np.sin(2 * np.pi * vib_f * t[idx])
        cur += n

    # Add noise
    rng = np.random.default_rng(0)
    noise = 0.1 * (rng.standard_normal(t.shape) + 1j * rng.standard_normal(t.shape))
    x = np.exp(1j * phase) + noise

    res = find_pattern_start(
        x=x,
        fs=fs,
        pattern=pattern,
        win_sec=0.05,
        hop_sec=0.02,
        method='diff_rms',
    )

    print('Detected start_time (s):', res['start_time'])
    print('Detected score:', res['score'])
    return res

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


def detrend_amplitude(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Remove slow amplitude trend by dividing out a best-fit linear magnitude trend.
    Preserves phase. Safer than subtractive detrend on magnitude because it avoids negatives.
    """
    n = x.size
    if n == 0:
        return x
    t = np.arange(n, dtype=float)
    mag = np.abs(x) + eps
    # Fit linear trend in least squares sense: mag ~ a*t + b
    a, b = np.polyfit(t, mag, 1)
    trend = a * t + b
    # Avoid division by very small values
    trend = np.maximum(trend, eps)
    y = x / trend * np.mean(trend)
    return y.astype(np.complex128)


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
):
    """
    Full pipeline for the provided MAT file and MATLAB-like slicing.
    1) Load and slice per MATLAB code
    2) Truncate at first zero-amplitude point
    3) Detrend amplitude
    4) Run find_pattern_start
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
    x_dt = detrend_amplitude(x_used)
    x_filt = bandpass(x_dt, fs, 10, 200, 3)
    
    t_raw = t_axis(x_raw, fs)
    t_used = t_axis(x_used, fs)
    t_dt = t_axis(x_dt, fs)
    t_filt = t_axis(x_filt, fs)
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_raw, np.abs(x_raw), label='raw')
    plt.plot(t_used, np.abs(x_used), label='used')
    plt.plot(t_dt, np.abs(x_dt), label='dt')
    
    plt.figure(figsize=(10, 4))
    plt.plot(t_filt, np.abs(x_filt), label='filt')

    res = find_pattern_start(
        x=x_filt, fs=fs, pattern=pattern, win_sec=win_sec, hop_sec=hop_sec, method=method
    )
    res['meta'] = {**res.get('meta', {}), 'end_idx': end_idx, 'numLoops': numLoops, 'numFrames': numFrames}
    return res

def bandpass(x, fs, fmin, fmax, order):
    wn = [2* fmin / fs, 2*fmax / fs]
    b, a = butter(order, wn, btype='band')
    return filtfilt(b, a, x) 

def t_axis(x, fs):
    x = np.asarray(x)
    return np.arange(x.shape[0]) / fs

def _parse_pattern_arg(arg: str) -> List[Tuple[SegmentType, float]]:
    """
    Parse a simple pattern string like: v:0.4,s:0.3,v:0.4,s:0.3
    """
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    pat: List[Tuple[SegmentType, float]] = []
    for p in parts:
        kind, dur = p.split(':')
        kind = kind.strip().lower()
        if kind not in ('v', 's'):
            raise ValueError(f"Invalid segment kind: {kind}")
        pat.append((kind, float(dur)))
    return pat

if __name__ == '__main__':
    mat_path="E:/high_DiskSpy/1028/saveData/25_1028_exp3_disk_8_75_50_v11.mat"
    fs=255/16.8769*1000
    pattern=_parse_pattern_arg("v:0.075,v:0.075,s:0.05,s:0.05,v:0.075,v:0.075,s:0.05,s:0.05,v:0.075,v:0.075")
    win_sec=0.01
    hop_sec=0.001
    # Literal['diff_rms', 'mag_rms', 'phase_std'] = 'diff_rms',
    # * diff_rms：一阶差分幅度的 RMS（对直流/慢漂移更鲁棒，默认）。
    # * mag_rms：幅度的 RMS（幅度被振动调制时有效）。
    # * phase_std：解包裹相位的标准差（适合微小相位抖动）。
    method='mag_rms'
    antenna_idx=0
    sample_idx=19
    angle_idx=0
    loop_start=0
    zero_atol=0
    var_name="signal_raw"
    
    res=run_find_start_on_mat(
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
        var_name=var_name
    )
    
    print("start_frame: ",res['start_frame'])
    print("start_sample: ",res['start_sample'])
    print("start_time: ",res['start_time'])
    plt.show()
