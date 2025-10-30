import numpy as np
from typing import List, Tuple, Literal, Dict, Any


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


if __name__ == '__main__':
    # Run a quick demo when executed directly.
    _demo_synthetic()
