import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

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

def bandpass(x, fs, fmin, fmax, order):
    wn = [2* fmin / fs, 2*fmax / fs]
    b, a = butter(order, wn, btype='band')
    return filtfilt(b, a, x) 

def preprocess_signal(x: np.ndarray, fs: float, bandpass_kwargs: dict | None = None, zero_atol: float = 0.0) -> np.ndarray:
    """
    对信号进行预处理
    """
    # step1:找到结束点
    end_idx = find_end_by_zero(x, atol=zero_atol)
    x_used = x[:end_idx]\
    # step2:计算相位，解缠绕
    phase_unwrap = np.unwrap(np.angle(x_used))
    # step3:去趋势
    phase_dt = detrend_phase(phase_unwrap)
    # step4:滤波
    kwargs = dict(dict(bandpass_kwargs))
    fmin=kwargs.pop('fmin', 10)
    fmax=kwargs.pop('fmax', 200.0)
    order=kwargs.pop('order', 3)
    phase_filt = bandpass(phase_dt, fs, fmin, fmax, order)
    return phase_filt