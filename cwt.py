from typing import Tuple
import numpy as np
import pywt

# cwt_kwargs = {'totalscal': 256, 'wavename': 'morl', 'fmin': 10, 'fmax': 200}
def compute_cwt_power(
    x: np.ndarray,
    fs: float,
    cwt_kwargs: dict | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    kwargs=dict(dict(cwt_kwargs))
    totalscal = kwargs.pop('totalscal', 256)
    wavename = kwargs.pop('wavename', 'morl')
    fmin = kwargs.pop('fmin', 10)
    fmax = kwargs.pop('fmax', 200.0)

    # 小波变换
    dt = 1.0 / float(fs) if fs > 1.0 else float(fs)
    fc = pywt.central_frequency(wavename)
    cparam = 2 * fc * totalscal
    scales = cparam / np.arange(totalscal, 0, -1)

    coeff, freqs = pywt.cwt(x, scales, wavename, dt)
    amp = np.abs(coeff)
    power = amp ** 2
    
    # 频带选择
    fi = np.ones_like(freqs, dtype=bool)
    if fmin is not None:
        fi &= (freqs >= float(fmin))
    if fmax is not None:
        fi &= (freqs <= float(fmax))
    if not np.any(fi):
        fi[:] = True  # 若范围无效，则使用全频带
    return amp[fi,:], power[fi,:], freqs[fi]


import pandas as pd

def remove_cwt_background(power: np.ndarray, t: np.ndarray, 
                          bg_window_sec: float = 3.0, 
                          interp_step_sec: float = 0.05) -> np.ndarray:
    """
    对 CWT 功率谱进行时间方向的背景去除。
    
    参数：
        power: np.ndarray, shape=(n_freqs, n_times)，CWT功率谱
        t: np.ndarray, shape=(n_times,), 时间向量（秒）
        bg_window_sec: 背景估计窗口（秒），相当于平滑窗口
        interp_step_sec: 插值的时间步长（秒）
    返回：
        去除背景后的功率谱 z (n_freqs, n_times)
    """
    rectime = pd.to_timedelta(t, 's')
    spg = pd.DataFrame(power.T, index=rectime)

    # 按指定时间窗口（例如3秒）做平均作为背景
    bg = spg.resample(f'{bg_window_sec}s').mean().copy()

    # 再以较小时间步长插值（例如0.05秒）
    bg = bg.resample(f'{interp_step_sec}s').interpolate(method='time')

    # 对齐到原始时间索引
    bg = bg.reindex(rectime, method='nearest')

    # 转置回来
    background = bg.values.T

    # 去背景
    z = power - background
    z[z < 0] = 0  # 去除负值（可选）
    return z
