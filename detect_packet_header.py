from load_from_mat import load_from_mat
from preprocess_signal import preprocess_signal
from pattern_utils import SegmentType, string_to_pattern, _build_pattern_template
from cwt import compute_cwt_power, remove_cwt_background
from match import detect_vibration_start_trend, detect_vibration_start, detect_vibration_start_energy
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

boolean = False
def t_axis(x, fs):
    x = np.asarray(x)
    return np.arange(x.shape[0]) / fs


def detect_pattern_start(
    x: np.ndarray,
    fs: float,
    pattern: List[Tuple[SegmentType, float]],
    detect_kwargs: dict | None = None,
    cwt_kwargs: dict | None = None
) -> dict:
    amp, power, freqs = compute_cwt_power(x, fs, cwt_kwargs)
    db=10*np.log10(power)
    t = t_axis(x, 1)
    db_remove_bg = remove_cwt_background(db, t, 30000, 1)

    band_amp = amp.mean(axis=0)
    band_db_remove_bg = db_remove_bg.mean(axis=0)
    
    # detect参数读取
    kwargs=dict(dict(detect_kwargs))
    win_sec = detect_kwargs.get('win_sec', 0.01)
    hop_sec = detect_kwargs.get('hop_sec', 0.001)
    smooth_sec = kwargs.pop('smooth_sec', 0.01)
    
    #时间平滑
    # plt.figure(figsize=(10, 4))
    # plt.plot(band_amp)
    # plt.plot(band_db_remove_bg)
    
    k = max(1, int(round(smooth_sec * fs)))
    if k > 1:
        kernel = np.ones(k, dtype=float) / k
        band_amp = np.convolve(band_amp, kernel, mode='same')
        band_db_remove_bg = np.convolve(band_db_remove_bg, kernel, mode='same')
    # if boolean:
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(band_amp)
    #     plt.plot(band_db_remove_bg)
    
    # 模板
    template, frames_per_segment = _build_pattern_template(pattern, fs=fs)

    # 匹配
    n_hop = int(round(hop_sec * fs))
    
    start_idx, corr, hop_idxs = detect_vibration_start_energy(band_db_remove_bg, template, n_hop, eps=1e-12, energy_ratio_thresh=2, use_trend=False)
    # print(f"检测到静止起点: {start_idx}")
    # if boolean:
    #     print(f"检测到振动起点: {(int)(start_idx+4*time_still*fs)}")
    # plt.figure(figsize=(10,4))
    # plt.plot(hop_idxs, corr, label='Correlation')
    # plt.axvline(start_idx, color='r', linestyle='--', label='Detected start')
    # plt.title("Correlation with Energy Constraint")
    # plt.xlabel("Sample index")
    # plt.ylabel("Correlation")
    # plt.legend()
    
    # 绘制频谱
    if boolean:
        t = t_axis(x, 1)
        plt.figure(figsize=(10, 4))
        plt.contourf(t, freqs, amp, cmap='jet')
        plt.axvline(start_idx, color='r', linestyle='--', label='Detected start')
        plt.axvline(start_idx+4*time_still*fs,color='g', linestyle='--', label='Detected start')
        plt.figure(figsize=(10, 4))
        plt.contourf(t, freqs, db_remove_bg, cmap='jet')
        plt.axvline(start_idx, color='r', linestyle='--', label='Detected start')
        plt.axvline(start_idx+4*time_still*fs,color='g', linestyle='--', label='Detected start')
    
    # plt.figure(figsize=(10, 4))
    # plt.pcolormesh(t, freqs, amp, cmap='jet')
    # plt.axvline(start_idx, color='r', linestyle='--', label='Detected start')
    
    return start_idx


def run_find_start_on_mat(
    mat_path: str,
    pattern: List[Tuple[SegmentType, float]],
    fs: float,
    antenna_idx: int = 0,
    sample_idx: int = 19,
    angle_idx: int = 0,
    loop_start: int = 1,
    zero_atol: float = 0.0,
    var_name: str = 'signal_raw',
    bandpass_kwargs: dict | None = None,
    cwt_kwargs: dict | None = None,
    detect_kwargs: dict | None = None,
):
    x_raw, numLoops, numFrames = load_from_mat(
        mat_path=mat_path,
        antenna_idx=antenna_idx,
        sample_idx=sample_idx,
        loop_start=loop_start,
        angle_idx=angle_idx,
        var_name=var_name,
    )

    phase_filt=preprocess_signal(x_raw, fs, bandpass_kwargs, zero_atol)
    
    # Plot
    # t_filt = t_axis(phase_filt, fs)
    # t_all = t_axis(x_raw, fs)
    # plt.figure(figsize=(10, 4))
    # plt.plot(t_all, np.angle(x_raw), label='dt', linewidth=1)
    # plt.plot(t_filt, phase_filt, label='filt', linewidth=1)
    
    
    # Detect pattern
    start_idx = detect_pattern_start(
        x=phase_filt,
        fs=fs,
        pattern=pattern,
        cwt_kwargs=cwt_kwargs,
        detect_kwargs=detect_kwargs,
    )
    return start_idx


def detect_packet_header(
    mat_path: str,
    pattern_str: str,
    time_vib: float,
    time_still: float,
    antenna_idx: int,
    sample_idx : int,
):
    pattern = string_to_pattern(pattern_str, vib_sec=time_vib, still_sec=time_still)
    # 数据选取参数
    
    angle_idx = 0
    loop_start = 0 if angle_idx == 0 else 1
    var_name = "signal_raw"
    zero_atol = 0.0
    
    # 滤波参数
    fs = 255 / 16.8769 * 1000
    bandpass_kwargs = {'fmin': 10, 'fmax': 200, 'order': 3}
    
    # 小波变换参数
    cwt_kwargs = {'totalscal': 256, 'wavename': 'morl', 'fmin': 10, 'fmax': 200}
    
    # 检测器参数
    detect_kwargs = {'win_sec':0.01, 'hop_sec':0.001, 'smooth_sec':0.02}

    start_idx = run_find_start_on_mat(
        mat_path=mat_path,
        antenna_idx=antenna_idx,
        sample_idx=sample_idx,
        angle_idx=angle_idx,
        loop_start=loop_start,
        zero_atol=zero_atol,
        var_name=var_name,
        pattern=pattern,
        fs=fs,
        bandpass_kwargs=bandpass_kwargs,
        cwt_kwargs=cwt_kwargs,
        detect_kwargs=detect_kwargs,
    )
    # return int(round(start_idx+4*time_still*fs))
    return start_idx
    
    


if __name__ == '__main__':
    # 数据路径
    mat_path = "E:/high_DiskSpy/1028/saveData/25_1028_exp3_disk_8_75_25_v18.mat"
    # 数据选取参数
    antenna_idx = 3
    sample_idx = 19
    angle_idx = 0
    loop_start = 0
    var_name = "signal_raw"
    zero_atol = 0.0
    
    # 振动模式
    time_vib = 0.075
    time_still = 0.025
    pattern = string_to_pattern("0000110000110000110000", vib_sec=time_vib, still_sec=time_still)
    
    # 滤波参数
    fs = 255 / 16.8769 * 1000
    bandpass_kwargs = {'fmin': 10, 'fmax': 200, 'order': 3}
    
    # 小波变换参数
    cwt_kwargs = {'totalscal': 256, 'wavename': 'morl', 'fmin': 10, 'fmax': 200}
    
    # 检测器参数
    detect_kwargs = {'win_sec':0.01, 'hop_sec':0.001, 'smooth_sec':0.02}

    start_idx = run_find_start_on_mat(
        mat_path=mat_path,
        antenna_idx=antenna_idx,
        sample_idx=sample_idx,
        angle_idx=angle_idx,
        loop_start=loop_start,
        zero_atol=zero_atol,
        var_name=var_name,
        pattern=pattern,
        fs=fs,
        bandpass_kwargs=bandpass_kwargs,
        cwt_kwargs=cwt_kwargs,
        detect_kwargs=detect_kwargs,
    )
    
    # print(f"振动采样数：{time_vib * fs}")
    # print(f"静止采样数：{time_still * fs}")
    if boolean:
        plt.show()