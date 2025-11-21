from load_from_mat import load_from_mat
from preprocess_signal import preprocess_signal
from cwt import compute_cwt_power, remove_cwt_background
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from cut_image import cut_image,save_image
from generate_sigs import generate_sigs
import os
import csv

def t_axis(x, fs):
    x = np.asarray(x)
    return np.arange(x.shape[0]) / fs

if __name__ == '__main__':
    root = "E:\\high_DiskSpy\\csv\\"
    input_csv = os.path.join(root, 'disk_9_results_with_unwrap.csv')

    rows = []
    with open(input_csv, 'r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 读取参数
    # sample_idx = 19
    angle_idx = 0
    loop_start = 0
    var_name = "signal_raw"
    zero_atol = 0.0
    # 滤波参数
    fs = 255 / 16.8769 * 1000
    bandpass_kwargs = {'fmin': 10, 'fmax': 200, 'order': 3}
    # 小波变换参数
    cwt_kwargs = {'totalscal': 256, 'wavename': 'morl', 'fmin': 10, 'fmax': 200}

    results = []
    for row in rows:
        number = (row.get('number') or '').strip()
        subdir = (row.get('subdir') or '').strip()
        path = (row.get('path') or '').strip()
        pattern_str = (row.get('pattern') or '').strip().strip('"')
        time_vib_s = float((row.get('time_vib_s') or '').strip())
        time_still_s = float((row.get('time_still_s') or '').strip())
        sample_idx = int((row.get('sample_idx') or '').strip())
        antenna_idx = int((row.get('antenna_idx') or '').strip())
        start_sample = int((row.get('start_sample') or '').strip())
        status = (row.get('status') or '').strip()
        
        # if status == '0' or time_still_s == 0.05:
        #     continue
        
        if "test" in subdir:
            continue
        
        # load .mat
        x_raw, numLoops, numFrames = load_from_mat(
            mat_path=path,
            antenna_idx=antenna_idx,
            sample_idx=sample_idx,
            loop_start=loop_start,
            angle_idx=angle_idx,
            var_name=var_name,
        )

        # signal process
        phase_filt=preprocess_signal(x_raw, fs, bandpass_kwargs, zero_atol)
        amp, power, freqs = compute_cwt_power(phase_filt, fs, cwt_kwargs)
        db=10*np.log10(power)
        t = t_axis(phase_filt, 1)
        db_remove_bg = remove_cwt_background(db, t, 30000, 1)
        # amp_remove_bg = remove_cwt_background(amp, t, 30000, 1)
        
        # img
        length = 17000
        start = start_sample - 2000
        if start < 0:
            print(f"[start] 超出范围，无法截取图片，文件是{subdir}")
            continue
        end = start + length
        if end > amp.shape[1]:
            print(f"[end] 超出范围，无法截取图片，文件是{subdir}")
            continue
        n_sample_vib = int(time_vib_s * fs)
        n_sample_still = int(time_still_s * fs)
        
        time_segs = generate_sigs(start_sample, n_sample_vib, n_sample_still, pattern_str)
        
        # # 保存每段振动信号，方便后续单独分析
        # seg_root = r"E:\\high_DiskSpy\\signalSeg\\disk_8\\amp"
        # for seg_idx, (vib_pattern, _, seg_start, seg_end) in enumerate(time_segs, start=1):
        #     seg_dir = os.path.join(seg_root, str(vib_pattern))
        #     os.makedirs(seg_dir, exist_ok=True)
        #     segment = amp[:,seg_start:seg_end]
        #     seg_name = f"number_{number}_ant_{antenna_idx}_segment_{seg_idx:02d}.npy"
        #     np.save(os.path.join(seg_dir, seg_name), segment)
            
        # seg_root = r"E:\\high_DiskSpy\\signalSeg\\disk_8\\db"
        # for seg_idx, (vib_pattern, _, seg_start, seg_end) in enumerate(time_segs, start=1):
        #     seg_dir = os.path.join(seg_root, str(vib_pattern))
        #     os.makedirs(seg_dir, exist_ok=True)
        #     segment = db_remove_bg[:,seg_start:seg_end]
        #     seg_name = f"number_{number}_ant_{antenna_idx}_segment_{seg_idx:02d}.npy"
        #     np.save(os.path.join(seg_dir, seg_name), segment)
        
        # img_path = f"E:\\high_DiskSpy\\img\\disk_17_75ms_50ms_append\\amp\\"
        # img_name = f"number_{number}_ant_{antenna_idx}"
        # full_path=save_image(t, freqs, amp, start, end, img_path, img_name)
        # cut_image(full_path, start, end, time_segs, img_path, img_name)
        
        img_path = f"E:\\high_DiskSpy\\img\\disk_9_75ms_50ms\\db\\"
        img_name = f"number_{number}_sample_{sample_idx}_ant_{antenna_idx}"
        full_path=save_image(t, freqs, db_remove_bg, start, end, img_path, img_name)
        cut_image(full_path, start, end, time_segs, img_path, img_name)