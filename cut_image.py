import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

def save_image(t, freqs, amp, t_start, t_end, img_path, img_name):
    t_seg = t[t_start:t_end]
    amp_seg = amp[:, t_start:t_end]

    save_dir = f"{img_path}\\all\\"
    os.makedirs(save_dir, exist_ok=True)
    full_path = os.path.join(save_dir, f"{img_name}.png")

    fig, ax = plt.subplots(figsize=(12, 1))
    cf = ax.contourf(t_seg, freqs, amp_seg, cmap='jet')

    # 去除坐标轴、边距和colorbar
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # 保存纯净时频谱（像素与时间线性对应）
    plt.savefig(full_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return full_path

# 时间→像素映射
def time_to_pixel(t_value,t_start, t_end, width):
    return int((t_value - t_start) / (t_end - t_start) * width)

def cut_image(full_path, t_start, t_end, time_segs, img_path, img_name):

    # 读取图片
    img = Image.open(full_path)
    width, height = img.size

    segment_paths = []
    for i, (vib_pattern, bit_value, seg_start, seg_end) in enumerate(time_segs):
        left = time_to_pixel(seg_start, t_start=t_start, t_end=t_end, width=width)
        right = time_to_pixel(seg_end, t_start=t_start, t_end=t_end, width=width)
        box = (left, 0, right, height)
        cropped = img.crop(box)
        
        save_dir = f"{img_path}\\cut\\{vib_pattern}\\"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{img_name}_bit_{bit_value}_segment_{i+1:02d}.png")
        cropped.save(save_path)
        segment_paths.append(save_path)
        
