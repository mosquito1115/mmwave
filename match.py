import numpy as np


import numpy as np

def detect_vibration_start_energy(
    x, template, n_hop=1, eps=1e-12, energy_ratio_thresh=1.0, use_trend=False
):
    """
    基于归一化互相关 + 能量约束的振动起点检测。
    要求匹配到的区域信号能量均值 > 全局能量均值 * energy_ratio_thresh。

    参数：
    ----------
    x : np.ndarray
        实际信号（1D）
    template : np.ndarray
        模板信号（1D）
    n_hop : int
        滑动步长（采样点数）
    energy_ratio_thresh : float
        能量阈值比例，默认1.0，表示匹配段平均能量需大于全局平均能量
    use_trend : bool
        是否使用趋势匹配（差分信号）
    eps : float
        防止除零的小数
    
    返回：
    ----------
    start_idx : int
        检测到的振动起点索引
    corr : np.ndarray
        全部相关值序列
    hop_idxs : np.ndarray
        对应的滑动起点索引数组
    """

    x = np.asarray(x, dtype=float)
    t = np.asarray(template, dtype=float)

    if use_trend:
        x = np.diff(x)
        t = np.diff(t)

    N, M = len(x), len(t)
    if M > N:
        raise ValueError("Template longer than signal")

    # === 计算全局平均能量 ===
    global_energy = np.mean(x**2)

    # === 预计算局部统计量（O(N)) ===
    csum = np.cumsum(np.concatenate(([0.0], x)))
    csum2 = np.cumsum(np.concatenate(([0.0], x * x)))
    win_sum = csum[M:] - csum[:-M]
    win_sum2 = csum2[M:] - csum2[:-M]
    win_mean = win_sum / M
    win_var = np.maximum(win_sum2 / M - win_mean**2, 0.0)
    win_std = np.sqrt(win_var) + eps

    hop_idxs = np.arange(0, N - M + 1, n_hop, dtype=int)
    corr = np.empty(len(hop_idxs), dtype=float)

    # === 滑动匹配 ===
    for i, k in enumerate(hop_idxs):
        seg = x[k:k+M]
        seg_energy = np.mean(seg**2)
        if seg_energy >= energy_ratio_thresh * global_energy:
            z = (seg - win_mean[k]) / win_std[k]
            corr[i] = np.dot(z, t)
        else:
            corr[i] = 0.0
    
    best_i = np.argmax(corr)
    start_idx = hop_idxs[best_i]

    return start_idx, corr, hop_idxs


def detect_vibration_start(x, template, n_hop=1, eps=1e-12):
    """
    基于归一化互相关的模板匹配算法，检测振动起始采样点。
    x: 实际信号 (1D)
    template: 理想模板 (1D)
    n_hop: 滑动步长（采样点数）
    eps: 防止除零的小数
    
    返回：
        start_idx: 振动起始采样点索引
        corr: 全部相关值序列
        hop_idxs: 对应的滑动起点索引数组
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(template, dtype=float)
    N = len(x)
    M = len(t)
    if M > N:
        raise ValueError("Template longer than signal")

    # 预计算滑动均值和方差（高效）
    csum = np.cumsum(np.concatenate(([0.0], x)))
    csum2 = np.cumsum(np.concatenate(([0.0], x * x)))
    win_sum = csum[M:] - csum[:-M]
    win_sum2 = csum2[M:] - csum2[:-M]
    win_mean = win_sum / M
    win_var = np.maximum(win_sum2 / M - win_mean * win_mean, 0.0)
    win_std = np.sqrt(win_var) + eps

    # 滑动匹配（每 n_hop 步）
    hop_idxs = np.arange(0, N - M + 1, n_hop, dtype=int)
    corr = np.empty(len(hop_idxs), dtype=float)
    for i, k in enumerate(hop_idxs):
        seg = x[k:k+M]
        z = (seg - win_mean[k]) / win_std[k]
        corr[i] = np.dot(z, t)
    
    # 找最大值对应的起点
    best_i = np.argmax(corr)
    start_idx = hop_idxs[best_i]
    return start_idx, corr, hop_idxs

def detect_vibration_start_trend(x, template, n_hop=1, eps=1e-12, diff_order=1):
    """
    基于趋势（差分）匹配的振动起点检测。
    与幅值无关，关注波形的变化方向。
    
    x: 实际信号 (1D)
    template: 模板信号 (1D)
    n_hop: 滑动步长（采样点数）
    diff_order: 差分阶数（一般为1）
    eps: 防止除零的小数
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(template, dtype=float)
    N = len(x)
    M = len(t)
    if M > N:
        raise ValueError("Template longer than signal")

    # 计算差分（趋势）
    dx = np.diff(x, n=diff_order)
    dt = np.diff(t, n=diff_order)
    M2 = len(dt)
    N2 = len(dx)
    if M2 > N2:
        raise ValueError("Template trend longer than signal trend")

    # 模板标准化
    dt = dt - dt.mean()
    dt = dt / (np.linalg.norm(dt) + eps)

    # 预计算滑动窗口均值/方差
    csum = np.cumsum(np.concatenate(([0.0], dx)))
    csum2 = np.cumsum(np.concatenate(([0.0], dx * dx)))
    win_sum = csum[M2:] - csum[:-M2]
    win_sum2 = csum2[M2:] - csum2[:-M2]
    win_mean = win_sum / M2
    win_var = np.maximum(win_sum2 / M2 - win_mean * win_mean, 0.0)
    win_std = np.sqrt(win_var) + eps

    hop_idxs = np.arange(0, N2 - M2 + 1, n_hop, dtype=int)
    corr = np.empty(len(hop_idxs), dtype=float)

    for i, k in enumerate(hop_idxs):
        seg = dx[k:k + M2]
        z = (seg - win_mean[k]) / win_std[k]
        corr[i] = np.dot(z, dt)

    best_i = np.argmax(corr)
    start_idx = hop_idxs[best_i] + diff_order  # 对齐原信号
    return start_idx, corr, hop_idxs
