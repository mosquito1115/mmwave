def generate_sigs(start_sample, n_sample_vib, n_sample_still, pattern_str):
    """
    根据给定参数生成信号段信息列表。

    参数：
        start_sample: int，起始采样点
        n_sample_vib: int，振动段长度
        n_sample_still: int，静止段长度
        pattern_str: str，由'0'和'1'组成，长度为3的倍数

    返回：
        sigs: list[(vib_pattern, bit_value, seg_start, seg_end)]
    """
    if len(pattern_str) % 3 != 0:
        raise ValueError("pattern_str 的长度必须是 3 的倍数")

    n = len(pattern_str)
    one_third = n // 3
    sigs = []

    cur_start = start_sample

    for idx, bit in enumerate(pattern_str):
        # bit_value 为 0 或 1
        bit_value = int(bit)

        # 根据bit_value和所在区域确定vib_pattern
        if bit_value == 0:
            vib_pattern = 0
            seg_len = n_sample_still
        else:
            if idx < one_third:
                vib_pattern = 1
            elif idx < 2 * one_third:
                vib_pattern = 2
            else:
                vib_pattern = 3
            seg_len = n_sample_vib

        seg_start = cur_start
        seg_end = cur_start + n_sample_vib
        cur_start = cur_start + seg_len  # 更新下一个段的起点

        sigs.append((vib_pattern, bit_value, seg_start, seg_end))

    return sigs


# 示例使用
if __name__ == "__main__":
    start_sample = 15
    n_sample_vibn = 2000
    n_sample_still = 1000
    pattern_str = "110011001100"

    sigs = generate_sigs(start_sample, n_sample_vibn, n_sample_still, pattern_str)
    for s in sigs:
        print(s)
