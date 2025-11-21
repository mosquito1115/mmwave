from __future__ import annotations

from typing import List, Tuple, Literal
import numpy as np


SegmentType = Literal['v', 's']

def string_to_pattern(bits: str, vib_sec: float, still_sec: float, compress_runs: bool = False) -> List[Tuple[SegmentType, float]]:
    pattern_str = segments_to_string(bits_to_segments(bits, vib_sec, still_sec))
    pattern = parse_pattern_string(pattern_str)
    return pattern


def bits_to_segments(bits: str, vib_sec: float, still_sec: float, compress_runs: bool = False) -> List[Tuple[SegmentType, float]]:
    """
    Convert bitstring like '1100110011' into [('v', vib_sec), ('s', still_sec), ...].
    When compress_runs=True, consecutive identical bits are merged and their durations multiplied.
    """
    if not isinstance(bits, str):
        raise TypeError('bits must be a string')
    seq = [c for c in bits.strip() if c in ('0', '1')]
    if not seq:
        return []
    if not compress_runs:
        return [('v' if b == '1' else 's', vib_sec if b == '1' else still_sec) for b in seq]
    out: List[Tuple[SegmentType, float]] = []
    cur = seq[0]
    cnt = 1
    for b in seq[1:]:
        if b == cur:
            cnt += 1
        else:
            dur = (vib_sec if cur == '1' else still_sec) * cnt
            out.append(('v' if cur == '1' else 's', float(dur)))
            cur = b
            cnt = 1
    dur = (vib_sec if cur == '1' else still_sec) * cnt
    out.append(('v' if cur == '1' else 's', float(dur)))
    return out

def segments_to_string(segments: List[Tuple[SegmentType, float]]) -> str:
    """Format [('v', 0.075), ('s', 0.05), ...] -> 'v:0.075,s:0.05,...'."""
    return ','.join(f"{k}:{float(d)}" for k, d in segments)

def parse_pattern_string(arg: str) -> List[Tuple[SegmentType, float]]:
    """
    Parse a simple pattern string like: v:0.4,s:0.3,v:0.4,s:0.3.
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

def _build_pattern_template(
    segments: List[Tuple[SegmentType, float]],
    fs: float,
    vib_value: float = 1.0,
    still_value: float = 0.0,
):
    tpl = []
    frames_per_segment: List[int] = []
    for kind, dur_sec in segments:
        n_sample = max(1, int(round(dur_sec * fs)))
        frames_per_segment.append(n_sample)
        val = vib_value if kind == 'v' else still_value
        tpl.extend([val] * n_sample)

    template = np.asarray(tpl, dtype=float)
    eps = 1e-12
    template = template - template.mean()
    norm = np.linalg.norm(template)
    if norm > 0:
        template = template / (norm+eps)
    return template, frames_per_segment