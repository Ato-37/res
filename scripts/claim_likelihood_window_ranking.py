"""Outcome-blind fixed-window partition and deterministic selection helpers."""
from __future__ import annotations
import math

def partition_window_indices(frame_count: int, window_count: int = 4) -> tuple[tuple[int, ...], ...]:
    if (type(frame_count) is not int or type(window_count) is not int
            or frame_count <= 0 or window_count <= 1
            or frame_count % window_count):
        raise ValueError("window_partition_dimensions_invalid")
    width = frame_count // window_count
    return tuple(tuple(range(i * width, (i + 1) * width))
                 for i in range(window_count))

def select_max_mean_logprob(scores) -> int:
    if not isinstance(scores, (list, tuple)) or len(scores) != 4:
        raise ValueError("window_score_count_invalid")
    values=[]
    for value in scores:
        if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(float(value)):
            raise ValueError("window_score_nonfinite")
        values.append(float(value))
    # Python max preserves the first position on ties: earliest window wins.
    return max(range(len(values)), key=values.__getitem__)

def selected_interval(sampled_times, window_index: int, window_count: int = 4):
    windows=partition_window_indices(len(sampled_times),window_count)
    if type(window_index) is not int or not 0 <= window_index < len(windows):
        raise ValueError("window_index_invalid")
    indices=windows[window_index]
    times=[]
    for value in sampled_times:
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)):
            raise ValueError("sampled_time_nonfinite")
        times.append(float(value))
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError("sampled_times_not_strictly_increasing")
    return indices,(times[indices[0]],times[indices[-1]])
