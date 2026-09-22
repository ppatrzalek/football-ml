"""Formation extraction from SkillCorner tracking data."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

GK_ROLE_NAME = "Goalkeeper"


def attack_sign(is_home: bool, period: float, home_team_side: str) -> int:
    """Sign to apply to ``player_x`` so larger ``sign * player_x`` = more advanced.

    Parameters
    ----------
    is_home : bool
        Whether the player's team is the home team.
    period : float
        Match period (1.0 or 2.0).
    home_team_side : str
        Home team's period-1 attacking side ("right_to_left" or "left_to_right").

    Returns
    -------
    int
        +1 or -1. Depth is ``attack_sign(...) * player_x``.
    """
    home_sign_p1 = -1 if home_team_side == "right_to_left" else 1
    if period == 1:
        return home_sign_p1 if is_home else -home_sign_p1
    return -home_sign_p1 if is_home else home_sign_p1


def _greedy_lines(depths_sorted: np.ndarray, tolerance_m: float) -> list[list[float]]:
    """Greedily group ascending depths into lines within ``tolerance_m`` of the running mean."""
    lines: list[list[float]] = []
    for d in depths_sorted:
        if lines and (d - float(np.mean(lines[-1]))) <= tolerance_m:
            lines[-1].append(float(d))
        else:
            lines.append([float(d)])
    return lines


def _merge_closest(lines: list[list[float]]) -> list[list[float]]:
    """Merge the adjacent pair of lines with the smallest gap between their means."""
    means = [float(np.mean(line)) for line in lines]
    gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    i = int(np.argmin(gaps))
    return lines[:i] + [lines[i] + lines[i + 1]] + lines[i + 2:]


def _split_widest(lines: list[list[float]]) -> list[list[float]]:
    """Split the line with the largest internal span at its largest internal gap."""
    best_i, best_span = -1, -1.0
    for i, line in enumerate(lines):
        if len(line) < 2:
            continue
        span = line[-1] - line[0]  # line is ascending
        if span > best_span:
            best_span, best_i = span, i
    line = lines[best_i]
    gaps = [line[k + 1] - line[k] for k in range(len(line) - 1)]
    j = int(np.argmax(gaps))
    return lines[:best_i] + [line[: j + 1], line[j + 1:]] + lines[best_i + 1:]


def _constrain_lines(lines: list[list[float]]) -> list[list[float]]:
    """Force the number of lines into [3, 4]."""
    while len(lines) > 4:
        lines = _merge_closest(lines)
    while len(lines) < 3:
        lines = _split_widest(lines)
    return lines


def classify_formation(depths: Sequence[float], tolerance_m: float) -> str:
    """Classify a set of outfielder depths into a formation string.

    Parameters
    ----------
    depths : Sequence[float]
        Depth (own-goal distance) of each outfielder in a frame (9 or 10 values).
    tolerance_m : float
        Line-keeping tolerance in meters.

    Returns
    -------
    str
        Formation ordered defence->attack, e.g. "4-4-2"; digits sum to len(depths).
    """
    d = np.sort(np.asarray(depths, dtype=float))
    lines = _constrain_lines(_greedy_lines(d, tolerance_m))
    return "-".join(str(len(line)) for line in lines)
