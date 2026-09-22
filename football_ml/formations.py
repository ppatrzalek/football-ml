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
