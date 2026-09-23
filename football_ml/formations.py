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


def ball_frame(
    seconds_since_midnight: float, period: float, p1_start: int, p2_start: int
) -> int:
    """Derive the player-tracking frame for a ball-tracking timestamp.

    Parameters
    ----------
    seconds_since_midnight : float
        Ball timestamp's elapsed seconds since 00:00:00 of its date.
    period : float
        Match period (1.0 or 2.0).
    p1_start, p2_start : int
        Player-tracking start frame of each period.

    Returns
    -------
    int
        The corresponding player-tracking frame.
    """
    half = seconds_since_midnight if period == 1 else seconds_since_midnight - 45 * 60
    start = p1_start if period == 1 else p2_start
    return start + int(round(half * 10))


def interval_key(
    frame: int, period: float, p1_start: int, p2_start: int, interval_min: int
) -> int:
    """Bucket a frame into its interval-start-minute key.

    First-half stoppage (>= 45 min) folds into the last regular first-half bucket
    to avoid colliding with the second half, which starts at 45.
    """
    if period == 1:
        minute = (frame - p1_start) / 600.0
        raw = int(minute // interval_min) * interval_min
        return min(raw, 45 - interval_min)
    minute = (frame - p2_start) / 600.0
    return 45 + int(minute // interval_min) * interval_min


def mode_label(labels: Sequence[str], min_frames: int) -> str | None:
    """Most common label, or None if fewer than ``min_frames`` valid labels.

    Parameters
    ----------
    labels : Sequence[str]
        Per-frame formation labels for one interval/team/possession phase.
    min_frames : int
        Minimum valid labels required to emit a formation.

    Returns
    -------
    str | None
    """
    valid = [label for label in labels if label is not None]
    if len(valid) < min_frames:
        return None
    return Counter(valid).most_common(1)[0][0]


def settled_open_play_frames(
    ball: pd.DataFrame,
    p1_start: int,
    p2_start: int,
    *,
    z_max: float = 2.0,
    settle_seconds: float = 5.0,
    half_length: float = 52.5,
    half_width: float = 34.0,
) -> set[int]:
    """Frames of settled open play, inferred from ball tracking.

    A frame is *in play* when the ball is in bounds, grounded, detected, and
    possessed. Since the data has no restart/event labels, this stands in for
    excluding throw-ins, corners, goal kicks, aerial deliveries, and dead/loose
    ball. In addition, the first ``settle_seconds`` (at 10 fps) after every
    return from a not-in-play stretch are dropped, so players have time to
    resettle into shape before a frame counts.

    Parameters
    ----------
    ball : pd.DataFrame
        Ball tracking for one match (timestamp, period, ball_x/y/z, is_detected,
        possession_team_group).
    p1_start, p2_start : int
        Player-tracking start frame of each period.
    z_max : float
        Maximum ball height (m) for a frame to count as grounded.
    settle_seconds : float
        Seconds to drop after each return to play.
    half_length, half_width : float
        Half pitch length/width (m); the ball is out when it crosses these.

    Returns
    -------
    set[int]
        Player-tracking frames to keep.
    """
    b = ball.copy()
    secs = (b["timestamp"] - b["timestamp"].dt.normalize()).dt.total_seconds()
    b["frame"] = [ball_frame(s, p, p1_start, p2_start) for s, p in zip(secs, b["period"])]
    b["in_play"] = (
        (b["ball_x"].abs() <= half_length)
        & (b["ball_y"].abs() <= half_width)
        & (b["ball_z"] <= z_max)
        & b["is_detected"].astype(bool)
        & b["possession_team_group"].notna()
    ).to_numpy()

    settle_frames = int(round(settle_seconds * 10))
    kept: set[int] = set()
    for _, g in b.groupby("period", sort=False):
        g = g.sort_values("frame")
        run = 0  # consecutive in-play frames
        for frame, in_play in zip(g["frame"], g["in_play"]):
            if in_play:
                run += 1
                if run > settle_frames:
                    kept.add(int(frame))
            else:
                run = 0
    return kept


def extract_match_formations(
    tracking: pd.DataFrame,
    ball: pd.DataFrame,
    roster: pd.DataFrame,
    positions: pd.DataFrame,
    match_row: pd.Series,
    *,
    tolerance_m: float = 6.0,
    interval_min: int = 5,
    min_frames: int = 200,
    z_max: float = 2.0,
    settle_seconds: float = 5.0,
) -> dict:
    """Extract per-interval attacking/defensive formations for one match.

    See the plan/spec for the shape of each input frame and the returned dict.
    """
    home_id = int(match_row["home_team.id"])
    away_id = int(match_row["away_team.id"])
    hts = match_row["home_team_side"]
    p1s = int(match_row["match_period_1st_start_frame"])
    p2s = int(match_row["match_period_2nd_start_frame"])

    roster = roster.merge(
        positions[["player_role_id", "player_role_name"]], on="player_role_id", how="left"
    )
    role_map = roster.set_index("player_id")["player_role_name"].to_dict()
    team_map = roster.set_index("player_id")["team_id"].to_dict()

    t = tracking.copy()
    t["player_role_name"] = t["player_id"].map(role_map)
    t["team_id"] = t["player_id"].map(team_map)
    # Drop tracking players missing from the roster: an unmatched player_id maps
    # to team_id NaN, which is not the GK and would otherwise default to the
    # away bucket below (NaN == home_id is False), silently corrupting that
    # frame's player count.
    t = t[t["team_id"].notna()]
    t = t[t["player_role_name"] != GK_ROLE_NAME]
    t["is_home"] = t["team_id"] == home_id
    t["depth"] = [
        attack_sign(h, p, hts) * x
        for h, p, x in zip(t["is_home"], t["period"], t["player_x"])
    ]
    t["interval"] = [
        interval_key(f, p, p1s, p2s, interval_min)
        for f, p in zip(t["frame"], t["period"])
    ]

    # Assumes each match's ball timestamps share one calendar day; a kickoff
    # spanning midnight would need a different anchor than day-normalize.
    secs = (ball["timestamp"] - ball["timestamp"].dt.normalize()).dt.total_seconds()
    bframes = [ball_frame(s, p, p1s, p2s) for s, p in zip(secs, ball["period"])]
    poss = pd.Series(ball["possession_team_group"].to_numpy(), index=bframes)
    poss = poss[~poss.index.duplicated()]
    t["possession"] = t["frame"].map(poss)

    # Keep only settled open-play frames (drops restarts, aerial deliveries,
    # dead/loose ball, and a resettle buffer) so formation reflects real shape.
    kept = settled_open_play_frames(
        ball, p1s, p2s, z_max=z_max, settle_seconds=settle_seconds
    )
    t = t[t["frame"].isin(kept)]

    records = []
    # POC-scoped: per-frame Python loop over ~87k team-frames; fine for this
    # 10-match POC but would need vectorization for a full-season run.
    for (interval, is_home), g in t.groupby(["interval", "is_home"], sort=False):
        for frame, gf in g.groupby("frame", sort=False):
            depths = gf["depth"].to_numpy()
            if len(depths) < 9:  # need a full-ish outfield; skip tracking gaps
                continue
            label = classify_formation(depths, tolerance_m)
            poss_val = gf["possession"].iloc[0]
            own = "home team" if is_home else "away team"
            opp = "away team" if is_home else "home team"
            records.append(
                {
                    "interval": interval,
                    "is_home": bool(is_home),
                    "in_poss": poss_val == own,
                    "out_poss": poss_val == opp,
                    "label": label,
                }
            )

    df = pd.DataFrame(records, columns=["interval", "is_home", "in_poss", "out_poss", "label"])
    intervals: dict[str, dict] = {}
    for interval in sorted(df["interval"].unique()) if not df.empty else []:
        cell = {}
        for is_home, key in [(True, "home"), (False, "away")]:
            sub = df[(df["interval"] == interval) & (df["is_home"] == is_home)]
            cell[key] = {
                "in_possession": mode_label(sub[sub["in_poss"]]["label"].tolist(), min_frames),
                "out_of_possession": mode_label(sub[sub["out_poss"]]["label"].tolist(), min_frames),
            }
        intervals[str(int(interval))] = cell

    return {
        "match_id": str(match_row["match_id"]),
        "params": {
            "interval_min": interval_min,
            "tolerance_m": tolerance_m,
            "min_frames": min_frames,
            "z_max": z_max,
            "settle_seconds": settle_seconds,
        },
        "teams": {"home": home_id, "away": away_id},
        "intervals": intervals,
    }


def write_formations_json(result: dict, out_path: Path) -> None:
    """Write a formations result dict to ``out_path`` as indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
