"""Per-player pitch-zone occupancy and spatial-dispersion features.

Consumes the tidy outfielder frame from
:func:`football_ml.formations.build_outfielder_frame`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from football_ml.formations import _constrain_lines, _greedy_lines

N_BANDS = 8
N_LANES = 4
N_ZONES = N_BANDS * N_LANES


def zone_index(
    depth,
    lateral,
    pitch_length: float,
    pitch_width: float,
    *,
    n_bands: int = N_BANDS,
    n_lanes: int = N_LANES,
):
    """Map normalized (depth, lateral) coordinates to pitch-zone indices.

    Parameters
    ----------
    depth, lateral : float or array-like
        Attack-normalized coordinates in meters (0 = pitch center). ``depth``
        runs own-goal(-L/2) -> opponent-goal(+L/2); ``lateral`` runs
        own-left(-W/2) -> own-right(+W/2).
    pitch_length, pitch_width : float
        Pitch dimensions (L, W) in meters.
    n_bands, n_lanes : int
        Grid resolution along length and width.

    Returns
    -------
    tuple
        ``(band, lane, zone)``. Scalars in -> Python ints out; arrays in ->
        numpy int arrays out. ``zone = band * n_lanes + lane``. Out-of-pitch
        coordinates are clipped into the edge zones.
    """
    depth = np.asarray(depth, dtype=float)
    lateral = np.asarray(lateral, dtype=float)
    band = np.floor((depth + pitch_length / 2) / (pitch_length / n_bands))
    lane = np.floor((lateral + pitch_width / 2) / (pitch_width / n_lanes))
    band = np.clip(band, 0, n_bands - 1).astype(int)
    lane = np.clip(lane, 0, n_lanes - 1).astype(int)
    zone = band * n_lanes + lane
    if band.ndim == 0:
        return int(band), int(lane), int(zone)
    return band, lane, zone


def zone_entropy(fracs) -> float:
    """Normalized Shannon entropy of an occupancy vector (0 = fixed, 1 = uniform)."""
    p = np.asarray(fracs, dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(len(fracs)))


def with_phase(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a ``phase`` column (``in``/``out``), dropping loose-ball rows.

    A row is ``in`` possession when the ball belongs to the player's own team,
    ``out`` when it belongs to the opponent. Rows with no possession (loose
    ball / ``None``) are dropped so they count toward neither phase.
    """
    f = frame[frame["possession"].notna()].copy()
    own = np.where(f["is_home"], "home team", "away team")
    f["phase"] = np.where(f["possession"].to_numpy() == own, "in", "out")
    return f


def player_interval_features(
    frame: pd.DataFrame,
    pitch_length: float,
    pitch_width: float,
    *,
    min_frames: int = 50,
    n_bands: int = N_BANDS,
    n_lanes: int = N_LANES,
) -> pd.DataFrame:
    """Compute per-(player, interval, phase) zone occupancy and spread features.

    Parameters
    ----------
    frame : pd.DataFrame
        Tidy outfielder frame from :func:`build_outfielder_frame`.
    pitch_length, pitch_width : float
        Pitch dimensions in meters (for zone binning).
    min_frames : int
        Rows with fewer valid frames than this are dropped.
    n_bands, n_lanes : int
        Zone grid resolution.

    Returns
    -------
    pd.DataFrame
        Keys ``player_id, team_id, is_home, interval, phase``; scalars
        ``n_frames, mean_depth, mean_lateral, std_depth, std_lateral,
        ellipse_area, zone_entropy``; and ``n_bands*n_lanes`` occupancy columns
        ``z_b{b}_l{l}`` summing to 1 per row.
    """
    n_zones = n_bands * n_lanes
    zcols = [f"z_b{b}_l{l}" for b in range(n_bands) for l in range(n_lanes)]
    f = with_phase(frame)
    rows = []
    for (player_id, interval, phase), g in f.groupby(
        ["player_id", "interval", "phase"], sort=False
    ):
        n = len(g)
        if n < min_frames:
            continue
        depth = g["depth"].to_numpy()
        lateral = g["lateral"].to_numpy()
        _, _, zone = zone_index(
            depth, lateral, pitch_length, pitch_width,
            n_bands=n_bands, n_lanes=n_lanes,
        )
        occ = np.bincount(zone, minlength=n_zones).astype(float) / n
        std_d = float(np.std(depth))
        std_l = float(np.std(lateral))
        row = {
            "player_id": player_id,
            "team_id": g["team_id"].iloc[0],
            "is_home": bool(g["is_home"].iloc[0]),
            "interval": int(interval),
            "phase": phase,
            "n_frames": n,
            "mean_depth": float(np.mean(depth)),
            "mean_lateral": float(np.mean(lateral)),
            "std_depth": std_d,
            "std_lateral": std_l,
            "ellipse_area": float(np.pi * std_d * std_l),
            "zone_entropy": zone_entropy(occ),
        }
        row.update(dict(zip(zcols, occ)))
        rows.append(row)
    return pd.DataFrame(rows, columns=[
        "player_id", "team_id", "is_home", "interval", "phase", "n_frames",
        "mean_depth", "mean_lateral", "std_depth", "std_lateral",
        "ellipse_area", "zone_entropy", *zcols,
    ])


def assign_players_to_lines(
    frame: pd.DataFrame,
    *,
    tolerance_m: float = 6.0,
    min_frames: int = 50,
) -> dict:
    """Assign on-pitch outfielders to formation lines by interval-average depth.

    For each (interval, team, phase): keep outfielders present in at least
    ``min_frames`` frames (the 10 most-present if a substitution inflates the
    count), cluster their mean depths with the formation line algorithm, and
    emit lines ordered defence->attack with player membership. Groups with
    fewer than 3 qualifying players are skipped (no dict entry), since at least
    3 players are required to form >=3 formation lines.

    Returns
    -------
    dict
        Keyed by ``(interval, is_home, phase)`` -> list of line dicts
        ``{"depth_rank", "size", "player_ids", "mean_depths"}``.
    """
    f = with_phase(frame)
    result: dict = {}
    for (interval, is_home, phase), g in f.groupby(
        ["interval", "is_home", "phase"], sort=False
    ):
        counts = g.groupby("player_id").size()
        means = g.groupby("player_id")["depth"].mean()
        present = counts[counts >= min_frames]
        if present.empty:
            continue
        # keep the 10 most-present (ties broken by higher count then player_id)
        keep = present.sort_values(ascending=False, kind="stable").head(10).index
        if len(keep) < 3:
            continue  # cannot form >=3 formation lines from <3 players
        player_means = means.loc[keep].sort_values()  # ascending depth
        pids = list(player_means.index)
        depths_sorted = player_means.to_numpy()
        lines = _constrain_lines(_greedy_lines(depths_sorted, tolerance_m))
        out_lines = []
        pos = 0
        for rank, line in enumerate(lines):
            size = len(line)
            out_lines.append({
                "depth_rank": rank,
                "size": size,
                "player_ids": pids[pos:pos + size],
                "mean_depths": [float(d) for d in depths_sorted[pos:pos + size]],
            })
            pos += size
        result[(int(interval), bool(is_home), str(phase))] = out_lines
    return result
