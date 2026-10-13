"""Entry point: player positioning features + player-annotated formations."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, logger
from football_ml.formations import (
    build_outfielder_frame,
    extract_match_formations,
    write_formations_json,
)
from football_ml.player_positioning import (
    assign_players_to_lines,
    enrich_formations,
    player_interval_features,
    rollup_player_features,
    write_player_positioning,
)

DEFAULT_PITCH_LENGTH = 105.0
DEFAULT_PITCH_WIDTH = 68.0


def _pitch_dims(stadium: pd.DataFrame, home_id: int) -> tuple[float, float]:
    """Look up pitch dimensions for the home team's stadium.

    Parameters
    ----------
    stadium : pd.DataFrame
        The ``dim_stadium`` table with ``team_id``, ``pitch_length_meters``,
        and ``pitch_width_meters`` columns.
    home_id : int
        Home team identifier used to filter the stadium table.

    Returns
    -------
    tuple[float, float]
        ``(pitch_length_meters, pitch_width_meters)``. Falls back to
        105.0 x 68.0 if no row is found or values are NaN.
    """
    row = stadium[stadium["team_id"] == home_id]
    if row.empty:
        return DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH
    r = row.iloc[0]
    length = r["pitch_length_meters"]
    width = r["pitch_width_meters"]
    if pd.isna(length) or pd.isna(width):
        return DEFAULT_PITCH_LENGTH, DEFAULT_PITCH_WIDTH
    return float(length), float(width)


def run(
    match_id: str = "1886347",
    *,
    tolerance_m: float = 6.0,
    interval_min: int = 5,
    min_frames: int = 50,
    formation_min_frames: int = 200,
    z_max: float = 2.0,
    settle_seconds: float = 5.0,
) -> dict:
    """Build player-annotated formations + player positioning tables for a match.

    Two thresholds: ``formation_min_frames`` (200) gates the formation mode
    string inside ``extract_match_formations``; ``min_frames`` (50) gates
    player-positioning feature and assignment rows.

    Parameters
    ----------
    match_id : str
        Match identifier. Defaults to the SkillCorner open-data sample match.
    tolerance_m : float
        Line-keeping tolerance in meters for formation classification.
    interval_min : int
        Interval length in minutes for temporal bucketing.
    min_frames : int
        Minimum valid frames for player-positioning feature/assignment rows.
    formation_min_frames : int
        Minimum valid frames for the formation mode string (passed to
        ``extract_match_formations`` as its ``min_frames`` argument).
    z_max : float
        Maximum ball height (m) to consider a frame as grounded open play.
    settle_seconds : float
        Seconds to drop after each return to open play.

    Returns
    -------
    dict
        ``{"json": Path, "interval": Path, "rollup": Path}`` pointing to the
        three written artifacts.
    """
    match_id = str(match_id)
    tracking = pd.read_parquet(RAW_DATA_DIR / "fct_players_tracking.parquet")
    tracking = tracking[tracking["match_id"].astype(str) == match_id]
    ball = pd.read_parquet(RAW_DATA_DIR / "fct_ball_tracking.parquet")
    ball = ball[ball["match_id"].astype(str) == match_id]
    roster = pd.read_parquet(RAW_DATA_DIR / "fct_match_players.parquet")
    roster = roster[roster["match_id"].astype(str) == match_id]
    positions = pd.read_parquet(RAW_DATA_DIR / "dim_players_position.parquet")
    dim_match = pd.read_parquet(RAW_DATA_DIR / "dim_match.parquet")
    match_row = dim_match[dim_match["match_id"].astype(str) == match_id].iloc[0]
    stadium = pd.read_parquet(RAW_DATA_DIR / "dim_stadium.parquet")
    players = pd.read_parquet(RAW_DATA_DIR / "dim_players.parquet")

    home_id = int(match_row["home_team.id"])
    length, width = _pitch_dims(stadium, home_id)
    name_map = players.set_index("player_id")["player_short_name"].to_dict()

    result = extract_match_formations(
        tracking, ball, roster, positions, match_row,
        tolerance_m=tolerance_m, interval_min=interval_min,
        min_frames=formation_min_frames, z_max=z_max, settle_seconds=settle_seconds,
    )
    frame = build_outfielder_frame(
        tracking, ball, roster, positions, match_row, interval_min=interval_min
    )
    assignment = assign_players_to_lines(
        frame, tolerance_m=tolerance_m, min_frames=min_frames
    )
    result = enrich_formations(result, assignment, name_map)

    interval_df = player_interval_features(frame, length, width, min_frames=min_frames)
    interval_df.insert(0, "match_id", match_id)
    interval_df["player_name"] = interval_df["player_id"].map(name_map)
    rollup_df = rollup_player_features(interval_df)
    rollup_df.insert(0, "match_id", match_id)
    rollup_df["player_name"] = rollup_df["player_id"].map(name_map)

    json_path = PROCESSED_DATA_DIR / "formations" / f"match_{match_id}.json"
    write_formations_json(result, json_path)
    out_dir = PROCESSED_DATA_DIR / "player_positioning"
    p1, p2 = write_player_positioning(interval_df, rollup_df, match_id, out_dir)
    logger.info("Wrote %s, %s, %s", json_path, p1, p2)
    return {"json": json_path, "interval": p1, "rollup": p2}


if __name__ == "__main__":
    run()
