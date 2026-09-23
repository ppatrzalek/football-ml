"""Entry point: extract formations for a match and write JSON."""
from __future__ import annotations

import pandas as pd

from football_ml.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, logger
from football_ml.formations import extract_match_formations, write_formations_json


def run(
    match_id: str = "1886347",
    *,
    tolerance_m: float = 6.0,
    interval_min: int = 5,
    min_frames: int = 50,
) -> dict:
    """Load a match, extract formations, and write the JSON output.

    Parameters
    ----------
    match_id : str
        Identifier of the match to process (coerced to str for filtering).
    tolerance_m : float
        Line-keeping tolerance in meters passed to the classifier.
    interval_min : int
        Length in minutes of each aggregation interval.
    min_frames : int
        Minimum valid frames required to emit a formation for an interval.

    Returns
    -------
    dict
        The formations result (also written to disk).
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

    result = extract_match_formations(
        tracking, ball, roster, positions, match_row,
        tolerance_m=tolerance_m, interval_min=interval_min, min_frames=min_frames,
    )
    out_path = PROCESSED_DATA_DIR / "formations" / f"match_{match_id}.json"
    write_formations_json(result, out_path)
    logger.info("Wrote formations to %s", out_path)
    return result


if __name__ == "__main__":
    run()
