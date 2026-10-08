"""Tests for football_ml.player_positioning and the shared outfielder frame."""
import numpy as np
import pandas as pd
import pytest

from football_ml.formations import build_outfielder_frame


def _match_row():
    return pd.Series({
        "match_id": "T1",
        "home_team.id": 10,
        "away_team.id": 20,
        "home_team_side": "left_to_right",
        "match_period_1st_start_frame": 0,
        "match_period_2nd_start_frame": 1000,
    })


def _inputs():
    # 2 outfielders (ids 1,2 home; 3 away) + 1 GK (id 9 home) at frame 5, period 1
    tracking = pd.DataFrame({
        "player_id": [1, 2, 3, 9],
        "player_x": [-40.0, 0.0, 30.0, -50.0],
        "player_y": [10.0, -5.0, 0.0, 0.0],
        "period": [1.0, 1.0, 1.0, 1.0],
        "frame": [5, 5, 5, 5],
        "match_id": ["T1", "T1", "T1", "T1"],
    })
    ball = pd.DataFrame({
        "timestamp": pd.to_datetime(["2020-01-01 00:00:00.5"]),
        "period": [1.0],
        "possession_team_group": ["home team"],
        "match_id": ["T1"],
    })
    roster = pd.DataFrame({
        "player_id": [1, 2, 3, 9],
        "team_id": [10, 10, 20, 10],
        "player_role_id": [4, 5, 4, 0],
        "match_id": ["T1"] * 4,
    })
    positions = pd.DataFrame({
        "player_role_id": [0, 4, 5],
        "player_role_name": ["Goalkeeper", "Defender", "Midfielder"],
    })
    return tracking, ball, roster, positions


def test_build_outfielder_frame_drops_gk_and_adds_depth_lateral():
    tracking, ball, roster, positions = _inputs()
    out = build_outfielder_frame(tracking, ball, roster, positions, _match_row())
    # GK (id 9) dropped; 3 outfielders remain
    assert set(out["player_id"]) == {1, 2, 3}
    assert {"depth", "lateral", "interval", "possession"}.issubset(out.columns)
    # home_team_side left_to_right => home sign +1 in P1: depth == player_x, lateral == player_y
    row1 = out[out["player_id"] == 1].iloc[0]
    assert row1["depth"] == pytest.approx(-40.0)
    assert row1["lateral"] == pytest.approx(10.0)
    assert row1["is_home"] == True  # noqa: E712
    # away team (id 3) sign -1 in P1: depth == -player_x
    row3 = out[out["player_id"] == 3].iloc[0]
    assert row3["depth"] == pytest.approx(-30.0)
