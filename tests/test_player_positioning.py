"""Tests for football_ml.player_positioning and the shared outfielder frame."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from football_ml.formations import build_outfielder_frame
from football_ml.player_positioning import zone_index
from football_ml.player_positioning import (
    zone_entropy,
    with_phase,
    player_interval_features,
    assign_players_to_lines,
    rollup_player_features,
    N_ZONES,
    enrich_formations,
    write_player_positioning,
)


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


def test_zone_index_corners_and_center():
    L, W = 105.0, 68.0
    # own goal line (depth = -L/2) -> band 0; opponent goal (+L/2) -> band 7
    assert zone_index(-L / 2, 0.0, L, W)[0] == 0
    assert zone_index(+L / 2 - 1e-6, 0.0, L, W)[0] == 7
    # own-left touchline (lateral = -W/2) -> lane 0; own-right (+W/2) -> lane 3
    assert zone_index(0.0, -W / 2, L, W)[1] == 0
    assert zone_index(0.0, +W / 2 - 1e-6, L, W)[1] == 3
    # zone = band*4 + lane
    band, lane, zone = zone_index(+L / 2 - 1e-6, +W / 2 - 1e-6, L, W)
    assert zone == band * 4 + lane == 31


def test_zone_index_clips_out_of_bounds():
    L, W = 105.0, 68.0
    # far beyond both lines -> clipped to edge zones, never out of range
    assert zone_index(-999.0, -999.0, L, W) == (0, 0, 0)
    assert zone_index(+999.0, +999.0, L, W) == (7, 3, 31)


def test_zone_index_half_mirror_symmetry():
    # same physical point for a team attacking the opposite way in P2 maps
    # to the same zone, because attack_sign flips both depth and lateral.
    L, W = 105.0, 68.0
    p1 = zone_index(20.0, 15.0, L, W)       # depth/lateral already normalized
    p2 = zone_index(20.0, 15.0, L, W)
    assert p1 == p2


def test_zone_entropy_fixed_and_uniform():
    fixed = np.zeros(N_ZONES); fixed[5] = 1.0
    assert zone_entropy(fixed) == pytest.approx(0.0)
    four = np.zeros(N_ZONES); four[:4] = 0.25
    assert zone_entropy(four) == pytest.approx(np.log(4) / np.log(N_ZONES))


def test_with_phase_drops_loose_ball_and_labels():
    frame = pd.DataFrame({
        "player_id": [1, 1, 1],
        "is_home": [True, True, True],
        "possession": ["home team", "away team", None],
    })
    out = with_phase(frame)
    assert list(out["phase"]) == ["in", "out"]  # None row dropped


def _feature_frame():
    # player 1 (home) sits in one zone for 3 in-possession frames;
    # player 2 (home) has only 1 in-possession frame (below min_frames=2).
    return pd.DataFrame({
        "player_id": [1, 1, 1, 2],
        "team_id": [10, 10, 10, 10],
        "is_home": [True, True, True, True],
        "interval": [0, 0, 0, 0],
        "depth": [20.0, 20.0, 20.0, -30.0],
        "lateral": [10.0, 10.0, 10.0, 0.0],
        "possession": ["home team", "home team", "home team", "home team"],
    })


def test_player_interval_features_occupancy_and_drop():
    out = player_interval_features(_feature_frame(), 105.0, 68.0, min_frames=2)
    # player 2 has 1 frame < min_frames -> dropped
    assert set(out["player_id"]) == {1}
    row = out.iloc[0]
    assert row["n_frames"] == 3
    assert row["phase"] == "in"
    # occupancy sums to 1 and is concentrated in a single zone
    zcols = [c for c in out.columns if c.startswith("z_b")]
    assert len(zcols) == N_ZONES
    assert row[zcols].sum() == pytest.approx(1.0)
    assert row[zcols].max() == pytest.approx(1.0)
    assert row["std_depth"] == pytest.approx(0.0)
    assert row["ellipse_area"] == pytest.approx(0.0)
    assert row["zone_entropy"] == pytest.approx(0.0)


def test_player_interval_features_no_rows_when_all_loose_ball():
    frame = _feature_frame().assign(possession=None)
    out = player_interval_features(frame, 105.0, 68.0, min_frames=1)
    assert out.empty


def _lines_frame(player_depths, extra=None):
    # one in-possession frame per listed (player_id, depth); all home team.
    rows = []
    for frame_no, snapshot in enumerate([player_depths] + (extra or [])):
        for pid, d in snapshot.items():
            rows.append({
                "player_id": pid, "team_id": 10, "is_home": True,
                "interval": 0, "depth": d, "lateral": 0.0,
                "possession": "home team",
            })
    return pd.DataFrame(rows)


def test_assign_players_442_by_mean_depth():
    # 10 players: 4 deep, 3 mid, 3 advanced
    depths = {1: 0, 2: 1, 3: -1, 4: 0.5, 5: 20, 6: 19, 7: 21, 8: 40, 9: 41, 10: 39}
    out = assign_players_to_lines(_lines_frame(depths), min_frames=1)
    lines = out[(0, True, "in")]
    assert [ln["size"] for ln in lines] == [4, 3, 3]
    assert set(lines[0]["player_ids"]) == {1, 2, 3, 4}  # deepest line


def test_assign_red_card_sums_to_nine():
    depths = {1: 0, 2: 1, 3: -1, 4: 0.5, 5: 20, 6: 19, 7: 21, 8: 40, 9: 41}
    out = assign_players_to_lines(_lines_frame(depths), min_frames=1)
    assert sum(ln["size"] for ln in out[(0, True, "in")]) == 9


def test_assign_sub_keeps_ten_most_present():
    # 10 regulars present in 2 frames, 1 sub present in only 1 frame -> dropped
    base = {i: float(i) for i in range(1, 11)}
    out = assign_players_to_lines(
        _lines_frame(base, extra=[{**base, 11: 100.0}]), min_frames=2
    )
    assigned = {pid for ln in out[(0, True, "in")] for pid in ln["player_ids"]}
    assert 11 not in assigned
    assert sum(ln["size"] for ln in out[(0, True, "in")]) == 10


def test_assign_small_group_skipped_no_crash():
    # only 2 players reach min_frames for the phase -> cannot form >=3 lines;
    # the group is skipped (no key) and no exception is raised.
    depths = {1: 5.0, 2: 20.0}
    out = assign_players_to_lines(_lines_frame(depths), min_frames=1)
    assert (0, True, "in") not in out
    assert out == {}


def test_rollup_frame_weighted_occupancy_and_all_phase():
    zcols = [f"z_b{b}_l{l}" for b in range(8) for l in range(4)]
    def make(pid, interval, phase, n, mean_d, z):
        occ = np.zeros(32); occ[z] = 1.0
        base = {
            "player_id": pid, "team_id": 10, "is_home": True,
            "interval": interval, "phase": phase, "n_frames": n,
            "mean_depth": mean_d, "mean_lateral": 0.0,
            "std_depth": 0.0, "std_lateral": 0.0,
            "ellipse_area": 0.0, "zone_entropy": 0.0,
        }
        base.update(dict(zip(zcols, occ)))
        return base
    # player 1, in-possession: interval 0 (90 frames, zone 5), interval 5 (10 frames, zone 6)
    feats = pd.DataFrame([
        make(1, 0, "in", 90, 20.0, 5),
        make(1, 5, "in", 10, 20.0, 6),
    ])
    out = rollup_player_features(feats)
    phases = set(out["phase"])
    assert phases == {"in", "all"}  # no out-possession rows present
    in_row = out[(out["player_id"] == 1) & (out["phase"] == "in")].iloc[0]
    # frame-weighted: zone 5 -> 0.9, zone 6 -> 0.1
    assert in_row["z_b1_l1"] == pytest.approx(0.9)  # zone 5 = band1 lane1
    assert in_row["z_b1_l2"] == pytest.approx(0.1)  # zone 6 = band1 lane2
    assert in_row["n_frames"] == 100


def test_enrich_formations_adds_named_lines_and_nulls():
    result = {
        "match_id": "T1",
        "intervals": {
            "0": {
                "home": {"in_possession": "4-3-3", "out_of_possession": None},
                "away": {"in_possession": None, "out_of_possession": "4-4-2"},
            }
        },
    }
    assignment = {
        (0, True, "in"): [
            {"depth_rank": 0, "size": 1, "player_ids": [1], "mean_depths": [10.0]},
        ],
    }
    name_map = {1: "J. Smith"}
    out = enrich_formations(result, assignment, name_map)
    cell = out["intervals"]["0"]["home"]
    assert cell["in_possession_lines"][0]["players"] == [
        {"player_id": 1, "name": "J. Smith", "mean_depth": 10.0}
    ]
    # mode string was None -> lines null
    assert cell["out_of_possession_lines"] is None
    # no assignment entry for away/in -> lines null (string was None anyway)
    assert out["intervals"]["0"]["away"]["in_possession_lines"] is None


def test_write_player_positioning_roundtrip(tmp_path):
    df = pd.DataFrame({"player_id": [1], "phase": ["in"], "n_frames": [10]})
    roll = pd.DataFrame({"player_id": [1], "phase": ["all"], "n_frames": [10]})
    p1, p2 = write_player_positioning(df, roll, "T1", tmp_path)
    assert p1.exists() and p2.exists()
    assert pd.read_parquet(p1).shape[0] == 1
    assert pd.read_parquet(p2).iloc[0]["phase"] == "all"


@pytest.mark.skipif(
    not (Path("data/raw/fct_players_tracking.parquet").exists()),
    reason="raw match parquet not available",
)
def test_end_to_end_smoke_1886347():
    from football_ml.jobs import player_positioning_job
    out = player_positioning_job.run("1886347")
    interval = pd.read_parquet(out["interval"])
    assert not interval.empty
    zcols = [c for c in interval.columns if c.startswith("z_b")]
    assert len(zcols) == 32
    # every occupancy row sums to ~1
    assert np.allclose(interval[zcols].sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    # names resolved
    assert interval["player_name"].notna().any()
    import json
    result = json.loads(Path(out["json"]).read_text())
    # at least one cell has player-annotated lines summing to its formation digits
    found = False
    for cell in result["intervals"].values():
        for team in ("home", "away"):
            mode = cell[team].get("in_possession")
            lines = cell[team].get("in_possession_lines")
            if mode and lines:
                assert sum(ln["size"] for ln in lines) == sum(int(d) for d in mode.split("-"))
                found = True
    assert found
