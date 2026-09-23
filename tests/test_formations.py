"""Tests for football_ml.formations."""
import numpy as np

from football_ml.formations import attack_sign


def test_attack_sign_home_right_to_left_period1():
    # home attacks right->left (toward -x) in P1: own goal at +x,
    # so advanced players (-x) must get the LARGER depth => sign is -1
    assert attack_sign(is_home=True, period=1.0, home_team_side="right_to_left") == -1


def test_attack_sign_flips_between_halves():
    hts = "right_to_left"
    assert attack_sign(True, 1.0, hts) == -attack_sign(True, 2.0, hts)


def test_attack_sign_away_is_mirror_of_home():
    hts = "right_to_left"
    assert attack_sign(True, 1.0, hts) == -attack_sign(False, 1.0, hts)
    assert attack_sign(True, 2.0, hts) == -attack_sign(False, 2.0, hts)


def test_attack_sign_left_to_right():
    assert attack_sign(True, 1.0, "left_to_right") == 1


from football_ml.formations import classify_formation


def _sizes(label):
    return [int(n) for n in label.split("-")]


def test_classify_clean_442():
    # 4 defenders ~0m, 4 midfielders ~20m, 2 forwards ~40m
    depths = [0.0, 0.5, -0.5, 0.2, 20.0, 19.5, 20.5, 20.2, 40.0, 40.3]
    assert classify_formation(depths, tolerance_m=6.0) == "4-4-2"


def test_classify_bunched_forces_three_lines():
    # everyone within tolerance -> greedy gives 1 line -> split to 3
    depths = [float(i) * 0.5 for i in range(10)]  # 0.0 .. 4.5
    label = classify_formation(depths, tolerance_m=6.0)
    assert len(_sizes(label)) == 3
    assert sum(_sizes(label)) == 10


def test_classify_oversegmented_merges_to_four_lines():
    # 10 singletons spaced 10m apart -> 10 lines -> merge down to 4
    depths = [float(i) * 10.0 for i in range(10)]
    label = classify_formation(depths, tolerance_m=6.0)
    assert len(_sizes(label)) == 4
    assert sum(_sizes(label)) == 10


def test_classify_red_card_sums_to_nine():
    depths = [0.0, 0.3, -0.2, 0.1, 20.0, 19.7, 20.4, 40.0, 40.2]  # 9 outfielders
    label = classify_formation(depths, tolerance_m=6.0)
    assert sum(_sizes(label)) == 9


def test_classify_allows_four_line_formation():
    # 4 def ~0, 2 dm ~15, 3 am ~30, 1 fwd ~45 -> 4-2-3-1
    depths = [0.0, 0.4, -0.3, 0.2, 15.0, 15.4, 30.0, 30.3, 29.7, 45.0]
    assert classify_formation(depths, tolerance_m=6.0) == "4-2-3-1"


from football_ml.formations import ball_frame, interval_key


def test_ball_frame_period1_anchor():
    assert ball_frame(0.0, 1.0, 10, 27800) == 10
    assert ball_frame(0.1, 1.0, 10, 27800) == 11


def test_ball_frame_period2_anchor():
    # period 2 clock starts at 45:00 -> maps to p2_start
    assert ball_frame(45 * 60, 2.0, 10, 27800) == 27800
    assert ball_frame(45 * 60 + 0.1, 2.0, 10, 27800) == 27801


def test_interval_key_first_half_buckets():
    assert interval_key(10, 1.0, 10, 27800, 5) == 0          # kickoff
    assert interval_key(10 + 600 * 5, 1.0, 10, 27800, 5) == 5  # 5:00
    assert interval_key(10 + 600 * 6, 1.0, 10, 27800, 5) == 5  # 6:00


def test_interval_key_first_half_stoppage_caps_at_40():
    # 46:00 in first half must not collide with second-half 45
    assert interval_key(10 + 600 * 46, 1.0, 10, 27800, 5) == 40


def test_interval_key_second_half_starts_at_45():
    assert interval_key(27800, 2.0, 10, 27800, 5) == 45
    assert interval_key(27800 + 600 * 6, 2.0, 10, 27800, 5) == 50


from football_ml.formations import mode_label


def test_mode_label_returns_most_common():
    labels = ["4-4-2", "4-4-2", "4-5-1", "4-4-2"]
    assert mode_label(labels, min_frames=1) == "4-4-2"


def test_mode_label_none_when_too_few():
    assert mode_label(["4-4-2"], min_frames=50) is None


def test_mode_label_none_when_empty():
    assert mode_label([], min_frames=1) is None


import pandas as pd

from football_ml.formations import extract_match_formations


def _make_tracking(frames, team_players, period=1.0):
    """Build tracking rows placing each player at a fixed x giving a 4-4-2 shape."""
    rows = []
    for frame in frames:
        for player_id, x in team_players:
            rows.append(
                {"player_id": player_id, "player_x": x, "period": period, "frame": frame}
            )
    return pd.DataFrame(rows)


def test_extract_match_formations_structure_and_labels():
    # Home team (id 1): GK + 4-4-2 outfielders. home_team_side left_to_right,
    # so depth = +player_x. Defenders x=0, mid x=20, fwd x=40.
    outfield_x = [0, 0, 0, 0, 20, 20, 20, 20, 40, 40]
    home_players = [(100, 0)]  # GK at x=0, dropped by role
    home_players += [(101 + i, x) for i, x in enumerate(outfield_x)]

    frames = [10, 11, 12]
    tracking = _make_tracking(frames, home_players)

    # ball: home in possession for all three frames (period 1, ~1s in),
    # in-bounds and grounded so they survive the settled-open-play filter.
    ball = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-04 00:00:00.000", "2026-01-04 00:00:00.100", "2026-01-04 00:00:00.200"]
            ),
            "period": [1.0, 1.0, 1.0],
            "ball_x": [0.0, 0.0, 0.0],
            "ball_y": [0.0, 0.0, 0.0],
            "ball_z": [0.0, 0.0, 0.0],
            "is_detected": [True, True, True],
            "possession_team_group": ["home team", "home team", "home team"],
        }
    )

    roster = pd.DataFrame(
        {
            "player_id": [100] + [101 + i for i in range(10)],
            "team_id": [1] * 11,
            "player_role_id": [0] + [2] * 10,  # 0 = GK, 2 = outfield
        }
    )
    positions = pd.DataFrame(
        {"player_role_id": [0, 2], "player_role_name": ["Goalkeeper", "Center Back"]}
    )
    match_row = pd.Series(
        {
            "match_id": "TESTMATCH",
            "home_team.id": 1,
            "away_team.id": 2,
            "home_team_side": "left_to_right",
            "match_period_1st_start_frame": 10,
            "match_period_2nd_start_frame": 27800,
        }
    )

    # settle_seconds=0 so the 3-frame window isn't consumed by the resettle buffer.
    result = extract_match_formations(
        tracking, ball, roster, positions, match_row,
        tolerance_m=6.0, interval_min=5, min_frames=1, settle_seconds=0.0,
    )

    assert result["match_id"] == "TESTMATCH"
    assert result["teams"] == {"home": 1, "away": 2}
    assert result["params"] == {
        "interval_min": 5,
        "tolerance_m": 6.0,
        "min_frames": 1,
        "z_max": 2.0,
        "settle_seconds": 0.0,
    }
    # frames 10-12 are all in interval "0"; home in possession -> 4-4-2 attacking
    assert result["intervals"]["0"]["home"]["in_possession"] == "4-4-2"
    # home never out of possession in this window -> None
    assert result["intervals"]["0"]["home"]["out_of_possession"] is None


def test_extract_excludes_players_missing_from_roster():
    # Home and away both set up as clean 4-4-2s, plus one extra tracking
    # player_id (999) that is NOT present in the roster. Without the fix,
    # that player's team_id is NaN, survives the GK filter, and
    # `is_home = (NaN == home_id)` evaluates False -> it is silently counted
    # into the away team's frame (11 players -> "5-4-2"). The fix must drop
    # it, leaving both formations clean 4-4-2s.
    outfield_x = [0, 0, 0, 0, 20, 20, 20, 20, 40, 40]
    home_players = [(100, 0)] + [(101 + i, x) for i, x in enumerate(outfield_x)]
    away_players = [(200, 0)] + [(201 + i, -x) for i, x in enumerate(outfield_x)]
    stray = [(999, 5)]  # not in roster

    frames = [10, 11, 12]
    tracking = _make_tracking(frames, home_players + away_players + stray)

    ball = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-04 00:00:00.000", "2026-01-04 00:00:00.100", "2026-01-04 00:00:00.200"]
            ),
            "period": [1.0, 1.0, 1.0],
            "ball_x": [0.0, 0.0, 0.0],
            "ball_y": [0.0, 0.0, 0.0],
            "ball_z": [0.0, 0.0, 0.0],
            "is_detected": [True, True, True],
            "possession_team_group": ["home team", "home team", "home team"],
        }
    )

    roster = pd.DataFrame(
        {
            "player_id": [100] + [101 + i for i in range(10)] + [200] + [201 + i for i in range(10)],
            "team_id": [1] * 11 + [2] * 11,
            "player_role_id": ([0] + [2] * 10) * 2,  # 0 = GK, 2 = outfield
        }
    )
    positions = pd.DataFrame(
        {"player_role_id": [0, 2], "player_role_name": ["Goalkeeper", "Center Back"]}
    )
    match_row = pd.Series(
        {
            "match_id": "TESTMATCH",
            "home_team.id": 1,
            "away_team.id": 2,
            "home_team_side": "left_to_right",
            "match_period_1st_start_frame": 10,
            "match_period_2nd_start_frame": 27800,
        }
    )

    result = extract_match_formations(
        tracking, ball, roster, positions, match_row,
        tolerance_m=6.0, interval_min=5, min_frames=1, settle_seconds=0.0,
    )

    assert result["intervals"]["0"]["home"]["in_possession"] == "4-4-2"
    assert result["intervals"]["0"]["away"]["out_of_possession"] == "4-4-2"


import json

from football_ml.formations import write_formations_json


def test_write_formations_json_roundtrip(tmp_path):
    result = {"match_id": "X", "params": {}, "teams": {"home": 1, "away": 2}, "intervals": {}}
    out = tmp_path / "formations" / "match_X.json"
    write_formations_json(result, out)
    assert out.exists()
    assert json.loads(out.read_text()) == result


from football_ml.formations import settled_open_play_frames


def _synthetic_ball(n, p1_start=10, bad=None):
    """Ball rows at 10 fps in period 1; ``bad`` maps frame-index -> exclusion reason.

    Frame derived from the timestamp equals ``p1_start + index``.
    """
    bad = bad or {}
    ts = pd.date_range("2026-01-04 00:00:00", periods=n, freq="100ms")
    cols = {
        "timestamp": ts,
        "period": [1.0] * n,
        "ball_x": [0.0] * n,
        "ball_y": [0.0] * n,
        "ball_z": [0.0] * n,
        "is_detected": [True] * n,
        "possession_team_group": ["home team"] * n,
    }
    for i, reason in bad.items():
        if reason == "oob":
            cols["ball_x"][i] = 60.0
        elif reason == "air":
            cols["ball_z"][i] = 5.0
        elif reason == "dead":
            cols["possession_team_group"][i] = None
        elif reason == "undet":
            cols["is_detected"][i] = False
    return pd.DataFrame(cols)


def test_settled_open_play_buffer_and_out_of_bounds():
    # 60 frames, one out-of-bounds blip at index 30; settle buffer = 2s = 20 frames.
    ball = _synthetic_ball(60, bad={30: "oob"})
    kept = settled_open_play_frames(ball, 10, 27800, z_max=2.0, settle_seconds=2.0)
    assert 10 + 30 not in kept  # the out-of-bounds frame itself
    assert 10 + 29 in kept      # settled, just before the blip (past initial buffer)
    assert 10 + 19 not in kept  # inside the initial 20-frame settle buffer
    assert 10 + 50 not in kept  # index 50 -> run=20, still buffered after the blip
    assert 10 + 51 in kept      # index 51 -> run=21, first kept after the blip
    assert 10 + 55 in kept


def test_settled_open_play_excludes_aerial_dead_and_undetected():
    # No settle buffer, so only the per-frame reasons remove frames.
    ball = _synthetic_ball(40, bad={20: "air", 25: "dead", 30: "undet"})
    kept = settled_open_play_frames(ball, 10, 27800, z_max=2.0, settle_seconds=0.0)
    assert 10 + 20 not in kept  # aerial ball
    assert 10 + 25 not in kept  # loose/dead ball
    assert 10 + 30 not in kept  # ball not detected
    assert 10 + 19 in kept
    assert 10 + 21 in kept      # frame after aerial is settled again


def test_end_to_end_smoke_real_match():
    # Integration: runs the real pipeline on match 1886347 via the job entry point.
    from football_ml.jobs.formation_extraction_job import run

    result = run(match_id="1886347")

    assert result["match_id"] == "1886347"
    assert result["intervals"], "expected at least one interval"
    for cell in result["intervals"].values():
        for team in ("home", "away"):
            for phase in ("in_possession", "out_of_possession"):
                label = cell[team][phase]
                if label is None:
                    continue
                digits = [int(n) for n in label.split("-")]
                assert 3 <= len(digits) <= 4
                assert sum(digits) in (9, 10)
    # Guards the match_id-dtype fix: team ids resolve to non-null ints.
    assert isinstance(result["teams"]["home"], int) and result["teams"]["home"] is not None
    assert isinstance(result["teams"]["away"], int) and result["teams"]["away"] is not None
