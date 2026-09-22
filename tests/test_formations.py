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
