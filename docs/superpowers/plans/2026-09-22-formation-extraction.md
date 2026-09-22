# Formation Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract each team's attacking (in-possession) and defensive (out-of-possession) football formation in 5-minute intervals from SkillCorner tracking data, and write it to a per-match JSON file.

**Architecture:** One pure-pandas module `football_ml/formations.py` holding small, independently testable functions (direction normalization, ball-frame alignment, interval bucketing, line clustering, mode aggregation, orchestrator), plus a thin runner in `football_ml/jobs/`. Per frame we normalize each outfielder's depth relative to their own goal, greedily cluster players into 3–4 lines within a configurable tolerance, emit a formation string, then take the modal label per interval × team × possession phase. No star-schema changes; output is JSON under `data/processed/formations/`.

**Tech Stack:** Python 3.10+, pandas ≥2.3, numpy ≥2.2, pytest ≥8.

**Spec:** `docs/superpowers/specs/2026-09-22-formation-extraction-design.md`

## Global Constraints

- **Code style:** type hints on all function signatures; numpy-style docstrings with `Parameters`/`Returns`; pandas for processing; `index=False` when writing tabular data (N/A here — output is JSON).
- **Frame rate:** tracking is 10 fps. A 5-min interval ≈ 3000 frames.
- **Ball↔player alignment:** `fct_ball_tracking` has no `frame` column; it is keyed by `timestamp` on a match clock (period 1 anchored at 00:00:00, period 2 at 00:45:00). Derive frame as `period_start_frame + round(elapsed_seconds_in_half * 10)`. Verified: 43458/43458 ball frames match player frames exactly for match 1886347.
- **Interval keys:** first half → `0,5,…,40` (first-half stoppage time ≥45:00 folds into the `40` bucket to avoid colliding with the second half); second half → `45,50,…`. Keys are strings in the JSON.
- **Goalkeeper:** identified and dropped by `player_role_name == "Goalkeeper"` (role_id 0), not by live position.
- **Defaults:** `tolerance_m=6.0`, `interval_min=5`, `min_frames=50`.
- **POC scope:** match `1886347`. Same code generalizes to all matches by looping.
- **Test location:** `tests/test_formations.py` (module is top-level, mirroring `tests/preprocessors/` convention). Run tests with `pytest`.

---

### Task 1: Direction normalization — `attack_sign`

Converts a raw `player_x` into a sign so that, for any team in any period, larger `sign * player_x` = closer to the opponent goal (more advanced). Makes "deepest line = defenders" comparable across halves and teams.

**Files:**
- Create: `football_ml/formations.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Produces: `attack_sign(is_home: bool, period: float, home_team_side: str) -> int` returning `+1` or `-1`. `depth = attack_sign(...) * player_x`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k attack_sign -v`
Expected: FAIL with `ImportError` / `cannot import name 'attack_sign'`.

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_formations.py -k attack_sign -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py tests/test_formations.py
git commit -m "feat(formations): add attack_sign direction normalization"
```

---

### Task 2: Line clustering — `classify_formation`

The core algorithm: depths → `"4-4-2"`. Greedy 1-D clustering along depth within `tolerance_m`, then constrain to 3–4 lines (merge closest adjacent lines if >4; split widest line if <3).

**Files:**
- Modify: `football_ml/formations.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `classify_formation(depths: Sequence[float], tolerance_m: float) -> str` — returns a hyphen string ordered defence→attack whose digits sum to `len(depths)`.
  - Internal helpers `_greedy_lines`, `_merge_closest`, `_split_widest`, `_constrain_lines` (list-of-lists of sorted depths).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k classify -v`
Expected: FAIL with `cannot import name 'classify_formation'`.

- [ ] **Step 3: Write minimal implementation**

Append to `football_ml/formations.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_formations.py -k classify -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py tests/test_formations.py
git commit -m "feat(formations): add line-clustering classify_formation"
```

---

### Task 3: Ball-frame alignment & interval bucketing

Two clock helpers: derive a player-frame from a ball timestamp's elapsed seconds, and bucket a frame into its 5-min interval key.

**Files:**
- Modify: `football_ml/formations.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Produces:
  - `ball_frame(seconds_since_midnight: float, period: float, p1_start: int, p2_start: int) -> int`
  - `interval_key(frame: int, period: float, p1_start: int, p2_start: int, interval_min: int) -> int`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k "ball_frame or interval_key" -v`
Expected: FAIL with `cannot import name 'ball_frame'`.

- [ ] **Step 3: Write minimal implementation**

Append to `football_ml/formations.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_formations.py -k "ball_frame or interval_key" -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py tests/test_formations.py
git commit -m "feat(formations): add ball_frame alignment and interval bucketing"
```

---

### Task 4: Modal aggregation — `mode_label`

Reduce a window's per-frame labels to the single most common one, or `None` when there are too few frames.

**Files:**
- Modify: `football_ml/formations.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Produces: `mode_label(labels: Sequence[str], min_frames: int) -> str | None`

- [ ] **Step 1: Write the failing test**

```python
from football_ml.formations import mode_label


def test_mode_label_returns_most_common():
    labels = ["4-4-2", "4-4-2", "4-5-1", "4-4-2"]
    assert mode_label(labels, min_frames=1) == "4-4-2"


def test_mode_label_none_when_too_few():
    assert mode_label(["4-4-2"], min_frames=50) is None


def test_mode_label_none_when_empty():
    assert mode_label([], min_frames=1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k mode_label -v`
Expected: FAIL with `cannot import name 'mode_label'`.

- [ ] **Step 3: Write minimal implementation**

Append to `football_ml/formations.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_formations.py -k mode_label -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py tests/test_formations.py
git commit -m "feat(formations): add mode_label aggregation"
```

---

### Task 5: Orchestrator — `extract_match_formations`

Ties everything together: drop GK, normalize depth, bucket intervals, join possession, classify each frame, aggregate to the nested result dict.

**Files:**
- Modify: `football_ml/formations.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Consumes: `attack_sign`, `classify_formation`, `ball_frame`, `interval_key`, `mode_label`.
- Produces: `extract_match_formations(tracking, ball, roster, positions, match_row, *, tolerance_m=6.0, interval_min=5, min_frames=50) -> dict`
  - `tracking`: rows of `fct_players_tracking` for one match (`player_id, player_x, period, frame`).
  - `ball`: rows of `fct_ball_tracking` for the match (`timestamp, period, possession_team_group`).
  - `roster`: rows of `fct_match_players` for the match (`player_id, team_id, player_role_id`).
  - `positions`: `dim_players_position` (`player_role_id, player_role_name`).
  - `match_row`: one `dim_match` row (Series) with `match_id`, `home_team.id`, `away_team.id`, `home_team_side`, `match_period_1st_start_frame`, `match_period_2nd_start_frame`.
  - Returns dict: `{match_id, params:{interval_min,tolerance_m}, teams:{home,away}, intervals:{"0":{"home":{"in_possession","out_of_possession"}, "away":{...}}, ...}}`.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from football_ml.formations import extract_match_formations


def _make_tracking(frames, team_players, depth_x_by_role, period=1.0):
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

    # ball: home in possession for all three frames (period 1, ~1s in)
    ball = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-04 00:00:00.000", "2026-01-04 00:00:00.100", "2026-01-04 00:00:00.200"]
            ),
            "period": [1.0, 1.0, 1.0],
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

    result = extract_match_formations(
        tracking, ball, roster, positions, match_row,
        tolerance_m=6.0, interval_min=5, min_frames=1,
    )

    assert result["match_id"] == "TESTMATCH"
    assert result["teams"] == {"home": 1, "away": 2}
    assert result["params"] == {"interval_min": 5, "tolerance_m": 6.0}
    # frames 10-12 are all in interval "0"; home in possession -> 4-4-2 attacking
    assert result["intervals"]["0"]["home"]["in_possession"] == "4-4-2"
    # home never out of possession in this window -> None
    assert result["intervals"]["0"]["home"]["out_of_possession"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k extract_match_formations -v`
Expected: FAIL with `cannot import name 'extract_match_formations'`.

- [ ] **Step 3: Write minimal implementation**

Append to `football_ml/formations.py`:

```python
def extract_match_formations(
    tracking: pd.DataFrame,
    ball: pd.DataFrame,
    roster: pd.DataFrame,
    positions: pd.DataFrame,
    match_row: pd.Series,
    *,
    tolerance_m: float = 6.0,
    interval_min: int = 5,
    min_frames: int = 50,
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

    secs = (ball["timestamp"] - ball["timestamp"].dt.normalize()).dt.total_seconds()
    bframes = [ball_frame(s, p, p1s, p2s) for s, p in zip(secs, ball["period"])]
    poss = pd.Series(ball["possession_team_group"].to_numpy(), index=bframes)
    poss = poss[~poss.index.duplicated()]
    t["possession"] = t["frame"].map(poss)

    records = []
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
        "params": {"interval_min": interval_min, "tolerance_m": tolerance_m},
        "teams": {"home": home_id, "away": away_id},
        "intervals": intervals,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_formations.py -k extract_match_formations -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py tests/test_formations.py
git commit -m "feat(formations): add extract_match_formations orchestrator"
```

---

### Task 6: JSON writer + runner job + end-to-end smoke

Persist the result and provide a runnable entry point; verify against the real match `1886347`.

**Files:**
- Modify: `football_ml/formations.py`
- Create: `football_ml/jobs/formation_extraction_job.py`
- Test: `tests/test_formations.py`

**Interfaces:**
- Consumes: `extract_match_formations`.
- Produces:
  - `write_formations_json(result: dict, out_path: Path) -> None`
  - `formation_extraction_job.run(match_id: str = "1886347", *, tolerance_m=6.0, interval_min=5, min_frames=50) -> dict`

- [ ] **Step 1: Write the failing test**

```python
import json

from football_ml.formations import write_formations_json


def test_write_formations_json_roundtrip(tmp_path):
    result = {"match_id": "X", "params": {}, "teams": {"home": 1, "away": 2}, "intervals": {}}
    out = tmp_path / "formations" / "match_X.json"
    write_formations_json(result, out)
    assert out.exists()
    assert json.loads(out.read_text()) == result


def test_end_to_end_smoke_real_match():
    # Integration: runs the real pipeline on match 1886347.
    import pandas as pd
    from football_ml.config import RAW_DATA_DIR
    from football_ml.formations import extract_match_formations

    mid = "1886347"
    tracking = pd.read_parquet(RAW_DATA_DIR / "fct_players_tracking.parquet")
    tracking = tracking[tracking["match_id"] == mid]
    ball = pd.read_parquet(RAW_DATA_DIR / "fct_ball_tracking.parquet")
    ball = ball[ball["match_id"] == mid]
    roster = pd.read_parquet(RAW_DATA_DIR / "fct_match_players.parquet")
    roster = roster[roster["match_id"] == mid]
    positions = pd.read_parquet(RAW_DATA_DIR / "dim_players_position.parquet")
    dim_match = pd.read_parquet(RAW_DATA_DIR / "dim_match.parquet")
    match_row = dim_match[dim_match["match_id"] == mid].iloc[0]

    result = extract_match_formations(tracking, ball, roster, positions, match_row)

    assert result["match_id"] == mid
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_formations.py -k "write_formations_json or end_to_end" -v`
Expected: FAIL with `cannot import name 'write_formations_json'`.

- [ ] **Step 3: Write minimal implementation**

Append to `football_ml/formations.py`:

```python
def write_formations_json(result: dict, out_path: Path) -> None:
    """Write a formations result dict to ``out_path`` as indented JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
```

Create `football_ml/jobs/formation_extraction_job.py`:

```python
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

    Returns
    -------
    dict
        The formations result (also written to disk).
    """
    tracking = pd.read_parquet(RAW_DATA_DIR / "fct_players_tracking.parquet")
    tracking = tracking[tracking["match_id"] == match_id]
    ball = pd.read_parquet(RAW_DATA_DIR / "fct_ball_tracking.parquet")
    ball = ball[ball["match_id"] == match_id]
    roster = pd.read_parquet(RAW_DATA_DIR / "fct_match_players.parquet")
    roster = roster[roster["match_id"] == match_id]
    positions = pd.read_parquet(RAW_DATA_DIR / "dim_players_position.parquet")
    dim_match = pd.read_parquet(RAW_DATA_DIR / "dim_match.parquet")
    match_row = dim_match[dim_match["match_id"] == match_id].iloc[0]

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_formations.py -v`
Expected: PASS (all tests, incl. the real-match smoke test).

Then run the job for real and eyeball the output:

Run: `python -m football_ml.jobs.formation_extraction_job`
Expected: writes `data/processed/formations/match_1886347.json`; spot-check that early intervals show plausible formations (e.g. a back four) and both teams/phases are populated.

- [ ] **Step 5: Commit**

```bash
git add football_ml/formations.py football_ml/jobs/formation_extraction_job.py tests/test_formations.py
git commit -m "feat(formations): add JSON writer, runner job, and e2e smoke test"
```

---

## Self-Review Notes

- **Spec coverage:** §5 pipeline → Tasks 1,3,5; §6 clustering → Task 2; §7 aggregation → Task 4; §8 output → Task 6; §10 tests → each task's tests (clustering T2, normalization T1, bucketing T3, aggregation T4, e2e smoke T6). Ball-frame alignment (a spec §5.4 refinement) → Task 3.
- **Type consistency:** `classify_formation`, `attack_sign`, `ball_frame`, `interval_key`, `mode_label`, `extract_match_formations`, `write_formations_json` used with identical signatures across tasks and the runner.
- **No placeholders:** every step has real test and implementation code.
- **Deviation from spec defaults:** none. `min_frames` default 50 as specified; tests pass `min_frames=1` for tiny fixtures.
