# Formation Extraction from Tracking Data — Design

**Date:** 2026-09-22
**Status:** Approved (brainstorm) — ready for implementation planning
**Repo:** football-ml

## 1. Problem & Objective

During a match a team changes its defensive and attacking shape depending on
game state (score, red cards, injuries). An analyst preparing for an opponent
wants to see how that opponent's formation evolves over the game, split by phase
of play, so the coach can plan the optimal setup.

**Objective:** extract each team's **attacking** (in-possession) and **defensive**
(out-of-possession) formation in **5-minute intervals** from SkillCorner tracking
data, and write the result to a per-match JSON file. Formations are expressed as
line-count strings (goalkeeper omitted), e.g. `"4-4-2"`, `"4-3-3"`, `"4-2-3-1"`.

## 2. Scope

- **First build (POC):** one match, `match_id = 1886347`. Validate the output
  looks football-plausible, then generalize to all 10 matches (same code, loop
  over matches).
- **In scope:** formation string per interval × team × possession phase; JSON output.
- **Out of scope:** width/shape nuance beyond line counts; naming formations from a
  fixed catalog; visualization; any change to the star-schema tables.

## 3. Design Decisions (locked during brainstorm)

| Decision | Choice |
|---|---|
| Detection method | **Free line-clustering** by depth, using a configurable line tolerance (no fixed catalog) |
| Line count | **Allow 3 or 4 lines** (e.g. `4-2-3-1` permitted); capped at 4 |
| Interval aggregation | **Per-frame classify → take the mode** label over the window |
| Attack vs defence split | Driven by ball possession (`possession_team_group`) |
| Scope | **One-match POC first**, then all matches |

## 4. Data Inputs (verified)

- Tracking is **10 fps**; frames are 1 apart. A 5-min interval ≈ 3000 frames.
  A match runs from frame ~10 to ~59000 (~98 min including stoppage).
- `fct_players_tracking`: `player_x`, `player_y`, `period`, `frame`, `player_id`,
  `match_id`. ~22 players per frame; ~29 distinct players per match (subs).
- `fct_ball_tracking`: `possession_team_group` ∈ {`home team`, `away team`, `None`}
  (~73% attributed, ~27% loose ball), keyed by frame/period/match.
- `fct_match_players`: `player_id`, `team_id`, `player_role_id`, `red_card`,
  `playing_start_frame`, `playing_end_frame` per match (roster + who is on the pitch).
- `dim_players_position`: `player_role_name == "Goalkeeper"` (role_id 0) identifies
  the GK robustly, regardless of live position.
- `dim_match`: `home_team.id`, `away_team.id`, `home_team_side`
  (home team's period-1 attacking side), `match_period_{1st,2nd}_start_frame`.

## 5. Architecture

A single module `football_ml/formations.py` (pure pandas functions, type-hinted,
numpy-style docstrings per repo conventions) plus a thin runner that writes JSON.
No new star-schema tables — formations are a derived analysis.

**Output location:** `data/processed/formations/match_{match_id}.json`.

Data is read with pandas (filtered by `match_id`); the DuckDB layer is not required
for the POC.

### Pipeline (one match)

1. **Load & drop GK.** Load player tracking for the match; join roster
   (`fct_match_players`) to attach `team_id` and role; drop the goalkeeper
   (`player_role_name == "Goalkeeper"`). Result: outfielders only.
2. **Normalize attacking direction.** Using `home_team_side` + `period`, flip the
   x-axis so every team, in every period, attacks the same way. Define
   **`depth` = signed distance from the team's own goal** (0 = own goal, larger =
   toward opponent goal). This makes "deepest line = defenders" comparable across
   halves and between teams.
   - Home team attacks the direction implied by `home_team_side` in period 1 and
     the opposite in period 2; away team is the mirror of home in each period.
3. **Bucket frames into 5-min intervals** on the match clock:
   - first half: `minute = (frame − p1_start_frame) / 600`
   - second half: `minute = 45 + (frame − p2_start_frame) / 600`
   - interval key = `floor(minute / 5) * 5` → `"0","5",…,"45","50",…` (strings).
4. **Tag possession per frame** from `possession_team_group`. For a given team,
   frames where it holds possession → **attacking**; frames where the opponent
   holds possession → **defensive**; `None` (loose ball) frames are excluded from both.

## 6. Line-clustering algorithm (per frame, per team)

Input: the outfielders on the pitch for that team at that frame (10, or 9 after a
red card). Frames missing the full expected outfield count (tracking gaps) are
**skipped**.

1. **Sort** outfielders by `depth` (deepest → most advanced).
2. **Greedy 1-D clustering along depth** with `tolerance_m`: walk from deepest;
   add a player to the current line while within `tolerance_m` of the line's
   running mean depth, else start a new line. (This is the configurable
   line-keeping tolerance — players within ~tolerance are treated as one line.)
3. **Constrain to 3–4 lines:**
   - If **>4 lines**: repeatedly **merge the two adjacent lines with the smallest
     depth gap** until 4 remain.
   - If **<3 lines**: repeatedly **split the widest line at its largest internal
     depth gap** until 3 remain.
4. **Emit** line sizes ordered defence→attack as a hyphen string, e.g.
   `[4,4,2] → "4-4-2"`, `[4,2,3,1] → "4-2-3-1"`. Digits sum to 10 (or 9 with a red card).

**Deliberate simplifications:**
- Lines are split by **depth (x) only**, not width (y) — a flat four and a diamond
  four both read as "4" at that depth. This is the standard way a formation string
  is read.
- The merge/split-to-3–4 step is the main judgement call and follows directly from
  the "allow 3 or 4 lines" decision.

## 7. Interval aggregation

For each interval × team × possession phase:
- classify every **valid** frame in the window,
- take the **mode** of the resulting labels.

If a phase has too few valid frames to classify (configurable minimum; e.g. a team
that barely touched the ball in that window), emit **`null`** for that cell.

## 8. Output format

`data/processed/formations/match_{match_id}.json`:

```json
{
  "match_id": "1886347",
  "params": { "interval_min": 5, "tolerance_m": 6.0 },
  "teams": { "home": 4177, "away": 1805 },
  "intervals": {
    "0":  { "home": { "in_possession": "4-3-3", "out_of_possession": "4-5-1" },
            "away": { "in_possession": "4-4-2", "out_of_possession": "4-4-2" } },
    "45": { "home": { "in_possession": null,    "out_of_possession": "4-4-2" },
            "away": { "in_possession": "4-4-2", "out_of_possession": "4-4-1" } }
  }
}
```

- Interval keys are interval-start minutes as strings, per the 0-and-45 convention.
- `null` marks an unclassifiable cell.
- `teams` maps `home`/`away` → `team_id`; `params` records the tolerance/interval so
  each file is self-describing.

## 9. Configuration & defaults

| Param | Default | Meaning |
|---|---|---|
| `tolerance_m` | `6.0` | Line-keeping tolerance in meters for depth clustering |
| `interval_min` | `5` | Interval length in minutes |
| `min_frames_per_cell` | (small, e.g. `50`) | Below this a cell is `null` |

## 10. Testing (TDD, pytest, synthetic fixtures)

Small hand-built fixtures — no dependence on the 44 MB parquet — except one smoke test.

1. **Clustering:** clean 4-4-2 → `"4-4-2"`; bunched blob → forced-split to 3 lines;
   6 thin lines → merged to 4; 9-outfielder (red card) set → digits sum to 9.
2. **Direction normalization:** a team whose raw x flips between periods yields the
   same depth ordering in both halves.
3. **Interval bucketing:** frames map to the correct `0/5/45/…` bucket given period
   start frames.
4. **Aggregation:** mixed per-frame labels return the correct mode; an all-loose-ball
   window → `null`.
5. **End-to-end smoke** on match `1886347`: JSON is well-formed; every formation
   string's digits sum to 10 or 9; a couple of intervals spot-checked as plausible.

## 11. Open questions / future work

- Extend to all 10 matches (loop; same code).
- Optional later: correlate formation changes with events (goals, red cards) to
  answer the original "how does shape change after a goal/red card" question.
- Optional later: width/shape descriptors beyond line counts.
