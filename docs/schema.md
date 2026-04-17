# Schema Reference

Data is organized as a **star schema**: dimension tables describing entities (players, teams, matches) and fact tables containing measurements (tracking positions, match events). All tables are stored as Parquet files under `data/raw/`.

## Table Relationships

```
dim_competition_season ──────────────────────────────────┐
                                                          │
dim_stadium ── (team_id) ──── dim_team ──────────────────┤
                                  │                       │
                              dim_players ────────────────┤
                                  │                       │
                            dim_players_position          │
                                                          ▼
                                                      dim_match
                                                     /    |    \
                                        fct_match_players │     │
                                                          │     │
                                          fct_players_tracking  │
                                                                │
                                                   fct_ball_tracking
```

Foreign keys:
- `dim_match` → `dim_team` via `home_team.id` / `away_team.id`
- `dim_match` → `dim_competition_season` via `competition_id` + `season_id`
- `dim_players` → `dim_team` via `team_id`
- `dim_players` → `dim_players_position` via `player_role_id`
- `dim_stadium` → `dim_team` via `team_id` (home team)
- `fct_match_players` → `dim_match` via `match_id`
- `fct_match_players` → `dim_players` via `player_id`
- `fct_players_tracking` → `dim_match` via `match_id`
- `fct_players_tracking` → `dim_players` via `player_id`
- `fct_ball_tracking` → `dim_match` via `match_id`
- `fct_ball_tracking` → `dim_players` via `possession_player_id` (nullable)

---

## Dimension Tables

### `dim_competition_season`

One row per competition-season combination.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `competition_id` | int | no | SkillCorner competition ID |
| `season_id` | int | no | SkillCorner season ID |
| `competition_name` | str | no | e.g. `"Premier League"` |
| `season_name` | str | no | e.g. `"2022/2023"` |
| `competition_area` | str | yes | Country or region, e.g. `"England"` |
| `competition_edition_name` | str | yes | Full edition label, e.g. `"Premier League 2022/2023"` |

**Primary key**: `(competition_id, season_id)`

---

### `dim_match`

One row per match.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `match_id` | int | no | SkillCorner match ID |
| `date_time` | datetime | no | Kickoff datetime (UTC) |
| `home_team.id` | int | no | Home team ID → `dim_team` |
| `away_team.id` | int | no | Away team ID → `dim_team` |
| `home_team_score` | int | yes | Final score, home side |
| `away_team_score` | int | yes | Final score, away side |
| `home_team_side` | str | yes | Side home team attacks in 1st half: `"left"` or `"right"` |
| `match_period_1st_start_frame` | int | no | Frame number where 1st half begins |
| `match_period_1st_end_frame` | int | no | Frame number where 1st half ends |
| `match_period_1st_duration` | float | no | 1st half duration in minutes |
| `match_period_2nd_start_frame` | int | no | Frame number where 2nd half begins |
| `match_period_2nd_end_frame` | int | no | Frame number where 2nd half ends |
| `match_period_2nd_duration` | float | no | 2nd half duration in minutes |
| `competition_round.round_number` | int | yes | Matchday / round number |
| `competition_edition.competition.id` | int | no | → `dim_competition_season.competition_id` |
| `competition_edition.season.id` | int | no | → `dim_competition_season.season_id` |

**Primary key**: `match_id`

> **Note**: Several columns retain dot notation in their names (`home_team.id`, `competition_round.round_number`, etc.) — an artifact of `pd.json_normalize`. In DuckDB SQL, quote these with double quotes: `"home_team.id"`.

---

### `dim_team`

One row per team. Built by deduplicating home and away team entries across all matches.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `team_id` | int | no | SkillCorner team ID |
| `team_name` | str | no | Full team name, e.g. `"Manchester City"` |
| `team_short_name` | str | yes | Shortened name, e.g. `"Man City"` |
| `team_acronym` | str | yes | 3-letter code, e.g. `"MCI"` |

**Primary key**: `team_id`

---

### `dim_stadium`

One row per stadium. Linked to the home team since stadium data comes from match metadata.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `stadium_id` | int | no | SkillCorner stadium ID |
| `stadium_name` | str | no | Full stadium name |
| `stadium_city` | str | yes | City where the stadium is located |
| `stadium_capacity` | int | yes | Seating capacity |
| `pitch_length_meters` | float | no | Pitch length (x-axis span), typically ~105 |
| `pitch_width_meters` | float | no | Pitch width (y-axis span), typically ~68 |
| `team_id` | int | no | Home team using this stadium → `dim_team` |

**Primary key**: `stadium_id`

---

### `dim_players`

One row per player. Players are extracted from match metadata, so only players who appear in at least one match are included.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `player_id` | int | no | SkillCorner player ID |
| `player_first_name` | str | yes | First name |
| `player_last_name` | str | yes | Last name |
| `player_short_name` | str | no | Common display name, e.g. `"K. De Bruyne"` |
| `team_player_id` | int | yes | Team-scoped player ID (SkillCorner internal) |
| `team_id` | int | no | Current team → `dim_team` |
| `player_gender` | str | yes | `"male"` or `"female"` |
| `player_birthday` | str | yes | Date of birth, ISO format |
| `player_role_id` | int | no | → `dim_players_position` |
| `player_role.name` | str | yes | Position name, e.g. `"Center Back"` |
| `player_role.acronym` | str | yes | Position acronym, e.g. `"CB"` |
| `player_role.position_group` | str | yes | Broad group: `"Defender"`, `"Midfielder"`, `"Forward"`, `"Goalkeeper"` |

**Primary key**: `player_id`

> **Note**: `player_role.name`, `player_role.acronym`, and `player_role.position_group` duplicate information from `dim_players_position` — they exist here for convenience but `player_role_id` is the canonical FK. The dot-notation column names apply here too.

---

### `dim_players_position`

Lookup table for player role/position classifications.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `player_role_id` | int | no | Role ID |
| `player_role_name` | str | no | Full position name, e.g. `"Center Back"`, `"Striker"` |
| `player_role_acronym` | str | no | Short code, e.g. `"CB"`, `"ST"`, `"GK"` |
| `player_position_group` | str | no | Broad group: `"Defender"`, `"Midfielder"`, `"Forward"`, `"Goalkeeper"` |

**Primary key**: `player_role_id`

---

## Fact Tables

### `fct_match_players`

One row per player per match. Covers all squad members, including substitutes.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `match_id` | int | no | → `dim_match` |
| `player_id` | int | no | → `dim_players` |
| `team_id` | int | no | → `dim_team` |
| `player_role_id` | int | no | Role in this match → `dim_players_position` |
| `number` | int | yes | Jersey number |
| `start_time` | str | yes | Time player entered the pitch, `"HH:MM:SS"` format. Null if started |
| `end_time` | str | yes | Time player left the pitch. Null if played full match |
| `playing_start_frame` | int | yes | Frame number when player started playing |
| `playing_end_frame` | int | yes | Frame number when player stopped playing |
| `minutes_played` | float | yes | Total minutes played (TIP + OTIP) |
| `minutes_played_regular_time` | float | yes | Minutes played excluding extra time |
| `minutes_tip` | float | yes | Minutes in Time In Play (ball in play) |
| `minutes_otip` | float | yes | Minutes in Out of Time In Play (stoppages) |
| `minutes_total` | object | yes | Raw nested dict from SkillCorner — avoid using directly |
| `yellow_card` | bool | no | Received a yellow card |
| `red_card` | bool | no | Received a red card |
| `injured` | bool | yes | Left match injured |
| `goal` | int | no | Goals scored |
| `own_goal` | int | no | Own goals scored |

**Primary key**: `(match_id, player_id)`

> **Note**: `minutes_total` is an object column containing the raw nested JSON dict. It is redundant with the other `minutes_*` columns and should be ignored.

---

### `fct_players_tracking`

One row per player per frame. High-volume table — partitioned by match.

**Storage**: `data/raw/fct_players_tracking/match_{match_id}.parquet`

| Column | Type | Nullable | Description |
|---|---|---|---|
| `player_tracking_id` | str | no | Composite key: `"{match_id}_{frame}_{player_id}"` |
| `match_id` | int | no | → `dim_match` |
| `player_id` | int | no | → `dim_players` |
| `frame` | int | no | Frame number. SkillCorner tracks at **10 fps** |
| `timestamp` | str | no | Match clock, format `"HH:MM:SS.mmm"` |
| `period` | int | no | Match period: `1` (1st half), `2` (2nd half), `3`/`4` (extra time) |
| `player_x` | float | no | Player x position in meters — see [Coordinate System](#coordinate-system) |
| `player_y` | float | no | Player y position in meters — see [Coordinate System](#coordinate-system) |

**Primary key**: `player_tracking_id`

> Not all players are present in every frame — tracking may be partial for players outside the camera view. Filter by `player_id` and `match_id` before joining.

---

### `fct_ball_tracking`

One row per timestamp. Ball position for the full match, partitioned by match.

**Storage**: `data/raw/fct_ball_tracking/match_{match_id}.parquet`

| Column | Type | Nullable | Description |
|---|---|---|---|
| `ball_tracking_id` | str | no | Composite key: `"{match_id}_{timestamp}"` |
| `match_id` | int | no | → `dim_match` |
| `timestamp` | str | no | Match clock, format `"HH:MM:SS.mmm"` |
| `period` | int | no | Match period: `1`, `2`, `3`, `4` |
| `ball_x` | float | no | Ball x position in meters |
| `ball_y` | float | no | Ball y position in meters |
| `ball_z` | float | no | Ball height in meters (0 = ground) |
| `is_detected` | bool | no | `True` = physically detected by cameras; `False` = extrapolated |
| `possession_player_id` | int | yes | Player currently in possession → `dim_players`. Null if no clear possession |
| `possession_team_group` | str | yes | `"home"` or `"away"`. Null if no possession |

**Primary key**: `ball_tracking_id`

> Rows where `is_detected = False` are model-extrapolated positions, not real camera detections. For physical analysis (height, bounce detection), filter to `is_detected = True` only.

---

## Coordinate System

SkillCorner uses a **center-origin** coordinate system in meters:

```
                    y = +34 (near side)
                         |
                         |
  x = -52.5 ─────────── 0,0 ─────────── x = +52.5
  (left goal)            |            (right goal)
                         |
                    y = -34 (far side)
```

- **Origin**: center circle
- **x-axis**: spans the length of the pitch (~−52.5 to +52.5 for a 105m pitch)
- **y-axis**: spans the width (~−34 to +34 for a 68m pitch)
- **z-axis** (ball only): height above ground in meters

The actual pitch dimensions per match are in `dim_stadium.pitch_length_meters` and `dim_stadium.pitch_width_meters`.

### Direction of play

The tracking coordinate system is **fixed** — positions do not flip between halves. Use `dim_match.home_team_side` to determine which goal each team is attacking:

| `home_team_side` | Home team attacks in 1st half |
|---|---|
| `"right"` | Towards x = +52.5 |
| `"left"` | Towards x = −52.5 |

The home team switches sides in the 2nd half (standard football rules). Away team is always the opposite of home.

### Normalizing to attacking direction

To normalize all frames so a team always attacks left-to-right:

```python
# For a given team in a given half, determine if coordinates need flipping
attacks_right = (home_team_side == "right" and is_home_team and period == 1) or \
                (home_team_side == "left"  and is_home_team and period == 2)

if not attacks_right:
    df["player_x"] = -df["player_x"]
    df["player_y"] = -df["player_y"]
```

---

## Composite Keys

Surrogate keys are constructed as strings to remain stable across re-runs:

| Key | Table | Construction |
|---|---|---|
| `player_tracking_id` | `fct_players_tracking` | `f"{match_id}_{frame}_{player_id}"` |
| `ball_tracking_id` | `fct_ball_tracking` | `f"{match_id}_{timestamp}"` |

---

## Storage Layout

```
data/raw/
  fct_players_tracking/
    match_1234.parquet     # one file per match, ~millions of rows per match
    match_1235.parquet
    ...
  fct_ball_tracking/
    match_1234.parquet
    match_1235.parquet
    ...
  dim_match.parquet
  dim_team.parquet
  dim_stadium.parquet
  dim_players.parquet
  dim_players_position.parquet
  dim_competition_season.parquet
  fct_match_players.parquet
```

All tables are registered as SQL views via DuckDB. Querying tracking data for a single match will read only the relevant partition file:

```python
from football_ml.db import get_connection

conn = get_connection()

# Efficient — reads only match_1234.parquet
df = conn.execute("""
    SELECT t.player_id, p.player_short_name, t.player_x, t.player_y, t.timestamp
    FROM fct_players_tracking t
    JOIN dim_players p USING (player_id)
    WHERE t.match_id = 1234
      AND t.period = 1
""").df()
```
