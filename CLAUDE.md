# football-ml

Personal football analytics repository for data processing, visualization, and machine learning using free/open data sources.

## Project Goals

- Process and analyze football tracking data (SkillCorner open data as primary source)
- Build visualizations for tracking, event, and spatial data
- Develop ML models for player analysis, clustering, and tactical patterns
- Explore: player similarity, off-ball movement, set-piece patterns, pressing metrics

## Environment

- **Conda**: miniforge3 at `~/miniforge3`
- **Package**: `football_ml` (installed locally)
- **Storage format**: Parquet (raw and processed)
- **Primary data source**: SkillCorner open data via GitHub API

## Repository Structure

```
football_ml/          # Main Python package
  config/             # Paths (ROOT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR) and logger
  preprocessors/      # Data transformation functions
    base.py           # Fetch match list and metadata from SkillCorner GitHub API
    match.py          # Transform raw JSON into star schema dimension/fact tables
    tracking.py       # Preprocess player and ball tracking data
    transfermarkt.py  # Preprocess Transfermarkt market value data → fct_player_market_value
    utils.py          # Shared utilities (e.g. time_to_seconds)
  jobs/
    data_gathering_job.py  # Main ETL job: fetch all matches → preprocess → save to parquet
  readers/            # Readers for multiple data sources (in development)
  db.py               # DuckDB connection utility: get_connection() registers all Parquet views

data/
  raw/
    fct_players_tracking/  # Partitioned: match_{match_id}.parquet (one file per match)
    fct_ball_tracking/     # Partitioned: match_{match_id}.parquet (one file per match)
    dim_match.parquet       # Flat files for all dim/fct tables
    dim_team.parquet
    ...
  processed/          # Derived/transformed datasets

notebooks/            # Jupyter notebooks for exploration and POCs
```

## Data Schema (Star Schema)

### Dimension Tables
| Table | Key columns |
|---|---|
| `dim_match` | match_id, date_time, home/away team ids, scores, period frames, competition/season ids |
| `dim_competition_season` | competition_id, season_id, name, area |
| `dim_team` | team_id, team_name, short_name, acronym |
| `dim_stadium` | stadium_id, name, city, capacity, pitch_length_meters, pitch_width_meters |
| `dim_players` | player_id, first/last/short name, team_id, gender, birthday, player_role_id |
| `dim_players_position` | player_role_id, name, acronym, position_group |

### Fact Tables
| Table | Key columns | Storage |
|---|---|---|
| `fct_players_tracking` | player_tracking_id (`{match_id}_{frame}_{player_id}`), timestamp, player_x, player_y, period, frame, player_id, match_id | Partitioned by match |
| `fct_ball_tracking` | ball_tracking_id (`{match_id}_{timestamp}`), timestamp, ball_x, ball_y, ball_z, is_detected, period, match_id, possession_player_id, possession_team_group | Partitioned by match |
| `fct_match_players` | player_id, match_id, team_id, start/end_time, minutes_played, yellow/red_card, goal, own_goal, playing_start/end_frame | Flat file |
| `fct_player_market_value` | player_id, value_eur, value_date, contract_expiry, source | Flat file (Transfermarkt, planned) |

### Tracking coordinate system
- Player positions: `player_x`, `player_y` (pitch coordinates in meters)
- Ball positions: `ball_x`, `ball_y`, `ball_z` (3D, z = height)
- Pitch dimensions stored in `dim_stadium`: typically ~105m × 68m

## Data Sources

| Source | Type | Status |
|---|---|---|
| SkillCorner | Tracking + event | Active — open data via GitHub |
| StatsBomb | Event data | Planned |
| Transfermarkt | Player market values | Planned |
| Metrica Sports | Tracking (sample) | Planned |

## Code Conventions

- **pandas** for all data processing (no polars yet)
- **Type hints** on all function signatures
- **Docstrings** with `Parameters` / `Returns` sections (numpy style)
- Deprecated functions marked with `# Deprecated function` comment or `DEPRECATED_` prefix in name
- Star schema naming: `dim_*` for dimensions, `fct_*` for facts
- Data written to parquet with `index=False`

## Active Work (as of April 2026)

- `notebooks/tracking_visualization.ipynb` — visualizing player and ball tracking data
- `notebooks/poc_tracking_imputation.ipynb` — POC for imputing missing tracking positions
- `football_ml/readers/` — building standardized readers for multiple data sources
