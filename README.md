# football-ml

Personal football analytics repository for processing, visualizing, and modeling football data using open data sources.

## Setup

**Environment**: miniforge3 conda, Python 3.10+

```bash
conda activate <your-env>
pip install -e .
```

**Key dependencies**: pandas, numpy, scipy, duckdb, mplsoccer, kloppy, scikit-learn, plotly, matplotlib

## Data Sources

| Source | Type | Status |
|---|---|---|
| SkillCorner | Tracking + event | Active — open data via GitHub API |
| StatsBomb | Event data | Planned |
| Transfermarkt | Player market values | Planned |
| Metrica Sports | Tracking (sample) | Planned |

## Repository Structure

```
football_ml/
  config/               # Path constants (ROOT_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR) and logger
  preprocessors/
    base.py             # Fetch match list and metadata from SkillCorner GitHub API
    match.py            # Transform raw JSON → star schema dim/fct tables
    tracking.py         # Player and ball tracking preprocessing
    transfermarkt.py    # Market value preprocessing → fct_player_market_value (planned)
    utils.py            # Shared utilities (time_to_seconds)
  jobs/
    data_gathering_job.py  # Main ETL: fetch all matches → preprocess → save parquet
  readers/              # Standardized multi-source readers (in development)
  db.py                 # DuckDB connection: get_connection() registers all Parquet views

data/
  raw/
    fct_players_tracking/   # Partitioned: match_{match_id}.parquet
    fct_ball_tracking/      # Partitioned: match_{match_id}.parquet
    dim_match.parquet
    dim_team.parquet
    dim_stadium.parquet
    dim_players.parquet
    dim_players_position.parquet
    dim_competition_season.parquet
    fct_match_players.parquet
  processed/            # Derived datasets

notebooks/              # Exploration and POCs
```

## Data Schema (Star Schema)

Star schema with dimension tables (`dim_*`) and fact tables (`fct_*`). Full column-level documentation, type information, gotchas, and coordinate system details in [docs/schema.md](docs/schema.md).

| Table | Type | Storage |
|---|---|---|
| `dim_match` | Dimension | Flat |
| `dim_competition_season` | Dimension | Flat |
| `dim_team` | Dimension | Flat |
| `dim_stadium` | Dimension | Flat |
| `dim_players` | Dimension | Flat |
| `dim_players_position` | Dimension | Flat |
| `fct_players_tracking` | Fact | Partitioned by match |
| `fct_ball_tracking` | Fact | Partitioned by match |
| `fct_match_players` | Fact | Flat |

## Querying Data

All Parquet tables are registered as SQL views via DuckDB:

```python
from football_ml.db import get_connection

conn = get_connection()
df = conn.execute("""
    SELECT t.player_id, p.short_name, t.player_x, t.player_y
    FROM fct_players_tracking t
    JOIN dim_players p USING (player_id)
    WHERE t.match_id = '1234'
""").df()
```

## Running the ETL

```python
from football_ml.jobs.data_gathering_job import run

run()  # fetches all available SkillCorner matches → saves to data/raw/
```

## Analysis Ideas

- **Pass analysis**: passes by zone (defensive/offensive third), split by short/long
- **Player clustering**: off-ball movement profiles using tracking data
- **Set-piece patterns**: corner and free-kick routines — space creation, player runs
- **Pressing metrics**: PPDA, press triggers, counterpressing intensity
- **Physical profiling**: distance covered, high-intensity runs, sprint counts per player/match
