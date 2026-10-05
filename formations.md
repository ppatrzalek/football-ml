# Formation extraction

How `football_ml/formations.py` turns raw SkillCorner tracking data into each
team's **attacking** and **defensive** formation, per five-minute interval.

## What it produces

For one match, a JSON file at `data/processed/formations/match_{match_id}.json`.
For every 5-minute interval and both teams it records the formation while
**in possession** (attacking shape) and **out of possession** (defensive shape),
as a line-count string with the goalkeeper omitted — e.g. `4-4-2`, `4-3-3`,
`4-2-3-1`. The digits always sum to 10 (or 9 after a red card).

The goal is to see how a team's shape changes across a game — the foundation for
later linking shape changes to game state (goals, red cards, substitutions).

## Run it

```bash
python -m football_ml.jobs.formation_extraction_job          # default match 1886347
```

or from Python:

```python
from football_ml.jobs.formation_extraction_job import run
result = run(match_id="1886347", tolerance_m=6.0, min_frames=200, settle_seconds=5.0)
```

## Inputs

All from `data/raw/` (SkillCorner open data, 10 fps tracking):

| Table | Used for |
|---|---|
| `fct_players_tracking` | player `player_x`, `period`, `frame`, `player_id` |
| `fct_ball_tracking` | ball `x/y/z`, `is_detected`, `possession_team_group`, `timestamp` |
| `fct_match_players` | roster: `team_id`, `player_role_id` (to find the GK) |
| `dim_players_position` | `player_role_name` (`"Goalkeeper"`) |
| `dim_match` | `home_team.id`, `away_team.id`, `home_team_side`, period start frames |

## The pipeline

Everything runs inside `extract_match_formations()`. Per match:

### 1. Drop the goalkeeper
The roster (`fct_match_players`) is joined to `dim_players_position` and any
player whose role is `"Goalkeeper"` is removed, leaving 10 outfielders (9 after a
red card). Tracking players with no roster row are also dropped — otherwise an
unmatched `player_id` would silently be miscounted into the away team.

### 2. Normalize attacking direction
Teams switch ends at half-time, so a raw `player_x` isn't comparable across
halves. Using `home_team_side` and the period, `attack_sign()` flips the x-axis
so that **`depth = attack_sign(...) * player_x`** always means *distance from the
team's own goal*: small depth = near own goal (defenders), large depth = up the
pitch (attackers). This makes "the deepest line is the defence" true for both
teams in both halves.

### 3. Keep only settled open play
Formation is only meaningful when players are in shape — not during throw-ins,
corners, goal kicks, aerial deliveries or loose-ball scrambles. **The dataset has
no event labels** (no corners/free kicks/fouls are marked anywhere), so those
moments are *inferred from the ball* by `settled_open_play_frames()`. A frame is
kept only when the ball is:

- **in bounds** (`|ball_x| ≤ 52.5 m`, `|ball_y| ≤ 34 m`),
- **grounded** (`ball_z ≤ z_max`, default 2 m),
- **detected** (`is_detected`), and
- **possessed** (`possession_team_group` is set, i.e. not a loose ball).

On top of that, the first `settle_seconds` (default 5 s) after every return to
play are dropped — a resettle buffer so players have time to get back into shape
before a frame counts. This keeps roughly two-thirds of frames.

> **Ball ↔ player alignment.** Ball rows are keyed by `timestamp`, players by
> `frame`. Both sit on the same 10 fps match clock (1st half anchored at 00:00,
> 2nd half at 45:00), so a player frame is derived from a ball timestamp as
> `start_frame + round(elapsed_seconds_in_half × 10)`. This matched the player
> frames exactly (43458/43458) on the POC match.

### 4. Cluster outfielders into lines
For each remaining frame, `classify_formation()` reads the shape from the depth
values:

1. Sort the outfielders by depth.
2. **Greedy grouping** — walk from the deepest player, adding each to the current
   line while they're within `tolerance_m` (default 6 m) of the line's running
   mean depth; otherwise start a new line. This is the "line-keeping tolerance":
   players within ~6 m of each other count as one line.
3. **Constrain to 3–4 lines** — if greedy produced more than 4 lines, repeatedly
   merge the closest adjacent pair; if fewer than 3, split the widest line at its
   largest internal gap.
4. Emit the line sizes, defence → attack, as `"4-4-2"` etc.

Lines are split by **depth only**, not width — a flat back four and a diamond
both read as "4" at that depth, which is how a formation string is normally read.

### 5. Split by possession
The ball's `possession_team_group` tags each kept frame. Frames where a team has
the ball feed its **attacking** shape; frames where the opponent has the ball feed
its **defensive** shape.

### 6. Aggregate to the interval
Each frame is bucketed into a 5-minute interval on the match clock (1st half
`0,5,…,40`; 2nd half `45,50,…` — first-half stoppage folds into `40` so it never
collides with the second half). Within each interval × team × phase, every
frame's formation is classified and the **most common** label wins. If a cell has
fewer than `min_frames` settled frames (default 200 ≈ 20 s), it is reported as
`null` rather than guessing.

## Output format

```json
{
  "match_id": "1886347",
  "params": { "interval_min": 5, "tolerance_m": 6.0, "min_frames": 200,
              "z_max": 2.0, "settle_seconds": 5.0 },
  "teams": { "home": 4177, "away": 1805 },
  "intervals": {
    "0":  { "home": { "in_possession": null,    "out_of_possession": "4-3-3" },
            "away": { "in_possession": "3-2-4-1", "out_of_possession": null } },
    "10": { "home": { "in_possession": "2-4-4",  "out_of_possession": "4-3-3" },
            "away": { "in_possession": "3-2-4-1", "out_of_possession": "5-4-1" } }
  }
}
```

- Interval keys are interval-start minutes as strings.
- `null` = too little settled possession in that window to call a shape.
- `params` records the settings used, so each file is self-describing.

## Parameters

| Name | Default | Meaning |
|---|---|---|
| `tolerance_m` | `6.0` | Line-keeping tolerance in metres for depth clustering |
| `interval_min` | `5` | Interval length in minutes |
| `min_frames` | `200` | Minimum settled frames to emit a formation (else `null`) |
| `z_max` | `2.0` | Max ball height (m) for a frame to count as grounded |
| `settle_seconds` | `5.0` | Seconds dropped after each return to play |

`tolerance_m = 6` and `min_frames = 200` were chosen from a sweep over
`tolerance ∈ {4..8}` × `min_frames ∈ {150..300}`: tolerance 6 minimised
degenerate shapes (7–8 merged players into over-large lines), and 200 balanced
coverage against cleanliness.

## Known limitations

- **No back-line prior (by design).** After filtering, a back-2 or back-3 in
  possession is legitimate modern football (full-backs push high), so the code
  does **not** force conventional back-line sizes.
- **Residual degeneracy (~8%).** A few windows still produce shapes like `7-1-2`
  ("everyone deep" — e.g. defending a late lead). These are honest reflections of
  where players stood and a floor of depth-only clustering, not bugs.
- **`null` cells (~25%).** Low-possession windows legitimately lack the settled
  data to name a shape.
- **POC scope.** Validated on one match (`1886347`); the same code generalises to
  all matches by looping. Pitch bounds are constants (105 × 68 m) rather than read
  from `dim_stadium`, and the per-frame classification is a plain Python loop —
  both fine at this scale, wired up later if it goes to production.
