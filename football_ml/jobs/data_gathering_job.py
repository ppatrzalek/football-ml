"""Data gathering job for football ML project."""
import requests
import numpy as np
import pandas as pd
import sys
import os

print(os.getcwd())
print(sys.path)


from football_ml.config import CONTENT_URL, RAW_DATA_DIR
from football_ml.config import logger


def get_available_matches(content_url: str) -> pd.DataFrame:
    """Fetch the list of available matches from the GitHub repository.

    Args:
        content_url (str): The URL to fetch match data from.

    Returns:
        pd.DataFrame: DataFrame containing match information.
    """
    response = requests.get(content_url)
    matches = response.json()
    return pd.DataFrame(matches)


def preprocess_tracking_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess tracking data by normalizing nested JSON fields and extracting relevant information.

    Args: s
        raw_data (pd.DataFrame): Raw tracking data containing nested JSON fields.

    Returns:
        pd.DataFrame: Preprocessed tracking data with normalized fields.
    """
    raw_df = pd.json_normalize(
        raw_data.to_dict("records"),
        "player_data",
        ["frame", "timestamp", "period", "possession", "ball_data"],
    )
    raw_df["possession_player_id"] = raw_df["possession"].apply(
    lambda x: x.get("player_id")
        )
    raw_df["possession_group"] = raw_df["possession"].apply(lambda x: x.get("group"))
    
    # Expand the ball_data with json_normalize
    raw_df[["ball_x", "ball_y", "ball_z", "is_detected_ball"]] = pd.json_normalize(
        raw_df.ball_data
    )
    # Drop the original 'possession' column if you no longer need it
    raw_df = raw_df.drop(columns=["possession", "ball_data"])
    
    return raw_df


def get_match_tracking_data(match_id: str) -> pd.DataFrame:
    """Fetch and preprocess tracking data for a specific match.

    Args:
        match_id (str): The ID of the match to fetch data for.

    Returns:
        pd.DataFrame: Preprocessed tracking data for the specified match.
    """
    logger.info(f"Match ID: {match_id}")
    content_tracking_url = CONTENT_URL + f"/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    content_tracking_data = pd.read_json(content_tracking_url, lines=True)
    raw_tracking_data = pd.read_json(content_tracking_data['download_url'][0], lines=True)
    processed_tracking_df = preprocess_tracking_data(raw_tracking_data)
    processed_tracking_df["match_id"] = int(match_id)
    return processed_tracking_df


def get_match_meta_data(match_id: str) -> pd.DataFrame:
    """Fetch match metadata for a specific match.

    Args:
        match_id (str): The ID of the match to fetch metadata for.
    Returns:
        pd.DataFrame: Match metadata for the specified match.
    """
    content_match_url = CONTENT_URL + f"/{match_id}/{match_id}_match.json"  # Data is stored using GitLFS
    content_match_data = pd.read_json(content_match_url, lines=True)
    response = requests.get(content_match_data['download_url'][0])
    raw_match_data = response.json()
    raw_match_df = pd.json_normalize(raw_match_data, max_level=2)
    return raw_match_df


def time_to_seconds(time_str):
    if time_str is None:
        return 90 * 60  # 120 minutes = 7200 seconds
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def preprocess_players_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess match data by converting time strings to seconds.

    Args:
        raw_match_df (pd.DataFrame): Raw match data containing time strings.
    Returns:
        pd.DataFrame: Preprocessed players data with time in seconds.
    """
    # Extract relevant fields from nested JSON structures
    players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
        meta=[
            "home_team_score",
            "away_team_score",
            "date_time",
            "home_team_side",
            "home_team.name",
            "home_team.id",
            "away_team.name",
            "away_team.id",
        ],  # data we keep
    )
    players_df["match_id"] = raw_match_df["id"].iloc[0]
    players_df = players_df.rename(columns={"id": "player_id"})

    # Take only players who played and create their total time
    mask = (players_df.start_time.isna()) & (players_df.end_time.isna())
    players_df = players_df[~mask]
    players_df["total_time"] = players_df["end_time"].apply(time_to_seconds) - players_df["start_time"].apply(time_to_seconds)
    
    # Create a flag for GK
    players_df["is_gk"] = players_df["player_role.acronym"] == "GK"

    # Add a match name column
    players_df["match_name"] = (
        players_df["home_team.name"] + " vs " + players_df["away_team.name"]
    )
    # Add a flag if the given player is home or away
    players_df["home_away_player"] = np.where(
        players_df.team_id == players_df["home_team.id"], "Home", "Away"
    )
    # Create flag from player
    players_df["team_name"] = np.where(
        players_df.team_id == players_df["home_team.id"],
        players_df["home_team.name"],
        players_df["away_team.name"],
    )
    # Figure out sides
    players_df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
        players_df["home_team_side"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "")
        .str.split(", ", expand=True)
    )
    # Clean up sides
    players_df["direction_player_1st_half"] = np.where(
        players_df.home_away_player == "Home",
        players_df.home_team_side_1st_half,
        players_df.home_team_side_2nd_half,
    )
    players_df["direction_player_2nd_half"] = np.where(
        players_df.home_away_player == "Home",
        players_df.home_team_side_2nd_half,
        players_df.home_team_side_1st_half,
    )
    # Clean up and keep the columns that we want to keep about
    columns_to_keep = [
        "start_time",
        "end_time",
        "match_id",
        "match_name",
        "date_time",
        "home_team.name",
        "away_team.name",
        "id",
        "short_name",
        "number",
        "team_id",
        "team_name",
        "player_role.position_group",
        "total_time",
        "player_role.name",
        "player_role.acronym",
        "is_gk",
        "direction_player_1st_half",
        "direction_player_2nd_half",
    ]
    players_df = players_df[columns_to_keep]
    return players_df


def main():
    """Main function to gather and preprocess football match tracking data."""
    logger.info("Starting data gathering job...")

    # Fetch available matches   
    matches_df = get_available_matches(CONTENT_URL)

    # Display available match IDs
    match_ids = [match['name'] for match in matches_df.to_dict('records') if match['type'] == 'dir']
    logger.info(f"Available matches: {len(match_ids)}")
    
    all_matches_tracking, all_matches_players = [], []
    for match_id in match_ids:
        logger.info(f"Match ID: {match_id}")
        try:
            processed_tracking_df = get_match_tracking_data(match_id)
            all_matches_tracking.append(processed_tracking_df)
            logger.info(f"Processed match ID: {match_id}")
        except Exception as e:
            logger.error(f"Failed to process match ID: {match_id}. Error: {e}")
            
        try:
            raw_match_df = get_match_meta_data(match_id)
            processed_players_df = preprocess_players_data(raw_match_df)
            all_matches_players.append(processed_players_df)
            logger.info(f"Processed players data for match ID: {match_id}")
        except Exception as e:
            logger.error(f"Failed to process players data for match ID: {match_id}. Error: {e}")
    
    all_matches_tracking_df = pd.concat(all_matches_tracking, ignore_index=True)
    all_matches_players_df = pd.concat(all_matches_players, ignore_index=True)
    logger.info(f"Total tracking records gathered: {len(all_matches_tracking_df)}")
    logger.info(f"Total players records gathered: {len(all_matches_players_df)}")
    
    # Write into parquet file
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DATA_DIR / "all_matches_tracking.parquet"
    all_matches_tracking_df.to_parquet(output_file, index=False)
    logger.info(f"Processed tracking data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "all_matches_players.parquet"
    all_matches_players_df.to_parquet(output_file, index=False)
    logger.info(f"Processed players data saved to {output_file}")


if __name__ == "__main__":
    main()
