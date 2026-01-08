import pandas as pd

from football_ml.config import CONTENT_URL

def preprocess_tracking_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Preprocess tracking data by normalizing nested JSON fields and extracting relevant information.

    Parameters:
    -----------
    raw_data: pd.DataFrame
        Raw tracking data containing nested JSON fields.

    Returns:
    -----------
    pd.DataFrame
        Preprocessed tracking data with normalized fields.
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


def preprocess_ball_tracking(raw_tracking_data, match_id):
    """
    Preprocess raw tracking data to extract ball tracking information.
    
    Parameters:
    -----------
    raw_tracking_data : pd.DataFrame
        Raw tracking data containing ball_data and possession columns
    match_id : int
        Match identifier
        
    Returns:
    --------
    pd.DataFrame
        Processed ball tracking dataframe
    """
    # Expand the ball_data column
    ball_expanded = pd.json_normalize(raw_tracking_data['ball_data'])
    possession_expanded = pd.json_normalize(raw_tracking_data['possession'])
    
    # Combine with original dataframe
    ball_tracking_df = pd.concat([
        raw_tracking_data.drop(columns=["ball_data", "player_data", "image_corners_projection", "possession", "frame"], axis=1),
        ball_expanded.rename(columns={"x": "ball_x", "y": "ball_y", "z": "ball_z"}),
        possession_expanded.rename(columns={"player_id": "possession_player_id", "group": "possession_team_group"}),    
    ], 
    axis=1).drop_duplicates().dropna(subset=["ball_x", "ball_y", "ball_z"])
    
    ball_tracking_df["match_id"] = match_id
    ball_tracking_df = ball_tracking_df.reset_index(drop=True).reset_index().rename(columns={"index": "ball_tracking_id"})
    
    columns_ = [
        "ball_tracking_id", "timestamp",
        "ball_x", "ball_y", "ball_z",
        "is_detected", "period", "match_id", "possession_player_id", "possession_team_group"
    ]
    ball_tracking_df = ball_tracking_df[columns_]
    
    return ball_tracking_df


# Deprecated function
def preprocess_player_tracking(raw_tracking_data, match_id):
    """
    Preprocess raw tracking data to extract player tracking information.
    
    Parameters:
    -----------
    raw_tracking_data : pd.DataFrame
        Raw tracking data containing player_data column
    match_id : int
        Match identifier
        
    Returns:
    --------
    pd.DataFrame
        Processed player tracking dataframe
    """
    players_tracking_df = pd.json_normalize(
        raw_tracking_data.to_dict("records"),
        "player_data",
        ["frame", "timestamp", "period"],
    ).reset_index().rename(columns={"x": "player_x", "y": "player_y", "index": "player_tracking_id"})
    players_tracking_df["match_id"] = match_id
    columns_ = [
        "player_tracking_id",
        "timestamp",
        "player_x",
        "player_y",
        "period",
        "frame",
        "player_id",
        "match_id"
    ]
    players_tracking_df = players_tracking_df[columns_]
    
    return players_tracking_df


# Deprecated function
def get_match_tracking_data(match_id: str) -> pd.DataFrame:
    """Fetch and preprocess tracking data for a specific match.

    Args:
        match_id (str): The ID of the match to fetch data for.

    Returns:
        pd.DataFrame: Preprocessed tracking data for the specified match.
    """
    content_tracking_url = CONTENT_URL + f"/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    content_tracking_data = pd.read_json(content_tracking_url, lines=True)
    raw_tracking_data = pd.read_json(content_tracking_data['download_url'][0], lines=True)
    processed_tracking_df = preprocess_tracking_data(raw_tracking_data)
    processed_tracking_df["match_id"] = int(match_id)
    
    # Columns to keep
    columns_ = [
        "match_id",
        "player_id",
        "possession_player_id",
        "timestamp",
        "x",
        "y",
        "ball_x",
        "ball_y",
        "ball_z",
        "is_detected_ball",
        "possession_group",
        "period",
        "frame",
    ]
    return processed_tracking_df[columns_]