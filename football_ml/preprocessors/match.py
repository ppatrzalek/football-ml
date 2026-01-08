"""Match Data Preprocessing Module
This module provides functions to preprocess raw football match data into structured
dimension and fact tables following a star schema design pattern. It extracts and
transforms data related to competitions, matches, teams, stadiums, and players.
Functions:
    preprocess_competition_season_data: Creates dimension table for competition seasons
    preprocess_match_data: Creates dimension table for match details
    preprocess_teams_data: Creates dimension table for teams
    preprocess_stadium_data: Creates dimension table for stadiums
    preprocess_players_dimension_data: Creates dimension table for players
    preprocess_match_players_data: Creates fact table linking matches and players
"""
import numpy as np
import pandas as pd

from football_ml.preprocessors.utils import time_to_seconds


def preprocess_competition_season_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a dimension competition season DataFrame."""
    dim_competition_season_columns = [
        "competition_edition.competition.id",
        "competition_edition.season.id",
        "competition_edition.competition.name",
        "competition_edition.season.name",
        "competition_edition.competition.area",
        "competition_edition.name"
    ]
    
    dim_competition_season_df = raw_match_df[dim_competition_season_columns].copy()
    dim_competition_season_df = dim_competition_season_df.rename(
        columns={
            "competition_edition.competition.id": "competition_id",
            "competition_edition.season.id": "season_id",
            "competition_edition.competition.name": "competition_name",
            "competition_edition.season.name": "season_name",
            "competition_edition.competition.area": "competition_area",
            "competition_edition.name": "competition_edition_name",
        }
    )
    
    return dim_competition_season_df


def preprocess_match_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a dimension match DataFrame."""
    dim_match_columns = [
        "id",
        "date_time",
        "home_team.id",
        "away_team.id",
        "home_team_score",
        "away_team_score",
        "home_team_side",
        "match_periods",
        "competition_round.round_number",
        "competition_edition.competition.id",
        "competition_edition.season.id",
    ]

    dim_match_df = raw_match_df[dim_match_columns].copy()
    dim_match_df["home_team_side"] = dim_match_df["home_team_side"].apply(
        lambda x: x[0] if isinstance(x, list) else None
    )
    dim_match_df["match_period_1st_start_frame"] = dim_match_df["match_periods"].apply(
        lambda x: x[0]["start_frame"]
    )
    dim_match_df["match_period_1st_end_frame"] = dim_match_df["match_periods"].apply(
        lambda x: x[0]["end_frame"]
    )
    dim_match_df["match_period_1st_duration"] = dim_match_df["match_periods"].apply(
        lambda x: x[0]["duration_minutes"]
    )
    dim_match_df["match_period_2nd_start_frame"] = dim_match_df["match_periods"].apply(
        lambda x: x[1]["start_frame"]
    )
    dim_match_df["match_period_2nd_end_frame"] = dim_match_df["match_periods"].apply(
        lambda x: x[1]["end_frame"]
    )
    dim_match_df["match_period_2nd_duration"] = dim_match_df["match_periods"].apply(
        lambda x: x[1]["duration_minutes"]
    )
    dim_match_df = dim_match_df.drop(columns=["match_periods"])
    dim_match_df = dim_match_df.rename(columns={"id": "match_id"})
    
    columns_ = [
        "match_id",
        "date_time",
        "home_team.id",
        "away_team.id",
        "home_team_score",
        "away_team_score",
        "home_team_side",
        "match_period_1st_start_frame",
        "match_period_1st_end_frame",
        "match_period_1st_duration",
        "match_period_2nd_start_frame",
        "match_period_2nd_end_frame",
        "match_period_2nd_duration",
        "competition_round.round_number",
        "competition_edition.competition.id",
        "competition_edition.season.id",
    ]
    
    return dim_match_df[columns_]


def preprocess_teams_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a dimension teams DataFrame.
    
    Args:
        raw_match_df: Raw match DataFrame containing home and away team information
        
    Returns:
        DataFrame with unique teams and their attributes
    """
    def get_team_columns(team_prefix: str) -> list:
        """Generate column names for a specific team (home or away)."""
        return [
            f"{team_prefix}_team.id",
            f"{team_prefix}_team.name",
            f"{team_prefix}_team.short_name",
            f"{team_prefix}_team.acronym",
        ]
    
    # Define column mappings
    team_column_mapping = {
        "home_team.id": "team_id",
        "away_team.id": "team_id",
        "home_team.name": "team_name",
        "away_team.name": "team_name",
        "home_team.short_name": "team_short_name",
        "away_team.short_name": "team_short_name",
        "home_team.acronym": "team_acronym",
        "away_team.acronym": "team_acronym",
    }
    
    # Extract home team data
    dim_teams_home_df = raw_match_df[get_team_columns("home")].copy()
    dim_teams_home_df = dim_teams_home_df.rename(
        columns={k: v for k, v in team_column_mapping.items() if k.startswith("home")}
    )
    
    # Extract away team data
    dim_teams_away_df = raw_match_df[get_team_columns("away")].copy()
    dim_teams_away_df = dim_teams_away_df.rename(
        columns={k: v for k, v in team_column_mapping.items() if k.startswith("away")}
    )
    
    # Combine and remove duplicates
    dim_teams_df = pd.concat(
        [dim_teams_home_df, dim_teams_away_df], ignore_index=True
    ).drop_duplicates(subset=["team_id"]).reset_index(drop=True)
    
    return dim_teams_df


def preprocess_stadium_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a dimension stadium DataFrame.
    
    Args:
        raw_match_df: Raw match DataFrame containing stadium information
        
    Returns:
        DataFrame with stadium information and their attributes
    """
    dim_stadium_columns = [
        "stadium.id",
        "stadium.name",
        "stadium.city",
        "stadium.capacity",
        "pitch_length",
        "pitch_width",
        "home_team.id",
    ]
    
    dim_stadium_df = raw_match_df[dim_stadium_columns].copy()
    dim_stadium_df = dim_stadium_df.rename(
        columns={
            "stadium.id": "stadium_id",
            "stadium.name": "stadium_name",
            "stadium.city": "stadium_city",
            "stadium.capacity": "stadium_capacity",
            "pitch_length": "pitch_length_meters",
            "pitch_width": "pitch_width_meters",
            "home_team.id": "team_id",
        }
    )
    
    return dim_stadium_df


def preprocess_players_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a dimension players DataFrame.
    
    Args:
        raw_match_df: Raw match DataFrame containing players information
        
    Returns:
        DataFrame with players information and their attributes
    """
    dim_players_columns = [
        "id",
        "first_name",
        "last_name",
        "short_name",
        "team_player_id",
        "team_id",
        "gender",
        "birthday",
        "player_role.id",
        "player_role.name",
        "player_role.acronym",
        "player_role.position_group",
    ]
    
    dim_players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
    )[dim_players_columns]
    
    dim_players_df = dim_players_df.rename(
        columns={
            "id": "player_id",
            "player_role.id": "player_role_id",
            "first_name": "player_first_name",
            "last_name": "player_last_name",
            "short_name": "player_short_name",
            "gender": "player_gender",
            "birthday": "player_birthday",
        }
    )
    
    return dim_players_df


def preprocess_player_position_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    dim_player_position_columns = [
        "player_role.id",
        "player_role.name",
        "player_role.acronym",
        "player_role.position_group",
    ]
    dim_player_position_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
    )[dim_player_position_columns]
    dim_player_position_df = dim_player_position_df.rename(
        columns={
            "player_role.id": "player_role_id",
            "player_role.name": "player_role_name",
            "player_role.acronym": "player_role_acronym",
            "player_role.position_group": "player_position_group",
        }
    )
    return dim_player_position_df


def preprocess_match_players_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses raw match data to create a fact match players DataFrame.
    
    Args:
        raw_match_df: Raw match DataFrame containing players information
        
    Returns:
        DataFrame with match players fact table
    """
    fct_match_players_columns = [
        "id",
        "team_id",
        "start_time",
        "end_time",
        "number",
        "yellow_card",
        "red_card",
        "injured",
        "goal",
        "own_goal",
        "playing_time.total.minutes_tip",
        "playing_time.total.minutes_otip",
        "playing_time.total.start_frame",
        "playing_time.total.end_frame",
        "playing_time.total.minutes_played",
        "playing_time.total.minutes_played_regular_time",
        "playing_time.total",
        "player_role.id",
    ]
    
    fct_match_players_df = pd.json_normalize(
        raw_match_df.to_dict("records"),
        record_path="players",
    )[fct_match_players_columns]
    
    fct_match_players_df["match_id"] = raw_match_df["id"].values[0]
    fct_match_players_df = fct_match_players_df.rename(
        columns={
            "id": "player_id",
            "player_role.id": "player_role_id",
            "playing_time.total.minutes_tip": "minutes_tip",
            "playing_time.total.minutes_otip": "minutes_otip",
            "playing_time.total.minutes_played": "minutes_played",
            "playing_time.total": "minutes_total",
            "playing_time.total.start_frame": "playing_start_frame",
            "playing_time.total.end_frame": "playing_end_frame",
            "playing_time.total.minutes_played_regular_time": "minutes_played_regular_time",
        }
    )
    
    return fct_match_players_df


# Deprecated function
def DEPPRECATED_preprocess_players_data(raw_match_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess match data by converting time strings to seconds.

    Args:
        raw_match_df (pd.DataFrame): Raw match data containing time strings.
    Returns:
        pd.DataFrame: Preprocessed players data with time in seconds.
    """
    # Extract relevant fields from nested JSON structures
    df = pd.json_normalize(
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
    df["match_id"] = raw_match_df["id"].iloc[0]
    df = df.rename(columns={"id": "player_id"})

    # Take only players who played and create their total time
    mask = (df.start_time.isna()) & (df.end_time.isna())
    df = df[~mask]
    df["total_time"] = df["end_time"].apply(time_to_seconds) - df["start_time"].apply(time_to_seconds)
    
    # Create a flag for GK
    df["is_gk"] = df["player_role.acronym"] == "GK"
    # Add a match name column
    df["match_name"] = (
        df["home_team.name"] + " vs " + df["away_team.name"]
    )
    # Add a flag if the given player is home or away
    df["home_away_player"] = np.where(
        df.team_id == df["home_team.id"], "Home", "Away"
    )
    # Create flag from player
    df["team_name"] = np.where(
        df.team_id == df["home_team.id"],
        df["home_team.name"],
        df["away_team.name"],
    )
    # Figure out sides
    df[["home_team_side_1st_half", "home_team_side_2nd_half"]] = (
        df["home_team_side"]
        .astype(str)
        .str.strip("[]")
        .str.replace("'", "")
        .str.split(", ", expand=True)
    )
    # Clean up sides
    df["direction_player_1st_half"] = np.where(
        df.home_away_player == "Home",
        df.home_team_side_1st_half,
        df.home_team_side_2nd_half,
    )
    df["direction_player_2nd_half"] = np.where(
        df.home_away_player == "Home",
        df.home_team_side_2nd_half,
        df.home_team_side_1st_half,
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
        "player_id",
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
    df = df[columns_to_keep]
    return df