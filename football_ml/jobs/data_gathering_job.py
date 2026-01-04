"""Data gathering job for football ML project."""
import pandas as pd

from football_ml.config import CONTENT_URL, RAW_DATA_DIR
from football_ml.config import logger
from football_ml.preprocessors.base import (
    get_available_matches,
    get_match_meta_data,
)
from football_ml.preprocessors.tracking import (
    preprocess_player_tracking,
    preprocess_ball_tracking,
)
from football_ml.preprocessors.match import (
    preprocess_competition_season_data,
    preprocess_match_data,
    preprocess_teams_data,
    preprocess_stadium_data,
    preprocess_players_data,
    preprocess_match_players_data,
    preprocess_player_position_data,
)

def main():
    """Main function to gather and preprocess football match tracking data."""
    logger.info("Starting data gathering job...")

    # Fetch available matches   
    matches_df = get_available_matches(CONTENT_URL)

    # Display available match IDs
    match_ids = [match['name'] for match in matches_df.to_dict('records') if match['type'] == 'dir']
    logger.info(f"Available matches: {len(match_ids)}")
    
    fct_track_players, fct_track_ball, dim_competition_season, dim_match, dim_team, dim_stadium, dim_players, fct_match_players, dim_players_position = [], [], [], [], [], [], [], [], []

    for match_id in match_ids:
        logger.info(f"Match ID: {match_id}")
        content_tracking_url = CONTENT_URL + f"/{match_id}/{match_id}_tracking_extrapolated.jsonl"
        content_tracking_data = pd.read_json(content_tracking_url, lines=True)
        raw_tracking_data = pd.read_json(content_tracking_data['download_url'][0], lines=True)
        try:
            fct_track_players.append(
                preprocess_player_tracking(raw_tracking_data, match_id)
            )
            fct_track_ball.append(
                preprocess_ball_tracking(raw_tracking_data, match_id)
            )
        
            logger.info(f"Processed match ID: {match_id}")
        except Exception as e:
            logger.error(f"Failed to process match ID: {match_id}. Error: {e}")
            
        try:
            raw_match_df = get_match_meta_data(match_id)
            dim_competition_season.append(
                preprocess_competition_season_data(raw_match_df)
            )
            dim_match.append(
                preprocess_match_data(raw_match_df)
            )
            dim_team.append(
                preprocess_teams_data(raw_match_df)
            )
            dim_stadium.append(
                preprocess_stadium_data(raw_match_df)
            )
            dim_players.append(
                preprocess_players_data(raw_match_df)
            )
            dim_players_position.append(
                preprocess_player_position_data(raw_match_df)
            )
            fct_match_players.append(
                preprocess_match_players_data(raw_match_df)
            )
                        
            logger.info(f"Processed players data for match ID: {match_id}")
        except Exception as e:
            logger.error(f"Failed to process players data for match ID: {match_id}. Error: {e}")
    
    fct_track_players_df = pd.concat(fct_track_players, ignore_index=True)
    fct_track_ball_df = pd.concat(fct_track_ball, ignore_index=True)
    dim_competition_season_df = pd.concat(dim_competition_season, ignore_index=True).drop_duplicates().reset_index(drop=True)
    dim_match_df = pd.concat(dim_match, ignore_index=True)
    dim_team_df = pd.concat(dim_team, ignore_index=True).drop_duplicates().reset_index(drop=True)
    dim_stadium_df = pd.concat(dim_stadium, ignore_index=True).drop_duplicates().reset_index(drop=True)
    dim_players_df = pd.concat(dim_players, ignore_index=True).drop_duplicates().reset_index(drop=True)
    dim_players_position_df = pd.concat(dim_players_position, ignore_index=True).drop_duplicates().reset_index(drop=True)
    fct_match_players_df = pd.concat(fct_match_players, ignore_index=True)
    logger.info("All matches processed successfully.")
    
    # Write into parquet file
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RAW_DATA_DIR / "fct_players_tracking.parquet"
    fct_track_players_df.to_parquet(output_file, index=False)
    logger.info(f"Processed tracking data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "fct_ball_tracking.parquet"
    fct_track_ball_df.to_parquet(output_file, index=False)
    logger.info(f"Processed tracking data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "fct_match_players.parquet"
    fct_match_players_df.to_parquet(output_file, index=False)
    logger.info(f"Processed players data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_competition_season.parquet"
    dim_competition_season_df.to_parquet(output_file, index=False)
    logger.info(f"Processed competition season data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_match.parquet"
    dim_match_df.to_parquet(output_file, index=False)
    logger.info(f"Processed match data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_team.parquet"
    dim_team_df.to_parquet(output_file, index=False)
    logger.info(f"Processed team data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_stadium.parquet"
    dim_stadium_df.to_parquet(output_file, index=False)
    logger.info(f"Processed stadium data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_players.parquet"
    dim_players_df.to_parquet(output_file, index=False)
    logger.info(f"Processed players dimension data saved to {output_file}")
    
    output_file = RAW_DATA_DIR / "dim_players_position.parquet"
    dim_players_position_df.to_parquet(output_file, index=False)
    logger.info(f"Processed players position data saved to {output_file}")
    
    logger.info("Data gathering job completed successfully.")
    

if __name__ == "__main__":
    main()
