import requests
import pandas as pd

from football_ml.config import CONTENT_URL


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