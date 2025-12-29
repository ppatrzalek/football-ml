from pathlib import Path

CONTENT_URL = "https://api.github.com/repos/SkillCorner/opendata/contents/data/matches"
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"