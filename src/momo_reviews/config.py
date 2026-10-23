"""Đường dẫn và hằng số dùng chung cho toàn bộ project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_FILE = RAW_DIR / "momo_reviews_raw.parquet"
CLEAN_FILE = PROCESSED_DIR / "momo_reviews_clean.parquet"
SAMPLE_FILE = SAMPLE_DIR / "momo_reviews_sample.csv"
MODEL_FILE = MODELS_DIR / "sentiment_tfidf_linear.joblib"

APP_ID = "com.mservice.momotransfer"
LANG = "vi"
COUNTRY = "vn"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
RANDOM_STATE = 42
