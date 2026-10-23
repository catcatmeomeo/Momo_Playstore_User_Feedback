"""Làm sạch dữ liệu thô -> dataset phân tích/huấn luyện.

    python -m momo_reviews.preprocess
"""
from __future__ import annotations

import logging

import pandas as pd

from .config import CLEAN_FILE, PROCESSED_DIR, RAW_FILE, SAMPLE_DIR, SAMPLE_FILE, RANDOM_STATE
from .text import is_spam, normalize

log = logging.getLogger(__name__)


def score_to_sentiment(score: int) -> str:
    if score >= 4:
        return "positive"
    if score == 3:
        return "neutral"
    return "negative"


def major_version(v: str | None) -> str | None:
    if not isinstance(v, str) or not v:
        return None
    return v.split(".")[0]


def build_clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop_duplicates("reviewId").copy()
    df["at"] = pd.to_datetime(df["at"])
    df["repliedAt"] = pd.to_datetime(df["repliedAt"])

    # Review không có appVersion vẫn giữ lại cho phân tích text; chỉ phân tích
    # theo version mới cần lọc. (Notebook cũ drop toàn bộ -> mất ~10% dữ liệu.)
    df["version"] = df["appVersion"].fillna(df["reviewCreatedVersion"])
    df["major_version"] = df["version"].map(major_version)

    df["content"] = df["content"].fillna("").astype(str)
    df["text_clean"] = df["content"].map(normalize)
    df["n_words"] = df["text_clean"].str.split().str.len()
    df["is_spam"] = df["content"].map(is_spam)

    df["sentiment"] = df["score"].map(score_to_sentiment)
    df["has_reply"] = df["replyContent"].notna()
    df["reply_hours"] = (df["repliedAt"] - df["at"]).dt.total_seconds() / 3600

    df["year"] = df["at"].dt.year
    df["month"] = df["at"].dt.to_period("M").dt.to_timestamp()
    df["hour"] = df["at"].dt.hour
    df["weekday"] = df["at"].dt.day_of_week
    return df.sort_values("at").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    raw = pd.read_parquet(RAW_FILE)
    df = build_clean(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_FILE, index=False)
    log.info("Clean: %d -> %d review, lưu %s", len(raw), len(df), CLEAN_FILE)

    # Sample nhỏ commit lên GitHub để người xem chạy thử pipeline không cần crawl.
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample = df.groupby("sentiment", group_keys=False).sample(n=1000, random_state=RANDOM_STATE)
    sample[["reviewId", "content", "score", "at", "version", "thumbsUpCount"]].to_csv(SAMPLE_FILE, index=False)
    log.info("Sample %d dòng -> %s", len(sample), SAMPLE_FILE)


if __name__ == "__main__":
    main()
