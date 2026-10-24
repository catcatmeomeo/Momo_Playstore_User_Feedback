"""Crawl every MoMo e-wallet review on Google Play.

Reviews are fetched in batches with a continuation token. Progress is
checkpointed to a separate ``.partial`` file and only moved over the real
dataset once the crawl completes, so an interrupted run never overwrites a
previously complete dataset.

    python -m momo_reviews.crawl            # crawl everything
    python -m momo_reviews.crawl --max 5000 # quick run: 5,000 newest reviews
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from google_play_scraper import Sort, reviews

from .config import APP_ID, COUNTRY, LANG, RAW_DIR, RAW_FILE

PARTIAL_FILE = RAW_FILE.with_suffix(".partial.parquet")
log = logging.getLogger(__name__)

# userName / userImage are personal data and not needed for analysis -> never stored.
KEEP_COLS = [
    "reviewId", "content", "score", "thumbsUpCount", "reviewCreatedVersion",
    "at", "replyContent", "repliedAt", "appVersion",
]


def crawl(max_reviews: int | None = None, batch_size: int = 200,
          sleep_s: float = 0.5, checkpoint_every: int = 50) -> pd.DataFrame:
    rows: list[dict] = []
    token = None
    n_batches = 0
    while True:
        for attempt in range(5):
            try:
                batch, token = reviews(
                    APP_ID, lang=LANG, country=COUNTRY, sort=Sort.NEWEST,
                    count=batch_size, continuation_token=token,
                )
                break
            except Exception as exc:  # network error / rate limit -> back off
                wait = 2 ** attempt
                log.warning("Batch failed (%s), retrying in %ss", exc, wait)
                time.sleep(wait)
        else:
            # Keep the previous complete dataset untouched; progress stays in PARTIAL_FILE.
            raise RuntimeError(f"Giving up after 5 attempts; partial crawl kept in {PARTIAL_FILE}")

        rows.extend({k: r.get(k) for k in KEEP_COLS} for r in batch)
        n_batches += 1
        if n_batches % checkpoint_every == 0:
            _save(rows, PARTIAL_FILE)
            log.info("Crawled %d reviews", len(rows))

        if not batch or token is None or token.token is None:
            break
        if max_reviews and len(rows) >= max_reviews:
            break
        time.sleep(sleep_s)

    df = _save(rows, PARTIAL_FILE)
    PARTIAL_FILE.replace(RAW_FILE)  # atomic swap only after a complete crawl
    return df


def _save(rows: list[dict], path) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=KEEP_COLS).drop_duplicates("reviewId")
    df.to_parquet(path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=None, help="Maximum number of reviews")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = crawl(max_reviews=args.max)
    log.info("Done: %d reviews -> %s", len(df), RAW_FILE)


if __name__ == "__main__":
    main()
