"""Crawl toàn bộ review của ví MoMo trên Google Play Store.

Crawl theo batch kèm continuation token và ghi checkpoint định kỳ, nên nếu bị
ngắt giữa chừng có thể chạy lại mà không mất dữ liệu đã lấy.

    python -m momo_reviews.crawl            # crawl toàn bộ
    python -m momo_reviews.crawl --max 5000 # crawl thử 5000 review mới nhất
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
from google_play_scraper import Sort, reviews

from .config import APP_ID, COUNTRY, LANG, RAW_DIR, RAW_FILE

log = logging.getLogger(__name__)

# userName / userImage là dữ liệu cá nhân, không cần cho phân tích -> không lưu.
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
            except Exception as exc:  # lỗi mạng / rate limit -> backoff
                wait = 2 ** attempt
                log.warning("Batch lỗi (%s), thử lại sau %ss", exc, wait)
                time.sleep(wait)
        else:
            log.error("Bỏ cuộc sau 5 lần thử, lưu phần đã crawl.")
            break

        rows.extend({k: r.get(k) for k in KEEP_COLS} for r in batch)
        n_batches += 1
        if n_batches % checkpoint_every == 0:
            _save(rows)
            log.info("Đã crawl %d review", len(rows))

        if not batch or token is None or token.token is None:
            break
        if max_reviews and len(rows) >= max_reviews:
            break
        time.sleep(sleep_s)

    return _save(rows)


def _save(rows: list[dict]) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=KEEP_COLS).drop_duplicates("reviewId")
    df.to_parquet(RAW_FILE, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=None, help="Số review tối đa")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = crawl(max_reviews=args.max)
    log.info("Hoàn tất: %d review -> %s", len(df), RAW_FILE)


if __name__ == "__main__":
    main()
