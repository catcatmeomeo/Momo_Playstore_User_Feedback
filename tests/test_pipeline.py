import pandas as pd
import pytest

from momo_reviews.aspects import tag_aspects
from momo_reviews.model import candidate_models, temporal_split
from momo_reviews.preprocess import build_clean, score_to_sentiment
from momo_reviews.text import is_spam, normalize


@pytest.mark.parametrize("raw, expected", [
    ("Ko chuyển tiền dc!!!", "không chuyển tiền được"),
    ("App quá tệ😡", "ứng dụng quá tệ neg_emoji"),
    ("mất 500k rồi", "mất num rồi"),
    ("Tốtttt", "tốt"),
    (None, ""),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_spam_detection():
    assert is_spam("Good app, vào ( W E B ) VAY9 . C0M lấy 9tr")
    assert is_spam("Mọi người muốn kiếm thêm thì tìm TinChac nha")
    assert not is_spam("web lag quá, không đăng nhập được")
    assert not is_spam("nhập mã OTP mãi không được")


def test_score_to_sentiment():
    assert [score_to_sentiment(s) for s in range(1, 6)] == \
        ["negative", "negative", "neutral", "positive", "positive"]


def test_tag_aspects():
    tags = tag_aspects(normalize("Chuyển tiền bị trừ tiền, gọi tổng đài ko ai nghe"))
    assert {"Money transfer", "Lost money & refunds", "Customer support"} <= set(tags)
    assert "Loans & pay-later" in tag_aspects(normalize("Vay trả sau bị tính phí cao"))
    assert tag_aspects("") == []


@pytest.fixture
def raw_df():
    texts = ["ứng dụng tốt tiện lợi", "lỗi không chuyển tiền được", "bình thường",
             "rất tốt nhanh", "mất tiền lừa đảo", "tạm được"] * 20
    scores = [5, 1, 3, 5, 1, 3] * 20
    n = len(texts)
    return pd.DataFrame({
        "reviewId": [f"id{i}" for i in range(n)], "content": texts, "score": scores,
        "thumbsUpCount": 0, "reviewCreatedVersion": "5.1.0",
        "at": pd.date_range("2024-01-01", periods=n, freq="D"),
        "replyContent": None, "repliedAt": pd.NaT, "appVersion": "5.1.0",
    })


def test_build_clean_and_split(raw_df):
    df = build_clean(raw_df)
    assert {"text_clean", "sentiment", "major_version", "is_spam"} <= set(df.columns)
    assert df["at"].is_monotonic_increasing
    train, test = temporal_split(df, 0.25)
    assert train["at"].max() <= test["at"].min()


def test_models_fit(raw_df):
    df = build_clean(raw_df)
    pipe = candidate_models()["TF-IDF word + LogReg"]
    pipe.set_params(tfidf__min_df=1)
    pipe.fit(df["text_clean"], df["sentiment"])
    assert pipe.predict(["lỗi không chuyển tiền được"])[0] == "negative"
