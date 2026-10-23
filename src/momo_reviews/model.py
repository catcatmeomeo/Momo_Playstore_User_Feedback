"""Mô hình phân loại cảm xúc review (negative / neutral / positive).

Nhãn được suy ra từ số sao (weak supervision): 1-2 sao = negative, 3 = neutral,
4-5 = positive. Chia train/test THEO THỜI GIAN (test = 20% review mới nhất) để
đánh giá đúng kịch bản thực tế: train trên quá khứ, dự đoán review tương lai.

    python -m momo_reviews.model
"""
from __future__ import annotations

import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import CLEAN_FILE, MODEL_FILE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, SENTIMENT_ORDER

log = logging.getLogger(__name__)


def load_modeling_data(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ review rỗng và spam quảng cáo, giữ thứ tự thời gian."""
    out = df[(df["n_words"] > 0) & (~df["is_spam"])]
    return out.sort_values("at").reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    cut = int(len(df) * (1 - test_size))
    return df.iloc[:cut], df.iloc[cut:]


def word_vectorizer() -> TfidfVectorizer:
    # Tiếng Việt là ngôn ngữ đơn lập: 1 "từ" thường gồm 2 âm tiết (chuyển khoản,
    # tài khoản). Bigram âm tiết vì vậy xấp xỉ word segmentation mà không cần
    # thư viện tách từ.
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9,
                           sublinear_tf=True, max_features=150_000)


def char_vectorizer() -> TfidfVectorizer:
    # Char n-gram bền với lỗi chính tả / thiếu dấu ("chuyen tien", "chuyển tìên").
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                           sublinear_tf=True, max_features=200_000)


def candidate_models() -> dict[str, Pipeline]:
    words_chars = FeatureUnion([("word", word_vectorizer()), ("char", char_vectorizer())])
    return {
        "Majority baseline": Pipeline([("tfidf", word_vectorizer()),
                                       ("clf", DummyClassifier(strategy="most_frequent"))]),
        "TF-IDF word + ComplementNB": Pipeline([("tfidf", word_vectorizer()),
                                                ("clf", ComplementNB(alpha=0.3))]),
        "TF-IDF word + LogReg": Pipeline([("tfidf", word_vectorizer()),
                                          ("clf", LogisticRegression(C=4, max_iter=2000,
                                                                     class_weight="balanced"))]),
        "TF-IDF word+char + LogReg": Pipeline([("tfidf", clone(words_chars)),
                                               ("clf", LogisticRegression(C=4, max_iter=2000,
                                                                          class_weight="balanced"))]),
        "TF-IDF word+char + LinearSVC": Pipeline([("tfidf", clone(words_chars)),
                                                  ("clf", LinearSVC(C=0.3, class_weight="balanced",
                                                                    random_state=RANDOM_STATE))]),
    }


def evaluate(models: dict[str, Pipeline], train: pd.DataFrame, test: pd.DataFrame):
    X_tr, y_tr = train["text_clean"], train["sentiment"]
    X_te, y_te = test["text_clean"], test["sentiment"]
    rows, fitted = [], {}
    for name, pipe in models.items():
        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "model": name,
            "macro_f1": f1_score(y_te, pred, average="macro"),
            "weighted_f1": f1_score(y_te, pred, average="weighted"),
            "f1_negative": f1_score(y_te, pred, labels=["negative"], average="macro"),
            "f1_neutral": f1_score(y_te, pred, labels=["neutral"], average="macro"),
            "f1_positive": f1_score(y_te, pred, labels=["positive"], average="macro"),
            "train_seconds": time.time() - t0,
        })
        fitted[name] = pipe
        log.info("%-32s macro-F1 = %.3f", name, rows[-1]["macro_f1"])
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False), fitted


def select_model(results: pd.DataFrame, tolerance: float = 0.005) -> str:
    """Chọn mô hình huấn luyện nhanh nhất trong khoảng `tolerance` macro-F1 so với tốt nhất.

    Chênh lệch < 0.5 điểm F1 nằm trong nhiễu của việc chia dữ liệu, trong khi
    mô hình nhanh hơn rẻ hơn khi retrain và (với LogReg) cho xác suất dự đoán.
    """
    ok = results[results["macro_f1"] >= results["macro_f1"].max() - tolerance]
    return ok.sort_values("train_seconds").iloc[0]["model"]


def save_final_model(pipe: Pipeline, df: pd.DataFrame) -> Pipeline:
    """Refit trên toàn bộ dữ liệu rồi lưu để dùng cho inference (app)."""
    final = clone(pipe).fit(df["text_clean"], df["sentiment"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_FILE, compress=3)
    return final


def top_features(pipe: Pipeline, n: int = 20) -> pd.DataFrame:
    """Các n-gram có trọng số lớn nhất cho từng lớp (chỉ cho mô hình tuyến tính)."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.asarray(vec.get_feature_names_out())
    out = {}
    for i, cls in enumerate(clf.classes_):
        idx = np.argsort(clf.coef_[i])[::-1][:n]
        out[cls] = [n.split("__", 1)[-1] for n in names[idx]]
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    df = load_modeling_data(pd.read_parquet(CLEAN_FILE))
    train, test = temporal_split(df)
    log.info("Train %d (%s -> %s) | Test %d (%s -> %s)", len(train), train["at"].min().date(),
             train["at"].max().date(), len(test), test["at"].min().date(), test["at"].max().date())

    results, fitted = evaluate(candidate_models(), train, test)
    best_name = select_model(results)
    best = fitted[best_name]
    pred = best.predict(test["text_clean"])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    metrics = {
        "selected_model": best_name,
        "n_train": len(train), "n_test": len(test),
        "test_period": [str(test["at"].min().date()), str(test["at"].max().date())],
        "report": classification_report(test["sentiment"], pred, output_dict=True),
        "confusion_matrix": confusion_matrix(test["sentiment"], pred, labels=SENTIMENT_ORDER).tolist(),
        "labels": SENTIMENT_ORDER,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    save_final_model(best, df)
    log.info("Best: %s -> %s", best_name, MODEL_FILE)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
