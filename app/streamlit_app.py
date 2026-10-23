"""Demo: phân loại cảm xúc + gắn khía cạnh cho review ví MoMo.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
import streamlit as st

from momo_reviews.aspects import tag_aspects
from momo_reviews.config import MODEL_FILE
from momo_reviews.text import is_spam, normalize

LABEL_VI = {"negative": "😠 Tiêu cực", "neutral": "😐 Trung lập", "positive": "😊 Tích cực"}

st.set_page_config(page_title="MoMo Review Analyzer", page_icon="💬", layout="centered")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


def analyze(texts: pd.Series, model) -> pd.DataFrame:
    clean = texts.fillna("").astype(str).map(normalize)
    return pd.DataFrame({
        "review": texts,
        "sentiment": model.predict(clean),
        "aspects": clean.map(lambda t: ", ".join(tag_aspects(t))),
        "spam": texts.map(is_spam),
    })


st.title("💬 MoMo Review Analyzer")
st.caption("Mô hình TF-IDF + linear classifier huấn luyện trên review ví MoMo trên Google Play. "
           "Xem chi tiết tại notebooks/02_sentiment_model.ipynb.")

if not MODEL_FILE.exists():
    st.error("Chưa có model. Chạy `make data model` trước.")
    st.stop()
model = load_model()

tab1, tab2 = st.tabs(["Một review", "Nhiều review (CSV)"])

with tab1:
    text = st.text_area("Nhập review", "Chuyển tiền bị trừ tiền mà người nhận không nhận được, gọi tổng đài không ai nghe")
    if st.button("Phân tích", type="primary") and text.strip():
        res = analyze(pd.Series([text]), model).iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Cảm xúc", LABEL_VI[res["sentiment"]])
        c2.metric("Spam quảng cáo?", "Có" if res["spam"] else "Không")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([normalize(text)])[0]
            st.bar_chart(pd.Series(proba, index=[LABEL_VI[c] for c in model.classes_], name="xác suất"))
        st.write("**Khía cạnh được nhắc tới:**", res["aspects"] or "—")
        st.caption(f"Văn bản sau chuẩn hoá: `{normalize(text)}`")

with tab2:
    up = st.file_uploader("CSV có cột `content`", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "content" not in df.columns:
            st.error("File cần có cột `content`.")
        else:
            out = analyze(df["content"], model)
            st.bar_chart(out["sentiment"].value_counts())
            aspects = out["aspects"].str.split(", ").explode().replace("", pd.NA).dropna()
            st.write("**Khía cạnh phổ biến**")
            st.bar_chart(aspects.value_counts())
            st.dataframe(out, use_container_width=True)
            st.download_button("Tải kết quả", out.to_csv(index=False).encode("utf-8-sig"),
                               "momo_reviews_scored.csv", "text/csv")
