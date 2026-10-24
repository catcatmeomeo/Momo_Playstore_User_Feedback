"""Demo: sentiment classification + aspect tagging for MoMo e-wallet reviews.

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

LABELS = {"negative": "😠 Negative", "neutral": "😐 Neutral", "positive": "😊 Positive"}

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
st.caption("TF-IDF + linear classifier trained on 217k Vietnamese Google Play reviews of the MoMo e-wallet. "
           "Details in notebooks/02_sentiment_model.ipynb. Reviews must be written in Vietnamese.")

if not MODEL_FILE.exists():
    st.error("No model found. Run `make data model` first.")
    st.stop()
model = load_model()

tab1, tab2 = st.tabs(["Single review", "Batch (CSV)"])

with tab1:
    text = st.text_area("Review (Vietnamese)",
                        "Chuyển tiền bị trừ tiền mà người nhận không nhận được, gọi tổng đài không ai nghe")
    st.caption("Example above: *“Money was deducted but the recipient never got it, and nobody answers the hotline.”*")
    if st.button("Analyze", type="primary") and text.strip():
        res = analyze(pd.Series([text]), model).iloc[0]
        c1, c2 = st.columns(2)
        c1.metric("Sentiment", LABELS[res["sentiment"]])
        c2.metric("Promotional spam?", "Yes" if res["spam"] else "No")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([normalize(text)])[0]
            st.bar_chart(pd.Series(proba, index=[LABELS[c] for c in model.classes_], name="probability"))
        st.write("**Aspects mentioned:**", res["aspects"] or "—")
        st.caption(f"Normalised text: `{normalize(text)}`")

with tab2:
    up = st.file_uploader("CSV with a `content` column (try data/sample/momo_reviews_sample.csv)", type="csv")
    if up is not None:
        df = pd.read_csv(up)
        if "content" not in df.columns:
            st.error("The file needs a `content` column.")
        else:
            out = analyze(df["content"], model)
            st.write("**Predicted sentiment**")
            st.bar_chart(out["sentiment"].value_counts())
            aspects = out["aspects"].str.split(", ").explode().replace("", pd.NA).dropna()
            st.write("**Most mentioned aspects**")
            st.bar_chart(aspects.value_counts())
            st.dataframe(out, use_container_width=True)
            st.download_button("Download results", out.to_csv(index=False).encode("utf-8-sig"),
                               "momo_reviews_scored.csv", "text/csv")
