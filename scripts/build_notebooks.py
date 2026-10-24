"""Generate the three analysis notebooks from source kept here (small, reviewable git diffs).

    python scripts/build_notebooks.py        # build all, then `make notebooks` to execute
    python scripts/build_notebooks.py 01     # build only notebook 01
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """\
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from IPython.display import display, Markdown

from momo_reviews.config import CLEAN_FILE
from momo_reviews.viz import set_style, save, SENTIMENT_COLORS, BLUE, INK_2
set_style()
pd.set_option("display.max_colwidth", 160)

df = pd.read_parquet(CLEAN_FILE)
print(f"{len(df):,} reviews | {df['at'].min():%Y-%m-%d} → {df['at'].max():%Y-%m-%d}")"""

def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# ----------------------------------------------------------------------------
# 01 — EDA & business insights (Data Analyst)
# ----------------------------------------------------------------------------
eda = [
    md("""
# 01 · EDA & Business Insights — MoMo e-wallet reviews on Google Play

**Business questions**
1. How does user satisfaction evolve over time and across app versions?
2. Which releases made the experience worse (early-warning signal)?
3. How quickly and how often does MoMo respond to users?
4. What is unusual about the rating distribution (polarisation, spam) that matters when ratings are used as ML labels?

Data: every public review, crawled with `google-play-scraper` (`src/momo_reviews/crawl.py`) and cleaned by `src/momo_reviews/preprocess.py`. User names and avatars are never stored. Review text is in Vietnamese.
"""),
    code(SETUP),
    md("## 1. Data overview & KPIs"),
    code("""
kpi = pd.Series({
    "Reviews": f"{len(df):,}",
    "Average rating": f"{df['score'].mean():.2f} ★",
    "% negative (1–2★)": f"{(df['sentiment']=='negative').mean():.1%}",
    "% positive (4–5★)": f"{(df['sentiment']=='positive').mean():.1%}",
    "% with text": f"{(df['n_words']>0).mean():.1%}",
    "% replied by MoMo": f"{df['has_reply'].mean():.1%}",
    "% promotional spam": f"{df['is_spam'].mean():.2%}",
    "% missing appVersion": f"{df['version'].isna().mean():.1%}",
}, name="Value")
kpi.to_frame()
"""),
    code("""
missing = df[["content", "version", "replyContent", "thumbsUpCount"]].isna().mean().rename("missing rate")
missing.to_frame().style.format("{:.1%}")
"""),
    md("## 2. Rating distribution — a polarised \"J-shape\""),
    code("""
dist = df["score"].value_counts().sort_index()
colors = [SENTIMENT_COLORS[s] for s in ["negative", "negative", "neutral", "positive", "positive"]]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(dist.index.astype(str) + "★", dist.values, color=colors, width=0.6)
for b, v in zip(bars, dist.values):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v/dist.sum():.0%}", ha="center", va="bottom", fontsize=9, color=INK_2)
ax.set_title("Star rating distribution"); ax.set_ylabel("Reviews")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.grid(axis="x", visible=False)
save(fig, "01_rating_distribution"); plt.show()
"""),
    md("""
**Takeaway.** Ratings are strongly polarised: lots of 5★ and 1★, very few 2–3★ — typical for app reviews (people write when they are delighted or furious).
Implication for ML: the *neutral* class is rare → use **macro-F1** instead of accuracy and balance class weights.
"""),
    md("## 3. Trends over time"),
    code("""
monthly = df.groupby("month").agg(n=("reviewId", "size"), avg=("score", "mean"), neg=("sentiment", lambda s: (s == "negative").mean()))
monthly = monthly[monthly.index < monthly.index.max()]  # drop the current, incomplete month

fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
axes[0].bar(monthly.index, monthly["n"], width=25, color=BLUE, alpha=0.85)
axes[0].set_title("Reviews per month"); axes[0].grid(axis="x", visible=False)
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

axes[1].plot(monthly.index, monthly["neg"], color=SENTIMENT_COLORS["negative"], alpha=0.35, linewidth=1)
axes[1].plot(monthly.index, monthly["neg"].rolling(3, center=True).mean(),
             color=SENTIMENT_COLORS["negative"], label="3-month rolling average")
axes[1].set_title("Share of negative reviews (1–2★) per month")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1)); axes[1].legend(loc="upper left")
fig.tight_layout(); save(fig, "01_monthly_trend"); plt.show()
"""),
    code("""
yearly = df.groupby("year").agg(reviews=("reviewId", "size"), avg_rating=("score", "mean"),
                                negative_share=("sentiment", lambda s: (s == "negative").mean()),
                                reply_rate=("has_reply", "mean"))
yearly["yoy_growth"] = yearly["reviews"].pct_change()
yearly.style.format({"reviews": "{:,}", "avg_rating": "{:.2f}", "negative_share": "{:.1%}",
                     "reply_rate": "{:.1%}", "yoy_growth": "{:+.0%}"}, na_rep="—") \\
      .background_gradient(subset=["negative_share"], cmap="Reds")
"""),
    code("""
roll = monthly["neg"].rolling(12, min_periods=6)
spikes = monthly.assign(z=(monthly["neg"] - roll.mean().shift(1)) / roll.std().shift(1))
display(Markdown("**Months where the negative share spiked (z-score vs. the previous 12 months > 2):**"))
spikes[spikes["z"] > 2][["n", "avg", "neg", "z"]].sort_values("z", ascending=False).head(10) \\
    .style.format({"n": "{:,}", "avg": "{:.2f}", "neg": "{:.1%}", "z": "{:.1f}"})
"""),
    md("""
**Use case.** This table is a simple **anomaly alert**: run weekly, it tells the product team as soon as an incident or release pushes complaints up — before the overall Store rating moves.
"""),
    md("## 4. Ratings by app version"),
    code("""
by_major = (df.dropna(subset=["major_version"])
              .groupby("major_version")
              .agg(n=("reviewId", "size"), rating=("score", "mean"), neg=("sentiment", lambda s: (s == "negative").mean()),
                   first_seen=("at", "min"), last_seen=("at", "max")))
by_major = by_major[by_major["n"] >= 200].sort_index(key=lambda i: i.astype(int))
by_major.style.format({"n": "{:,}", "rating": "{:.2f}", "neg": "{:.1%}",
                       "first_seen": "{:%Y-%m}", "last_seen": "{:%Y-%m}"})
"""),
    code("""
recent = df[df["at"] >= df["at"].max() - pd.DateOffset(months=24)].dropna(subset=["version"])
recent = recent.assign(minor=recent["version"].str.extract(r"^(\\d+\\.\\d+)")[0])
by_minor = recent.groupby("minor").agg(n=("reviewId", "size"), neg=("sentiment", lambda s: (s == "negative").mean()))
by_minor = by_minor[by_minor["n"] >= 300]
by_minor = by_minor.loc[sorted(by_minor.index, key=lambda v: tuple(int(x) for x in v.split(".")))]
base = (recent["sentiment"] == "negative").mean()

fig, ax = plt.subplots(figsize=(11, 3.8))
col = [SENTIMENT_COLORS["negative"] if v > base * 1.15 else "#c9c7c0" for v in by_minor["neg"]]
ax.bar(by_minor.index, by_minor["neg"], color=col, width=0.65)
ax.axhline(base, color=INK_2, linestyle="--", linewidth=1)
ax.text(len(by_minor) - 0.5, base, f" 24-month avg: {base:.0%}", va="bottom", ha="right", fontsize=9, color=INK_2)
ax.set_title("Share of negative reviews by app version (last 24 months, ≥300 reviews)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.tick_params(axis="x", rotation=60)
ax.grid(axis="x", visible=False)
save(fig, "01_negative_by_version"); plt.show()
display(Markdown("Versions in red are >15% above the average negative share — candidates for a release-notes / crash-log review."))
"""),
    md("## 5. How does MoMo respond to users?"),
    code("""
reply = df.groupby("year").agg(rate=("has_reply", "mean"), n_reply=("has_reply", "sum"))
replied = df[df["has_reply"] & (df["reply_hours"] >= 0)]
reply_time = replied.groupby("year")["reply_hours"].median()
reply_time = reply_time[reply.loc[reply_time.index, "n_reply"] >= 100]   # skip years with too few replies

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(reply.index.astype(str), reply["rate"], color=BLUE, width=0.65)
axes[0].set_title("Share of reviews answered by MoMo"); axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[0].tick_params(axis="x", rotation=45); axes[0].grid(axis="x", visible=False)
axes[1].bar(reply_time.index.astype(str), reply_time.values, color=BLUE, width=0.65)
for x, v in zip(reply_time.index.astype(str), reply_time.values):
    axes[1].text(x, v, f"{v:.1f}h" if v < 1 else f"{v:.0f}h", ha="center", va="bottom", fontsize=8, color=INK_2)
axes[1].set_title("Median time to reply (hours)"); axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "01_reply_behaviour"); plt.show()
print("Reply rate by sentiment: " + ", ".join(f"{k}: {v:.0%}" for k, v in df.groupby("sentiment")["has_reply"].mean().items()))
print(f"Overall median time to reply: {replied['reply_hours'].median():.1f} hours")
"""),
    md("""
**Takeaway.** The reply policy changed sharply: between 2017 and 2023 MoMo answered very few reviews, while from 2025 it answers ~90%+ within a median of under an hour — most likely automated or semi-automated replies. The natural next step is to measure whether replies actually *resolve* issues (e.g. users raising their rating afterwards).
"""),
    md("## 6. Which complaints does the community agree with most?"),
    code("""
helpful = df.groupby("sentiment")["thumbsUpCount"].agg(["mean", "median", lambda s: (s > 0).mean()])
helpful.columns = ["avg_thumbs_up", "median", "% with ≥1 thumbs-up"]
display(helpful.style.format({"avg_thumbs_up": "{:.2f}", "median": "{:.0f}", "% with ≥1 thumbs-up": "{:.1%}"}))
display(Markdown("**Top 8 most 'helpful'-voted negative reviews in the last 12 months** — complaints that speak for many users:"))
last12 = df[df["at"] >= df["at"].max() - pd.DateOffset(months=12)]
last12[last12.sentiment == "negative"].nlargest(8, "thumbsUpCount")[["at", "score", "thumbsUpCount", "version", "content"]]
"""),
    md("## 7. Review length & time of day"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
med_len = df[df.n_words > 0].groupby("score")["n_words"].median()
axes[0].bar(med_len.index.astype(str) + "★", med_len.values, color=BLUE, width=0.6)
axes[0].set_title("Median words per review, by rating"); axes[0].grid(axis="x", visible=False)

neg_hour = df.groupby("hour")["sentiment"].apply(lambda s: (s == "negative").mean())
axes[1].plot(neg_hour.index, neg_hour.values, marker="o", markersize=4, color=SENTIMENT_COLORS["negative"])
axes[1].set_title("Share of negative reviews by hour of day"); axes[1].set_xticks(range(0, 24, 3))
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
fig.tight_layout(); save(fig, "01_length_and_hour"); plt.show()
"""),
    md("""
**Takeaway.** Negative reviews are much longer than positive ones ("Tốt" = "Good", "Ok") and carry far more detail about *specific problems* — which is why notebook 03 mines the text of negative reviews.
5★ reviews are often one or two words: there is little to learn from them, but they are easy to classify.
"""),
    md("""
## 8. Summary for stakeholders
Exact figures are printed in each section above; the key points:
- Ratings are **strongly polarised** — the negative share is a more sensitive KPI than the average rating.
- The negative share **moves in waves tied to specific releases** → monitor per release, not just per month.
- Reply rate and time-to-reply are measurable customer-support levers.
- Negative reviews are long and information-rich → good input for topic analysis (notebook 03) and modelling (notebook 02).
"""),
]

# ----------------------------------------------------------------------------
# 02 — Sentiment model (ML / DS)
# ----------------------------------------------------------------------------
model_nb = [
    md("""
# 02 · Vietnamese Review Sentiment Classification

**Goal.** Predict sentiment (negative / neutral / positive) from the *review text*. Use cases:
- label feedback from channels **without star ratings** (support chat, social media, transcribed hotline calls);
- flag reviews whose **rating contradicts the text** (e.g. 5★ but a complaint) so issues are not missed.

**Experimental design**
| Decision | Why |
|---|---|
| Labels from stars (1–2 / 3 / 4–5) | Weak supervision — no labelling cost |
| **Time-based split** (newest 20% as test) | Mirrors deployment; prevents leakage from the future; lets us measure *data drift* |
| Primary metric **macro-F1** | The neutral class is rare, so accuracy would be misleading |
| TF-IDF word 1–2-grams + char 2–5-grams | Syllable bigrams ≈ Vietnamese compound words; char n-grams are robust to typos and missing diacritics |
| `class_weight="balanced"` | Compensates for class imbalance |
"""),
    code(SETUP + """
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from momo_reviews.model import (load_modeling_data, temporal_split, candidate_models, evaluate,
                                top_features, select_model, save_final_model)
from momo_reviews.config import SENTIMENT_ORDER, REPORTS_DIR"""),
    md("## 1. Training data"),
    code("""
data = load_modeling_data(df)
train, test = temporal_split(data, test_size=0.2)
print(f"Dropped {len(df)-len(data):,} empty/spam reviews → {len(data):,} remain")
print(f"Train: {len(train):,}  ({train['at'].min():%Y-%m} – {train['at'].max():%Y-%m})")
print(f"Test : {len(test):,}  ({test['at'].min():%Y-%m} – {test['at'].max():%Y-%m})")
pd.DataFrame({"train": train["sentiment"].value_counts(normalize=True),
              "test": test["sentiment"].value_counts(normalize=True)}).loc[SENTIMENT_ORDER].style.format("{:.1%}")
"""),
    md("Raw review vs. normalised text (teencode expanded, emoji → tokens, numbers → `num`):"),
    code("""
train[["content", "text_clean", "sentiment"]].sample(6, random_state=1)
"""),
    md("## 2. Model comparison"),
    code("""
results, fitted = evaluate(candidate_models(), train, test)
results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
results.style.format({c: "{:.3f}" for c in results.columns if c not in ("model", "train_seconds")} | {"train_seconds": "{:.0f}s"}) \\
       .background_gradient(subset=["macro_f1"], cmap="Blues").hide(axis="index")
"""),
    code("""
best_name = select_model(results)
r = results.sort_values("macro_f1")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.barh(r["model"], r["macro_f1"], color=[BLUE if m == best_name else "#c9c7c0" for m in r["model"]], height=0.6)
for y, v in enumerate(r["macro_f1"]):
    ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
ax.set_xlim(0, max(r["macro_f1"]) * 1.12); ax.set_title("Macro-F1 on the test set (newest reviews) — selected model in blue")
ax.grid(axis="y", visible=False)
save(fig, "02_model_comparison"); plt.show()
"""),
    md("""
**Model selection.** The linear TF-IDF models are very close. Selection rule: *the fastest-to-train model within 0.5 macro-F1 points of the best one* — smaller gaps are within noise, while a simpler model is cheaper to retrain regularly and (for Logistic Regression) yields **probabilities** for alert thresholds.
"""),
    md("## 3. Detailed evaluation of the selected model"),
    code("""
best = fitted[best_name]
test = test.assign(pred=best.predict(test["text_clean"]))
print(best_name)
print(classification_report(test["sentiment"], test["pred"], labels=SENTIMENT_ORDER, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, norm, title in [(axes[0], None, "Counts"), (axes[1], "true", "Normalised by true label (recall)")]:
    ConfusionMatrixDisplay.from_predictions(test["sentiment"], test["pred"], labels=SENTIMENT_ORDER, normalize=norm,
                                            cmap="Blues", ax=ax, colorbar=False, values_format=".2f" if norm else ",d")
    ax.set_title(title); ax.grid(False)
fig.tight_layout(); save(fig, "02_confusion_matrix"); plt.show()
"""),
    md("""
**Reading the confusion matrix.** Negative and positive separate well. Neutral is the hardest class — which *matches the nature of the data*: 3★ reviews mix praise and criticism ("convenient app but often buggy"), and users themselves are inconsistent when choosing 3★.
"""),
    md("""
### What if we drop the neutral class?
In practice the business question is usually *"is this review a complaint?"*. Retraining the same pipeline as a binary negative-vs-positive task:
"""),
    code("""
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
tr_b, te_b = train[train.sentiment != "neutral"], test[test.sentiment != "neutral"]
bin_model = clone(fitted["TF-IDF word+char + LogReg"]).fit(tr_b["text_clean"], tr_b["sentiment"])
proba = bin_model.predict_proba(te_b["text_clean"])[:, list(bin_model.classes_).index("negative")]
y_true = (te_b["sentiment"] == "negative").astype(int)
print(f"Binary  F1(negative) = {f1_score(y_true, proba > 0.5):.3f}   "
      f"macro-F1 = {f1_score(y_true, proba > 0.5, average='macro'):.3f}   ROC-AUC = {roc_auc_score(y_true, proba):.3f}")
"""),
    md("## 4. What did the model learn? (explainability)"),
    code("""
feats = top_features(fitted["TF-IDF word + LogReg"], n=15)[["negative", "positive"]]
feats.index = range(1, len(feats) + 1)
feats
"""),
    md("""
The highest-weighted n-grams make business sense: negative ↔ *lừa* (scam), *rác* (garbage), *không nên* (don't use), *rất tệ* (very bad); positive ↔ *tiện* (convenient), *nhanh gọn* (quick), *tốt* (good). This builds stakeholder trust and is a sanity check that the model did not latch onto artefacts.
"""),
    md("## 5. Error analysis — when the stars \"lie\""),
    code("""
mismatch = test[(test["sentiment"] == "positive") & (test["pred"] == "negative") & (test["n_words"] >= 6)]
print(f"{len(mismatch):,} reviews rated 4–5★ whose text the model reads as negative ({len(mismatch)/len(test):.1%} of the test set)")
mismatch.sample(min(8, len(mismatch)), random_state=0)[["score", "content"]]
"""),
    code("""
mismatch2 = test[(test["sentiment"] == "negative") & (test["pred"] == "positive") & (test["n_words"] >= 4)]
print(f"{len(mismatch2):,} reviews rated 1–2★ whose text reads as positive")
mismatch2.sample(min(6, len(mismatch2)), random_state=0)[["score", "content"]]
"""),
    md("""
**Key insight.** Many of the model's "errors" are really **label noise**: users give 5★ but write a complaint (e.g. *"3 tiếng đồng hồ rồi tôi không vào được ứng dụng"* — "I haven't been able to open the app for 3 hours"). Looking only at stars, a business would *miss* these complaints.
→ A text-based model has its own business value: **flag high-star reviews with negative content** for customer support.
→ It also shows that the achievable ceiling is limited by label quality, not just by the model.
"""),
    md("## 6. Does the model age? (drift)"),
    code("""
q = test.assign(quarter=test["at"].dt.to_period("Q").astype(str))
drift = q.groupby("quarter").apply(lambda g: pd.Series({"n": len(g),
        "macro_f1": f1_score(g["sentiment"], g["pred"], average="macro")}))
drift = drift[drift["n"] >= 500]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(drift.index, drift["macro_f1"], marker="o", color=BLUE)
ax.set_title("Macro-F1 per quarter over the test period"); ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=45)
save(fig, "02_drift"); plt.show()
drift.style.format({"n": "{:,.0f}", "macro_f1": "{:.3f}"})
"""),
    md("""
Whether performance holds or decays across quarters tells us how often to **retrain**. In production the pipeline would crawl new reviews weekly, track this metric, and retrain when it drops below a threshold.

## 7. Save the model for the demo app (refit on all data)
"""),
    code("""
save_final_model(fitted[best_name], data)
print("Saved", best_name)
"""),
    md("""
## 8. Next steps
- Fine-tune **PhoBERT** (`vinai/phobert-base-v2`) with word segmentation (VnCoreNLP) — expected gains mostly on the neutral class; needs a GPU.
- Hand-label ~1–2k reviews to get a "clean" test set and measure the effect of label noise precisely.
- Aspect-based sentiment (sentiment per aspect) instead of one label per review — first steps in notebook 03.
"""),
]

# ----------------------------------------------------------------------------
# 03 — Topics & aspects (NLP + DA)
# ----------------------------------------------------------------------------
topics_nb = [
    md("""
# 03 · What do users complain about? — Topic Modelling & Aspect Analysis

1. **Discover** (unsupervised): NMF on TF-IDF of negative reviews to find the main complaint topics.
2. **Measure** (rule-based): turn topics into aspect keyword lists (`src/momo_reviews/aspects.py`) — transparent, stable and trackable over time.
3. **Prioritise**: which aspects are both *frequently mentioned* and *most negative* → recommended actions.

Topic words and example reviews are in Vietnamese (the source language); aspect names are in English.
"""),
    code(SETUP + """
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from momo_reviews.aspects import aspect_matrix, ASPECTS

# Vietnamese stop words (function words, pronouns, fillers) + tokens introduced by normalisation
STOP = ["và", "là", "thì", "mà", "có", "của", "cho", "này", "đã", "được", "bị", "với", "các", "những", "một",
        "rồi", "lại", "cũng", "nữa", "khi", "để", "nên", "vì", "như", "ra", "vào", "lên", "đi", "làm", "rất",
        "quá", "nhưng", "hay", "thế", "gì", "vậy", "không", "tôi", "mình", "em", "anh", "chị", "bạn", "momo",
        "ứng", "dụng", "ứng dụng", "num", "neg_emoji", "pos_emoji", "nó", "ai", "sao", "nào", "luôn", "còn",
        "mới", "mãi", "thôi", "đâu", "ạ", "nha", "nhé", "ơi", "ko", "trong", "từ", "đến", "tại", "sẽ", "đang"]"""),
    md("## 1. Topic modelling (NMF) on negative reviews"),
    code("""
neg = df[(df["sentiment"] == "negative") & (df["n_words"] >= 4) & (~df["is_spam"])]
neg = neg[neg["at"] >= neg["at"].max() - pd.DateOffset(years=3)]   # last 3 years = issues that are still "alive"
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, stop_words=STOP, max_features=30_000)
X = vec.fit_transform(neg["text_clean"])
K = 10
nmf = NMF(n_components=K, random_state=42, init="nndsvd", max_iter=400)
W = nmf.fit_transform(X)
terms = vec.get_feature_names_out()
topics = pd.DataFrame({f"Topic {k+1}": [terms[i] for i in nmf.components_[k].argsort()[::-1][:10]] for k in range(K)}).T
topics["share"] = pd.Series(W.argmax(1)).value_counts(normalize=True).sort_index().values
print(f"{len(neg):,} negative reviews (last 3 years)")
topics.sort_values("share", ascending=False).style.format({"share": "{:.1%}"})
"""),
    md("Most representative reviews for the five largest topics:"),
    code("""
for k in topics.sort_values("share", ascending=False).index[:5]:
    i = int(k.split()[1]) - 1
    ex = neg.iloc[W[:, i].argsort()[::-1][:3]]["content"].str.slice(0, 160).tolist()
    display(Markdown(f"**{k}** — " + ", ".join(topics.loc[k, range(5)]) + "\\n\\n" + "\\n".join(f"> {e}" for e in ex)))
"""),
    md("""
The NMF topics converge on clear problem groups — e.g. *chuyển tiền / trừ tiền* (money transfer, money deducted), *ví trả sau* (pay-later wallet), *lỗi liên tục* (constant errors), *chăm sóc khách hàng* (customer support), *thanh toán* (payments). These informed the aspect keyword lists in `aspects.py`, which **measure consistently** across years.
"""),
    md("## 2. Aspect coverage"),
    code("""
A = aspect_matrix(df["text_clean"])
df_a = pd.concat([df[["at", "score", "sentiment", "n_words", "is_spam", "version"]], A], axis=1)
df_a = df_a[(df_a["n_words"] >= 3) & (~df_a["is_spam"])]
cov = (df_a[list(ASPECTS)].sum(1) > 0).groupby(df_a["sentiment"]).mean()
display(Markdown("**Share of reviews (≥3 words) tagged with at least one aspect:** " +
                 ", ".join(f"{k}: {v:.0%}" for k, v in cov.items())))
"""),
    md("## 3. Priority matrix: which aspects to fix first?"),
    code("""
recent = df_a[df_a["at"] >= df_a["at"].max() - pd.DateOffset(months=12)]
stats = pd.DataFrame({
    "reviews": recent[list(ASPECTS)].sum(),
    "mention_rate": recent[list(ASPECTS)].mean(),
    "negative_share": [(recent.loc[recent[a] == 1, "sentiment"] == "negative").mean() for a in ASPECTS],
    "avg_rating": [recent.loc[recent[a] == 1, "score"].mean() for a in ASPECTS],
}, index=list(ASPECTS)).sort_values("reviews", ascending=False)
base_neg = (recent["sentiment"] == "negative").mean()
stats.style.format({"reviews": "{:,}", "mention_rate": "{:.1%}", "negative_share": "{:.0%}", "avg_rating": "{:.2f}"}) \\
     .background_gradient(subset=["negative_share"], cmap="Reds")
"""),
    code("""
fig, ax = plt.subplots(figsize=(9.5, 6))
x, y = stats["mention_rate"], stats["negative_share"]
hot = (y > base_neg) & (x > x.median())
ax.scatter(x, y, s=np.sqrt(stats["reviews"]) * 6, c=np.where(hot, SENTIMENT_COLORS["negative"], BLUE),
           alpha=0.75, edgecolor="white", linewidth=2)
# Place labels without overlap: try candidate offsets, skip any that collide with labels already placed
fig.canvas.draw(); placed = []
for name, xi, yi in sorted(zip(stats.index, x, y), key=lambda t: -t[2]):
    px, py = ax.transData.transform((xi, yi))
    for dx, dy in [(8, 4), (8, -14), (-8, 4), (-8, -14), (8, 16)]:
        w = 6.5 * len(name)
        box = (px + dx - (w if dx < 0 else 0), py + dy, w, 12)
        if not any(abs(box[0] - b[0]) < (w + b[2]) / 2 + 4 and abs(box[1] - b[1]) < 13 for b in placed):
            break
    placed.append(box)
    ax.annotate(name, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK_2,
                ha="right" if dx < 0 else "left")
ax.axhline(base_neg, color=INK_2, linestyle="--", linewidth=1)
ax.axvline(x.median(), color=INK_2, linestyle=":", linewidth=1)
ax.text(ax.get_xlim()[1], base_neg, f"overall negative share (reviews ≥3 words) {base_neg:.0%} ",
        ha="right", va="bottom", fontsize=8, color=INK_2)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel("How often the aspect is mentioned (% of reviews)"); ax.set_ylabel("% negative among reviews mentioning it")
ax.set_title("Aspect priority matrix — last 12 months")
save(fig, "03_aspect_priority_matrix"); plt.show()
display(Markdown("**Top-right (red) = high priority:** " + ", ".join(stats.index[hot])))
"""),
    md("## 4. How negative aspects evolve over time"),
    code("""
neg_a = df_a[(df_a["sentiment"] == "negative") & (df_a["at"] >= df_a["at"].max() - pd.DateOffset(years=4))]
trend = neg_a.groupby(neg_a["at"].dt.to_period("Q"))[list(ASPECTS)].mean()
trend = trend.iloc[:-1] if len(trend) > 1 else trend   # drop the unfinished quarter
top = stats.index[:6]
fig, ax = plt.subplots(figsize=(11, 4.5))
for a in top:
    ax.plot(trend.index.astype(str), trend[a], marker="o", markersize=4, label=a)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.tick_params(axis="x", rotation=45)
ax.set_title("% of negative reviews mentioning each aspect, by quarter")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22))
save(fig, "03_aspect_trend"); plt.show()
"""),
    code("""
change = (trend.iloc[-2:].mean() - trend.iloc[-6:-2].mean()).sort_values(ascending=False)
display(Markdown("**Change in share (last 2 quarters vs. the 4 before), percentage points:**"))
(change * 100).round(1).to_frame("Δ pp")
"""),
    md("""
## 5. Recommended actions (data-driven)

| Finding | Recommendation | KPI to track |
|---|---|---|
| Aspects in the red corner of the priority matrix | Put them at the top of the backlog; assign an owner per aspect | % of negative reviews mentioning the aspect, per release |
| Negative share spikes tied to specific versions (notebook 01) | Automatic post-release alert (daily z-score) | Time to detect an incident |
| 4–5★ reviews containing complaints (notebook 02) | Use the sentiment model to flag them for customer support | % of flagged reviews handled within 24h |
| Fast-rising aspects (table above) | Investigate root cause before they reach the red corner | Quarter-over-quarter change |

*Limitations:* keyword lists cannot catch every phrasing; Store reviews are a self-selected sample (very happy / very unhappy users write more) and do not represent all users.
"""),
]


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, NB_DIR / name)


NOTEBOOKS = {
    "01_eda_business_insights.ipynb": eda,
    "02_sentiment_model.ipynb": model_nb,
    "03_topics_and_aspects.ipynb": topics_nb,
}

if __name__ == "__main__":
    import sys
    NB_DIR.mkdir(exist_ok=True)
    only = sys.argv[1:]
    for name, cells in NOTEBOOKS.items():
        if not only or any(name.startswith(o) for o in only):
            write(cells, name)
            print("built", name)
