"""Sinh 3 notebook phân tích từ code nguồn ở đây (giữ notebook gọn, review được trong git).

    python scripts/build_notebooks.py && make notebooks   # build rồi execute
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
print(f"{len(df):,} review | {df['at'].min():%d/%m/%Y} → {df['at'].max():%d/%m/%Y}")"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


# ----------------------------------------------------------------------------
# 01 — EDA & business insights (Data Analyst)
# ----------------------------------------------------------------------------
eda = [
    md("""
# 01 · EDA & Business Insights — Review ví MoMo trên Google Play

**Câu hỏi nghiệp vụ**
1. Mức độ hài lòng của người dùng thay đổi thế nào theo thời gian và theo phiên bản app?
2. Bản cập nhật nào làm trải nghiệm xấu đi (cần cảnh báo sớm)?
3. MoMo phản hồi người dùng nhanh và đầy đủ đến đâu?
4. Phân phối rating có gì bất thường (phân cực, spam) cần lưu ý khi dùng làm nhãn cho ML?

Dữ liệu: toàn bộ review công khai crawl bằng `google-play-scraper` (`src/momo_reviews/crawl.py`), đã làm sạch bằng `src/momo_reviews/preprocess.py`. Không lưu tên/ảnh người dùng.
"""),
    code(SETUP),
    md("## 1. Tổng quan dữ liệu & KPI"),
    code("""
kpi = pd.Series({
    "Số review": f"{len(df):,}",
    "Rating trung bình": f"{df['score'].mean():.2f} ★",
    "% tiêu cực (1–2★)": f"{(df['sentiment']=='negative').mean():.1%}",
    "% tích cực (4–5★)": f"{(df['sentiment']=='positive').mean():.1%}",
    "% có nội dung text": f"{(df['n_words']>0).mean():.1%}",
    "% được MoMo trả lời": f"{df['has_reply'].mean():.1%}",
    "% review spam/quảng cáo": f"{df['is_spam'].mean():.2%}",
    "% thiếu appVersion": f"{df['version'].isna().mean():.1%}",
}, name="Giá trị")
kpi.to_frame()
"""),
    code("""
missing = df[["content", "version", "replyContent", "thumbsUpCount"]].isna().mean().rename("tỷ lệ thiếu")
missing.to_frame().style.format("{:.1%}")
"""),
    md("## 2. Phân phối rating — phân cực \"J-shape\""),
    code("""
dist = df["score"].value_counts().sort_index()
colors = [SENTIMENT_COLORS[s] for s in ["negative", "negative", "neutral", "positive", "positive"]]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(dist.index.astype(str) + "★", dist.values, color=colors, width=0.6)
for b, v in zip(bars, dist.values):
    ax.text(b.get_x() + b.get_width()/2, v, f"{v/dist.sum():.0%}", ha="center", va="bottom", fontsize=9, color=INK_2)
ax.set_title("Phân phối số sao"); ax.set_ylabel("Số review")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
ax.grid(axis="x", visible=False)
save(fig, "01_rating_distribution"); plt.show()
"""),
    md("""
**Nhận xét.** Rating phân cực mạnh: rất nhiều 5★ và 1★, rất ít 2–3★. Đây là đặc trưng của review app (người dùng chỉ viết khi rất vui hoặc rất bực).
Hệ quả cho ML: lớp *neutral* rất hiếm → phải dùng **macro-F1** thay vì accuracy và cân bằng trọng số lớp.
"""),
    md("## 3. Xu hướng theo thời gian"),
    code("""
monthly = df.groupby("month").agg(n=("reviewId", "size"), avg=("score", "mean"),
                                  neg=("sentiment", lambda s: (s == "negative").mean()))
monthly = monthly[monthly.index < monthly.index.max()]  # bỏ tháng hiện tại chưa đủ dữ liệu

fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
axes[0].bar(monthly.index, monthly["n"], width=25, color=BLUE, alpha=0.85)
axes[0].set_title("Số review mỗi tháng"); axes[0].grid(axis="x", visible=False)
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))

axes[1].plot(monthly.index, monthly["neg"], color=SENTIMENT_COLORS["negative"], alpha=0.35, linewidth=1)
axes[1].plot(monthly.index, monthly["neg"].rolling(3, center=True).mean(),
             color=SENTIMENT_COLORS["negative"], label="Trung bình trượt 3 tháng")
axes[1].set_title("Tỷ lệ review tiêu cực (1–2★) theo tháng")
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1)); axes[1].legend(loc="upper left")
fig.tight_layout(); save(fig, "01_monthly_trend"); plt.show()
"""),
    code("""
yearly = df.groupby("year").agg(review=("reviewId", "size"), rating_tb=("score", "mean"),
                                ty_le_tieu_cuc=("sentiment", lambda s: (s == "negative").mean()),
                                ty_le_tra_loi=("has_reply", "mean"))
yearly["tang_truong_yoy"] = yearly["review"].pct_change()
yearly.style.format({"review": "{:,}", "rating_tb": "{:.2f}", "ty_le_tieu_cuc": "{:.1%}",
                     "ty_le_tra_loi": "{:.1%}", "tang_truong_yoy": "{:+.0%}"}, na_rep="—") \\
      .background_gradient(subset=["ty_le_tieu_cuc"], cmap="Reds")
"""),
    code("""
top_months = monthly.assign(z=(monthly["neg"] - monthly["neg"].rolling(12, min_periods=6).mean())
                              / monthly["neg"].rolling(12, min_periods=6).std())
display(Markdown("**Các tháng tỷ lệ tiêu cực tăng đột biến (z-score so với 12 tháng trước > 2):**"))
top_months[top_months["z"] > 2][["n", "avg", "neg", "z"]].sort_values("z", ascending=False).head(10) \\
    .style.format({"n": "{:,}", "avg": "{:.2f}", "neg": "{:.1%}", "z": "{:.1f}"})
"""),
    md("""
**Ứng dụng.** Bảng trên là một **cơ chế cảnh báo bất thường** đơn giản: nếu chạy hằng tuần, team sản phẩm sẽ biết ngay khi một sự cố/bản cập nhật làm tỷ lệ phàn nàn vọt lên — trước khi rating tổng trên Store bị ảnh hưởng.
"""),
    md("## 4. Rating theo phiên bản app"),
    code("""
by_major = (df.dropna(subset=["major_version"])
              .groupby("major_version")
              .agg(n=("reviewId", "size"), rating=("score", "mean"),
                   neg=("sentiment", lambda s: (s == "negative").mean()),
                   tu=("at", "min"), den=("at", "max")))
by_major = by_major[by_major["n"] >= 200].sort_index(key=lambda i: i.astype(int))
by_major.style.format({"n": "{:,}", "rating": "{:.2f}", "neg": "{:.1%}",
                       "tu": "{:%m/%Y}", "den": "{:%m/%Y}"})
"""),
    code("""
recent = df[df["at"] >= df["at"].max() - pd.DateOffset(months=24)].dropna(subset=["version"])
recent = recent.assign(minor=recent["version"].str.extract(r"^(\\d+\\.\\d+)")[0])
by_minor = recent.groupby("minor").agg(n=("reviewId", "size"), neg=("sentiment", lambda s: (s == "negative").mean()),
                                       first=("at", "min"))
by_minor = by_minor[by_minor["n"] >= 300]
by_minor = by_minor.loc[sorted(by_minor.index, key=lambda v: tuple(int(x) for x in v.split(".")))]
base = (recent["sentiment"] == "negative").mean()

fig, ax = plt.subplots(figsize=(11, 3.8))
col = [SENTIMENT_COLORS["negative"] if v > base * 1.15 else "#c9c7c0" for v in by_minor["neg"]]
ax.bar(by_minor.index, by_minor["neg"], color=col, width=0.65)
ax.axhline(base, color=INK_2, linestyle="--", linewidth=1)
ax.text(len(by_minor) - 0.5, base, f" TB 24 tháng: {base:.0%}", va="bottom", ha="right", fontsize=9, color=INK_2)
ax.set_title("Tỷ lệ review tiêu cực theo phiên bản (24 tháng gần nhất, ≥300 review)")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.tick_params(axis="x", rotation=60)
ax.grid(axis="x", visible=False)
save(fig, "01_negative_by_version"); plt.show()
display(Markdown("Phiên bản tô đỏ có tỷ lệ tiêu cực cao hơn trung bình >15% — ứng viên để soát lại release notes / crash log."))
"""),
    md("## 5. MoMo phản hồi người dùng như thế nào?"),
    code("""
reply = df.groupby("year").agg(rate=("has_reply", "mean"), n_reply=("has_reply", "sum"))
replied = df[df["has_reply"] & (df["reply_hours"] >= 0)]
reply_time = replied.groupby("year")["reply_hours"].median()
reply_time = reply_time[reply.loc[reply_time.index, "n_reply"] >= 100]   # bỏ năm quá ít reply

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
axes[0].bar(reply.index.astype(str), reply["rate"], color=BLUE, width=0.65)
axes[0].set_title("Tỷ lệ review được MoMo trả lời, theo năm"); axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
axes[0].tick_params(axis="x", rotation=45); axes[0].grid(axis="x", visible=False)
axes[1].bar(reply_time.index.astype(str), reply_time.values, color=BLUE, width=0.65)
for x, v in zip(reply_time.index.astype(str), reply_time.values):
    axes[1].text(x, v, f"{v:.1f}h" if v < 1 else f"{v:.0f}h", ha="center", va="bottom", fontsize=8, color=INK_2)
axes[1].set_title("Thời gian phản hồi trung vị (giờ)"); axes[1].tick_params(axis="x", rotation=45)
axes[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "01_reply_behaviour"); plt.show()
print("Tỷ lệ trả lời theo nhóm cảm xúc: " + ", ".join(f"{k}: {v:.0%}" for k, v in df.groupby("sentiment")["has_reply"].mean().items()))
print(f"trung vị thời gian phản hồi: {replied['reply_hours'].median():.1f} giờ")
"""),
    md("""
**Nhận xét.** Chính sách phản hồi thay đổi rõ rệt: giai đoạn 2017–2023 gần như không trả lời review, từ 2025 tỷ lệ trả lời vọt lên ~90%+ với thời gian phản hồi < 1 giờ — nhiều khả năng là phản hồi tự động/bán tự động. Bước tiếp theo nên đo xem phản hồi có *giải quyết được vấn đề* hay không (vd: người dùng sửa rating sau khi được trả lời).
"""),
    md("## 6. Review nào được cộng đồng đồng tình nhất?"),
    code("""
helpful = df.groupby("sentiment")["thumbsUpCount"].agg(["mean", "median", lambda s: (s > 0).mean()])
helpful.columns = ["thumbs_up_tb", "trung_vi", "% có ≥1 like"]
display(helpful.style.format({"thumbs_up_tb": "{:.2f}", "trung_vi": "{:.0f}", "% có ≥1 like": "{:.1%}"}))
display(Markdown("**Top 8 review tiêu cực được nhiều người 'hữu ích' nhất trong 12 tháng gần đây** — tiếng nói đại diện cho nhiều người dùng:"))
last12 = df[df["at"] >= df["at"].max() - pd.DateOffset(months=12)]
last12[last12.sentiment == "negative"].nlargest(8, "thumbsUpCount")[["at", "score", "thumbsUpCount", "version", "content"]]
"""),
    md("## 7. Độ dài review & thời điểm viết"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
med_len = df[df.n_words > 0].groupby("score")["n_words"].median()
axes[0].bar(med_len.index.astype(str) + "★", med_len.values, color=BLUE, width=0.6)
axes[0].set_title("Số từ trung vị theo số sao"); axes[0].grid(axis="x", visible=False)

neg_hour = df.groupby("hour")["sentiment"].apply(lambda s: (s == "negative").mean())
axes[1].plot(neg_hour.index, neg_hour.values, marker="o", markersize=4, color=SENTIMENT_COLORS["negative"])
axes[1].set_title("Tỷ lệ tiêu cực theo giờ trong ngày"); axes[1].set_xticks(range(0, 24, 3))
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1))
fig.tight_layout(); save(fig, "01_length_and_hour"); plt.show()
"""),
    md("""
**Nhận xét.** Review tiêu cực dài hơn hẳn review tích cực ("Tốt", "Ok") → chứa nhiều thông tin hơn về *vấn đề cụ thể*. Đây là lý do notebook 03 tập trung khai thác text của review tiêu cực.
Review 5★ ngắn, nhiều review chỉ 1–2 từ → giới hạn tự nhiên của việc học cảm xúc từ text cho lớp positive là thấp, nhưng lại dễ phân loại.
"""),
    md("""
## 8. Tóm tắt cho stakeholder
Các con số cụ thể được in ở từng mục phía trên; ý chính:
- Rating **phân cực mạnh** — tỷ lệ tiêu cực là KPI nhạy hơn rating trung bình.
- Tỷ lệ tiêu cực **biến động theo đợt**, gắn với phiên bản cụ thể → nên theo dõi theo release thay vì theo tháng.
- Tỷ lệ trả lời review & thời gian phản hồi là đòn bẩy CSKH đo lường được.
- Review tiêu cực dài, giàu thông tin → đầu vào tốt cho phân tích chủ đề (notebook 03) và mô hình (notebook 02).
"""),
]

# ----------------------------------------------------------------------------
# 02 — Sentiment model (ML / DS)
# ----------------------------------------------------------------------------
model_nb = [
    md("""
# 02 · Mô hình phân loại cảm xúc review (Vietnamese Sentiment Classification)

**Mục tiêu.** Dự đoán cảm xúc (negative / neutral / positive) từ *nội dung text*. Ứng dụng:
- gắn nhãn cho phản hồi đến từ kênh **không có số sao** (chat CSKH, mạng xã hội, hotline đã chuyển text);
- phát hiện review mà **số sao mâu thuẫn với nội dung** (vd 5★ nhưng than phiền) để không bỏ sót vấn đề.

**Thiết kế thí nghiệm**
| Quyết định | Lý do |
|---|---|
| Nhãn từ số sao (1–2 / 3 / 4–5) | Weak supervision, không tốn chi phí gán nhãn |
| Chia **theo thời gian** (20% mới nhất làm test) | Mô phỏng triển khai thật; tránh rò rỉ thông tin từ tương lai; đo được *data drift* |
| Metric chính **macro-F1** | Lớp neutral rất hiếm, accuracy sẽ đánh lừa |
| TF-IDF word 1–2 gram + char 2–5 gram | Bigram âm tiết ≈ từ ghép tiếng Việt; char n-gram bền với lỗi chính tả, thiếu dấu |
| `class_weight="balanced"` | Bù mất cân bằng lớp |
"""),
    code(SETUP + """
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from momo_reviews.model import load_modeling_data, temporal_split, candidate_models, evaluate, top_features, select_model, save_final_model
from momo_reviews.config import SENTIMENT_ORDER, REPORTS_DIR"""),
    md("## 1. Dữ liệu huấn luyện"),
    code("""
data = load_modeling_data(df)
train, test = temporal_split(data, test_size=0.2)
print(f"Bỏ {len(df)-len(data):,} review rỗng/spam → còn {len(data):,}")
print(f"Train: {len(train):,}  ({train['at'].min():%m/%Y} – {train['at'].max():%m/%Y})")
print(f"Test : {len(test):,}  ({test['at'].min():%m/%Y} – {test['at'].max():%m/%Y})")
pd.DataFrame({"train": train["sentiment"].value_counts(normalize=True),
              "test": test["sentiment"].value_counts(normalize=True)}).loc[SENTIMENT_ORDER].style.format("{:.1%}")
"""),
    code("""
train[["content", "text_clean", "sentiment"]].sample(6, random_state=1)
"""),
    md("## 2. So sánh mô hình"),
    code("""
results, fitted = evaluate(candidate_models(), train, test)
results.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
results.style.format({c: "{:.3f}" for c in results.columns if c not in ("model", "train_seconds")} | {"train_seconds": "{:.0f}s"}) \\
       .background_gradient(subset=["macro_f1"], cmap="Blues").hide(axis="index")
"""),
    code("""
r = results.sort_values("macro_f1")
fig, ax = plt.subplots(figsize=(8, 3.4))
ax.barh(r["model"], r["macro_f1"], color=["#c9c7c0"] * (len(r) - 1) + [BLUE], height=0.6)
for y, v in enumerate(r["macro_f1"]):
    ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=9, color=INK_2)
ax.set_xlim(0, max(r["macro_f1"]) * 1.12); ax.set_title("Macro-F1 trên tập test (review mới nhất)")
ax.grid(axis="y", visible=False)
save(fig, "02_model_comparison"); plt.show()
"""),
    md("""
**Chọn mô hình.** Các mô hình tuyến tính trên TF-IDF cho kết quả rất sát nhau. Quy tắc chọn: *mô hình huấn luyện nhanh nhất trong khoảng 0.5 điểm macro-F1 so với mô hình tốt nhất* — chênh lệch nhỏ hơn mức này không có ý nghĩa thực tế, còn mô hình đơn giản rẻ hơn khi retrain định kỳ và (với Logistic Regression) cho **xác suất** để đặt ngưỡng cảnh báo.
"""),
    md("## 3. Đánh giá chi tiết mô hình được chọn"),
    code("""
best_name = select_model(results)
best = fitted[best_name]
test = test.assign(pred=best.predict(test["text_clean"]))
print(best_name)
print(classification_report(test["sentiment"], test["pred"], labels=SENTIMENT_ORDER, digits=3))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, norm, title in [(axes[0], None, "Số lượng"), (axes[1], "true", "Chuẩn hoá theo nhãn thật (recall)")]:
    ConfusionMatrixDisplay.from_predictions(test["sentiment"], test["pred"], labels=SENTIMENT_ORDER, normalize=norm,
                                            cmap="Blues", ax=ax, colorbar=False, values_format=".2f" if norm else ",d")
    ax.set_title(title); ax.grid(False)
fig.tight_layout(); save(fig, "02_confusion_matrix"); plt.show()
"""),
    md("""
**Đọc confusion matrix.** Negative và positive được tách tốt. Neutral là lớp khó nhất — điều này *đúng với bản chất dữ liệu*: review 3★ thường lẫn khen-chê ("app tiện nhưng hay lỗi"), và chính người dùng cũng không nhất quán khi chọn 3★.
"""),
    md("""
### Nếu bỏ lớp neutral?
Trong thực tế, câu hỏi kinh doanh thường là *"review này có phải phàn nàn không?"*. Huấn luyện lại cùng pipeline cho bài toán nhị phân negative vs positive:
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
    md("## 4. Mô hình học được gì? (Explainability)"),
    code("""
lr_name = "TF-IDF word + LogReg"
feats = top_features(fitted[lr_name], n=15)[["negative", "positive"]]
feats.index = range(1, len(feats) + 1)
feats
"""),
    md("""
Các n-gram có trọng số cao nhất rất "hợp lý nghiệp vụ": lớp negative gắn với *lỗi, không chuyển được, mất tiền, lừa đảo*, lớp positive gắn với *tiện lợi, tốt, nhanh*. Điều này giúp stakeholder tin mô hình và cũng là kiểm tra sanity rằng không học nhầm artefact.
"""),
    md("## 5. Error analysis — khi số sao \"nói dối\""),
    code("""
mismatch = test[(test["sentiment"] == "positive") & (test["pred"] == "negative") & (test["n_words"] >= 6)]
print(f"{len(mismatch):,} review 4–5★ nhưng mô hình đọc nội dung là tiêu cực ({len(mismatch)/len(test):.1%} tập test)")
mismatch.sample(min(8, len(mismatch)), random_state=0)[["score", "content"]]
"""),
    code("""
mismatch2 = test[(test["sentiment"] == "negative") & (test["pred"] == "positive") & (test["n_words"] >= 4)]
print(f"{len(mismatch2):,} review 1–2★ nhưng nội dung tích cực")
mismatch2.sample(min(6, len(mismatch2)), random_state=0)[["score", "content"]]
"""),
    md("""
**Insight quan trọng.** Phần lớn "lỗi" của mô hình thực ra là **nhiễu nhãn**: người dùng cho 5★ nhưng viết lời phàn nàn (hoặc ngược lại). Nếu chỉ nhìn số sao, doanh nghiệp *bỏ sót* những phàn nàn này.
→ Mô hình text-based có giá trị kinh doanh riêng: dùng để **gắn cờ review có sao cao nhưng nội dung tiêu cực** đưa về đội CSKH.
→ Đồng thời cho thấy trần hiệu năng thực tế bị giới hạn bởi chất lượng nhãn, không chỉ bởi mô hình.
"""),
    md("## 6. Model có bị \"lão hoá\" theo thời gian? (drift)"),
    code("""
from sklearn.metrics import f1_score
q = test.assign(quarter=test["at"].dt.to_period("Q").astype(str))
drift = q.groupby("quarter").apply(lambda g: pd.Series({"n": len(g),
        "macro_f1": f1_score(g["sentiment"], g["pred"], average="macro")}))
drift = drift[drift["n"] >= 500]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(drift.index, drift["macro_f1"], marker="o", color=BLUE)
ax.set_title("Macro-F1 theo quý trong giai đoạn test"); ax.set_ylim(0, 1)
save(fig, "02_drift"); plt.show()
drift.style.format({"n": "{:,.0f}", "macro_f1": "{:.3f}"})
"""),
    md("""
Hiệu năng giữ ổn định hay giảm dần qua các quý cho biết tần suất cần **retrain**. Trong production, pipeline sẽ crawl review mới hằng tuần, theo dõi metric này và retrain khi giảm quá ngưỡng.

## 7. Lưu mô hình cho app demo (refit trên toàn bộ dữ liệu)
"""),
    code("""
save_final_model(fitted[best_name], data)
print("Đã lưu", best_name)
"""),
    md("""
## 8. Hướng phát triển
- Fine-tune **PhoBERT** (`vinai/phobert-base-v2`) + word segmentation (VnCoreNLP) — kỳ vọng tăng chủ yếu ở lớp neutral; cần GPU.
- Gán nhãn thủ công ~1–2k review để có tập test "sạch", đo chính xác hơn tác động của nhiễu nhãn.
- Aspect-based sentiment (cảm xúc theo từng khía cạnh) thay vì 1 nhãn/review — xem bước đầu ở notebook 03.
"""),
]

# ----------------------------------------------------------------------------
# 03 — Topics & aspects (NLP + DA)
# ----------------------------------------------------------------------------
topics_nb = [
    md("""
# 03 · Người dùng phàn nàn về điều gì? — Topic Modeling & Aspect Analysis

1. **Khám phá** (unsupervised): NMF trên TF-IDF của review tiêu cực để tìm các chủ đề phàn nàn chính.
2. **Đo lường** (rule-based): chuyển các chủ đề thành bộ từ khoá khía cạnh (`src/momo_reviews/aspects.py`) — minh bạch, ổn định, theo dõi được theo thời gian.
3. **Ưu tiên hoá**: khía cạnh nào vừa *nhiều người nhắc* vừa *tiêu cực nhất* → đề xuất hành động.
"""),
    code(SETUP + """
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from momo_reviews.aspects import aspect_matrix, ASPECTS

STOP = ["và", "là", "thì", "mà", "có", "của", "cho", "này", "đã", "được", "bị", "với", "các", "những", "một",
        "rồi", "lại", "cũng", "nữa", "khi", "để", "nên", "vì", "như", "ra", "vào", "lên", "đi", "làm", "rất",
        "quá", "nhưng", "hay", "thế", "gì", "vậy", "không", "tôi", "mình", "em", "anh", "chị", "bạn", "momo",
        "ứng", "dụng", "ứng dụng", "num", "neg_emoji", "pos_emoji", "nó", "ai", "sao", "nào", "luôn", "còn",
        "mới", "mãi", "thôi", "đâu", "ạ", "nha", "nhé", "ơi", "ko", "trong", "từ", "đến", "tại", "sẽ", "đang"]"""),
    md("## 1. Topic modeling (NMF) trên review tiêu cực"),
    code("""
neg = df[(df["sentiment"] == "negative") & (df["n_words"] >= 4) & (~df["is_spam"])]
neg = neg[neg["at"] >= neg["at"].max() - pd.DateOffset(years=3)]   # 3 năm gần nhất = vấn đề còn "sống"
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=10, max_df=0.5, stop_words=STOP, max_features=30_000)
X = vec.fit_transform(neg["text_clean"])
K = 10
nmf = NMF(n_components=K, random_state=42, init="nndsvd", max_iter=400)
W = nmf.fit_transform(X)
terms = vec.get_feature_names_out()
topics = pd.DataFrame({f"Topic {k+1}": [terms[i] for i in nmf.components_[k].argsort()[::-1][:10]] for k in range(K)}).T
topics["share"] = pd.Series(W.argmax(1)).value_counts(normalize=True).sort_index().values
print(f"{len(neg):,} review tiêu cực (3 năm gần nhất)")
topics.sort_values("share", ascending=False).style.format({"share": "{:.1%}"})
"""),
    code("""
for k in topics.sort_values("share", ascending=False).index[:5]:
    i = int(k.split()[1]) - 1
    ex = neg.iloc[W[:, i].argsort()[::-1][:3]]["content"].str.slice(0, 160).tolist()
    display(Markdown(f"**{k}** — " + ", ".join(topics.loc[k, range(5)]) + "\\n\\n" + "\\n".join(f"> {e}" for e in ex)))
"""),
    md("""
Các topic NMF hội tụ về những nhóm vấn đề rõ ràng (đăng nhập/OTP, xác thực danh tính, chuyển–nạp–rút tiền, lỗi/chậm, CSKH, mất tiền…). Dựa trên đó, bộ từ khoá khía cạnh trong `aspects.py` được xây dựng để **đo lường nhất quán** qua các năm.
"""),
    md("## 2. Phủ của bộ khía cạnh"),
    code("""
A = aspect_matrix(df["text_clean"])
df_a = pd.concat([df[["at", "score", "sentiment", "n_words", "is_spam", "version"]], A], axis=1)
df_a = df_a[(df_a["n_words"] >= 3) & (~df_a["is_spam"])]
cov = (df_a[list(ASPECTS)].sum(1) > 0).groupby(df_a["sentiment"]).mean()
display(Markdown("**Tỷ lệ review (≥3 từ) được gắn ít nhất 1 khía cạnh:** " +
                 ", ".join(f"{k}: {v:.0%}" for k, v in cov.items())))
"""),
    md("## 3. Ma trận ưu tiên: khía cạnh nào cần xử lý trước?"),
    code("""
recent = df_a[df_a["at"] >= df_a["at"].max() - pd.DateOffset(months=12)]
stats = pd.DataFrame({
    "so_review": recent[list(ASPECTS)].sum(),
    "ty_le_nhac_den": recent[list(ASPECTS)].mean(),
    "ty_le_tieu_cuc": [(recent.loc[recent[a] == 1, "sentiment"] == "negative").mean() for a in ASPECTS],
    "rating_tb": [recent.loc[recent[a] == 1, "score"].mean() for a in ASPECTS],
}, index=list(ASPECTS)).sort_values("so_review", ascending=False)
base_neg = (recent["sentiment"] == "negative").mean()
stats.style.format({"so_review": "{:,}", "ty_le_nhac_den": "{:.1%}", "ty_le_tieu_cuc": "{:.0%}", "rating_tb": "{:.2f}"}) \\
     .background_gradient(subset=["ty_le_tieu_cuc"], cmap="Reds")
"""),
    code("""
fig, ax = plt.subplots(figsize=(9.5, 6))
x, y = stats["ty_le_nhac_den"], stats["ty_le_tieu_cuc"]
hot = (y > base_neg) & (x > x.median())
ax.scatter(x, y, s=np.sqrt(stats["so_review"]) * 6, c=np.where(hot, SENTIMENT_COLORS["negative"], BLUE),
           alpha=0.75, edgecolor="white", linewidth=2)
# Đặt nhãn tránh chồng chữ: thử lần lượt các vị trí, bỏ vị trí trùng với nhãn đã đặt
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
ax.text(ax.get_xlim()[1], base_neg, f"tỷ lệ tiêu cực chung (review ≥3 từ) {base_neg:.0%} ", ha="right", va="bottom", fontsize=8, color=INK_2)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
ax.set_xlabel("Mức độ được nhắc đến (% review)"); ax.set_ylabel("% review tiêu cực khi nhắc đến khía cạnh")
ax.set_title("Ma trận ưu tiên khía cạnh — 12 tháng gần nhất")
save(fig, "03_aspect_priority_matrix"); plt.show()
display(Markdown("**Góc phải-trên (đỏ) = ưu tiên cao:** " + ", ".join(stats.index[hot])))
"""),
    md("## 4. Xu hướng các khía cạnh tiêu cực theo thời gian"),
    code("""
neg_a = df_a[(df_a["sentiment"] == "negative") & (df_a["at"] >= df_a["at"].max() - pd.DateOffset(years=4))]
trend = neg_a.groupby(neg_a["at"].dt.to_period("Q"))[list(ASPECTS)].mean()
trend = trend.iloc[:-1] if len(trend) > 1 else trend   # bỏ quý chưa kết thúc
top = stats.index[:6]
fig, ax = plt.subplots(figsize=(11, 4.5))
for a in top:
    ax.plot(trend.index.astype(str), trend[a], marker="o", markersize=4, label=a)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1)); ax.tick_params(axis="x", rotation=45)
ax.set_title("% review tiêu cực nhắc tới từng khía cạnh, theo quý")
ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.22))
save(fig, "03_aspect_trend"); plt.show()
"""),
    code("""
change = (trend.iloc[-2:].mean() - trend.iloc[-6:-2].mean()).sort_values(ascending=False)
display(Markdown("**Thay đổi tỷ trọng (2 quý gần nhất so với 4 quý trước đó), điểm %:**"))
(change * 100).round(1).to_frame("Δ điểm %")
"""),
    md("""
## 5. Đề xuất hành động (dựa trên dữ liệu)

| Phát hiện | Đề xuất | KPI theo dõi |
|---|---|---|
| Các khía cạnh ở góc đỏ của ma trận ưu tiên | Đưa vào backlog ưu tiên; gắn owner theo từng khía cạnh | % review tiêu cực nhắc tới khía cạnh, theo release |
| Tỷ lệ tiêu cực tăng đột biến theo phiên bản (notebook 01) | Thêm cảnh báo tự động sau mỗi release (z-score theo ngày) | Thời gian phát hiện sự cố |
| Review 4–5★ có nội dung phàn nàn (notebook 02) | Dùng mô hình sentiment gắn cờ để CSKH xử lý | % review được xử lý trong 24h |
| Review tiêu cực được trả lời ít hơn / chậm | Ưu tiên trả lời review tiêu cực có nhiều 👍 | Tỷ lệ trả lời & thời gian phản hồi |

*Giới hạn:* bộ từ khoá không bắt được mọi cách diễn đạt; review trên Store là mẫu thiên lệch (người rất vui/rất bực viết nhiều hơn) — không đại diện cho toàn bộ người dùng.
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
    only = sys.argv[1:]  # vd: python scripts/build_notebooks.py 01  -> chỉ build notebook 01
    for name, cells in NOTEBOOKS.items():
        if not only or any(name.startswith(o) for o in only):
            write(cells, name)
            print("built", name)
