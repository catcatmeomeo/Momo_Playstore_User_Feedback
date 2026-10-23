# 💬 MoMo Play Store Reviews — Phân tích trải nghiệm & mô hình cảm xúc tiếng Việt

![Python](https://img.shields.io/badge/Python-3.12-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange) ![Streamlit](https://img.shields.io/badge/demo-Streamlit-red) ![CI](https://github.com/catcatmeomeo/Momo_Playstore_User_Feedback/actions/workflows/ci.yml/badge.svg)

Dự án end-to-end trên **217,336 review** công khai của ví điện tử MoMo trên Google Play (05/2014 → 10/2026): **crawl dữ liệu → làm sạch văn bản tiếng Việt → phân tích nghiệp vụ (DA) → mô hình phân loại cảm xúc (ML) → topic modeling & phân tích khía cạnh (NLP) → app demo**.

> **TL;DR (EN)** — End-to-end analytics & NLP project on 217k Vietnamese app reviews of Vietnam's largest e-wallet. Includes a resumable scraper, Vietnamese text normalisation (teencode, emoji, spam), business EDA with release-level anomaly flags, a sentiment classifier evaluated on a **time-based split** (macro-F1 0.65 on 3 classes; F1 0.94 / ROC-AUC 0.975 for complaint detection), NMF topic modelling feeding a transparent aspect taxonomy, a priority matrix for product teams, and a Streamlit demo. Reproducible with `make all`, unit-tested in CI.

---

## 1. Bài toán

MoMo nhận hàng trăm review mỗi ngày. Đọc thủ công là không khả thi, còn chỉ nhìn **số sao trung bình** thì che mất câu hỏi quan trọng: *người dùng đang khó chịu vì điều gì, từ khi nào, ở phiên bản nào?*

| Câu hỏi | Góc nhìn | Notebook |
|---|---|---|
| Mức hài lòng thay đổi ra sao theo thời gian / phiên bản? MoMo phản hồi thế nào? | Data Analyst | [`01_eda_business_insights`](notebooks/01_eda_business_insights.ipynb) |
| Có thể tự động nhận diện review tiêu cực từ *nội dung* (kể cả khi không có số sao)? | ML / Data Scientist | [`02_sentiment_model`](notebooks/02_sentiment_model.ipynb) |
| Người dùng phàn nàn về chủ đề nào, cái nào cần ưu tiên? | NLP + DA | [`03_topics_and_aspects`](notebooks/03_topics_and_aspects.ipynb) |

## 2. Kết quả chính

### 📉 Trải nghiệm xấu đi rõ rệt từ 2022
![Xu hướng theo tháng](reports/figures/01_monthly_trend.png)

| Năm | Review | Rating TB | % tiêu cực (1–2★) |
|---|---:|---:|---:|
| 2019 | 36,043 | 4.00 | 22% |
| 2021 | 36,880 | 3.58 | 33% |
| 2023 | 24,568 | 2.69 | **56%** |
| 2025 | 9,230 | 2.47 | **61%** |
| 2026 (đến 10/2026) | 9,413 | 3.21 | 43% |

- Rating **phân cực mạnh** (5★ và 1★ chiếm phần lớn) → *tỷ lệ tiêu cực* là KPI nhạy hơn rating trung bình.
- Notebook 01 có bảng **cảnh báo bất thường** (z-score so với 12 tháng trước) và tỷ lệ tiêu cực **theo phiên bản** để truy ra release gây sự cố.

### 🔎 Người dùng phàn nàn về điều gì? (12 tháng gần nhất)
![Ma trận ưu tiên](reports/figures/03_aspect_priority_matrix.png)

| Khía cạnh | % review nhắc tới | % trong đó là tiêu cực |
|---|---:|---:|
| Lỗi & hiệu năng (lag, treo, văng) | 22.4% | 82% |
| Mất tiền & hoàn tiền | 12.9% | **95%** |
| Chăm sóc khách hàng | 11.3% | 93% |
| Nạp/Rút & liên kết ngân hàng | 10.8% | 88% |
| Chuyển & nhận tiền | 10.6% | 82% |

→ **Ưu tiên** các khía cạnh ở góc đỏ (nhắc nhiều *và* rất tiêu cực). *Chăm sóc khách hàng* (+4.0 điểm %) và *eKYC/định danh* (+3.4 điểm %) là hai khía cạnh **tăng nhanh nhất** trong 2 quý gần đây, trong khi phàn nàn về *lỗi & hiệu năng* giảm 8.5 điểm %.

### 🤖 Mô hình phân loại cảm xúc
![So sánh mô hình](reports/figures/02_model_comparison.png)

| Mô hình (test = 43k review mới nhất, 12/2023 → 10/2026) | Macro-F1 | F1 neg | F1 pos | Train |
|---|---:|---:|---:|---:|
| Majority baseline | 0.225 | – | 0.675 | – |
| TF-IDF + Complement NB | 0.613 | 0.897 | 0.914 | 9s |
| **TF-IDF word + Logistic Regression** ✅ | **0.651** | 0.887 | 0.922 | 73s |
| TF-IDF word+char + Logistic Regression | 0.652 | 0.888 | 0.923 | 203s |
| TF-IDF word+char + LinearSVC | 0.655 | 0.908 | 0.930 | 523s |

- **Bài toán nhị phân "có phải phàn nàn không?"**: F1 = **0.937**, ROC-AUC = **0.975**.
- Lớp *neutral* (3★, chỉ ~3% dữ liệu) khó nhất (F1 0.14): review 3★ thường vừa khen vừa chê, và chính nhãn từ số sao cũng không nhất quán.
- Chọn **Logistic Regression** dù kém LinearSVC 0.5 điểm: chênh lệch nằm trong nhiễu, nhưng train nhanh hơn **7 lần** (rẻ khi retrain định kỳ) và cho **xác suất** để đặt ngưỡng cảnh báo.
- **Error analysis**: 1,047 review (2.4% tập test) chấm 4–5★ nhưng nội dung là phàn nàn (vd: *"3 tiếng đồng hồ rồi tôi không vào được ứng dụng"* — 5★). Chỉ nhìn số sao sẽ bỏ sót → mô hình text có giá trị riêng cho CSKH.

### 🧹 Phát hiện khác
- **Spam quảng cáo vay tiền trong review**: 221 review thuộc các chiến dịch spam (tên miền cố tình viết lệch kiểu `VAY9 . C0M`, `( W E B )`, app "TinChac"…), **62% trong số đó chấm 5★** → làm đẹp giả rating. Đã lọc khỏi dữ liệu huấn luyện.
- **Chính sách phản hồi thay đổi hẳn**: giai đoạn 2017–2023 MoMo trả lời rất ít review (đa số năm < 5%, cao nhất 21% năm 2020); từ 2025 trả lời ~90–95% với thời gian trung vị **< 30 phút** — nhiều khả năng là phản hồi tự động.
- Review tiêu cực dài hơn nhiều so với review tích cực (trung vị 13 vs 2 từ) → giàu thông tin cho phân tích chủ đề.

## 3. Phương pháp

```
Google Play ─crawl.py─▶ raw.parquet ─preprocess.py─▶ clean.parquet ─┬─▶ 01 EDA & KPI
 (217k review,          (không lưu tên/   (chuẩn hoá text, cờ spam,  ├─▶ 02 Sentiment ─▶ model.joblib ─▶ Streamlit app
  checkpoint + retry)    ảnh người dùng)   nhãn, version, thời gian)  └─▶ 03 NMF topics ─▶ aspects.py ─▶ ma trận ưu tiên
```

| Bước | Điểm kỹ thuật |
|---|---|
| **Crawl** ([`crawl.py`](src/momo_reviews/crawl.py)) | Phân trang bằng continuation token, retry + exponential backoff, checkpoint định kỳ; **không lưu dữ liệu cá nhân** (tên, avatar). |
| **Chuẩn hoá tiếng Việt** ([`text.py`](src/momo_reviews/text.py)) | Unicode NFC, ~80 teencode (`ko`→`không`, `dc`→`được`), emoji cảm xúc → token, rút gọn ký tự lặp (`tốtttt`→`tốt`), ẩn số điện thoại/URL, regex phát hiện spam. |
| **Gán nhãn** | Weak supervision từ số sao: 1–2★ negative, 3★ neutral, 4–5★ positive. |
| **Đánh giá** ([`model.py`](src/momo_reviews/model.py)) | **Chia theo thời gian** (train quá khứ, test 20% mới nhất) thay vì random — mô phỏng đúng triển khai thật, tránh rò rỉ tương lai; theo dõi **drift** theo quý. Macro-F1 vì lớp mất cân bằng. |
| **Đặc trưng** | TF-IDF bigram âm tiết (≈ từ ghép tiếng Việt, không cần thư viện tách từ) + char n-gram 2–5 (bền với lỗi gõ/thiếu dấu). |
| **Topic → Aspect** ([`aspects.py`](src/momo_reviews/aspects.py)) | NMF để *khám phá* chủ đề, rồi cố định thành bộ từ khoá để *đo lường* nhất quán theo thời gian và dễ giải thích cho team sản phẩm. |

## 4. Chạy lại dự án

```bash
git clone https://github.com/catcatmeomeo/Momo_Playstore_User_Feedback.git
cd Momo_Playstore_User_Feedback
python -m venv .venv && source .venv/bin/activate
make install
```

```bash
make app        # app demo với model đã train sẵn
make test       # unit test
make all        # crawl lại toàn bộ (~30 phút) -> train -> chạy lại 3 notebook
```

Dữ liệu đầy đủ không đưa lên GitHub (tái tạo bằng `make data`); có sẵn mẫu 3,000 review tại [`data/sample/`](data/sample/momo_reviews_sample.csv) để thử tab "Nhiều review (CSV)" của app.

## 5. Cấu trúc thư mục

```
├── src/momo_reviews/           # package: crawl, preprocess, text, model, aspects, viz, config
├── notebooks/                  # 01 EDA · 02 model · 03 topics (đã chạy sẵn, có output)
├── scripts/build_notebooks.py  # sinh notebook từ code -> diff gọn khi review
├── app/streamlit_app.py        # demo: dự đoán 1 review hoặc cả file CSV
├── models/                     # model đã train (~3MB)
├── reports/                    # figures/, model_comparison.csv, metrics.json
├── data/sample/                # mẫu dữ liệu nhỏ
├── tests/                      # pytest, chạy trong GitHub Actions
└── Makefile                    # make data | model | notebooks | app | test | all
```

## 6. Giới hạn & hướng phát triển
- Review trên Store là **mẫu tự chọn** (người rất vui/rất bực viết nhiều hơn) — không đại diện cho toàn bộ người dùng.
- Nhãn từ số sao có nhiễu → gán nhãn tay ~1–2k review để có tập test sạch.
- Fine-tune **PhoBERT** (cần GPU), kỳ vọng cải thiện chủ yếu ở lớp neutral; **aspect-based sentiment** thay vì 1 nhãn/review.
- Tự động hoá: crawl hằng tuần + cảnh báo khi tỷ lệ tiêu cực của một release vượt ngưỡng.

---
*Dữ liệu là review công khai trên Google Play, chỉ dùng cho mục đích học tập/nghiên cứu; không lưu thông tin định danh người dùng.*
