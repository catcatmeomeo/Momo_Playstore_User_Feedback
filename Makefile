PY ?= python
export PYTHONPATH := src

.PHONY: install data crawl preprocess model notebooks app test all

install:
	$(PY) -m pip install -r requirements.txt

crawl:          ## Crawl toàn bộ review từ Google Play (~20–30 phút)
	$(PY) -m momo_reviews.crawl

preprocess:     ## Làm sạch dữ liệu + tạo sample
	$(PY) -m momo_reviews.preprocess

data: crawl preprocess

model:          ## So sánh mô hình, lưu metrics + model tốt nhất
	$(PY) -m momo_reviews.model

notebooks:      ## Sinh lại và chạy toàn bộ notebook (cập nhật hình trong reports/figures)
	$(PY) scripts/build_notebooks.py
	cd notebooks && for nb in *.ipynb; do $(PY) -m jupyter nbconvert --to notebook --execute --inplace $$nb --ExecutePreprocessor.timeout=3600; done

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: data model notebooks
