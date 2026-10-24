PY ?= python
export PYTHONPATH := src

.PHONY: install data crawl preprocess model notebooks app test all

install:
	$(PY) -m pip install -r requirements.txt

crawl:          ## Crawl every review from Google Play (~30 min)
	$(PY) -m momo_reviews.crawl

preprocess:     ## Clean data + write sample
	$(PY) -m momo_reviews.preprocess

data: crawl preprocess

model:          ## Compare models, save metrics + selected model
	$(PY) -m momo_reviews.model

notebooks:      ## Rebuild and execute all notebooks (refreshes reports/figures)
	$(PY) scripts/build_notebooks.py
	cd notebooks && for nb in *.ipynb; do $(PY) -m jupyter nbconvert --to notebook --execute --inplace $$nb --ExecutePreprocessor.timeout=3600; done

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

all: data model notebooks
