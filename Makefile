.PHONY: help setup data audit train export test test-fast serve docker paper demo all clean

PY := .venv/bin/python
export PYTHONPATH := src

help:
	@echo "make setup    create .venv and install pinned dependencies"
	@echo "make data     download PhiUSIIL, Tranco, PhishTank, OpenPhish, PSL"
	@echo "make audit    reproduce and quantify the two label leaks"
	@echo "make train    train the host-name model and evaluate it"
	@echo "make export   write docs/model.json + docs/psl.json for the browser"
	@echo "make test     full suite, including Python<->JS parity"
	@echo "make serve    run the FastAPI service on :8000"
	@echo "make demo     serve the static demo on :8899"
	@echo "make paper    compile paper/report.pdf from the real run outputs"
	@echo "make all      data -> audit -> train -> export -> test -> paper"

setup:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -r requirements.txt
	@echo "macOS note: LightGBM needs libomp -> brew install libomp"

data:
	$(PY) scripts/fetch_data.py

audit:
	$(PY) scripts/leak_audit.py

train:
	$(PY) -m phishurl.train

export:
	$(PY) -m phishurl.export_model
	cp reports/metrics.json docs/metrics.json

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

serve:
	$(PY) -m uvicorn phishurl.api:app --reload --port 8000

demo:
	@echo "http://localhost:8899"
	cd docs && ../$(PY) -m http.server 8899

docker:
	docker build -t phishurl .

paper:
	cd paper && tectonic report.tex

all: data audit train export test paper

clean:
	rm -rf .pytest_cache **/__pycache__ paper/*.aux paper/*.log paper/*.out
