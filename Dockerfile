# Container for the scoring service.
#
# The image carries the trained model and the frozen reference lists, but no raw
# data: the datasets are fetched by `make data` at development time and are not
# redistributable in all cases (see scripts/fetch_data.py).
#
#   docker build -t phishurl .
#   docker run --rm -p 8000:8000 phishurl
#   curl -s localhost:8000/v1/score -H 'content-type: application/json' \
#        -d '{"url":"https://appleid.apple.com.secure-login-verify.xyz"}'

FROM python:3.12-slim

# LightGBM needs libgomp at runtime; without it `import lightgbm` fails with a
# dlopen error rather than anything self-explanatory.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml ./
COPY src/ ./src/
COPY models/model.txt ./models/model.txt
COPY reports/metrics.json ./reports/metrics.json
COPY data/public_suffix_list.dat data/brand_tokens.txt ./data/

ENV PYTHONPATH=/app/src
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "phishurl.api:app", "--host", "0.0.0.0", "--port", "8000"]
