# Container for the report-card demo (app.py) on Cloud Run.
#
# Only what the app reads at runtime is copied in: the Gradio app, the goal model it imports,
# the Annex C table match_engine loads at import time, the two locked rating parquets, and the
# flag PNGs the report embeds as base64. The pipeline scripts (09-18), notebooks and reports
# are deliberately left out — they are how the numbers were produced, not what serves them.
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py match_engine.py annex_c_third_allocation.csv ./
COPY assets/flags/ ./assets/flags/
COPY data/processed/wc_final_ratings.parquet data/processed/supremacy_params.parquet ./data/processed/

# Cloud Run sets PORT and expects the server on it; app.py reads PORT and binds 0.0.0.0.
ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
