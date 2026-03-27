FROM python:3.11-slim

WORKDIR /app

COPY req_inference.txt .

RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r req_inference.txt

COPY predict_pipeline.py schema.py main.py ./
COPY model/pipeline/ model/pipeline/
COPY model/best_vae.pt model/
COPY model/iso_forest.pkl model/
COPY Metadata/ Metadata/

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]