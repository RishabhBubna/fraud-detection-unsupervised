from fastapi import FastAPI
from contextlib import asynccontextmanager
import pandas as pd

from predict_pipeline import (VAE_MODEL_PATH , ISO_MODEL_PATH, VAE_PIPELINE_PATH,
                               ISO_PIPELINE_PATH, COLUMN_METADATA_PATH, INPUT_METADATA_PATH,
                               VAE_SCALER_PATH, ISO_SCALER_PATH, load_scaler,
                               load_models, load_pipeline, load_columns_info, run_pipeline)

from schema import PredictionRequest, PredictionResponse

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    z_dim = 3
    ml_models["columns"], ml_models["input_dim"] = load_columns_info(COLUMN_METADATA_PATH, INPUT_METADATA_PATH)
    ml_models["vae"], ml_models["iso"], ml_models["device"] = load_models(VAE_MODEL_PATH , ISO_MODEL_PATH, ml_models["input_dim"], z_dim)
    ml_models["vae_pipeline"], ml_models["iso_pipeline"] = load_pipeline(VAE_PIPELINE_PATH,ISO_PIPELINE_PATH)
    ml_models["vae_scaler"], ml_models["iso_scaler"] = load_scaler(VAE_SCALER_PATH, ISO_SCALER_PATH)
    yield
    ml_models.clear()


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    df_t = pd.DataFrame([request.transaction.model_dump()])
    df_i = pd.DataFrame([request.identity.model_dump()]) if request.identity else pd.DataFrame()
    total_score = run_pipeline(df_t,df_i,ml_models["vae"],ml_models["iso"],ml_models["vae_pipeline"],
                               ml_models["iso_pipeline"],ml_models["columns"],ml_models["device"])
    return PredictionResponse(
    ensemble_score=float(total_score["ensemble_score"].iloc[0]),
    prediction=int(total_score["prediction"].iloc[0])
)
