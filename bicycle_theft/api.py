import os
import json
from typing import cast, AsyncGenerator, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException

import lightgbm as lgb

from .models import PredictionResponse, PredictionRequest
from .predictor import Predictor
from .config import PROCESSED_DATA, MODEL_PATH


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Any, None]:
    with open(os.path.join(PROCESSED_DATA, "preproc.json"), "r") as f:
        app.state.artifacts = json.load(f)
    app.state.model = lgb.Booster(model_file=MODEL_PATH)
    app.state.service = Predictor(model=app.state.model, artifacts=app.state.artifacts)
    yield
    app.state.model = None
    app.state.artifacts = None
    app.state.service = None


def get_service() -> Predictor:
    return cast(Predictor, api.state.service)


api = FastAPI(
    title="Berlin Bicycle Thefts Prediction API",
    description="Application for predicting number of bicycle thefts in Berlin",
    version="1.0.0",
    lifespan=lifespan,
)


@api.post("/bicycle_thefts/predict", response_model=PredictionResponse)
async def predict(
    req: PredictionRequest, service: Predictor = Depends(get_service)
) -> PredictionResponse:
    df = service.build(req.items)
    result = service.predict(df)
    return PredictionResponse(items=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bicycle_theft.api:api",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
