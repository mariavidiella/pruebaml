from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
from utils.transformations import ExtendedTransformation, SimpleTransformation
from utils.filters import SimpleFilter
from sklearn.ensemble import GradientBoostingRegressor
import json


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# Variables globales
preprocessor = None
filter = None
model_80 = None
model_90 = None
model_99 = None

class PredictRequest(BaseModel):
    X: dict


# Cargar modelos
@app.post("/cargar_modelos/")
async def cargar_modelos(
    file_pre: UploadFile = File(...),
    file_filter: UploadFile = File(...),
    file_model_80: UploadFile = File(...),
    file_model_90: UploadFile = File(...),
    file_model_99: UploadFile = File(...),
):
    global preprocessor, filter, model_80, model_90, model_99

    if any([not f.filename.endswith(".pkl") for f in [file_pre, file_filter, file_model_80, file_model_90, file_model_99]]):
        raise HTTPException(status_code=400, detail="Todos los archivos deben ser .pkl")

    content_pre = await file_pre.read()
    content_filter = await file_filter.read()
    content_model_80 = await file_model_80.read()
    content_model_90 = await file_model_90.read()
    content_model_99 = await file_model_99.read()

    try:

        preprocessor = pickle.loads(content_pre)
        filter = pickle.loads(content_filter)
        model_80 = pickle.loads(content_model_80)
        model_90 = pickle.loads(content_model_90)
        model_99 = pickle.loads(content_model_99)

        return {"status": "Modelos cargados correctamente"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar modelos: {str(e)}")


def hacer_prediccion(modelo, request: PredictRequest):
    if any([m is None for m in [modelo, preprocessor, filter]]):
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    try:
        x_dict = request.X
        x_pd = pd.DataFrame(x_dict)
        dummy_y = pd.DataFrame([0]*len(x_pd))  # <- Añadido para cumplir la firma
        x_transform, _= preprocessor.transform(x_pd, dummy_y)
        x_filtered, _ = filter.transform(x_transform, None)
        y_pred, intervals = modelo.predict(x_filtered)
        y_pred_un = preprocessor.inverse_transform(y_pred.reshape(-1, 1)).tolist()
        y_low = preprocessor.inverse_transform(intervals[:, 0]).tolist()
        y_up = preprocessor.inverse_transform(intervals[:, 1]).tolist()

        return {"y_pred": y_pred_un, "y_low": y_low, "y_up": y_up}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_80/")
async def predict_80(request: PredictRequest):
    return hacer_prediccion(model_80, request)

@app.post("/predict_90/")
async def predict_90(request: PredictRequest):
    return hacer_prediccion(model_90, request)

@app.post("/predict_99/")
async def predict_99(request: PredictRequest):
    return hacer_prediccion(model_99, request)

