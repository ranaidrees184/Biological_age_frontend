from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
import joblib
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import time
import uvicorn
import os


from fastapi.middleware.cors import CORSMiddleware

# Path to your saved pickle file
model_path = "xgb_model_reg.pkl"  

# Load the model
model = joblib.load(model_path)

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, use specific domains
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

#pydantic Model to validate data
class UserInput(BaseModel):
    age: Annotated[int, Field(gt=0,description='Age of the Patient')]
    albumin_gL: Annotated[float, Field(gt=0,description='Quantity of Albumin in gL')]
    creat_umol: Annotated[float, Field(gt=0,description='Quantity of Creatnine in umol')]
    glucose_mmol: Annotated[float, Field(gt=0,description='Qunatity of Glucose in mmol')]
    lncrp:Annotated[float, Field(gt=0,description='Log of Crp')]
    lymph: Annotated[float, Field(gt=0,description='lym ph')]
    mcv: Annotated[float, Field(gt=0,description='mcv')]
    rdw: Annotated[float, Field(gt=0,description='rdw')]
    alp: Annotated[float, Field(gt=0,description='alp')]
    wbc: Annotated[float, Field(gt=0,description='white blood cell')]

@app.post('/predict')
def predict_premium(data: UserInput):
    try:
        input_df = pd.DataFrame(
            [
                {
                    'age': data.age,
                    'albumin_gL': data.albumin_gL,
                    'creat_umol': data.creat_umol,
                    'glucose_mmol': data.glucose_mmol,
                    'lncrp': data.lncrp,
                    'lymph': data.lymph,
                    'mcv': data.mcv,
                    'rdw': data.rdw,
                    'alp': data.alp,
                    'wbc': data.wbc
                }
            ]
        )

        prediction_value = float(model.predict(input_df)[0])  # <-- FIXED HERE

        return JSONResponse(
            status_code=200,
            content={"Predicted Biological Age of Patient": prediction_value}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

if __name__ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=10000)