from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError, validator
from typing import Annotated
import joblib
import pandas as pd
import numpy as np
import time
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware
import traceback
from math import log

# Path to your saved pickle file
model_path = "xgb_model_reg.pkl"

# Load the model
model = joblib.load(model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Pydantic Model to validate data
class UserInput(BaseModel):
    age_years: Annotated[int, Field(gt=0, lt=120, description="Chronological age in years")]
    albumin_gl: Annotated[float, Field(ge=20, le=60, description="Serum albumin concentration in g/L")]
    creatinine_umoll: Annotated[float, Field(gt=0, description="Serum creatinine level in μmol/L")]
    glucose_mmoll: Annotated[float, Field(gt=0, description="Blood glucose level in mmol/L")]
    crp_mgdl: Annotated[float, Field(gt=0, description="C-reactive protein level in mg/dL")]
    lymphocyte_percent: Annotated[float, Field(ge=0, le=100, description="Lymphocyte percentage")]
    mcv_fl: Annotated[float, Field(gt=0, description="Mean corpuscular volume in fL")]
    rdw_percent: Annotated[float, Field(ge=0, le=20, description="Red blood cell distribution width in %")]
    alkp_ul: Annotated[float, Field(gt=0, description="Alkaline phosphatase level in U/L")]
    wbc_10_9l: Annotated[float, Field(gt=0, description="White blood cell count in ×10^9/L")]

    # Validator for CRP to ensure it's positive for log transformation
    @validator('crp_mgdl')
    def crp_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('CRP must be greater than 0')
        return v

@app.post('/predict')
async def predict_premium(data: UserInput):
    try:
        start_time = time.time()
        
        # Calculate lnCRP from CRP
        lncrp = log(data.crp_mgdl)
        
        input_df = pd.DataFrame(
            [
                {
                    'age': data.age_years,
                    'albumin_gL': data.albumin_gl,
                    'creat_umol': data.creatinine_umoll,
                    'glucose_mmol': data.glucose_mmoll,
                    'lncrp': lncrp,
                    'lymph': data.lymphocyte_percent,
                    'mcv': data.mcv_fl,
                    'rdw': data.rdw_percent,
                    'alp': data.alkp_ul,
                    'wbc': data.wbc_10_9l
                }
            ]
        )

        prediction_value = float(model.predict(input_df)[0])
        processing_time = time.time() - start_time

        return JSONResponse(
            status_code=200,
            content={
                "predicted_biological_age": prediction_value,
                "status": "success",
                "model_type": "XGBoost Regression",
                "processing_time": f"{processing_time:.2f} seconds"
            }
        )
        
    except ValidationError as e:
        # Handle Pydantic validation errors
        return JSONResponse(
            status_code=422,
            content={"detail": e.errors()}
        )
        
    except ValueError as e:
        # Handle specific value errors
        return JSONResponse(
            status_code=400,
            content={"detail": [{"msg": str(e)}]}
        )
        
    except Exception as e:
        # Log the full error for debugging
        traceback.print_exc()
        
        # Return a generic server error
        return JSONResponse(
            status_code=500,
            content={
                "detail": [{
                    "msg": "Internal server error",
                    "type": "server_error"
                }]
            }
        )

# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)