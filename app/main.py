from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from app.schemas import LoanInput
from app.model_loader import load_models

import numpy as np

app = FastAPI(title="Smart Loan Predictor")

# Load models ONCE at startup
model, preprocessor = load_models()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

@app.post("/predict")
def predict(data: LoanInput):

    if model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Models not loaded")

    # Encoding
    GENDER_MAP = {"male": 0, "female": 1}
    EDUCATION_MAP = {"High School": 0, "Bachelor": 1, "Master": 2, "Doctor": 3}
    HOME_OWNERSHIP_MAP = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
    LOAN_INTENT_MAP = {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2, "VENTURE": 3, "HOME": 4}
    DEFAULTS_MAP = {"NO": 0, "YES": 1}

    X = np.array([[
        GENDER_MAP.get(data.person_gender.lower(), 0),
        data.person_age,
        0,
        EDUCATION_MAP.get(data.person_education, 0),
        0,
        data.person_emp_exp,
        data.person_income,
        HOME_OWNERSHIP_MAP.get(data.person_home_ownership, 0),
        data.credit_score,
        DEFAULTS_MAP.get(data.previous_loan_defaults_on_file, 0),
        data.loan_amnt,
        LOAN_INTENT_MAP.get(data.loan_intent, 0)
    ]], dtype=np.float64)

    X_scaled = preprocessor.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": "Approved" if prediction == 1 else "Rejected",
        "probability": round(float(probability), 3)
    }

