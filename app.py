from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

regression_model = joblib.load("models/regression_model.joblib")
classification_model = joblib.load("models/classification_model.joblib")


class RegressionInput(BaseModel):
    gender: str
    occupation_type: str
    avg_work_hours_per_day: float
    avg_rest_hours_per_day: float
    avg_sleep_hours_per_day: float
    avg_exercise_hours_per_day: float


class ClassificationInput(BaseModel):
    gender: str
    avg_work_hours_per_day: float
    avg_rest_hours_per_day: float
    avg_sleep_hours_per_day: float
    avg_exercise_hours_per_day: float


@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()


@app.post("/predict/regression")
def predict_regression(data: RegressionInput):
    values = pd.DataFrame([data.model_dump()])
    prediction = regression_model.predict(values)[0]
    return {"prediction": round(float(prediction), 2)}


@app.post("/predict/classification")
def predict_classification(data: ClassificationInput):
    values = pd.DataFrame([data.model_dump()])
    prediction = classification_model.predict(values)[0]
    return {"prediction": str(prediction)}
