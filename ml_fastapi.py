from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pydantic import BaseModel
from functools import lru_cache

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache()
def load_model():
    return joblib.load("src/models/XGBClassifier.joblib")


app.mount("/image", StaticFiles(directory="image"), name="image")


@app.get("/", response_class=FileResponse)
async def serve_homepage():
    return FileResponse(
        "index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
        media_type="text/html"
    )


class InputData(BaseModel):
    Age: float
    Gender: str
    Tenure: float
    Usage_Frequency: float
    Support_Calls: float
    Payment_Delay: float
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: float


@app.post("/predict")
def predict(data: InputData):
    pipe = load_model()
    df = pd.DataFrame([data.dict()])
    df.rename(columns={
        "Usage_Frequency": "Usage Frequency",
        "Support_Calls": "Support Calls",
        "Payment_Delay": "Payment Delay",
        "Subscription_Type": "Subscription Type",
        "Contract_Length": "Contract Length",
        "Total_Spend": "Total Spend",
        "Last_Interaction": "Last Interaction"
    }, inplace=True)
    pred = pipe.predict(df)
    prob = pipe.predict_proba(df)
    return {"churn_prediction": int(pred[0]), "churn_probability": prob[0].tolist()}
