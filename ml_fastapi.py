from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import joblib
from pydantic import BaseModel

pipe = joblib.load('src/models/XGBClassifier.joblib')

app = FastAPI()

#  Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Serve static and HTML
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_homepage():
    return FileResponse("index.html")


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
