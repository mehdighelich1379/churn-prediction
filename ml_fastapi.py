from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from pydantic import BaseModel

# Load model
pipe = joblib.load('src/models/XGBClassifier.joblib')

# Create app
app = FastAPI()

# ✅ Allow all origins (for testing / public)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
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

@app.get("/")
def home():
    return {"message": "Customer churn API is live with CORS ✅"}

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
    return {
        "churn_prediction": int(pred[0]),
        "churn_probability": prob[0].tolist()
    }
