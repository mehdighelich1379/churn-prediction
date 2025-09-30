import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Load the pre-trained pipeline
pipe = joblib.load('src/models/XGBClassifier.joblib')
print("Model loaded successfully.")

# class for input data validation
#  1   Age                440832 non-null  float64
#  2   Gender             440832 non-null  object 
#  3   Tenure             440832 non-null  float64
#  4   Usage Frequency    440832 non-null  float64
#  5   Support Calls      440832 non-null  float64
#  6   Payment Delay      440832 non-null  float64
#  7   Subscription Type  440832 non-null  object 
#  8   Contract Length    440832 non-null  object 
#  9   Total Spend        440832 non-null  float64
#  10  Last Interaction   440832 non-null  float64
#  11  Churn              440832 non-null  float64
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


# Initialize FastAPI app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Customer Churn Prediction API!"}

@app.post("/predict")
def predict_churn(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    
    # rename columns to match training data
    input_df.rename(columns={
        "Usage_Frequency": "Usage Frequency",
        "Support_Calls": "Support Calls",
        "Payment_Delay": "Payment Delay",
        "Subscription_Type": "Subscription Type",
        "Contract_Length": "Contract Length",
        "Total_Spend": "Total Spend",
        "Last_Interaction": "Last Interaction"
    }, inplace=True)
    
    prediction = pipe.predict(input_df)
    prediction_proba = pipe.predict_proba(input_df)

    return {
        "churn_prediction": int(prediction[0]),
        "churn_probability": prediction_proba[0].tolist()
    }
