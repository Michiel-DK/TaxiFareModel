from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pandas as pd
import pytz
import joblib
from TaxiFareModel.gcp import BUCKET_NAME, download_model

from fastapi import FastAPI

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True, # Allows cookies
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {"greeting": "hello"}

@app.get("/predict")
def predict(pickup_datetime,  # 2013-07-06 17:18:00
            lon1,             # -73.950655
            lat1,             # 40.783282
            lon2,             # -73.984365
            lat2,             # 40.769802
            passcount):       # 1
    
    X_pred = pd.DataFrame({
    "key": ["truc"],
    "pickup_datetime": [pickup_datetime + " UTC"],
    "pickup_longitude": [float(lon1)],
    "pickup_latitude": [float(lat1)],
    "dropoff_longitude": [float(lon2)],
    "dropoff_latitude": [float(lat2)],
    "passenger_count": [int(passcount)]})
    
    pipeline = download_model()
    
    # make a prediction for the parameters passed to the API
    y_pred = pipeline.predict(X_pred)

    # return the prediction
    prediction = y_pred[0]
    
    return {"fare": prediction}