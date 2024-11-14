from fastapi import FastAPI, HTTPException

import streamlit as st
import numpy as np
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Tuwaiq Academy"}

@app.get("/items/")
def create_item(item: dict):
    return {"item": item}

import joblib
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

from pydantic import BaseModel
class InputFeatures(BaseModel):
    age: int
    appearance: int
    minutes_played: int
    award: int
    highest_value: int
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
    'age': input_features.age,
    'appearance': input_features.appearance,
    'minutes_played': input_features.minutes_played,
    'award': input_features.award,
    'highest_value': input_features.highest_value
    }

    features_list = [dict_f[key] for key in sorted(dict_f)]


    scaled_features = scaler.transform([list(dict_f.values
    ())])

    return scaled_features
@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}



st.title("PLayer value prediction")

age = st.number_input("age", min_value=15, max_value=45, value=20, step=1)
appearance = st.number_input("appearance", min_value=0,max_value=500, value=50)
minutes_played = st.number_input("minutes_played", min_value=0,max_value=11000, value=1000)
award = st.number_input("award", min_value=0,max_value=100, value=25)
highest_value = st.number_input("highest_value", min_value=0,max_value=30000000000, value=50000000)

if st.button("Predict"):
    input_data = InputFeatures(
        age=age,
        appearance=appearance,
        minutes_played=minutes_played,
        award=award,
        highest_value=highest_value,
    )

    processed_data = preprocessing(input_data)
    prediction = model.predict(processed_data)

    st.success(f"Predicted Cluster: {prediction[0]}")
