from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load the trained scikit-learn model
with open("lr_model.pkl", "rb") as f:
    model = pickle.load(f)


# Define the input schema
class ModelInput(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.post("/predict")
def predict(data: ModelInput):
    """Predict using the trained scikit-learn model."""
    features = np.array([[data.feature1, data.feature2, data.feature3, data.feature4]])
    prediction = model.predict(features)[0]
    return {"prediction": prediction}

# Install requirements
# python -m pip install --upgrade pip
# pip install "fastapi[standard]" scikit-learn numpy

# fastapi dev main.py