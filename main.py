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
# pip install "fastapi[standard]"
# pip install "fastapi[standard]" scikit-learn numpy

# fastapi dev main.py

# https://github.com/sandeepmohamed666/Fast-API---Scikit-learn-trained-model-serving.git 

# =========================================================================================================
# Your system is blocking pip.exe (App Control policy), so use Python to invoke pip instead.
# Run these in PowerShell:

# PowerShell
# # inside your project folder
# .\.venv\Scripts\Activate.ps1

# python -m pip install --upgrade pip
# python -m pip install "fastapi[standard]"
# python -m pip install scikit-learn==1.6.0
# python -m fastapi dev main.py
# ------------------------------------------------------------------------------------------------------------------
# If python points to global Python, force venv Python directly:

# PowerShell
# .\.venv\Scripts\python.exe -m pip install "fastapi[standard]"
# .\.venv\Scripts\python.exe -m pip install scikit-learn==1.6.0
# .\.venv\Scripts\python.exe -m fastapi dev main.py
# If execution policy blocks activation, skip activation and always use .\.venv\Scripts\python.exe -m ....

# If you want, I can give you a one-shot script that installs everything and starts FastAPI without activation.
# =====================================================================================================================