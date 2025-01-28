import pandas as pd
from fastapi import FastAPI
from nn_from_scratch.api import HeartData
from nn_from_scratch.api import load_pretrained_model
import torch
import numpy as np

model, preprocessing = load_pretrained_model("../models")

app = FastAPI()

numeric_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
binary_columns = ['sex', 'fbs', 'exang']
categorical_columns = ['cp', 'restecg', 'slope', 'ca', 'thal']

all_columns = (numeric_columns + binary_columns + categorical_columns)


@app.post("/", response_model=dict)
async def predict(data: HeartData) -> dict:
    input_data = pd.DataFrame([data.model_dump()])[all_columns].replace({None: np.nan})

    preprocessed_data = torch.tensor(preprocessing.transform(input_data)).to(torch.float32)

    with torch.no_grad():
        prediction = model(preprocessed_data).detach().item()

    return {"prediction": prediction}


@app.post("/batch", response_model=dict)
async def predict_batch(data: list[HeartData]) -> dict:
    """
    Batch inference endpoint
    """

    input_data = pd.DataFrame([record.model_dump() for record in data])[all_columns].replace({None: np.nan})

    preprocessed_data = torch.tensor(preprocessing.transform(input_data)).to(torch.float32)

    with torch.no_grad():
        predictions = model(preprocessed_data).squeeze(-1).detach().numpy().tolist()

    return {"predictions": predictions}


@app.get("/", response_model=dict)
async def health() -> dict:
    return {
        "version": "Heart-disease detector 0.1",
        "status": "OK"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=7878, reload=True)
