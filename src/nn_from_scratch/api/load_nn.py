import json
import os.path
import torch
from ..train import HeartDiseaseDetector
import joblib


def load_pretrained_model(model_path):
    with open(os.path.join(model_path, "nn_parameters.json"), "r") as f:
        params = json.load(f)

    print(params)

    model = HeartDiseaseDetector(
        input_size=params['input_size'],
        hidden_sizes=params["hidden_sizes"],
        output_size=1,
        use_dropout=params["use_dropout"],
        use_batch_norm=params["use_batch_norm"],
        dropout_rate=params["dropout_rate"],
    )
    model = model.eval()
    model.load_state_dict(torch.load(os.path.join(model_path, "heart_disease_detector.pt"), map_location=torch.device("cpu"), weights_only=True))

    preprocessing = joblib.load(os.path.join(model_path, 'preprocess.joblib'))

    return model, preprocessing

