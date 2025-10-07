
import pickle
import json
import torch
import os

def save_fn(model, model_type: str, PATH: str) -> None:
    metadata = {"model_type": model_type}
    with open(os.path.join(PATH, "model_metadata.json"), "w") as f:
            json.dump(metadata, f)
    if model_type in ["xgboost", "mlp", "sklearn", "randomforest"]:
        with open(os.path.join(PATH, "model.pkl"), "wb") as file:
            pickle.dump(model, file)
    elif model_type == "pytorch":
        torch.save(model, os.path.join(PATH, "model.pkl"))
    else:
        raise TypeError("Model is not one of sklearn, xgboost, or pytorch")
    return None