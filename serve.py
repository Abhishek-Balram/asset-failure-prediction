import os
import json
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import torch
from torch import nn

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
PREPROCESSOR_PATH = os.path.join(ARTIFACT_DIR, "preprocessing.pkl")
MODEL_WEIGHTS_PATH = os.path.join(ARTIFACT_DIR, "model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------
# Model definition (must match training script)
# --------------------------------------------------
class FeedForwardNet(nn.Module):
    """Simple feed-forward neural network matching the training pipeline."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
            nn.Linear(32, 1),  # output logits
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def load_artifacts():
    """Load the preprocessing pipeline and the trained PyTorch model."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessing pipeline not found at {PREPROCESSOR_PATH}")
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f"Model weights not found at {MODEL_WEIGHTS_PATH}")

    # Load preprocessing pipeline
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Determine input dimension from the saved model weights
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
    first_layer_weight_key = "net.0.weight"  # as defined in FeedForwardNet
    input_dim = state_dict[first_layer_weight_key].shape[1]

    # Instantiate model architecture and load weights
    model = FeedForwardNet(input_dim)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return preprocessor, model


def preprocess_input(records: List[Dict[str, Any]], preprocessor) -> np.ndarray:
    """Convert list of JSON records to the pre-processed NumPy array expected by the model."""
    df = pd.DataFrame.from_records(records)
    return preprocessor.transform(df)


def predict_proba(preprocessed_x: np.ndarray, model) -> np.ndarray:
    """Run model inference and return failure probabilities."""
    with torch.no_grad():
        tensor_x = torch.tensor(preprocessed_x, dtype=torch.float32, device=DEVICE)
        logits = model(tensor_x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return probs


# --------------------------------------------------
# Flask application
# --------------------------------------------------
app = Flask(__name__)
# Enable CORS so the API can be called from a separately hosted frontend
# For a demo setup we allow all origins. In production, restrict via env.
_cors_origins = os.getenv("CORS_ORIGINS", "*")
CORS(app, resources={r"/predict": {"origins": _cors_origins}, r"/health": {"origins": _cors_origins}})

# Load artifacts once at startup
PREPROCESSOR, MODEL = load_artifacts()


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict machine failure probability for given records.

    Expects JSON with the following schema:
    {
        "data": [
            {
                "Type": "L",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.5,
                "Rotational speed [rpm]": 1550,
                "Torque [Nm]": 40.5,
                "Tool wear [min]": 120,
                ... (all other numeric feature columns used in training)
            },
            ...
        ]
    }
    Returns:
    {
        "predictions": [
            {"failure_probability": 0.123, "predicted_label": 0},
            ...
        ]
    }
    """
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format."}), 415

    payload = request.get_json()
    if "data" not in payload:
        return jsonify({"error": "Missing 'data' field in JSON."}), 400

    records = payload["data"]
    if not isinstance(records, list):
        return jsonify({"error": "'data' must be a list of records."}), 400

    try:
        x_processed = preprocess_input(records, PREPROCESSOR)
        probs = predict_proba(x_processed, MODEL)
        preds = (probs >= 0.5).astype(int)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    results = [
        {"failure_probability": float(p), "predicted_label": int(lbl)}
        for p, lbl in zip(probs, preds)
    ]

    return jsonify({"predictions": results})


@app.route("/", methods=["GET"])
def index():
    """Serve a simple HTML page with a form to collect feature inputs and call the model API."""
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False) 