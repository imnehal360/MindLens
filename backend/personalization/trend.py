import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path

torch.set_num_threads(2)

from personalization.history import get_last_7


# LSTM Model
class EmotionLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()


# AI_Agent/ is a sibling of backend/ — both live under project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
AI_AGENT_DIR = ROOT_DIR / "AI_Agent"

MODEL_PATH = AI_AGENT_DIR / "models" / "emotion_lstm.pth"
SCALER_PATH = AI_AGENT_DIR / "scaler.pkl"

# Load LSTM model
model = EmotionLSTM()
model.load_state_dict(
    torch.load(str(MODEL_PATH), map_location="cpu")
)
model.eval()

# Load scaler
print("Loading scaler from:", SCALER_PATH)
scaler = joblib.load(str(SCALER_PATH))


def predict_future_risk(user_id):

    history = get_last_7(user_id)

    if len(history) < 7:
        return None

    arr = []

    for h in history:
        arr.append([
            h["depression"],
            h["anxiety"],
            h["stress"],
            h["risk"]
        ])

    arr = pd.DataFrame(arr, columns=[
        "depression",
        "anxiety",
        "stress",
        "risk"
    ])

    arr = scaler.transform(arr)

    x = torch.tensor(arr).float().unsqueeze(0)

    with torch.no_grad():
        pred = model(x).item()

    pred = max(0, min(1, pred))

    return round(pred, 4)