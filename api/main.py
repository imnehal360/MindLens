import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import tempfile

import torch
import torch.nn as nn
import librosa
import timm
import shap
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

from PIL import Image
from torchvision import transforms

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from auth import router as auth_router
from auth import get_current_user


import sys

# Load root .env file (MINDLENS/.env)
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from personalization.history import save_history, get_last_7
from personalization.trend import predict_future_risk

# Environment variables
MONGODB_URI = os.getenv("MONGODB_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")




# ---------------- APP CONFIG ----------------
app = FastAPI(
    title="MindLens API",
    description="Multimodal Mental Health Analysis API",
    version="5.0.0"
)

app.include_router(auth_router, tags=["Authentication"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# DEVICE
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================================
# PATHS
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEXT_MODEL_DIR = os.path.join(BASE_DIR, "models", "text_model")
AUDIO_MODEL_DIR = os.path.join(BASE_DIR, "models", "audio_model")
FACE_MODEL_DIR = os.path.join(BASE_DIR, "models", "face_model")

# ==========================================================
# GLOBAL MODELS
# ==========================================================
vectorizer = None
text_model = None

feature_extractor = None
audio_model = None

face_model = None

# ==========================================================
# LABELS
# ==========================================================
audio_id2label = {
    0: "angry",
    1: "calm",
    2: "disgust",
    3: "fearful",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprised"
}

face_id2label = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# ==========================================================
# REQUEST MODEL
# ==========================================================
class TextRequest(BaseModel):
    text: str

# ==========================================================
# FACE MODEL
# ==========================================================
NUM_CLASSES = 7

class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=0
        )

        in_features = self.backbone.num_features

        self.embedding = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.40)
        )

        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, pixel_values):
        feat = self.backbone(pixel_values)
        emb = self.embedding(feat)
        logits = self.classifier(emb)
        return logits

# ==========================================================
# TRANSFORMS
# ==========================================================
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================================
# STARTUP LOAD MODELS
# ==========================================================
@app.on_event("startup")
def load_models():
    global vectorizer, text_model
    global feature_extractor, audio_model
    global face_model

    # TEXT
    try:
        with open(os.path.join(TEXT_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)

        with open(os.path.join(TEXT_MODEL_DIR, "logistic_regression_model.pkl"), "rb") as f:
            text_model = pickle.load(f)

        print("✅ Text model loaded")

    except Exception as e:
        print("❌ Text model error:", e)

    # AUDIO
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(AUDIO_MODEL_DIR)

        audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_DIR
        )

        audio_model.to(device)
        audio_model.eval()

        print("✅ Audio model loaded")

    except Exception as e:
        print("❌ Audio model error:", e)

    # FACE
    try:
        face_model = FaceModel()

        face_model.load_state_dict(
            torch.load(
                os.path.join(FACE_MODEL_DIR, "face_model.pt"),
                map_location=device
            )
        )

        face_model.to(device)
        face_model.eval()

        print("✅ Face model loaded")

    except Exception as e:
        print("❌ Face model error:", e)

# ==========================================================
# ROOT
# ==========================================================
@app.get("/")
def root():
    return {
        "status": "MindLens API Running",
        "version": "5.0.0"
    }

# ==========================================================
# TEXT
# ==========================================================
def predict_text_internal(text):

    vec = vectorizer.transform([text])

    pred = int(text_model.predict(vec)[0])

    probs = text_model.predict_proba(vec)[0]
    confidence = float(max(probs))

    if pred == 1:
        risk_score = confidence * 100
    else:
        risk_score = (1 - confidence) * 100

    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "risk_score": round(risk_score, 2)
    }

@app.post("/predict/text")
def predict_text(req: TextRequest):
    return predict_text_internal(req.text)

# ==========================================================
# TEXT EXPLAINABILITY
# ==========================================================

# Load X_train for SHAP explainer
# with open(os.path.join(TEXT_MODEL_DIR, "X_train.pkl"), "rb") as f:
#     X_train = pickle.load(f)

# Initialize explainer once at startup
# explainer = shap.LinearExplainer(text_model, X_train)

# @app.post("/explain/text")
# def explain_text(req: TextRequest):
#     x_input = vectorizer.transform([req.text])
#     shap_values = explainer(x_input)
#     feature_names = vectorizer.get_feature_names_out()
#     vals = shap_values.values[0]
#     top_idx = np.argsort(np.abs(vals))[::-1][:10]
#     important_words = []
#     for i in top_idx:
#         if vals[i] != 0:
#             important_words.append({
#                 "word": feature_names[i],
#                 "impact": round(float(vals[i]), 4)
#             })
#     return {
#         "important_words": important_words
#     }

# ==========================================================
# AUDIO
# ==========================================================
def audio_emotion_to_risk(emotion):
    mapping = {
        "sad": 85,
        "fearful": 78,
        "angry": 72,
        "disgust": 65,
        "neutral": 40,
        "calm": 20,
        "happy": 15,
        "surprised": 30
    }
    return mapping.get(emotion, 50)

async def predict_audio_internal(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        audio, sr = librosa.load(path, sr=16000)

        inputs = feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = audio_model(
                inputs.input_values.to(device)
            ).logits

            probs = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

        emotion = audio_id2label[pred_id]

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "risk_score": audio_emotion_to_risk(emotion)
        }

    finally:
        os.remove(path)

@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    return await predict_audio_internal(file)

# ==========================================================
# FACE
# ==========================================================
def face_emotion_to_risk(emotion):
    mapping = {
        "sad": 88,
        "fear": 78,
        "angry": 74,
        "disgust": 66,
        "neutral": 42,
        "surprise": 30,
        "happy": 12
    }
    return mapping.get(emotion, 50)

async def predict_face_internal(file: UploadFile):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        img = Image.open(path).convert("RGB")
        img = face_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = face_model(img)
            probs = torch.softmax(logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()

        emotion = face_id2label[pred_id]

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "risk_score": face_emotion_to_risk(emotion)
        }

    finally:
        os.remove(path)

@app.post("/predict/face")
async def predict_face(file: UploadFile = File(...)):
    return await predict_face_internal(file)

# ==========================================================
# FUSION
# ==========================================================
def fuse_predictions(text_result=None, audio_result=None, face_result=None):

    weights = {
        "text": 0.45,
        "audio": 0.20,
        "face": 0.35
    }

    numerator = 0
    denominator = 0

    if text_result:
        c = weights["text"] * text_result["confidence"]
        numerator += text_result["risk_score"] * c
        denominator += c

    if audio_result:
        c = weights["audio"] * audio_result["confidence"]
        numerator += audio_result["risk_score"] * c
        denominator += c

    if face_result:
        c = weights["face"] * face_result["confidence"]
        numerator += face_result["risk_score"] * c
        denominator += c

    if denominator == 0:
        return {"risk_score": 0, "risk_level": "Unknown"}

    final_score = round(numerator / denominator, 2)

    if final_score < 35:
        level = "Low"
    elif final_score < 65:
        level = "Moderate"
    else:
        level = "High"

    return {
        "risk_score": final_score,
        "risk_level": level
    }

# ==========================================================
# MAIN FUSION + PERSONALIZATION
# ==========================================================

@app.post("/predict/fusion")
async def predict_fusion(
    text: str = Form(None),
    audio: UploadFile = File(None),
    face: UploadFile = File(None),
    current_user: dict = Depends(get_current_user)
):
    user_id = current_user["id"]

    text_result = None
    audio_result = None
    face_result = None

    if text and text.strip():
        text_result = predict_text_internal(text)

    if audio:
        audio_result = await predict_audio_internal(audio)

    if face:
        face_result = await predict_face_internal(face)

    if not text_result and not audio_result and not face_result:
        raise HTTPException(status_code=400, detail="Provide input")

    final_result = fuse_predictions(
        text_result,
        audio_result,
        face_result
    )

    depression = final_result["risk_score"] / 100
    anxiety = depression
    stress = depression
    risk = depression

    save_history(
        user_id=user_id,
        depression=depression,
        anxiety=anxiety,
        stress=stress,
        risk=risk
    )

    future_risk = predict_future_risk(user_id)

    return {
        "status": "success",
        "user_id": user_id,
        "text_result": text_result,
        "audio_result": audio_result,
        "face_result": face_result,
        "final_result": final_result,
        "future_risk": future_risk
    }




# ==========================================================
# TREND API
# ==========================================================

@app.get("/trend")
def trend(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]

    future_risk = predict_future_risk(user_id)

    return {
        "user_id": user_id,
        "future_risk": future_risk
    }

# ==========================================================
# FULL REPORT
# ==========================================================
@app.get("/mental-health-report")
def report(current_user: dict = Depends(get_current_user)):
    user_id = current_user["id"]

    history = get_last_7(user_id)

    future_risk = predict_future_risk(user_id)

    if future_risk is None:
        status = "Need more data"
    elif future_risk < 0.35:
        status = "Stable"
    elif future_risk < 0.65:
        status = "Moderate Attention"
    else:
        status = "High Risk Trend"

    return {
        "user_id": user_id,
        "records_found": len(history),
        "future_risk": future_risk,
        "status": status,
        "history": history
    }