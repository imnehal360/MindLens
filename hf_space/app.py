# spaces MUST be imported before ANY other package (especially torch)
# This is a hard requirement for ZeroGPU Spaces on Hugging Face.
import spaces

import os
import sys
import pickle
import tempfile
import subprocess
import numpy as np
import gradio as gr

import torch
import torch.nn as nn
import librosa
import timm

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
from PIL import Image
from torchvision import transforms

# With ZeroGPU, CUDA is available inside @spaces.GPU decorated functions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Startup device: {device} (will use GPU inside @spaces.GPU calls)")

# ==========================================================
# PATHS
# All model files are uploaded flat to the root of the Space:
# tfidf_vectorizer.pkl, logistic_regression_model.pkl,
# config.json, preprocessor_config.json, model.safetensors,
# training_args.bin, face_model.pt
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_MODEL_DIR = BASE_DIR
AUDIO_MODEL_DIR = BASE_DIR
FACE_MODEL_DIR = BASE_DIR

# ==========================================================
# GLOBAL MODELS (loaded on CPU at startup)
# ==========================================================
vectorizer = None
text_model = None
feature_extractor = None
audio_model = None
face_model = None

audio_id2label = {
    0: "angry", 1: "calm", 2: "disgust", 3: "fearful",
    4: "happy", 5: "neutral", 6: "sad", 7: "surprised"
}

face_id2label = {
    0: "angry", 1: "disgust", 2: "fear", 3: "happy",
    4: "sad", 5: "surprise", 6: "neutral"
}

NUM_CLASSES = 7

class FaceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("convnext_tiny", pretrained=False, num_classes=0)
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

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    global vectorizer, text_model, feature_extractor, audio_model, face_model

    print(f"📁 BASE_DIR: {BASE_DIR}")
    print(f"📁 Files: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'NOT FOUND'}")

    # Text
    try:
        with open(os.path.join(TEXT_MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(TEXT_MODEL_DIR, "logistic_regression_model.pkl"), "rb") as f:
            text_model = pickle.load(f)
        print("✅ Text model loaded")
    except Exception as e:
        print("❌ Text model error:", repr(e))

    # Audio — loaded on CPU; moved to GPU inside @spaces.GPU call
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            AUDIO_MODEL_DIR, local_files_only=True
        )
        audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            AUDIO_MODEL_DIR, local_files_only=True
        )
        audio_model.eval()
        print("✅ Audio model loaded (CPU)")
    except Exception as e:
        print("❌ Audio model error:", repr(e))

    # Face — loaded on CPU; moved to GPU inside @spaces.GPU call
    try:
        face_model_obj = FaceModel()
        face_model_obj.load_state_dict(
            torch.load(
                os.path.join(FACE_MODEL_DIR, "face_model.pt"),
                map_location="cpu",
                weights_only=True
            )
        )
        face_model_obj.eval()
        face_model = face_model_obj
        print("✅ Face model loaded (CPU)")
    except Exception as e:
        print("❌ Face model error:", repr(e))

load_models()

# ==================== TEXT ====================
@spaces.GPU
def predict_text(text: str):
    if not text or not text.strip():
        raise gr.Error("Text cannot be empty")
    if vectorizer is None or text_model is None:
        raise gr.Error("Text model is not loaded")

    vec = vectorizer.transform([text])
    pred = int(text_model.predict(vec)[0])
    probs = text_model.predict_proba(vec)[0]
    confidence = float(max(probs))
    risk_score = confidence * 100 if pred == 1 else (1 - confidence) * 100

    return {
        "prediction": pred,
        "confidence": round(confidence, 4),
        "risk_score": round(risk_score, 2)
    }

# ==================== AUDIO ====================
def audio_emotion_to_risk(emotion):
    mapping = {"sad": 85, "fearful": 78, "angry": 72, "disgust": 65,
               "neutral": 40, "calm": 20, "happy": 15, "surprised": 30}
    return mapping.get(emotion, 50)

def convert_audio_to_wav(input_path: str) -> str:
    output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = output_file.name
    output_file.close()
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-i", input_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", output_path]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    if result.returncode != 0:
        raise gr.Error("FFmpeg could not decode the uploaded audio file.")
    return output_path

@spaces.GPU
def predict_audio(audio_path: str):
    if audio_model is None or feature_extractor is None:
        raise gr.Error("Audio model is not loaded")
    if not audio_path:
        raise gr.Error("Audio file is empty")

    # Move model to available device inside @spaces.GPU context
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_model.to(current_device)

    converted_path = None
    try:
        converted_path = convert_audio_to_wav(audio_path)
        audio, sr = librosa.load(converted_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise gr.Error("Audio file contains no samples.")
        audio = np.nan_to_num(audio).astype(np.float32)

        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(current_device)
        attention_mask = (
            inputs.attention_mask.to(current_device)
            if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None
            else None
        )

        with torch.no_grad():
            outputs = (
                audio_model(input_values=input_values, attention_mask=attention_mask)
                if attention_mask is not None
                else audio_model(input_values=input_values)
            )
            probs = torch.softmax(outputs.logits, dim=1)

        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
        emotion = audio_id2label[pred_id]

        return {
            "emotion": emotion,
            "confidence": round(confidence, 4),
            "risk_score": audio_emotion_to_risk(emotion)
        }
    except Exception as e:
        raise gr.Error(f"Audio error: {str(e)}")
    finally:
        if converted_path and os.path.exists(converted_path):
            os.remove(converted_path)

# ==================== FACE ====================
def face_emotion_to_risk(emotion):
    mapping = {"sad": 88, "fear": 78, "angry": 74, "disgust": 66,
               "neutral": 42, "surprise": 30, "happy": 12}
    return mapping.get(emotion, 50)

@spaces.GPU
def predict_face(image_path: str):
    if face_model is None:
        raise gr.Error("Face model is not loaded")
    if not image_path:
        raise gr.Error("Image file is empty")

    # Move model to available device inside @spaces.GPU context
    current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    face_model.to(current_device)

    try:
        img = Image.open(image_path).convert("RGB")
        img = face_transform(img).unsqueeze(0).to(current_device)

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
    except Exception as e:
        raise gr.Error(f"Face error: {str(e)}")

# ==================== GRADIO INTERFACE ====================
with gr.Blocks() as app:
    gr.Markdown("# MindLens Inference API")

    with gr.Tab("Text Prediction"):
        text_in = gr.Textbox(label="Input Text")
        text_out = gr.JSON(label="Prediction")
        text_btn = gr.Button("Predict")
        text_btn.click(fn=predict_text, inputs=text_in, outputs=text_out, api_name="predict_text")

    with gr.Tab("Audio Prediction"):
        audio_in = gr.Audio(type="filepath", label="Upload Audio")
        audio_out = gr.JSON(label="Prediction")
        audio_btn = gr.Button("Predict")
        audio_btn.click(fn=predict_audio, inputs=audio_in, outputs=audio_out, api_name="predict_audio")

    with gr.Tab("Face Prediction"):
        face_in = gr.Image(type="filepath", label="Upload Face Image")
        face_out = gr.JSON(label="Prediction")
        face_btn = gr.Button("Predict")
        face_btn.click(fn=predict_face, inputs=face_in, outputs=face_out, api_name="predict_face")

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
