
import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import tempfile
import subprocess



from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    Depends
)

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dotenv import load_dotenv

from auth import router as auth_router
from auth import get_current_user


# ==========================================================
# ROOT / ENVIRONMENT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")


from personalization.history import save_history, get_last_7
from personalization.trend import predict_future_risk


MONGO_URI = os.getenv("MONGO_URI")
JWT_SECRET = os.getenv("JWT_SECRET")
API_BASE = os.getenv("API_BASE", "http://localhost:8000")
HF_SPACE_URL = os.getenv("HF_SPACE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

from gradio_client import Client, handle_file

def get_hf_client():
    if not HF_SPACE_URL:
        return None
    try:
        return Client(HF_SPACE_URL, hf_token=HF_TOKEN)
    except Exception as e:
        print(f"Error connecting to HF Space: {e}")
        return None


# ==========================================================
# APP CONFIG
# ==========================================================

app = FastAPI(
    title="MindLens API",
    description="Multimodal Mental Health Analysis API",
    version="5.0.1"
)


app.include_router(
    auth_router,
    tags=["Authentication"]
)


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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"🖥️ Using device: {device}")


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEXT_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "text_model"
)

AUDIO_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "audio_model"
)

FACE_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "face_model"
)


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
            nn.Linear(
                in_features,
                512
            ),
            nn.GELU(),
            nn.Dropout(0.40)
        )

        self.classifier = nn.Linear(
            512,
            NUM_CLASSES
        )


    def forward(self, pixel_values):

        feat = self.backbone(
            pixel_values
        )

        emb = self.embedding(
            feat
        )

        logits = self.classifier(
            emb
        )

        return logits


# ==========================================================
# FACE TRANSFORMS
# ==========================================================

face_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================================
# STARTUP - LOAD MODELS
# ==========================================================

@app.on_event("startup")
def check_env():
    if not HF_SPACE_URL:
        print("⚠️ Warning: HF_SPACE_URL is not set!")
    else:
        print(f"✅ HF Space URL configured: {HF_SPACE_URL}")


# ==========================================================
# ROOT
# ==========================================================

@app.get("/")
def root():

    return {
        "status": "MindLens API Running",
        "version": "5.0.1"
    }


# ==========================================================
# TEXT
# ==========================================================

def predict_text_internal(text):
    client = get_hf_client()
    if not client:
        raise HTTPException(status_code=503, detail="HF_SPACE_URL not configured or unreachable")
    try:
        result = client.predict(text=text, api_name="/predict_text")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/text")
def predict_text(
    req: TextRequest
):

    if not req.text or not req.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )


    return predict_text_internal(
        req.text
    )


# ==========================================================
# AUDIO
# ==========================================================

def audio_emotion_to_risk(
    emotion
):

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

    return mapping.get(
        emotion,
        50
    )


# ==========================================================
# CHECK FFMPEG
# ==========================================================

def check_ffmpeg():

    try:

        result = subprocess.run(
            [
                "ffmpeg",
                "-version"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )

        return result.returncode == 0

    except (
        FileNotFoundError,
        subprocess.TimeoutExpired
    ):

        return False


# ==========================================================
# CONVERT AUDIO
# ==========================================================

def convert_audio_to_wav(
    input_path: str
) -> str:

    """
    Convert uploaded audio to a real
    16kHz mono PCM WAV.

    This handles common browser formats such as:

    - WebM
    - OGG
    - MP3
    - MP4/M4A
    - WAV
    - other FFmpeg-supported formats
    """


    if not check_ffmpeg():

        raise RuntimeError(
            "FFmpeg is not installed or not available "
            "in PATH. Install it with: brew install ffmpeg"
        )


    output_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    )

    output_path = output_file.name

    output_file.close()


    try:

        command = [

            "ffmpeg",

            "-y",

            "-hide_banner",

            "-loglevel",
            "error",

            "-i",
            input_path,

            # Convert to mono
            "-ac",
            "1",

            # 16 kHz required by Wav2Vec2
            "-ar",
            "16000",

            # Standard PCM WAV
            "-c:a",
            "pcm_s16le",

            output_path
        ]


        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )


        if result.returncode != 0:

            error_message = (
                result.stderr
                .decode(
                    "utf-8",
                    errors="replace"
                )
                .strip()
            )


            raise RuntimeError(
                "FFmpeg could not decode the uploaded "
                f"audio file: {error_message[:1000]}"
            )


        if not os.path.exists(
            output_path
        ):

            raise RuntimeError(
                "FFmpeg completed but did not "
                "create the WAV file."
            )


        if os.path.getsize(
            output_path
        ) == 0:

            raise RuntimeError(
                "FFmpeg created an empty WAV file."
            )


        return output_path


    except subprocess.TimeoutExpired:

        if os.path.exists(output_path):

            os.remove(output_path)


        raise RuntimeError(
            "Audio conversion timed out."
        )


    except Exception:

        if os.path.exists(output_path):

            os.remove(output_path)

        raise


# ==========================================================
# AUDIO PREDICTION
# ==========================================================

async def predict_audio_internal(file: UploadFile):
    client = get_hf_client()
    if not client:
        raise HTTPException(status_code=503, detail="HF_SPACE_URL not configured")
    
    raw_bytes = await file.read()
    import tempfile
    original_suffix = os.path.splitext(file.filename)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
        
    try:
        result = client.predict(audio_path=handle_file(tmp_path), api_name="/predict_audio")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


        if converted_path:

            try:

                if os.path.exists(
                    converted_path
                ):

                    os.remove(
                        converted_path
                    )

            except Exception:

                pass


# ==========================================================
# AUDIO ENDPOINT
# ==========================================================

@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(...)
):

    return await predict_audio_internal(
        file
    )


# ==========================================================
# FACE
# ==========================================================

def face_emotion_to_risk(
    emotion
):

    mapping = {

        "sad": 88,

        "fear": 78,

        "angry": 74,

        "disgust": 66,

        "neutral": 42,

        "surprise": 30,

        "happy": 12

    }

    return mapping.get(
        emotion,
        50
    )


async def predict_face_internal(file: UploadFile):
    client = get_hf_client()
    if not client:
        raise HTTPException(status_code=503, detail="HF_SPACE_URL not configured")
    
    raw_bytes = await file.read()
    import tempfile
    original_suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
        
    try:
        result = client.predict(image_path=handle_file(tmp_path), api_name="/predict_face")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ==========================================================
# FACE ENDPOINT
# ==========================================================

@app.post("/predict/face")
async def predict_face(
    file: UploadFile = File(...)
):

    return await predict_face_internal(
        file
    )


# ==========================================================
# FUSION
# ==========================================================

def fuse_predictions(
    text_result=None,
    audio_result=None,
    face_result=None
):

    weights = {

        "text": 0.45,

        "audio": 0.20,

        "face": 0.35

    }


    numerator = 0.0
    denominator = 0.0


    # ------------------------------------------------------
    # TEXT
    # ------------------------------------------------------

    if text_result:

        confidence = float(
            text_result["confidence"]
        )


        contribution = (
            weights["text"]
            * confidence
        )


        numerator += (
            text_result["risk_score"]
            * contribution
        )


        denominator += contribution


    # ------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------

    if audio_result:

        confidence = float(
            audio_result["confidence"]
        )


        contribution = (
            weights["audio"]
            * confidence
        )


        numerator += (
            audio_result["risk_score"]
            * contribution
        )


        denominator += contribution


    # ------------------------------------------------------
    # FACE
    # ------------------------------------------------------

    if face_result:

        confidence = float(
            face_result["confidence"]
        )


        contribution = (
            weights["face"]
            * confidence
        )


        numerator += (
            face_result["risk_score"]
            * contribution
        )


        denominator += contribution


    # ------------------------------------------------------
    # No valid prediction
    # ------------------------------------------------------

    if denominator == 0:

        return {
            "risk_score": 0,
            "risk_level": "Unknown"
        }


    final_score = round(
        numerator / denominator,
        2
    )


    # Clamp score to 0-100
    final_score = max(
        0,
        min(
            100,
            final_score
        )
    )


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

    current_user: dict = Depends(
        get_current_user
    )

):

    user_id = current_user["id"]


    text_result = None
    audio_result = None
    face_result = None


    # ------------------------------------------------------
    # TEXT
    # ------------------------------------------------------

    if text and text.strip():

        text_result = (
            predict_text_internal(
                text
            )
        )


    # ------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------

    if audio:

        audio_result = (
            await predict_audio_internal(
                audio
            )
        )


    # ------------------------------------------------------
    # FACE
    # ------------------------------------------------------

    if face:

        face_result = (
            await predict_face_internal(
                face
            )
        )


    # ------------------------------------------------------
    # Validate input
    # ------------------------------------------------------

    if (
        not text_result
        and not audio_result
        and not face_result
    ):

        raise HTTPException(
            status_code=400,
            detail="Provide at least one input: text, audio, or face"
        )


    # ------------------------------------------------------
    # Fusion
    # ------------------------------------------------------

    final_result = fuse_predictions(

        text_result=text_result,

        audio_result=audio_result,

        face_result=face_result

    )


    # ------------------------------------------------------
    # Personalization
    # ------------------------------------------------------

    depression = (
        final_result["risk_score"]
        / 100
    )

    anxiety = depression

    stress = depression

    risk = depression


    try:

        save_history(

            user_id=user_id,

            depression=depression,

            anxiety=anxiety,

            stress=stress,

            risk=risk

        )


    except Exception as e:

        print(
            "⚠️ History save failed:",
            repr(e)
        )


    # ------------------------------------------------------
    # Future risk
    # ------------------------------------------------------

    try:

        future_risk = (
            predict_future_risk(
                user_id
            )
        )

    except Exception as e:

        print(
            "⚠️ Future risk prediction failed:",
            repr(e)
        )

        future_risk = None


    # ------------------------------------------------------
    # Response
    # ------------------------------------------------------

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
def trend(

    current_user: dict = Depends(
        get_current_user
    )

):

    user_id = current_user["id"]


    future_risk = (
        predict_future_risk(
            user_id
        )
    )


    return {

        "user_id": user_id,

        "future_risk": future_risk

    }


# ==========================================================
# FULL REPORT
# ==========================================================

@app.get("/mental-health-report")
def report(

    current_user: dict = Depends(
        get_current_user
    )

):

    user_id = current_user["id"]


    history = get_last_7(
        user_id
    )


    future_risk = (
        predict_future_risk(
            user_id
        )
    )


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

        "records_found": len(
            history
        ),

        "future_risk": future_risk,

        "status": status,

        "history": history

    }
