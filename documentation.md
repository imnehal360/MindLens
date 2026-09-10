# MindLens: Multimodal Mental Health Assessment & Forecasting Platform
### System Architecture, Model Specifications, and Technical Documentation

---

## 1. Executive Summary & Problem Statement

Traditional psychiatric and psychological evaluations heavily depend on periodic, subjective self-reporting questionnaires (e.g., PHQ-9, GAD-7). These assessment methods are prone to recall bias, social stigma, and blunted patient self-disclosure. 

**MindLens** is an AI-powered multimodal platform engineered to provide objective, non-invasive, real-time mental wellness assessments and longitudinal trend forecasting. By fusing signals across three distinct human communication modalities—**verbal semantics (Text)**, **acoustic prosody (Voice)**, and **facial micro-expressions (Vision)**—MindLens delivers a holistic mental health risk index. The system incorporates:
1. **Explainable AI (SHAP)** to ensure clinical transparency.
2. **Confidence-Weighted Late Fusion** to handle missing or noisy sensory data.
3. **Deep Sequence Modeling (LSTM)** to track temporal wellness trajectories over a multi-day sliding window.

---

## 2. End-to-End System Architecture

The following Mermaid diagram outlines the complete end-to-end data pipeline, including client presentation, ingestion gateway, multimodal inference backbones, fusion layer, persistent storage, and temporal forecasting:

```mermaid
flowchart TB
    %% Styling Configuration
    classDef client fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4;
    classDef gateway fill:#181825,stroke:#cba6f7,stroke-width:2px,color:#cdd6f4;
    classDef model fill:#11111b,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4;
    classDef fusion fill:#313244,stroke:#f38ba8,stroke-width:2px,color:#cdd6f4;
    classDef storage fill:#181825,stroke:#fab387,stroke-width:2px,color:#cdd6f4;
    classDef xai fill:#181825,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4;

    %% Client Layer
    subgraph Frontend["Streamlit Presentation Layer"]
        UI_Auth["Authentication View (JWT Login / Signup)"]:::client
        UI_Input["Assessment Portal (Text / Audio Upload / Webcam Snapshot)"]:::client
        UI_Dash["Analytics Dashboard (7-Day Longitudinal Trajectory & SHAP View)"]:::client
    end

    %% Gateway & Preprocessing Layer
    subgraph Backend["FastAPI Orchestration Gateway"]
        API_Auth["OAuth2 / JWT Security Layer (HS256, Bcrypt)"]:::gateway
        API_Routes["Routing Controller (/predict/fusion, /mental-health-report)"]:::gateway
        Audio_Conv["FFmpeg Transcoding Pipeline (Mono 16kHz PCM WAV)"]:::gateway
        Img_Prep["Torchvision Pipeline (224x224, Normalized)"]:::gateway
    end

    %% Multimodal Inference Engines
    subgraph ModelLayer["Multimodal Model Inference Layer"]
        subgraph TextMod["Text Processing Engine"]
            TFIDF["TF-IDF Vectorizer (N-gram Vocabulary)"]:::model
            LogReg["Logistic Regression Classifier"]:::model
            SHAP["SHAP LinearExplainer (Word Impact Attribution)"]:::xai
        end

        subgraph AudioMod["Acoustic Emotion Recognition (SER) Engine"]
            W2V_FE["Wav2Vec2 Feature Extractor"]:::model
            W2V["Wav2Vec2 Sequence Classifier (Fine-tuned for Emotion)"]:::model
        end

        subgraph VisionMod["Facial Emotion Recognition Engine"]
            ConvNeXt["ConvNeXt-Tiny CNN Backbone"]:::model
            CustomHead["512-dim Embedding Head + GELU + 40% Dropout"]:::model
            FaceClassifier["7-Class Linear Softmax Classifier"]:::model
        end
    end

    %% Decision & Fusion Layer
    subgraph FusionEngine["Decision & Fusion Layer"]
        LateFusion["Confidence-Weighted Late Fusion
        Score = Σ(Weight × Confidence × Risk) / Σ(Weight × Confidence)
        Weights: Text = 0.45, Face = 0.35, Audio = 0.20"]:::fusion
        Stratification["Risk Level Categorization
        • Low: < 35
        • Moderate: 35 - 65
        • High: ≥ 65"]:::fusion
    end

    %% Persistence & Temporal Forecasting
    subgraph Persistence["Persistence & Longitudinal Forecasting"]
        MongoDB[("MongoDB Atlas (User Profiles & Longitudinal History)")]:::storage
        Scaler["StandardScaler (Feature Normalization)"]:::storage
        LSTM["EmotionLSTM (Stacked 2-Layer LSTM, Hidden=64)"]:::model
        Forecast["Trajectory Classifier (Stable / Moderate Attention / High Risk Trend)"]:::fusion
    end

    %% Data Flow Connections
    UI_Auth -->|User Credentials| API_Auth
    API_Auth -->|JWT Bearer Token| UI_Input
    UI_Input -->|Linguistic Text| API_Routes
    UI_Input -->|Browser Audio| Audio_Conv
    Audio_Conv -->|16kHz PCM WAV| API_Routes
    UI_Input -->|Facial Image| Img_Prep
    Img_Prep -->|Tensor| API_Routes

    API_Routes --> TFIDF --> LogReg
    TFIDF -.-> SHAP -.-> UI_Dash
    API_Routes --> W2V_FE --> W2V
    API_Routes --> ConvNeXt --> CustomHead --> FaceClassifier

    LogReg -->|Risk & Probability| LateFusion
    W2V -->|Emotion Class & Confidence| LateFusion
    FaceClassifier -->|Emotion Class & Confidence| LateFusion

    LateFusion --> Stratification
    Stratification -->|Session Risk Vector| MongoDB
    Stratification -->|Assessment Summary| UI_Dash

    MongoDB -->|Sliding Window (Last 7 Sessions)| Scaler
    Scaler --> LSTM --> Forecast
    Forecast -->|Longitudinal Status| UI_Dash
```

---

## 3. Deep Dive: Which Model is Used for What and Why?

| Modality / Task | Model Architecture | Core Task | Why This Specific Model? (Design Rationale) |
| :--- | :--- | :--- | :--- |
| **Linguistic Analysis (Text)** | **TF-IDF Vectorizer + Logistic Regression** | Detect depressive & anxious markers in user notes, journal entries, or transcripts. | **1. High Explainability via SHAP:** Black-box models are unsafe for clinical guidance. Logistic regression coefficients cleanly integrate with `SHAP LinearExplainer` to reveal token-level attributions.<br>**2. Sub-5ms CPU Latency:** Ultra-fast, zero GPU dependency.<br>**3. Deterministic Calibration:** Does not suffer from large generative model hallucinations or prompt brittleness. |
| **Speech Emotion Recognition (Audio)** | **Wav2Vec2** (`Wav2Vec2ForSequenceClassification`) | Classifies 8 speech emotional states (*Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, Surprised*). | **1. End-to-End Self-Supervised Acoustic Representations:** Pretrained on 10,000+ hours of raw audio; captures micro-prosodic cues (pitch jitter, speech cadence, vocal tremors) without hand-crafted spectrogram/MFCC loss.<br>**2. Acoustic Biomarker Sensitivity:** Mental health conditions like clinical depression manifest as flattened or monotonous vocal inflections, which Wav2Vec2 detects directly from raw 16kHz waveforms. |
| **Facial Expression Analysis (Vision)** | **ConvNeXt-Tiny** (`timm` backbone + custom 512-dim embedding head) | Classifies 7 facial expressions (*Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*). | **1. Modernized ConvNet with ViT Performance:** Adopts large 7×7 depthwise kernels, inverted bottlenecks, and GELU activations, achieving Vision Transformer accuracy while maintaining convolutional inductive bias.<br>**2. Spatial Locality for Micro-Expressions:** ConvNeXt captures local facial action units (furrowed brow, downturned lip corners) effectively on smaller datasets.<br>**3. Low Footprint:** ~28M parameters, offering fast real-time inference without high VRAM requirements. |
| **Longitudinal Trend Forecasting** | **Stacked 2-Layer LSTM** (`EmotionLSTM`, 64 hidden units, 20% Dropout) | Ingests a 7-step sequence of multi-attribute vectors `[depression, anxiety, stress, risk]` to forecast future risk. | **1. Temporal Memory vs. Static Snapshots:** A single bad day is an emotional fluctuation, not depression. LSTMs model recurring patterns, momentum, and temporal deterioration over time.<br>**2. Vanishing Gradient Prevention:** Internal gating mechanisms retain dependencies over 7+ observation steps. |

---

## 4. Multimodal Fusion Engine

### Dynamic Confidence-Weighted Late Fusion
MindLens implements **Late Decision Fusion** rather than early feature concatenation. The final score is computed as:

$$\text{Final Risk Score} = \frac{\sum_{m \in M} w_m \cdot c_m \cdot R_m}{\sum_{m \in M} w_m \cdot c_m}$$

Where:
* $M \subseteq \{\text{text}, \text{audio}, \text{face}\}$: The active subset of available modalities.
* $w_m$: Static modality base weight:
  * $\mathbf{w_{\text{text}} = 0.45}$ (Highest weight; explicit verbal self-disclosure contains direct semantic intent).
  * $\mathbf{w_{\text{face}} = 0.35}$ (Visual non-verbal micro-expressions provide clear affective validation).
  * $\mathbf{w_{\text{audio}} = 0.20}$ (Acoustic prosody serves as auxiliary emotional confirmation).
* $c_m \in [0.0, 1.0]$: Model output confidence / predicted class probability for modality $m$.
* $R_m \in [0, 100]$: Modality-specific clinical risk score.

### Emotion-to-Clinical-Risk Mapping Matrix
Because Wav2Vec2 and ConvNeXt output discrete emotional classes, an empirical clinical mapping translates these classes into continuous risk points:

| Emotion Class | Audio Risk Value ($R_{\text{audio}}$) | Face Risk Value ($R_{\text{face}}$) | Clinical Rationale |
| :--- | :---: | :---: | :--- |
| **Sad** | 85 | 88 | Core diagnostic criteria for depressive episodes and distress. |
| **Fear / Fearful** | 78 | 78 | Primary indicator of severe panic, acute anxiety, and trauma response. |
| **Angry** | 72 | 74 | Correlates with irritability, agitation, and emotional dysregulation. |
| **Disgust** | 65 | 66 | Often associated with somatic distress and severe aversion. |
| **Neutral** | 40 | 42 | Baseline baseline affect; neutral/blunted affect carries moderate monitoring weight. |
| **Surprise** | 30 | 30 | Transient arousal state; low sustained risk. |
| **Calm** | 20 | — | Parasympathetic equilibrium; indicative of emotional stability. |
| **Happy** | 15 | 12 | Positive affect and emotional resilience; minimal risk. |

### Risk Level Stratification
* **Low Risk:** $\text{Final Score} < 35$
* **Moderate Risk:** $35 \le \text{Final Score} < 65$
* **High Risk:** $\text{Final Score} \ge 65$

---

## 5. Temporal Forecasting & Personalization (LSTM)

### Pipeline
1. **Event Ingestion:** Following each `/predict/fusion` call, the session vector $(\text{depression}, \text{anxiety}, \text{stress}, \text{risk})$ is stored under the authenticated `user_id`.
2. **Sliding Window:** When computing `/mental-health-report` or `/trend`, the system fetches the user's last 7 records.
3. **Scaling:** Data passes through a pre-fitted `StandardScaler` (`scaler.pkl`) to normalize variance across dimensions.
4. **LSTM Inference:** The normalized $1 \times 7 \times 4$ tensor passes through `EmotionLSTM`:
   * Input features: 4
   * Hidden units: 64 (2 layers)
   * Dropout: 0.2
   * Linear Output: 1 scalar bounded between $[0, 1]$.
5. **Categorization:**
   * $\text{Forecast} < 0.35 \implies \textbf{Stable}$
   * $0.35 \le \text{Forecast} < 0.65 \implies \textbf{Moderate Attention}$
   * $\text{Forecast} \ge 0.65 \implies \textbf{High Risk Trend}$
   * $< 7 \text{ entries} \implies \textbf{Need more data}$ (Cold-start safety policy)

---

## 6. Explainability & Trust (SHAP)

MindLens integrates `shap.LinearExplainer` on the text classification pipeline:
* **Token-Level Attribution:** Extracts the exact positive and negative SHAP values for individual words in the user's input.
* **Clinical Value:** Highlights whether terms such as *"hopeless"*, *"worthless"*, or *"overwhelmed"* contributed positively to the risk score, or if stabilizing words like *"hope"*, *"peace"*, or *"calm"* reduced the score.
* **Patient Trust:** Patients and clinicians can inspect the underlying factors driving automated scoring, mitigating the "black box" concern.

---

## 7. Media Preprocessing & Production Engineering

### Audio Ingestion with FFmpeg
* **Challenge:** Audio recorded in web browsers through WebRTC / Streamlit arrives in variable formats (`.webm`, `.ogg`, `.mp4`) and varying sample rates.
* **Solution:** An automated subprocess converts any incoming audio payload to a standardized **16 kHz, single-channel (mono), 16-bit PCM WAV** stream using `ffmpeg -y -i input.bin -ac 1 -ar 16000 -c:a pcm_s16le output.wav`.
* **Sanitization:** Temporary files are generated via `tempfile.NamedTemporaryFile` and removed inside `finally` blocks to prevent memory leaks and disk bloat.

### Image Pipeline
* Uses PIL and `torchvision.transforms` to convert incoming files into normalized $(3, 224, 224)$ tensors using standard ImageNet mean and variance:
  * Mean: `[0.485, 0.456, 0.406]`
  * Std: `[0.229, 0.224, 0.225]`

---

## 8. API Specifications & Endpoints

All protected endpoints require an `Authorization: Bearer <JWT_TOKEN>` header.

### Authentication
* `POST /signup`: Registers a new user with name, email, and bcrypt-hashed password.
* `POST /login`: Validates credentials and returns an HS256-signed JWT token valid for 7 days.
* `GET /me`: Returns the profile information of the current authenticated user.

### Inference & Analytics
* `POST /predict/text`: Accepts JSON `{"text": "..."}`, returns text prediction, confidence, and risk score.
* `POST /predict/audio`: Multipart form upload (`file: UploadFile`), returns audio emotion, confidence, and risk score.
* `POST /predict/face`: Multipart form upload (`file: UploadFile`), returns facial emotion, confidence, and risk score.
* `POST /predict/fusion`: Form data (`text`, `audio`, `face`). Executes late fusion, saves assessment to history, and runs trend prediction.
* `GET /trend`: Returns the 7-day predicted future risk for the current user.
* `GET /mental-health-report`: Returns full history records, longitudinal trend status, and future risk projection.

---

## 9. Interview Defense: Common Questions & Sample Answers

### Q1: Why use Confidence-Weighted Late Fusion instead of Early Feature Fusion?
> **Answer:** *"Early fusion concatenates feature embeddings into a shared dense representation. However, in real-world clinical or mobile usage, users rarely provide all modalities simultaneously (e.g., a patient may refuse camera access or type without audio). Early fusion either fails or requires complex vector imputation when a modality is missing. Late fusion gracefully computes the weighted average of whichever subset of modalities is present. Furthermore, weighting by model confidence ensures that noisy signals (e.g., distorted audio with 30% confidence) do not degrade the overall assessment."*

### Q2: Why didn't you use a large language model (e.g., LLaMA or GPT) for text classification?
> **Answer:** *"In mental health applications, explainability, predictability, and low latency are paramount. Logistic regression with TF-IDF allows direct feature attribution via SHAP LinearExplainer, showing the exact words that contributed to the score. It executes in less than 5 milliseconds on CPU, eliminates hallucination risks, and operates without costly GPU infrastructure."*

### Q3: How do you handle cold-start users who have fewer than 7 check-ins?
> **Answer:** *"For instant sessions, the system evaluates the real-time multimodal inputs via the late fusion layer immediately. The longitudinal LSTM model requires a minimum sequence length of 7 records. Until the user logs at least 7 check-ins, the trend endpoint safely returns `status: Need more data` while still displaying their real-time session risk."*
