import os

if os.name == "nt":
    os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import whisper
import tempfile
import shutil
import numpy as np
import librosa

app = FastAPI(title="AI Spam Call Detection API", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_NAME = "Salmansheik/spam-call-detector"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded successfully!")

whisper_model = None

LANGUAGE_MAP = {
    "auto": None,
    "en": "en",
    "hi": "hi",
    "te": "te"
}


def get_whisper():
    global whisper_model
    if whisper_model is None:
        print("Loading Whisper...")
        whisper_model = whisper.load_model("base")
        print("Whisper loaded!")
    return whisper_model


class TextRequest(BaseModel):
    text: str


def classify(text: str) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]
    prediction = probs.argmax().item()
    label = "Spam" if prediction == 1 else "Not Spam"
    return {
        "label": label,
        "confidence": round(probs[prediction].item() * 100, 2),
        "probabilities": {
            "Not Spam": round(probs[0].item() * 100, 2),
            "Spam": round(probs[1].item() * 100, 2),
        }
    }


def detect_ai_voice(audio_path: str) -> dict:
    """
    Detect if voice is AI-generated or human using audio features.
    AI voices tend to have: low pitch variance, high spectral flatness consistency,
    unnatural zero crossing rate patterns.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)

        # Pitch variance — AI voices are unnaturally consistent
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        pitch_variance = float(np.std(f0_clean)) if len(f0_clean) > 0 else 0.0

        # Spectral flatness — AI voices have higher flatness
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_mean = float(np.mean(spec_flatness))

        # Zero crossing rate variance — AI voices are more regular
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_variance = float(np.var(zcr))

        # MFCC variance — human voices have more variation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_variance = float(np.mean(np.var(mfcc, axis=1)))

        # Scoring: low pitch variance + high flatness + low zcr variance = AI
        ai_score = 0
        if pitch_variance < 20:
            ai_score += 35
        if flatness_mean > 0.1:
            ai_score += 25
        if zcr_variance < 0.001:
            ai_score += 20
        if mfcc_variance < 50:
            ai_score += 20

        is_ai = ai_score >= 50
        return {
            "voice_type": "AI Generated" if is_ai else "Human",
            "ai_score": ai_score,
            "confidence": ai_score if is_ai else (100 - ai_score)
        }
    except Exception:
        return {"voice_type": "Unknown", "ai_score": 0, "confidence": 0}


def transcribe(audio_path: str, language: str = None) -> dict:
    wmodel = get_whisper()
    options = {"task": "transcribe"}
    if language:
        options["language"] = language
    result = wmodel.transcribe(audio_path, **options)
    transcript = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    return {"transcript": transcript, "detected_language": detected_lang}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {"status": "OK", "device": str(device)}


SPAM_KEYWORDS = [
    ("otp", "OTP sharing request — never share OTP with anyone"),
    ("prize", "Prize/lottery scam — you didn't enter any contest"),
    ("won", "Fake winner alert — common scam tactic"),
    ("lottery", "Lottery scam — ignore and block"),
    ("gift card", "Gift card scam — no legitimate company asks for gift cards"),
    ("loan", "Loan offer scam — verify with official bank before responding"),
    ("pre-approved", "Pre-approved loan scam — do not share personal details"),
    ("account blocked", "Account block threat — call your bank directly to verify"),
    ("verify your account", "Phishing attempt — do not click any links"),
    ("click here", "Suspicious link — do not click unknown links"),
    ("call now", "Urgency tactic — scammers create fake urgency"),
    ("limited time", "Urgency tactic — pressure to act fast is a red flag"),
    ("free", "Free offer bait — verify before responding"),
    ("congratulations", "Fake congratulations — common spam opener"),
    ("irs", "IRS/tax scam — government agencies don't call unexpectedly"),
    ("tax", "Tax scam — verify with official tax authority"),
    ("arrest", "Threat scam — police/agencies don't threaten via calls"),
    ("social security", "SSN scam — never share social security number over call"),
    ("password", "Password phishing — never share passwords"),
    ("credit card", "Credit card scam — verify with your bank directly"),
]


@app.post("/suggestions")
def get_suggestions(data: TextRequest):
    text_lower = data.text.lower()
    found = [msg for kw, msg in SPAM_KEYWORDS if kw in text_lower]
    return {"suggestions": list(dict.fromkeys(found))}


@app.post("/predict/text")
def predict_text(data: TextRequest):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return classify(data.text)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="auto")
):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        lang_code = LANGUAGE_MAP.get(language)
        result = transcribe(tmp_path, lang_code)
        if not result["transcript"]:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        voice_info = detect_ai_voice(tmp_path)
        return {**result, **voice_info}
    finally:
        os.unlink(tmp_path)


@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    language: str = Form(default="auto")
):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        lang_code = LANGUAGE_MAP.get(language)
        trans_result = transcribe(tmp_path, lang_code)
        if not trans_result["transcript"]:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        voice_info = detect_ai_voice(tmp_path)
        response = classify(trans_result["transcript"])
        response["transcript"] = trans_result["transcript"]
        response["detected_language"] = trans_result["detected_language"]
        response.update(voice_info)
        return response
    finally:
        os.unlink(tmp_path)
