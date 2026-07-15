import os
import warnings

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
if os.name == "nt":
    os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
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
        whisper_model = whisper.load_model("tiny")
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


def detect_ai_voice(audio_path: str, transcript: str = "") -> dict:
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 1. Pitch analysis — AI voices have very low pitch variance
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 10 else 999.0

        # 2. Spectral flatness — AI voices are unnaturally flat
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_std = float(np.std(flatness))  # low std = too consistent = AI

        # 3. Spectral rolloff variance — AI voices lack natural rolloff variation
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_std = float(np.std(rolloff))

        # 4. MFCC delta — measures how much voice changes over time
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = float(np.mean(np.abs(mfcc_delta)))

        # 5. Harmonic-to-noise ratio — AI voices are too clean
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6))

        # 6. Tempo regularity — AI voices have robotic rhythm
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_intervals = np.diff(beats) if len(beats) > 1 else np.array([0])
        beat_regularity = float(np.std(beat_intervals))

        ai_score = 0

        # Low pitch variance = AI
        if pitch_std < 20:
            ai_score += 30
        elif pitch_std < 35:
            ai_score += 15

        # Low spectral flatness std = AI (too consistent)
        if flatness_std < 0.025:
            ai_score += 20
        elif flatness_std < 0.04:
            ai_score += 10

        # Low rolloff std = AI
        if rolloff_std < 350:
            ai_score += 15
        elif rolloff_std < 600:
            ai_score += 5

        # Low MFCC delta = AI (not enough natural variation)
        if mfcc_delta_mean < 1.8:
            ai_score += 20
        elif mfcc_delta_mean < 3.0:
            ai_score += 10

        # Very high HNR = AI (too clean, no background noise)
        if hnr > 10:
            ai_score += 10

        # Very regular beat = AI
        if beat_regularity < 2.0:
            ai_score += 10

        # Grammar & Fluency Heuristics
        if transcript:
            text_lower = transcript.lower()
            # Lack of filler words (perfect fluency is robotic)
            fillers = [" uh", " um", " like ", " you know", " ah", " hmm"]
            filler_count = sum(text_lower.count(f) for f in fillers)
            
            if filler_count == 0:
                ai_score += 10  # Reduced: Whisper often cleans transcripts, so lack of fillers shouldn't be overly penalized
            else:
                ai_score -= 20 * filler_count  # Heavily reward hesitations as human
            
            # Stammering / Repetition detection (e.g., "the the", "I I") typical in human grammar errors
            words = text_lower.split()
            stammer_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1] and len(words[i]) > 0)
            if stammer_count > 0:
                ai_score -= 30 * stammer_count  # Repeated words are very human

            # Extremely formal phrasing typical of automated robocalls
            formal_phrases = ["hello", "sir", "madam", "dear customer", "urgent call", "has been suspended", "press 1", "press one", "account"]
            for phrase in formal_phrases:
                if phrase in text_lower:
                    ai_score += 5

        is_ai = ai_score >= 50
        
        # Calculate bound percentages
        ai_prob = max(0, min(100, int(ai_score * 1.5)))  # Scale so 50+ is strong AI
        if is_ai:
            ai_prob = max(51, ai_prob)
        else:
            ai_prob = min(49, ai_prob)
            
        human_prob = 100 - ai_prob

        return {
            "voice_type": "AI Generated" if is_ai else "Human",
            "ai_score": ai_score,
            "ai_prob": ai_prob,
            "human_prob": human_prob,
            "voice_confidence": max(ai_prob, human_prob),
            "features": {
                "pitch_std": round(pitch_std, 2) if pitch_std != 999.0 else 0,
                "flatness_std": round(flatness_std, 5),
                "rolloff_std": round(rolloff_std, 2),
                "mfcc_delta_mean": round(mfcc_delta_mean, 2),
                "hnr": round(hnr, 2)
            }
        }
    except Exception as e:
        return {
            "voice_type": "Unknown",
            "ai_score": 0,
            "ai_prob": 0,
            "human_prob": 0,
            "voice_confidence": 0,
            "features": {
                "pitch_std": 0,
                "flatness_std": 0,
                "rolloff_std": 0,
                "mfcc_delta_mean": 0,
                "hnr": 0
            }
        }


def transcribe(audio_path: str, language: str = None) -> dict:
    wmodel = get_whisper()
    # Transcribe in the original language (Telugu, Hindi, etc.)
    options = {"task": "transcribe", "fp16": False}
    if language and language not in ("auto",):
        options["language"] = language  # Hint for better detection accuracy
    result = wmodel.transcribe(audio_path, **options)
    transcript = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    return {"transcript": transcript, "detected_language": detected_lang}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.png", media_type="image/png")


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
def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="auto")
):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    contents = file.file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"detail": "Uploaded file is empty."})
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        lang_code = LANGUAGE_MAP.get(language)
        result = transcribe(tmp_path, lang_code)
        if not result["transcript"]:
            return JSONResponse(status_code=422, content={"detail": "Could not transcribe audio. Make sure the audio has speech."})
        voice_info = detect_ai_voice(tmp_path, transcript=result["transcript"])
        return {**result, **voice_info}
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Transcription failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/predict/audio")
def predict_audio(
    file: UploadFile = File(...),
    language: str = Form(default="auto")
):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    contents = file.file.read()
    if not contents:
        return JSONResponse(status_code=400, content={"detail": "Uploaded file is empty."})
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        lang_code = LANGUAGE_MAP.get(language)
        trans_result = transcribe(tmp_path, lang_code)
        if not trans_result["transcript"]:
            return JSONResponse(status_code=422, content={"detail": "Could not transcribe audio. Make sure the audio has speech."})
        voice_info = detect_ai_voice(tmp_path, transcript=trans_result["transcript"])
        response = classify(trans_result["transcript"])
        response["transcript"] = trans_result["transcript"]
        response["detected_language"] = trans_result["detected_language"]
        response.update(voice_info)
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Audio analysis failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
