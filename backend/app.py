import os
import warnings
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

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
import threading

try:
    from google import genai as genai_client
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[!] google-genai not installed.")

try:
    from deep_translator import GoogleTranslator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False
    print("[!] deep-translator not installed. Run: pip install deep-translator")

app = FastAPI(title="AI Spam Call Detection API", version="1.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_NAME = "Salmansheik/spam-call-detector"

tokenizer = None
model = None
device = torch.device("cpu")

def load_classifier():
    global tokenizer, model, device
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.to(device)
        model.eval()

whisper_model = None
model_lock = threading.Lock()

LANGUAGE_MAP = {
    "auto": None,
    "en": "en",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml"
}

# Configure Gemini API
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_AVAILABLE:
    try:
        gemini_client = genai_client.Client(api_key=GEMINI_API_KEY)
        print("[OK] Gemini API configured")
    except Exception as e:
        gemini_client = None
        print("[ERR] Gemini client init failed: " + str(e))
else:
    gemini_client = None
    print("[!] Gemini API not available - using fallback translation")

LANGUAGE_NAMES_FULL = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam"
}

TRANSLATION_MAP = {
    "en": "en",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml"
}


def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
    return whisper_model


class TextRequest(BaseModel):
    text: str


def classify(text: str) -> dict:
    load_classifier()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with model_lock:
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
        y, sr = librosa.load(audio_path, sr=16000, mono=True, duration=30.0)

        # 1. Pitch std — AI voices have unnaturally low pitch variance
        f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        pitch_std = float(np.std(f0_clean)) if len(f0_clean) > 10 else 999.0

        # 2. Spectral flatness std — AI voices are too spectrally consistent
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        flatness_std = float(np.std(flatness))

        # 3. Spectral rolloff std
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_std = float(np.std(rolloff))

        # 4. MFCC delta — AI voices lack natural temporal variation
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_mean = float(np.mean(np.abs(mfcc_delta)))

        # 5. HNR — AI voices are too clean (very high harmonic ratio)
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = float(np.mean(np.abs(harmonic)) / (np.mean(np.abs(percussive)) + 1e-6))

        # 6. Zero-crossing rate std — AI voices have very uniform ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = float(np.std(zcr))

        # 7. Spectral centroid std — AI voices have less centroid variation
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        centroid_std = float(np.std(centroid))

        # 8. Energy variance — AI voices have very steady energy levels
        rms = librosa.feature.rms(y=y)[0]
        rms_std = float(np.std(rms))
        rms_mean = float(np.mean(rms)) + 1e-6
        energy_cv = rms_std / rms_mean  # coefficient of variation

        # ── Scoring (each feature votes independently, no cross-cancellation) ──
        ai_votes = 0
        total_votes = 8

        # Pitch: AI < 25 Hz std
        if pitch_std < 25:
            ai_votes += 1

        # Flatness std: AI < 0.03
        if flatness_std < 0.03:
            ai_votes += 1

        # Rolloff std: AI < 500
        if rolloff_std < 500:
            ai_votes += 1

        # MFCC delta: AI < 2.5
        if mfcc_delta_mean < 2.5:
            ai_votes += 1

        # HNR: AI > 8 (too clean)
        if hnr > 8:
            ai_votes += 1

        # ZCR std: AI < 0.02
        if zcr_std < 0.02:
            ai_votes += 1

        # Centroid std: AI < 400
        if centroid_std < 400:
            ai_votes += 1

        # Energy CV: AI < 0.4 (very steady volume)
        if energy_cv < 0.4:
            ai_votes += 1

        # Transcript heuristics — weak signals only, capped contribution
        transcript_ai_bonus = 0
        if transcript:
            text_lower = transcript.lower()
            fillers = [" uh", " um", " like ", " you know", " ah", " hmm"]
            filler_count = sum(text_lower.count(f) for f in fillers)
            words = text_lower.split()
            stammer_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1] and len(words[i]) > 1)

            if filler_count == 0 and stammer_count == 0:
                transcript_ai_bonus = 1   # perfect fluency = slight AI signal
            elif filler_count >= 2 or stammer_count >= 1:
                transcript_ai_bonus = -1  # clear human disfluency

        # Final decision: majority vote on acoustic features + optional transcript nudge
        ai_score_pct = (ai_votes / total_votes) * 100
        adjusted = ai_score_pct + (transcript_ai_bonus * 8)
        adjusted = max(0, min(100, adjusted))

        is_ai = adjusted >= 50

        ai_prob = int(adjusted)
        if is_ai:
            ai_prob = max(52, ai_prob)
        else:
            ai_prob = min(48, ai_prob)
        human_prob = 100 - ai_prob

        return {
            "voice_type": "AI Generated" if is_ai else "Human",
            "ai_score": round(adjusted, 1),
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
    options = {"task": "transcribe", "fp16": False}
    if language and language not in ("auto",):
        options["language"] = language
    with model_lock:
        result = wmodel.transcribe(audio_path, **options)
    transcript = result["text"].strip()
    detected_lang = result.get("language", "unknown")
    return {"transcript": transcript, "detected_language": detected_lang}


def translate_text(text: str, target_lang: str) -> str:
    if not text:
        return text
    if target_lang in ("en", "auto", None):
        return text

    # Primary: deep-translator (Google Translate, free, no quota)
    if DEEP_TRANSLATOR_AVAILABLE:
        try:
            translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
            if translated:
                print("[OK] deep-translator -> " + target_lang)
                return translated
        except Exception as e:
            print("[WARN] deep-translator failed: " + str(e))

    # Fallback: Gemini API
    if GEMINI_AVAILABLE and gemini_client:
        try:
            lang_name = LANGUAGE_NAMES_FULL.get(target_lang, target_lang)
            prompt = "Translate the following text to " + lang_name + ". Only provide the translation, nothing else.\n\nText: " + text
            response = gemini_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            translated = response.text.strip() if response.text else ""
            if translated:
                print("[OK] Gemini -> " + lang_name)
                return translated
        except Exception as e:
            print("[ERR] Gemini translation failed: " + str(e))

    print("[WARN] All translation methods failed, returning original text")
    return text


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse("static/favicon.png", media_type="image/png")


@app.get("/health")
def health():
    return {"status": "OK", "device": str(device)}


@app.post("/test-translate")
def test_translate(data: TextRequest):
    """Test endpoint to verify translation works"""
    translations = {}
    for lang_code in ["en", "hi", "te", "ta", "kn", "ml"]:
        translated = translate_text(data.text, lang_code)
        translations[lang_code] = translated
    return {"original": data.text, "translations": translations}


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
        
        # Step 1: Transcribe in original language (auto-detect)
        result = transcribe(tmp_path, None)
        if not result["transcript"]:
            return JSONResponse(status_code=422, content={"detail": "Could not transcribe audio. Make sure the audio has speech."})
        
        original_transcript = result["transcript"]
        detected_lang = result["detected_language"]
        
        # Step 2: Translate to selected output language
        if language == "auto":
            target_lang_code = "en"
            output_lang = detected_lang
        else:
            target_lang_code = TRANSLATION_MAP.get(language, "en")
            output_lang = language
        
        translated_transcript = translate_text(original_transcript, target_lang_code)
        
        # Step 3: Detect voice type
        voice_info = detect_ai_voice(tmp_path, transcript=original_transcript)
        
        return {
            "transcript": original_transcript,
            "translated_transcript": translated_transcript,
            "detected_language": detected_lang,
            "output_language": output_lang,
            **voice_info
        }
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
        
        # Step 1: Transcribe audio in original language (auto-detect)
        trans_result = transcribe(tmp_path, None)
        if not trans_result["transcript"]:
            return JSONResponse(status_code=422, content={"detail": "Could not transcribe audio. Make sure the audio has speech."})
        
        original_transcript = trans_result["transcript"]
        detected_lang = trans_result["detected_language"]
        
        # Step 2: Translate to selected output language
        if language == "auto":
            target_lang_code = "en"
            output_lang = detected_lang
        else:
            target_lang_code = TRANSLATION_MAP.get(language, "en")
            output_lang = language
        
        translated_transcript = translate_text(original_transcript, target_lang_code)
        
        # Step 3: Analyze original transcript for spam/ham
        response = classify(original_transcript)
        
        # Step 4: Detect voice type
        voice_info = detect_ai_voice(tmp_path, transcript=original_transcript)
        
        # Step 5: Build response with both transcripts
        response["transcript"] = original_transcript
        response["translated_transcript"] = translated_transcript
        response["detected_language"] = detected_lang
        response["output_language"] = output_lang
        response.update(voice_info)
        
        return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Audio analysis failed: {str(e)}"})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
