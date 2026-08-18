import os
import warnings
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
if os.name == "nt":
    os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

# Detect lightweight mode (for Render free tier / low-memory environments)
LIGHTWEIGHT_MODE = os.environ.get("LIGHTWEIGHT_MODE", "0") == "1"
if LIGHTWEIGHT_MODE:
    print("[MODE] Running in LIGHTWEIGHT mode — keyword-based classifier, no ML models")
else:
    print("[MODE] Running in FULL mode — ML models enabled")

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import shutil
import threading

# Heavy imports only in full mode
if not LIGHTWEIGHT_MODE:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import whisper
    import numpy as np
else:
    torch = None
    whisper = None
    np = None

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.on_event("startup")
def preload_models():
    """Preload the spam classifier in a background thread (full mode only)."""
    if LIGHTWEIGHT_MODE:
        print("[STARTUP] Lightweight mode — no models to preload. Server ready!")
        return
    def _load():
        try:
            print("[STARTUP] Preloading spam classifier in background...")
            load_classifier()
            print("[STARTUP] Spam classifier ready!")
        except Exception as e:
            print(f"[STARTUP WARNING] Classifier preload failed: {e}")
    threading.Thread(target=_load, daemon=True).start()
    print("[STARTUP] Server is up — classifier loading in background...")

MODEL_NAME = "Salmansheik/spam-call-detector"

tokenizer = None
model = None
device = None if LIGHTWEIGHT_MODE else torch.device("cpu")

def load_classifier():
    global tokenizer, model, device
    if LIGHTWEIGHT_MODE:
        return  # No ML model in lightweight mode
    if model is None:
        try:
            print("[INFO] Loading spam classifier...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            print("[INFO] Tokenizer loaded")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            print("[INFO] Model loaded")
            model.to(device)
            model.eval()
            print("[OK] Spam classifier ready")
        except Exception as e:
            print(f"[ERR] Failed to load classifier: {str(e)}")
            raise

whisper_model = None
model_lock = threading.Lock()

LANGUAGE_MAP = {
    "auto": "en-US",
    "en": "en-US",
    "hi": "hi-IN",
    "te": "en-US",  # Telugu not supported by Google Speech API, fallback to English
    "ta": "ta-IN",
    "kn": "en-US",  # Kannada not supported, fallback to English
    "ml": "ml-IN"
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
    if LIGHTWEIGHT_MODE:
        raise RuntimeError("Whisper not available in lightweight mode")
    if whisper_model is None:
        print("[INFO] Loading Whisper model...")
        whisper_model = whisper.load_model("tiny")
        print("[OK] Whisper model ready")
    return whisper_model


class TextRequest(BaseModel):
    text: str


# Enhanced keyword classifier for lightweight mode
SPAM_KEYWORDS_WEIGHTED = [
    ('otp', 3), ('prize', 3), ('won', 2), ('lottery', 3), ('gift card', 3),
    ('loan', 2), ('pre-approved', 2), ('account blocked', 3), ('verify your account', 3),
    ('click here', 2), ('call now', 2), ('limited time', 2), ('free', 1),
    ('congratulations', 2), ('irs', 3), ('tax', 1), ('arrest', 3),
    ('social security', 3), ('password', 3), ('credit card', 2),
    ('urgent', 2), ('immediately', 2), ('suspended', 2), ('unauthorized', 2),
    ('claim', 2), ('reward', 2), ('winner', 3), ('offer', 1), ('deal', 1),
    ('bank', 1), ('blocked', 2), ('expire', 2), ('act now', 2),
]

def _keyword_classify(text: str) -> dict:
    """Enhanced keyword-based spam classifier (no ML model needed)."""
    text_lower = text.lower()
    total_weight = sum(w for kw, w in SPAM_KEYWORDS_WEIGHTED if kw in text_lower)
    # Score: 0-5 = safe, 6+ = spam
    if total_weight >= 6:
        spam_pct = min(97, 60 + (total_weight * 3))
    elif total_weight >= 3:
        spam_pct = 45 + (total_weight * 3)
    else:
        spam_pct = max(5, total_weight * 10)
    safe_pct = 100 - spam_pct
    is_spam = spam_pct > 50
    return {
        "label": "Spam" if is_spam else "Not Spam",
        "confidence": round(max(spam_pct, safe_pct), 2),
        "probabilities": {
            "Not Spam": round(safe_pct, 2),
            "Spam": round(spam_pct, 2),
        }
    }


def classify(text: str) -> dict:
    # In lightweight mode, always use keyword classifier
    if LIGHTWEIGHT_MODE:
        return _keyword_classify(text)
    try:
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
    except Exception as e:
        print(f"[ERR] Classification failed, using keyword fallback: {str(e)}")
        return _keyword_classify(text)


def detect_ai_voice(audio_path: str, transcript: str = "") -> dict:
    try:
        import scipy.io.wavfile as wavfile
        from scipy import signal
        
        sr, y = wavfile.read(audio_path)
        if len(y.shape) > 1:
            y = y[:, 0]
        y = y.astype(float) / np.max(np.abs(y))
        
        # Resample to 16kHz if needed
        if sr != 16000:
            num_samples = int(len(y) * 16000 / sr)
            y = signal.resample(y, num_samples)
            sr = 16000
        
        # Limit to 30 seconds
        max_samples = int(30 * sr)
        if len(y) > max_samples:
            y = y[:max_samples]
        
        # Simple AI voice detection without Numba
        ai_votes = 0
        total_votes = 5
        
        # 1. Energy variance (AI voices have steady volume)
        frame_length = int(0.02 * sr)
        frames = [np.sqrt(np.mean(y[i:i+frame_length]**2)) for i in range(0, len(y)-frame_length, frame_length)]
        if len(frames) > 1:
            energy_std = np.std(frames)
            energy_mean = np.mean(frames)
            energy_cv = energy_std / (energy_mean + 1e-6)
            if energy_cv < 0.4:
                ai_votes += 1
        
        # 2. Zero-crossing rate (AI voices have uniform ZCR)
        zcr = np.mean(np.abs(np.diff(np.sign(y))))
        if zcr < 0.1:
            ai_votes += 1
        
        # 3. Spectral centroid (AI voices have less variation)
        fft = np.abs(np.fft.fft(y[:min(len(y), sr*2)]))
        freqs = np.fft.fftfreq(len(fft), 1/sr)
        centroid = np.sum(freqs[:len(freqs)//2] * fft[:len(fft)//2]) / (np.sum(fft[:len(fft)//2]) + 1e-6)
        if centroid < 2000:
            ai_votes += 1
        
        # 4. Harmonic content (AI voices are too clean)
        autocorr = np.correlate(y, y, mode='full')[len(y)-1:]
        autocorr = autocorr / (autocorr[0] + 1e-6)
        if len(autocorr) > sr//100:
            harmonic_strength = np.max(autocorr[sr//200:sr//100])
            if harmonic_strength > 0.7:
                ai_votes += 1
        
        # 5. Pitch regularity (AI voices have robotic pitch)
        if len(y) > sr:
            segment = y[:sr]
            autocorr_seg = np.correlate(segment, segment, mode='full')[len(segment)-1:]
            autocorr_seg = autocorr_seg / (autocorr_seg[0] + 1e-6)
            if len(autocorr_seg) > sr//200:
                pitch_regularity = np.max(autocorr_seg[sr//200:sr//100])
                if pitch_regularity > 0.6:
                    ai_votes += 1
        
        # Transcript heuristics
        transcript_ai_bonus = 0
        if transcript:
            text_lower = transcript.lower()
            fillers = [" uh", " um", " like ", " you know", " ah", " hmm"]
            filler_count = sum(text_lower.count(f) for f in fillers)
            words = text_lower.split()
            stammer_count = sum(1 for i in range(len(words) - 1) if words[i] == words[i+1] and len(words[i]) > 1)
            
            if filler_count == 0 and stammer_count == 0:
                transcript_ai_bonus = 1
            elif filler_count >= 2 or stammer_count >= 1:
                transcript_ai_bonus = -1
        
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
                "pitch_std": 0,
                "flatness_std": 0,
                "rolloff_std": 0,
                "mfcc_delta_mean": 0,
                "hnr": 0
            }
        }
    except Exception as e:
        print(f"[ERR] AI voice detection failed: {str(e)}")
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
    try:
        return classify(data.text)
    except Exception as e:
        print(f"[ERR] /predict/text error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
