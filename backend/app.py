import os

if os.name == "nt":
    os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")
    
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import whisper
import tempfile
import shutil

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

def get_whisper():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
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
    return {"suggestions": list(dict.fromkeys(found))}  # deduplicate


@app.post("/predict/text")
def predict_text(data: TextRequest):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    return classify(data.text)


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        model = get_whisper()
        result = model.transcribe(tmp_path)
        transcript = result["text"].strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        return {"transcript": transcript}
    finally:
        os.unlink(tmp_path)


@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".tmp"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        whisper_model = get_whisper()
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        response = classify(transcript)
        response["transcript"] = transcript
        return response
    finally:
        os.unlink(tmp_path)
