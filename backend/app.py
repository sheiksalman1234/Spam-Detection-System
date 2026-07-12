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

MODEL_PATH = "model/SpamCallDetector"

print("Loading AI model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print("Model loaded!")

print("Loading Whisper...")
whisper_model = whisper.load_model("base")
print("Whisper loaded!")


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
        result = whisper_model.transcribe(tmp_path)
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
        result = whisper_model.transcribe(tmp_path)
        transcript = result["text"].strip()
        if not transcript:
            raise HTTPException(status_code=422, detail="Could not transcribe audio")
        response = classify(transcript)
        response["transcript"] = transcript
        return response
    finally:
        os.unlink(tmp_path)
