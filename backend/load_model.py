from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "Salmansheik/spam-call-detector"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

print("Model loaded successfully!")

text = "Congratulations! You have won ₹50,000. Click the link now."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

outputs = model(**inputs)

predicted_class = outputs.logits.argmax(dim=1).item()

print("Prediction:", predicted_class)