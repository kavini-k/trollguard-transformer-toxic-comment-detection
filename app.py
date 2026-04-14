from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
import sqlite3
import json
import langid
import re
from datetime import datetime

app = Flask(__name__)

# ====== Model Setup ======
MODEL_DIR = "models/transformer_trollguard"
TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR, output_attentions=True
)
model.eval()

# ====== Language Mapping ======
LANGUAGE_NAMES = {
    "en": "English", "tl": "Tagalog", "ta": "Tamil", "hi": "Hindi",
    "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ur": "Urdu", "ar": "Arabic", "bn": "Bengali",
    "ml": "Malayalam", "te": "Telugu", "kn": "Kannada", "pa": "Punjabi",
    "gu": "Gujarati", "mr": "Marathi", "tr": "Turkish", "ru": "Russian",
    "zh-cn": "Chinese (Simplified)", "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean"
}

# ====== Database Setup ======
conn = sqlite3.connect("predictions.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    labels TEXT,
    probs TEXT,
    language TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ====== Utility Functions ======
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def detect_language(text):
    try:
        lang_code, confidence = langid.classify(text)
        if confidence < 0.5:
            lang_code = "en"
    except:
        lang_code = "unknown"

    return LANGUAGE_NAMES.get(lang_code.lower(), lang_code.upper())


def predict_text(text):
    # Detect language
    lang_full = detect_language(text)

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Model prediction
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits.cpu().numpy()[0]
    probs = sigmoid(logits)
    labels = (probs >= 0.5).astype(int).tolist()

    # ===== Attention Highlights =====
    highlights = []
    try:
        attentions = outputs.attentions
        attn_tensor = torch.stack(attentions).mean(dim=0).mean(dim=1)
        cls_attn = attn_tensor[0, 0, :].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        idxs = np.argsort(cls_attn)[-6:]

        top_tokens = [tokens[i] for i in sorted(idxs)]
        highlights = [
            re.sub(r'^▁|^##|^Ġ', '', t)
            for t in top_tokens if t.isalpha()
        ]
    except:
        highlights = []

    # ===== Store in DB =====
    try:
        c.execute(
            "INSERT INTO predictions (text, labels, probs, language) VALUES (?, ?, ?, ?)",
            (text, json.dumps(labels), json.dumps(list(map(float, probs))), lang_full)
        )
        conn.commit()
    except Exception as e:
        print("DB Error:", e)

    return {
        "labels": labels,
        "probs": probs.tolist(),
        "lang": lang_full,
        "highlights": highlights
    }


# ====== ROUTES ======

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    text = ""
    error = None
    highlights = []

    if request.method == "POST":
        text = request.form.get("text", "").strip()

        if not text:
            error = "Please enter text!"
        else:
            try:
                result = predict_text(text)
                prediction = {
                    "labels": result["labels"],
                    "probs": result["probs"],
                    "lang": result["lang"]
                }
                highlights = result["highlights"]
            except Exception as e:
                error = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        text=text,
        error=error,
        highlights=highlights,
        TARGET_COLS=TARGET_COLS
    )


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/dashboard/data")
def dashboard_data():
    try:
        c.execute("SELECT labels, created_at, language FROM predictions")
        rows = c.fetchall()

        counts = {col: 0 for col in TARGET_COLS}
        times = {}
        lang_counts = {}

        for labels_json, created, lang in rows:
            labels = json.loads(labels_json)

            # Category count
            for i, val in enumerate(labels):
                counts[TARGET_COLS[i]] += int(val)

            # Time series
            day = created.split(" ")[0]
            times[day] = times.get(day, 0) + 1

            # Language count
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        return jsonify({
            "counts": counts,
            "timeseries": times,
            "languages": lang_counts
        })

    except Exception as e:
        print("Dashboard Error:", e)
        return jsonify({"counts": {}, "timeseries": {}, "languages": {}})


# ====== RUN APP ======
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
