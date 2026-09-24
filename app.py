"""
Fraud Detection web app.

Serves an HTML/CSS/JS frontend and a small JSON API backed by the
RandomForest model trained in model/train_model.py. Also offers two
optional AI-powered features (receipt-photo extraction and a chat
assistant) that use the Anthropic API when an API key is configured.

Run with:
    python app.py
then open http://127.0.0.1:5000
"""
import base64
import json
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_model.joblib")
META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    META = json.load(f)

FEATURE_COLUMNS = META["feature_columns"]
TRANSACTION_TYPES = META["transaction_types"]

# ---------------------------------------------------------------------------
# Optional AI features (receipt-photo extraction + chat assistant).
# These only work if an ANTHROPIC_API_KEY is set in the environment. The app
# runs completely fine without one — those two features just explain that
# they're not configured yet instead of erroring out.
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

_anthropic_client = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        _anthropic_client = None


def ai_enabled():
    return _anthropic_client is not None


def build_feature_row(payload: dict) -> pd.DataFrame:
    tx_type = payload["type"]
    amount = float(payload["amount"])
    old_orig = float(payload["oldbalanceOrg"])
    new_orig = float(payload["newbalanceOrig"])
    old_dest = float(payload["oldbalanceDest"])
    new_dest = float(payload["newbalanceDest"])
    step = float(payload.get("step", 1))

    row = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "errorBalanceOrig": new_orig + amount - old_orig,
        "errorBalanceDest": old_dest + amount - new_dest,
        "CASH_OUT": 1 if tx_type == "CASH_OUT" else 0,
        "DEBIT": 1 if tx_type == "DEBIT" else 0,
        "PAYMENT": 1 if tx_type == "PAYMENT" else 0,
        "TRANSFER": 1 if tx_type == "TRANSFER" else 0,
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@app.route("/")
def index():
    return render_template(
        "index.html",
        transaction_types=TRANSACTION_TYPES,
        model_name=META["best_model"].replace("_", " ").title(),
        model_auc=round(META["metrics"][META["best_model"]]["auc"] * 100, 2),
        model_acc=round(META["metrics"][META["best_model"]]["accuracy"] * 100, 2),
        trained_rows=f"{META['trained_rows']:,}",
        fraud_rows=f"{META['fraud_rows']:,}",
        ai_enabled=ai_enabled(),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    required = [
        "type", "amount", "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest",
    ]
    missing = [f for f in required if f not in payload or payload[f] == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        X = build_feature_row(payload)
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)

    if proba >= 0.85:
        risk = "critical"
    elif proba >= 0.5:
        risk = "high"
    elif proba >= 0.15:
        risk = "medium"
    else:
        risk = "low"

    return jsonify({
        "isFraud": prediction,
        "fraudProbability": round(proba, 6),
        "riskLevel": risk,
    })


@app.route("/api/model-info")
def model_info():
    return jsonify(META)


NO_AI_MESSAGE = (
    "AI features aren't turned on yet. Set an ANTHROPIC_API_KEY environment "
    "variable and restart the app to enable receipt scanning and the chat "
    "assistant. (See the README for how.)"
)


@app.route("/api/analyze-receipt", methods=["POST"])
def analyze_receipt():
    """Look at an uploaded/captured receipt photo and pull out what it can
    (amount, merchant, date) so the form can be autofilled. Best-effort —
    receipts don't carry bank-balance fields, so only some fields fill in."""
    if not ai_enabled():
        return jsonify({"error": NO_AI_MESSAGE}), 503

    payload = request.get_json(force=True)
    image_b64 = payload.get("image")
    media_type = payload.get("mediaType", "image/jpeg")
    if not image_b64:
        return jsonify({"error": "No image provided."}), 400

    try:
        response = _anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a photo of a payment receipt or transfer "
                            "confirmation. Reply with ONLY a JSON object, no "
                            "other text, with these keys: amount (number, or "
                            "null if unreadable), merchant (string or null), "
                            "date (string or null), suggestedType (one of "
                            "CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER, or "
                            "null), notes (a short plain-language note about "
                            "anything on the receipt worth flagging, or null)."
                        ),
                    },
                ],
            }],
        )
        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[-1]
        extracted = json.loads(text)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message either way
        return jsonify({"error": f"Couldn't read that receipt: {exc}"}), 502

    return jsonify(extracted)


@app.route("/api/chat", methods=["POST"])
def chat():
    """A small chat assistant that can explain a score or answer general
    questions about the fraud-detection model."""
    if not ai_enabled():
        return jsonify({"error": NO_AI_MESSAGE}), 503

    payload = request.get_json(force=True)
    message = (payload.get("message") or "").strip()
    context = payload.get("context")
    history = payload.get("history") or []
    if not message:
        return jsonify({"error": "Message is empty."}), 400

    system_prompt = (
        "You are the assistant built into a fraud-detection console called "
        "Ledger. You help the person understand transaction risk scores, "
        "the model behind them, and general fraud-awareness questions. "
        f"The active model is a {META['best_model'].replace('_', ' ')} "
        f"trained on {META['trained_rows']:,} transactions "
        f"({META['fraud_rows']:,} confirmed fraud), reaching "
        f"{META['metrics'][META['best_model']]['auc']*100:.1f}% ROC AUC. "
        "Keep answers short, concrete, and non-alarmist. You are not a "
        "substitute for a bank's fraud team or law enforcement — say so if "
        "someone describes an active scam in progress and point them to "
        "their bank's fraud line."
    )
    if context:
        system_prompt += f"\n\nThe most recently scored transaction: {json.dumps(context)}"

    messages = list(history) + [{"role": "user", "content": message}]

    try:
        response = _anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=messages,
        )
        reply = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Chat assistant error: {exc}"}), 502

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True, port=5156)