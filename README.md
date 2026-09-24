# Ledger — Fraud Detection Console

A web frontend for the fraud-detection model, built on top of the original
`FraudDetection.py` analysis. It replaces the notebook-style script with:

- **`model/train_model.py`** — loads `onlinefraud.csv`, engineers balance-error
  features, trains a Logistic Regression and a Random Forest, and saves the
  better-performing model (by ROC AUC) plus its metadata.
- **`app.py`** — a Flask backend that loads the trained model and exposes:
  - `GET  /` — the web UI
  - `POST /api/predict` — scores a single transaction, returns fraud
    probability and a risk level
  - `GET  /api/model-info` — full training metrics
- **`templates/index.html` + `static/css/style.css` + `static/js/script.js`**
  — the frontend: a form for entering a transaction, a live gauge showing
  fraud probability, and a plain-language verdict.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## (Re-)training the model

The trained model is already included (`model/fraud_model.joblib` +
`model/model_meta.json`), trained on a stratified sample of the dataset
(every fraud row + 200,000 legitimate rows) for speed. To retrain on your
own copy of `onlinefraud.csv`:

```bash
cd model
cp /path/to/onlinefraud.csv .
python train_model.py
```

This writes a fresh `fraud_model.joblib` and `model_meta.json` in the
`model/` folder, which `app.py` loads automatically.

## Running the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Notes

- The model was trained on the PaySim-style `onlinefraud.csv` schema
  (`step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
  oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud`). The frontend
  form only asks for the fields the model actually uses.
- Two "Try" buttons in the UI prefill a routine payment and a suspicious
  transfer so you can see the console react without typing numbers by hand.
