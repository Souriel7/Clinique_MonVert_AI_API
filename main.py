from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
import re
import os

app = FastAPI(title="API IT - Clinique MontVert")

# ── Load models once at startup ───────────────────────────────
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

try:
    with open("data/model/rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    model_loaded = True
    print("RF model loaded successfully")
except Exception as e:
    model_loaded = False
    print(f"RF model not found: {e}")

# ── Normalize text — same as pipeline.py ─────────────────────
noise_phrases = [
    "Hi IT team", "I ", "My ", "Please ", "I'm ", "Our ",
    "Greetings,", "Greetings ", "Dear IT ", "Hello IT ",
    "Dear team ", "Hello", ",", "We are ", "Good morning ",
    "Hi team,", "Hi there,", "Hello,", "Hi IT,", "Dear team,",
    "Greetings.", "Hi team.", "Hi there.", "Good morning.",
    "Hello.", "Dear team.", "Good morning IT.", "Dear IT.",
    "Hello IT.", "Hi IT.", "."
]

def normalize_text(text):
    if not isinstance(text, str): return ""
    text = text.replace('"', '')
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    noise_pattern = '|'.join([
        re.escape(p).replace(r'\ ', r'\s+')
        for p in sorted(noise_phrases, key=len, reverse=True)
    ])
    text = re.sub(r'(?i)\b(' + noise_pattern + r')\b', '', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# ── Helper ────────────────────────────────────────────────────
def charger_donnees_fraiches():
    try:
        df_clean   = pd.read_csv("data/processed/Clinique_MontVert_Cleaned_API.csv")
        df_clean   = df_clean.replace([float('inf'), float('-inf')], None)
        df_clean   = df_clean.where(pd.notnull(df_clean), None)
        # Convert to JSON and back to ensure all values are serializable
        import json
        tickets_db = json.loads(df_clean.to_json(orient="records"))
        return df_clean, tickets_db
    except FileNotFoundError:
        return pd.DataFrame(), []

# ── GET /api/ai/tickets ───────────────────────────────────────
@app.get("/api/ai/tickets")
def get_backlog():
    df_clean, tickets_db = charger_donnees_fraiches()
    if df_clean.empty:
        raise HTTPException(status_code=404, detail="Fichier introuvable")
    return {
        "status"       : "success",
        "total_tickets": len(tickets_db),
        "data"         : tickets_db
    }

# ── GET /api/ai/dashboard/stats ───────────────────────────────
@app.get("/api/ai/categories")
def get_dashboard_stats():
    df_clean, _ = charger_donnees_fraiches()
    if df_clean.empty:
        return {"error": "Aucune donnée disponible"}
    return {
        "status"                : "success",
        "repartition_categories": df_clean['category'].value_counts().to_dict(),
        "repartition_priorites" : df_clean['final_priority_score'].describe().to_dict(),
        "total_tickets"         : len(df_clean),
        "tickets_redondants"    : int(df_clean['master_ticket'].notna().sum()),
        "tickets_mal_categories": int(df_clean['suggested_category'].notna().sum())
    }

# ── GET /api/ai/tickets/service/{service_name} ────────────────
@app.get("/api/ai/tickets/service/{service_name}")
def get_tickets_by_service(service_name: str):
    df_clean, tickets_db = charger_donnees_fraiches()
    filtered = [t for t in tickets_db if t.get("clinic_service") == service_name]
    return {"status": "success", "total": len(filtered), "data": filtered}

# ── POST /api/ai/predict ──────────────────────────────────────
class TicketInput(BaseModel):
    ticket_label      : str
    ticket_description: str

@app.post("/api/ai/predict")
def predict_category(ticket: TicketInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Modèle IA non disponible")

    combined  = normalize_text(ticket.ticket_label + " " + ticket.ticket_description)
    embedding = sbert_model.encode([combined])

    prediction = rf_model.predict(embedding)[0]
    proba      = rf_model.predict_proba(embedding)[0]
    confidence = float(proba.max())
    all_scores = {
        cat: round(float(p), 4)
        for cat, p in zip(rf_model.classes_, proba)
    }

    return {
        "status"            : "success",
        "suggested_category": prediction,
        "confidence"        : round(confidence, 4),
        "all_scores"        : all_scores
    }

# ── GET /api/ai/refresh ───────────────────────────────────────
@app.get("/api/ai/refresh")
def refresh_info():
    return {
        "status" : "success",
        "message": "Dans ce POC, les données sont mises à jour en réexécutant pipeline.py",
        "pipeline_steps": [
            "1. Chargement et normalisation du dataset",
            "2. Génération des embeddings SBERT",
            "3. Détection des tickets mal catégorisés (Random Forest)",
            "4. Détection des doublons (Cosine Similarity)",
            "5. Calcul des scores de priorité",
            "6. Export CSV enrichi + sauvegarde modèle RF"
        ]
    }