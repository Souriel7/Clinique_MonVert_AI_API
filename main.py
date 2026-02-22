from fastapi import FastAPI, HTTPException
import pandas as pd

app = FastAPI(title="API IT - Clinique MontVert")

# 1. Helper Function : Force le serveur à lire le disque dur à chaque requête
def charger_donnees_fraiches():
    try:
        df_clean = pd.read_csv("data/processed/Clinique_MontVert_Cleaned_API.csv")
        tickets_db = df_clean.to_dict(orient="records")
        return df_clean, tickets_db
    except FileNotFoundError:
        return pd.DataFrame(), []

# 2. Endpoint : Récupérer le Smart Backlog complet
@app.get("/api/ai/tickets")
def get_backlog():
    df_clean, tickets_db = charger_donnees_fraiches()
    
    if df_clean.empty:
        raise HTTPException(status_code=404, detail="Fichier introuvable")
        
    return {
        "status": "success",
        "total_tickets": len(tickets_db),
        "data": tickets_db
    }

# 3. Endpoint : Statistiques pour le Tableau de Bord (Dashboard)
@app.get("/api/ai/dashboard/stats")
def get_dashboard_stats():
    df_clean, tickets_db = charger_donnees_fraiches()
    
    if df_clean.empty:
        return {"error": "Aucune donnée disponible"}
    
    # Compter les tickets par action requise
    action_counts = df_clean['ticket_action'].value_counts().to_dict()
    category_counts = df_clean['category'].value_counts().to_dict()
    
    return {
        "status": "success",
        "repartition_actions": action_counts,
        "repartition_categories": category_counts
    }

# 4. Endpoint : Filtrer par département
@app.get("/api/tickets/service/{service_name}")
def get_tickets_by_service(service_name: str):
    df_clean, tickets_db = charger_donnees_fraiches()
    
    filtered = [t for t in tickets_db if t.get("clinic_service") == service_name]
    return {
        "status": "success",
        "total": len(filtered),
        "data": filtered
    }