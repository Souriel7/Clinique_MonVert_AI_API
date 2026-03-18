import pandas as pd
import numpy as np
import re
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

print("=" * 50)
print("  Clinique MontVert — AI Pipeline")
print("=" * 50)

# ── Phase 1 — Charger et normaliser les données ───────────────
print("\n[Phase 1] Chargement des données...")

df = pd.read_csv("data/source/Clinique_MonVert_SourceDataset.csv")
df_copy = df.copy()

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

df_copy['normalized_text'] = (
    df_copy['ticket_label'].fillna('') + " " +
    df_copy['ticket_description'].fillna('')
).apply(normalize_text)

print(f"  {len(df_copy)} tickets chargés — {df_copy['category'].nunique()} catégories")

# ── Phase 2 — Embeddings SBERT ────────────────────────────────
print("\n[Phase 2] Génération des embeddings SBERT...")

sbert = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = sbert.encode(df_copy['normalized_text'].tolist(), show_progress_bar=True)
X = np.array(embeddings)

print(f"  {X.shape[0]} tickets convertis en vecteurs de {X.shape[1]} dimensions")

# ── Phase 3 — Détection tickets mal catégorisés (RF) ─────────
print("\n[Phase 3] Détection des tickets mal catégorisés...")

X_train, X_test, y_train, y_test = train_test_split(
    X, df_copy['category'], test_size=0.2,
    random_state=42, stratify=df_copy['category']
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"  Random Forest accuracy: {rf.score(X_test, y_test):.2%}")

df_copy['ai_suggested_category'] = rf.predict(X)
proba = rf.predict_proba(X)

label_encoder = {cat: i for i, cat in enumerate(rf.classes_)}
df_copy['confidence_current'] = [
    proba[i][label_encoder[df_copy.iloc[i]['category']]]
    for i in range(len(df_copy))
]

threshold = df_copy['confidence_current'].quantile(0.075)
print(f"  Confidence threshold: {threshold:.4f}")

error_df = df_copy[
    (df_copy['category'] != df_copy['ai_suggested_category']) &
    (df_copy['confidence_current'] < threshold)
]
print(f"  {len(error_df)} tickets mal catégorisés détectés ({len(error_df)/len(df_copy)*100:.1f}%)")

df_copy['ai_suggested_category'] = df_copy.apply(
    lambda row: row['ai_suggested_category']
    if row['ticket_id'] in error_df['ticket_id'].values
    else None,
    axis=1
)

# ── Phase 4 — Détection doublons ─────────────────────────────
print("\n[Phase 4] Détection des tickets doublons...")

adj_matrix = (cosine_similarity(embeddings) > 0.80).astype(int)
np.fill_diagonal(adj_matrix, 0)
G = nx.from_numpy_array(adj_matrix)

df_copy['redundancy_type'] = None
df_copy['master_ticket']   = None
incident_data = []

for cluster_id, indices in enumerate(nx.connected_components(G)):
    idx_list = list(indices)
    if len(idx_list) < 2:
        continue

    cluster_df = df_copy.iloc[idx_list]
    master     = cluster_df.iloc[0]
    r_type     = 'Doublon Exact' if cluster_df['ticket_description'].nunique() == 1 \
                 else 'Similarité Sémantique'

    df_copy.loc[idx_list, 'redundancy_type'] = r_type
    df_copy.loc[idx_list, 'master_ticket']   = master['ticket_id']

    incident_data.append({
        'Cluster_ID'                  : cluster_id,
        'Total'                       : len(idx_list),
        'Type'                        : r_type,
        'Sujet'                       : master['ticket_description'],
        'ID_Maître'                   : master['ticket_id'],
        'IDs_Redondants_Ou_Similaires': cluster_df.iloc[1:]['ticket_id'].tolist()
    })

df_incidents = pd.DataFrame(incident_data).sort_values('Total', ascending=False)
total_repeats = df_copy['master_ticket'].notna().sum()
print(f"  {len(df_incidents)} groupes identifiés — {total_repeats} tickets redondants")

# ── Phase 5 — Priorisation ────────────────────────────────────
print("\n[Phase 5] Calcul des scores de priorité...")

FIXED_SCORES    = {'Security': 1000, 'Network': 800}
PRIORITY_SCORES = {'Critical': 400, 'High': 300, 'Medium': 200, 'Low': 100}
STATUS_SCORES   = {'Open': 150, 'In Progress': 50, 'Resolved': 0}
THREAT_KEYWORDS = ['phishing', 'malware', 'virus', 'ransomware', 'compromised',
                   'hacked', 'breached', 'keylogger', 'trojan', 'unauthorized', 'suspicious']
URGENT_KEYWORDS = ['urgent', 'critical', 'immediately', 'emergency',
                   'down', 'we must', 'isolate', 'asap', 'blocked']
THREAT_BONUS    = 100
URGENT_BONUS    = 50

def calculate_priority(row):
    if pd.notna(row.get('master_ticket')) and row['master_ticket'] != row['ticket_id']:
        return 0
    effective_cat = row['ai_suggested_category'] \
                    if pd.notna(row.get('ai_suggested_category')) \
                    else row['category']
    if effective_cat in FIXED_SCORES:
        return FIXED_SCORES[effective_cat]
    text  = str(row['normalized_text']).lower()
    score = (
        PRIORITY_SCORES.get(row['priority'], 100) +
        STATUS_SCORES.get(row['status'], 0) +
        (THREAT_BONUS if any(t in text for t in THREAT_KEYWORDS) else 0) +
        (URGENT_BONUS if any(u in text for u in URGENT_KEYWORDS) else 0)
    )
    return score

df_copy['final_priority_score'] = df_copy.apply(calculate_priority, axis=1)
df_final_prioritized = df_copy.sort_values('final_priority_score', ascending=False)

# ── Phase 6 — Export CSV + RF model ──────────────────────────
print("\n[Phase 6] Export du dataset enrichi et du modèle...")

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/model", exist_ok=True)

final_df = df[[
    'ticket_id', 'ticket_label', 'ticket_description',
    'category', 'priority', 'status', 'software',
    'clinic_service', 'created_at', 'created_by',
    'updated_by', 'assigned_to'
]].copy()

final_df['suggested_category']   = df_copy['ai_suggested_category'].values
final_df['redundancy_type']      = df_copy['redundancy_type'].values
final_df['master_ticket']        = df_copy['master_ticket'].values
final_df['final_priority_score'] = df_copy['final_priority_score'].values

final_df = final_df.sort_values('final_priority_score', ascending=False).reset_index(drop=True)
final_df.to_csv("data/processed/Clinique_MontVert_Cleaned_API.csv", index=False)
print(f"  CSV exporté : data/processed/Clinique_MontVert_Cleaned_API.csv")

with open("data/model/rf_model.pkl", "wb") as f:
    pickle.dump(rf, f)
print(f"  Modèle RF sauvegardé : data/model/rf_model.pkl")

print("\n" + "=" * 50)
print("  Pipeline terminé avec succès")
print("=" * 50)