from prometheus_client import Counter, Histogram

# ── Niveau système ────────────────────────────────────────────────────────────
# Counter : valeur strictement croissante, jamais réinitialisée
# Label "decade" : permet de filtrer par tranche dans Prometheus/Grafana
#   ex: predictions_total{decade="1990s"} 42
#PREDICTION_COUNTER = Counter(
#    name="dataprophet_predictions_total",
#    documentation="Nombre total de prédictions effectuées",
#    labelnames=["decade"],  # tranche de l'année prédite : "1980s", "1990s", etc.
#)

PREDICTION_COUNTER = Counter(
    name="dataprophet_predictions_total",
    documentation="Nombre total de prédictions effectuées",
    labelnames=["age_category"],  # "jeune" ou "vieux"
)
# ── Niveau modèle ─────────────────────────────────────────────────────────────
# Histogram : enregistre la distribution des durées
# buckets : seuils en secondes — Prometheus compte combien de requêtes tombent
#           en dessous de chaque seuil (0.01s, 0.05s, 0.1s, 0.5s, 1s, 2s)
PREDICTION_DURATION = Histogram(
    name="dataprophet_prediction_duration_seconds",
    documentation="Durée de chaque prédiction en secondes",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
)
