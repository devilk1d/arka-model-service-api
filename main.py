import os, sys, json, io, warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import httpx
from scipy.sparse import hstack, csr_matrix

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API", version="3.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

class IdentityMapping:
    def predict(self, x):
        return x

sys.modules["__main__"].IdentityMapping = IdentityMapping

# Load artifacts
print("Loading artifacts...")

A = joblib.load(os.getenv("ARTIFACTS_PATH", "arka_model_artifacts_v2.pkl"))
NLP = joblib.load(os.getenv("NLP_ARTIFACTS_PATH", "arka_nlp_artifacts_v2.pkl"))

# Notebook model
MODEL        = A.get("calibrated_model") or A["model"]
FEATURES     = A["production_features"]
LE_PLAN      = A["le_plan"]
LE_CONTRACT  = A["le_contract"]
SCALER_SEG   = A["scaler_seg"]
KMEANS       = A["kmeans"]
LABEL_MAP    = A["cluster_label_map"]
SEG_FEATURES = A["seg_features"]   # ['days_since_login','payment_count','log_revenue','log_usage','feature_adoption_pct','avg_nps_score']
SEG_PROFILES = A["seg_profiles"]   # list of dicts per segment cluster (notebook 1 key = seg_profiles)
SEG_ACTIONS  = A["seg_actions"]    # dict keyed by segment label: {description, retain, offer, priority}
RISK_LOW     = A["risk_thresholds"]["low"]
RISK_HIGH    = A["risk_thresholds"]["high"]
REF          = pd.Timestamp(A["reference_date"])

# Notebook NLP
CV_VEC      = NLP["cv_vec"]
LDA         = NLP["lda"]
TOPIC_NAMES = NLP["topic_names"]
URGENCY_LEX = NLP["urgency_lexicon"]
N_TOPICS    = NLP["n_topics"]

# NLP Sentiment artifacts (Section 7)
TFIDF_SENT           = NLP["tfidf_sent"]
SENT_LGBM            = NLP["sent_lgbm"]
SCALER_SENT          = NLP["scaler_sent"]
CHURN_INTENT_LEXICON = NLP["churn_intent_lexicon"]
NEG_PATTERNS = [
    r"not\s+\w*\s*(good|great|satisfied|worth|happy|working|reliable|stable|clear)",
    r"(issue|problem|bug|error|crash|fail|broken|slow|bad|poor|terrible|horrible|awful|nightmare|frustrat)",
    r"(can't|cannot|unable|impossible|doesn't|won't|wouldn't)\s+\w+",
    r"(worst|never|useless|waste|refund|overcharged|unexpected\s+fee|charged\s+twice)",
]
POSITIVE_ANCHORS = [
    "great","excellent","fantastic","love","amazing","perfect","best","recommend","happy","outstanding"
]
CONTRAST_PATTERN = r"\b(but|however|although|yet|despite|while|though|even though|nevertheless)\b"

# SHAP explainer
EXPLAINER = A.get("explainer")
if EXPLAINER is None:
    _base = MODEL.calibrated_classifiers_[0].estimator if hasattr(MODEL, "calibrated_classifiers_") else MODEL
    EXPLAINER = shap.TreeExplainer(_base)

ANALYZER = SentimentIntensityAnalyzer()

LLM_URL   = os.getenv("OLLAMA_URL",      "https://api.ollama.ai/v1")
LLM_KEY   = os.getenv("OLLAMA_API_KEY",  "")
LLM_MODEL = os.getenv("OLLAMA_MODEL",    "qwen3.5:397b-cloud")
FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "1.0"))

print("✅ Artifacts loaded (v2.1 — best model active)")
print(f"   model_version : {A.get('model_version', 'unknown')}")
print(f"   model_name    : {A.get('best_model_name', 'unknown')}")
print(f"   has_calibrated: {'calibrated_model' in A}")
print(f"   SEG_FEATURES  : {SEG_FEATURES}")
print(f"   FEATURES count: {len(FEATURES)}")
print(f"   RISK_LOW/HIGH : {RISK_LOW} / {RISK_HIGH}")
print(f"   N_TOPICS      : {N_TOPICS}")
print(f"✅ NLP Artifacts loaded (sentiment: TF-IDF + LightGBM, 4-tier)")
print(f"   nlp_auc_cv    : {NLP.get('nlp_auc_cv', 'unknown')}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_level(score: float) -> str:
    return "Low" if score <= RISK_LOW else ("Medium" if score <= RISK_HIGH else "High")


def sanitize_floats(obj):
    """
    Recursively replace NaN/Inf/-Inf with JSON-safe values.
    JSON spec does not allow these float values, causing a ValueError on serialize.
    """
    if isinstance(obj, float):
        if obj != obj:          # NaN
            return None
        if obj == float("inf"):
            return None
        if obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj


def _parse_feedback(text: str) -> list:
    """Split aggregated feedback string into a list of individual feedback items."""
    if not text or text.strip() in ("", "nan"):
        return []
    parts = [p.strip() for p in text.split(" | ") if p.strip() and p.strip() != "nan"]
    return parts


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic markers (**) from LLM output."""
    if not text:
        return text
    cleaned = re.sub(r'\*{1,3}', '', text)
    cleaned = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', cleaned)
    return cleaned.strip()


def get_top_shap(shap_row: pd.Series, top_n: int = 5) -> list:
    top = shap_row.abs().nlargest(top_n)
    return [
        {
            "feature":       k,
            "shap_value":    round(float(shap_row[k]), 4),
            "direction":     "increases_churn" if shap_row[k] > 0 else "decreases_churn",
            "importance":    round(abs(float(shap_row[k])), 4),
            "feature_label": k.replace("_", " ").title(),
        }
        for k in top.index
    ]


# NLP / Sentiment pipeline (Section 7)

def extract_sent_features(text: str) -> list:
    """
    Ekstrak 15 fitur linguistik + VADER untuk satu teks.
    Identik dengan notebook cell 7.1 — digunakan oleh sent_lgbm.
    """
    t  = str(text)
    tl = t.lower()
    sc = ANALYZER.polarity_scores(t)
    sents = [s.strip() for s in re.split(r"[.!?]+", t) if len(s.strip()) > 8]
    ss    = [ANALYZER.polarity_scores(s)["compound"] for s in sents] if sents else [0.0]
    return [
        sc["compound"], sc["neg"], sc["pos"], sc["neu"],
        min(ss),
        float(np.mean(ss)),
        float(np.std(ss)) if len(ss) > 1 else 0.0,
        sum(1 for s in ss if s < -0.05) / max(len(ss), 1),
        ss[-1],
        sum(1 for kw in CHURN_INTENT_LEXICON if kw in tl),
        sum(len(re.findall(p, tl)) for p in NEG_PATTERNS),
        sum(1 for kw in POSITIVE_ANCHORS if kw in tl),
        len(re.findall(CONTRAST_PATTERN, tl)),
        len(t.split()),
        t.count("!"),
    ]


def map_sentiment_tier(proba: float) -> str:
    if proba < 0.20:
        return "Satisfied"
    elif proba < 0.50:
        return "Neutral/Stable"
    elif proba < 0.80:
        return "At Risk (Unsatisfied)"
    else:
        return "Critical (Dissatisfied)"


def predict_sentiment(text: str) -> dict:
    """
    Prediksi sentimen satu teks menggunakan TF-IDF + LightGBM dari notebook.
    Mengembalikan label 4-tier + dissatisfaction_score (0–1).
    """
    ling_feats   = np.array([extract_sent_features(text)])
    X_tfidf      = TFIDF_SENT.transform([str(text)])
    X_extra      = csr_matrix(SCALER_SENT.transform(ling_feats))
    X_full       = hstack([X_tfidf, X_extra])
    proba        = float(SENT_LGBM.predict_proba(X_full)[0, 1])
    return {
        "label":                map_sentiment_tier(proba),
        "dissatisfaction_score": proba,
    }


def compute_vader_features(text: str) -> dict:
    """
    Hitung fitur VADER + urgency untuk satu teks pelanggan (digunakan pipeline utama).
    Menyertakan dissatisfaction_proba dari model TF-IDF+LGBM (notebook section 7).
    """
    empty = {k: 0.0 for k in [
        "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "vader_min_sent", "vader_std_sent", "pct_negative_sent",
        "urgency_score", "avg_words_per_sent",
        "dissatisfaction_proba",
    ]}
    if not text or pd.isna(text):
        return empty

    doc   = ANALYZER.polarity_scores(str(text))
    sents = [s.strip() for s in re.split(r"[.!?|]+", str(text)) if len(s.strip()) > 10]
    sc    = [ANALYZER.polarity_scores(s)["compound"] for s in sents] if sents else [0.0]

    # Sentiment classifier (notebook section 7)
    sent_result = predict_sentiment(text)

    return {
        "vader_compound":       doc["compound"],
        "vader_pos":            doc["pos"],
        "vader_neg":            doc["neg"],
        "vader_neu":            doc["neu"],
        "vader_min_sent":       min(sc),
        "vader_std_sent":       float(np.std(sc)),
        "pct_negative_sent":    sum(1 for s in sc if s < -0.05) / len(sc),
        "urgency_score":        float(sum(1 for w in URGENCY_LEX if w in str(text).lower())),
        "avg_words_per_sent":   float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
        "dissatisfaction_proba": sent_result["dissatisfaction_score"],
    }


def compute_topic_features(texts: list[str]) -> dict:
    """Run LDA topic modelling and return per-customer topic arrays (v10 logic)."""
    X_counts  = CV_VEC.transform(texts)
    X_topics  = LDA.transform(X_counts)
    dom_idx   = X_topics.argmax(axis=1)
    dom_score = X_topics.max(axis=1)
    dom_label = [TOPIC_NAMES[i] for i in dom_idx]
    return {
        "dominant_topic":        dom_idx,
        "dominant_topic_label":  dom_label,
        "dominant_topic_score":  dom_score,
        "topic_distribution":    X_topics,
    }


def compute_nlp_flags(master: pd.DataFrame) -> pd.DataFrame:
    """
    Compute nlp_red_flag and loyalty_risk_flag.
    Requires: vader_compound, urgency_score, churn_score, segment_label, tenure_days

    loyalty_risk_flag: pelanggan dengan churn_score rendah (Low risk) tetapi berada di
    segmen at-risk dengan tenure > 1 tahun — berpotensi under-estimated.
    """
    master = master.copy()
    master["nlp_red_flag"] = (
        (master["vader_compound"] < -0.2) & (master["urgency_score"] >= 1)
    ).astype(int)
    # FIX: "Critical" bukan nama segment yang valid.
    # Segment labels aktual: At-Risk Actives, Loyal Champions, High-Value At-Risk, Disengaged Payers
    at_risk_segs = {"At-Risk Actives", "High-Value At-Risk", "Disengaged Payers"}
    master["loyalty_risk_flag"] = (
        (master["churn_score"] <= RISK_LOW) &
        (master["segment_label"].isin(at_risk_segs)) &
        (master["tenure_days"] > 365)
    ).astype(int)
    return master


# Core pipeline
def run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df):
    """
    Full pipeline — setiap langkah selaras 1-to-1 dengan notebook_1 (section 2-8)
    dan notebook_2 (section 7 untuk sentimen).

    """
    # 1. Clean (notebook section 2.1-2.5)
    ca_df  = ca_df.copy()
    bd_df  = bd_df.copy()
    um_df  = um_df.copy()
    st_df  = st_df.copy()
    nps_df = nps_df.copy()

    # FIX (1): capitalize (bukan title) sesuai notebook cell 2.1
    ca_df["plan_type"]     = ca_df["plan_type"].str.capitalize().str.strip()
    # FIX (2): contract_type hanya di-strip, tanpa ubah case
    ca_df["contract_type"] = ca_df["contract_type"].str.strip()
    # clip NPS 0-10 (notebook 2.5)
    nps_df["nps_score"]    = nps_df["nps_score"].clip(lower=0, upper=10)

    # Parse dates
    for df, cols in [
        (ca_df,  ["subscription_date", "unsubscribed_date"]),
        (bd_df,  ["billing_date", "payment_date"]),
        (um_df,  ["last_login_date"]),
        (st_df,  ["created_date"]),
        (nps_df, ["survey_date"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # obs_date: unsubscribed_date untuk yang churn, REFERENCE_DATE untuk active (notebook 2.2)
    ca_df["obs_date"] = ca_df["unsubscribed_date"].fillna(REF)

    # Hapus record dengan tenure negatif (notebook 2.3)
    ca_df["_tenure_raw"] = (ca_df["obs_date"] - ca_df["subscription_date"]).dt.days
    ca_df = ca_df[ca_df["_tenure_raw"] >= 0].drop(columns="_tenure_raw").reset_index(drop=True)

    # Deduplicate billing (notebook 2.4)
    bd_df = bd_df.drop_duplicates().reset_index(drop=True)

    # 2. FIX (3): Filter record pasca-obs_date (notebook section 2.6) ────────
    bd_df  = (bd_df.merge(ca_df[["customer_id", "obs_date"]], on="customer_id")
                   .query("payment_date  <= obs_date")
                   .drop(columns="obs_date").reset_index(drop=True))
    st_df  = (st_df.merge(ca_df[["customer_id", "obs_date"]], on="customer_id")
                   .query("created_date  <= obs_date")
                   .drop(columns="obs_date").reset_index(drop=True))
    nps_df = (nps_df.merge(ca_df[["customer_id", "obs_date"]], on="customer_id")
                    .query("survey_date  <= obs_date")
                    .drop(columns="obs_date").reset_index(drop=True))
    um_df  = (um_df.merge(ca_df[["customer_id", "obs_date"]], on="customer_id")
                   .query("last_login_date <= obs_date")
                   .drop(columns="obs_date").reset_index(drop=True))

    # 3. Billing features (notebook section 4.1)
    payments = bd_df[bd_df["record_type"] == "payment"].copy()
    payments["delay_days"] = (payments["payment_date"] - payments["billing_date"]).dt.days
    bf = payments.groupby("customer_id").agg(
        total_revenue     =("payment_value", "sum"),
        avg_payment_value =("payment_value", "mean"),
        payment_count     =("payment_value", "count"),
        avg_payment_delay =("delay_days",    "mean"),
        max_payment_delay =("delay_days",    "max"),
    ).reset_index()
    dun = (bd_df[bd_df["record_type"] == "dunning"]
           .groupby("customer_id").size()
           .reset_index(name="dunning_count"))
    bf  = bf.merge(dun, on="customer_id", how="left")
    bf["dunning_count"] = bf["dunning_count"].fillna(0)

    # 4. Usage features (notebook section 4.2)
    # Re-merge obs_date karena sudah di-drop saat filter
    uf = um_df.copy()
    uf = uf.merge(ca_df[["customer_id", "obs_date"]], on="customer_id", how="left")
    # FIX (4): tambahkan .clip(lower=0)
    uf["days_since_login"] = (REF - uf["last_login_date"]).dt.days.clip(lower=0)
    uf = uf.drop(columns=["last_login_date", "obs_date"], errors="ignore")

    # 5. Ticket features (notebook section 4.3)
    tf = st_df.groupby("customer_id").agg(
        total_tickets     =("ticket_id",  "count"),
        open_tickets      =("status",     lambda x: (x == "Open").sum()),
        billing_tickets   =("category",   lambda x: (x == "Billing").sum()),
        technical_tickets =("category",   lambda x: (x == "Technical").sum()),
        critical_tickets  =("priority",   lambda x: (x == "Critical").sum()),
        high_tickets      =("priority",   lambda x: (x == "High").sum()),
    ).reset_index()
    tf["unresolved_ratio"] = tf["open_tickets"]     / tf["total_tickets"].replace(0, 1)
    tf["critical_ratio"]   = tf["critical_tickets"] / tf["total_tickets"].replace(0, 1)

    # 6. NPS tabular features (notebook section 4.4)
    nf = nps_df.groupby("customer_id").agg(
        avg_nps_score =("nps_score", "mean"),
        min_nps_score =("nps_score", "min"),
        survey_count  =("survey_id", "count"),
        pct_detractor =("segment",   lambda x: (x == "detractor").mean()),
    ).reset_index()
    nf["has_nps_data"] = 1

    # 7. NPS text: feedback agregasi per customer
    text_per = (nps_df.groupby("customer_id")["feedback_text"]
                .apply(lambda x: " | ".join(x.dropna().astype(str)))
                .reset_index())
    text_per.columns = ["customer_id", "all_feedback"]

    # 8. Master merge (notebook section 4.5)
    master = ca_df[["customer_id", "plan_type", "contract_type", "total_users",
                     "subscription_date", "obs_date"]].copy()
    master["tenure_days"] = (
        master["obs_date"] - master["subscription_date"]
    ).dt.days.clip(lower=1)

    master = (master
              .merge(uf, on="customer_id", how="left")
              .merge(bf, on="customer_id", how="left")
              .merge(tf, on="customer_id", how="left")
              .merge(nf, on="customer_id", how="left"))

    # FIX (5): zero-fill HANYA kolom yang harus 0 (sesuai notebook 4.5).
    # JANGAN zero-fill total_revenue / avg_payment_value / payment_count —
    # biarkan NaN agar ticket_per_revenue dan log features tidak rusak.
    fill_zero = ["total_tickets", "open_tickets", "billing_tickets", "technical_tickets",
                 "critical_tickets", "high_tickets", "unresolved_ratio", "critical_ratio",
                 "dunning_count", "avg_payment_delay", "max_payment_delay"]
    master[fill_zero]      = master[fill_zero].fillna(0)
    master["has_nps_data"] = master["has_nps_data"].fillna(0)

    # 9. Interaction & ratio features (notebook section 4.6)
    master["log_total_revenue"]     = np.log1p(master["total_revenue"])
    master["log_monthly_usage_hrs"] = np.log1p(master["monthly_usage_hrs"])
    master["log_total_tickets"]     = np.log1p(master["total_tickets"])
    master["log_total_users"]       = np.log1p(master["total_users"])

    # FIX (6): gunakan tenure_days (bukan tenure_capped) sesuai notebook 4.6
    master["dunning_per_tenure"] = (
        master["dunning_count"] /
        (master["tenure_days"] / 30).replace(0, 1)
    )
    master["usage_per_user"] = (
        master["monthly_usage_hrs"] / master["total_users"].replace(0, 1)
    )
    # FIX (7): denominator = total_revenue / 1000 (bukan log_total_revenue + 1e-3)
    # Ini mempertahankan unit "tiket per $1000 revenue" sesuai notebook
    master["ticket_per_revenue"] = (
        master["total_tickets"] / (master["total_revenue"].replace(0, 1) / 1000)
    )
    master["adoption_x_usage"] = (
        master["feature_adoption_pct"] * master["log_monthly_usage_hrs"]
    )
    # FIX (8): fillna(5) bukan fillna(0), tanpa perkalian has_nps_data
    master["nps_x_dunning"] = (
        master["avg_nps_score"].fillna(5) * (master["dunning_count"] + 1)
    )

    # 10. Encoding & NPS imputation (notebook section 4.8)
    # FIX (9): NPS diimputasi dengan global median SEBELUM segmentasi
    master["plan_enc"]     = LE_PLAN.transform(master["plan_type"])
    master["contract_enc"] = LE_CONTRACT.transform(master["contract_type"])

    for col in ["avg_nps_score", "min_nps_score", "survey_count", "pct_detractor"]:
        med = master[col].median()
        master[col] = master[col].fillna(med)

    # 11. Segmentation (notebook section 5)
    # SEG_FEATURES dari artifact = ['monthly_usage_hrs','feature_adoption_pct',
    #                                'total_revenue','payment_count','avg_nps_score']
    # SCALER_SEG di-fit pada raw values tersebut (bukan log)
    seg_data = master[SEG_FEATURES].copy()
    for c in SEG_FEATURES:
        seg_data[c] = seg_data[c].fillna(seg_data[c].median())
    X_seg = SCALER_SEG.transform(seg_data.values)
    master["segment_cluster"] = KMEANS.predict(X_seg)
    master["segment_label"]   = master["segment_cluster"].map(LABEL_MAP)

    # 12. Merge NPS text feedback
    master = master.merge(text_per, on="customer_id", how="left")
    master["all_feedback"] = master["all_feedback"].fillna("")

    # 13. VADER / NLP features
    vader_rows = master["all_feedback"].apply(compute_vader_features)
    vader_df   = pd.DataFrame(list(vader_rows))
    for col in vader_df.columns:
        master[col] = vader_df[col].values

    master["urgency_level"]   = master["urgency_score"].apply(
        lambda u: "high" if u >= 3 else ("medium" if u >= 1 else "low")
    )
    master["sentiment_label"] = master["dissatisfaction_proba"].apply(map_sentiment_tier)

    # 14. Topic features (LDA dari notebook 2)
    topic_feats = compute_topic_features(master["all_feedback"].tolist())
    master["dominant_topic_label"] = topic_feats["dominant_topic_label"]
    master["dominant_topic_score"] = topic_feats["dominant_topic_score"]

    # 5. Model prediction
    X_tab     = master[FEATURES].fillna(0).values
    tab_proba = MODEL.predict_proba(X_tab)[:, 1]

    # 16. SHAP
    shap_vals = EXPLAINER.shap_values(master[FEATURES].fillna(0))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=FEATURES)

    # 17. Score & risk
    churn_score = (tab_proba * 100).round(1)
    master["churn_proba"] = tab_proba.round(4)
    master["churn_score"] = churn_score
    master["risk_level"]  = [risk_level(s) for s in churn_score]

    # 18. Segment centroid RFM context
    centroids_raw = SCALER_SEG.inverse_transform(KMEANS.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_raw, columns=SEG_FEATURES)

    # 20. Build output
    results = []
    for i in range(len(master)):
        row      = master.iloc[i]
        seg      = row["segment_label"]
        seg_cl   = int(row["segment_cluster"])
        seg_prof = next((p for p in SEG_PROFILES if p["segment_label"] == seg), {})
        centroid = centroid_df.iloc[seg_cl].to_dict()

        # Segment actions — keyed by segment label
        seg_action_data = SEG_ACTIONS.get(seg, {})
        seg_act = {
            "description": seg_action_data.get("description", ""),
            "retain":      seg_action_data.get("retain", []),
            "offer":       seg_action_data.get("offer",  []),
            "priority":    seg_action_data.get("priority", row["risk_level"]),
        }

        # centroid keys = SEG_FEATURES = ['monthly_usage_hrs','feature_adoption_pct',
        #                                  'total_revenue','payment_count','avg_nps_score']
        seg_rfm_context = {
            "days_since_login":     {"customer": round(float(row.get("days_since_login", 0)), 1),
                                     "segment_avg": 0.0},
            "payment_count":        {"customer": round(float(row.get("payment_count", 0)), 1),
                                     "segment_avg": round(float(centroid.get("payment_count", 0)), 1)},
            "total_revenue":        {"customer": round(float(row.get("total_revenue", 0) if pd.notna(row.get("total_revenue")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("total_revenue", 0)), 1)},
            "monthly_usage_hrs":    {"customer": round(float(row.get("monthly_usage_hrs", 0) if pd.notna(row.get("monthly_usage_hrs")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("monthly_usage_hrs", 0)), 1)},
            "feature_adoption_pct": {"customer": round(float(row.get("feature_adoption_pct", 0) if pd.notna(row.get("feature_adoption_pct")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("feature_adoption_pct", 0)), 1)},
            "avg_nps_score":        {"customer": round(float(row.get("avg_nps_score", 0)), 2),
                                     "segment_avg": round(float(centroid.get("avg_nps_score", 0)), 2)},
        }

        results.append({
            # Identity
            "customer_id":          row["customer_id"],
            "plan_type":            row["plan_type"],
            "contract_type":        row["contract_type"],

            # Churn Score
            "churn_score":          float(row["churn_score"]),
            "churn_proba":          round(float(tab_proba[i]), 4),
            "tabular_proba":        round(float(tab_proba[i]), 4),
            "nlp_proba":            round(float(tab_proba[i]), 4),
            "risk_level":           row["risk_level"],

            # SHAP (tabular)
            "shap_top5":            get_top_shap(shap_df.iloc[i]),

            # NLP / Sentiment
            "sentiment": {
                "label":                 row["sentiment_label"],
                "dissatisfaction_score": round(float(row["dissatisfaction_proba"]), 4),
                "vader_compound":        round(float(row["vader_compound"]), 4),
                "vader_neg":             round(float(row["vader_neg"]), 4),
                "pct_negative_sent":     round(float(row["pct_negative_sent"]) * 100, 1),
                "urgency_level":         row["urgency_level"],
                "urgency_score":         int(row["urgency_score"]),
                "dominant_topic":        row["dominant_topic_label"],
                "topic_strength":        round(float(row["dominant_topic_score"]), 3),
                "feedback_texts":        _parse_feedback(str(row["all_feedback"])),
            },

            # NPS data flag
            "has_nps_data":         int(row["has_nps_data"]),

            # Segmentation
            "segment_label":        seg,
            "segment_cluster":      seg_cl,
            "segment_rfm_context":  seg_rfm_context,
            "segment_profile":      seg_prof,
            "segment_actions":      seg_act,
        })

    return results


# LLM narrative builders
def build_churn_xai_prompt(r: dict) -> str:
    """Churn Prediction XAI prompt. Output: strict JSON."""
    shap_lines = "\n".join([
        f"  {idx+1}. {f['feature_label']} "
        f"({'increases' if f['direction'] == 'increases_churn' else 'decreases'} churn risk, "
        f"SHAP: {f['shap_value']:+.3f})"
        for idx, f in enumerate(r["shap_top5"])
    ])
    sent    = r["sentiment"]
    rfm     = r["segment_rfm_context"]

    tenure   = rfm.get("days_since_login", {}).get("customer", 0)
    revenue  = rfm.get("total_revenue", {}).get("customer", 0)
    usage    = rfm.get("monthly_usage_hrs", {}).get("customer", 0)
    adoption = rfm.get("feature_adoption_pct", {}).get("customer", 0)
    nps      = rfm.get("avg_nps_score", {}).get("customer", 0)

    feedback_items = sent.get("feedback_texts", [])
    feedback_str   = " | ".join(feedback_items[:5]) if feedback_items else "No feedback available"

    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON, no other text, no markdown.

CUSTOMER: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']}) | Segment: {r['segment_label']}
Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}
Revenue: ${revenue:,.0f}/mo | Usage: {usage:.0f}h/mo | Feature Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10
Days since last login: {tenure:.0f}

TOP CHURN FACTORS (SHAP):
{shap_lines}

SENTIMENT: {sent['label']} | VADER: {sent['vader_compound']:+.3f} | Urgency: {sent['urgency_level']} | Topic: {sent['dominant_topic']}
Feedback: "{feedback_str[:300]}"

Instructions:
- score_reason: 1-2 plain sentences explaining why this customer has this churn score. Use the actual numbers. Be specific.
- risk_factors: 3 short phrases (max 8 words each), one per top churn driver.
- feedback_signal: 1 plain sentence summarizing what the customer is complaining about.
- retain: exactly 3 short action items (max 10 words each) specific to this customer's situation.
- offer: exactly 3 short offer items (max 10 words each) specific to this customer's plan and revenue.
Do not use asterisks, bold, or markdown in any value.

Reply JSON:
{{"score_reason":"...","risk_factors":["...","...","..."],"feedback_signal":"...","action":{{"retain":["...","...","..."],"offer":["...","...","..."],"reason":"..."}}}}"""


def build_segment_xai_prompt(r: dict) -> str:
    """Segment XAI prompt. Output: strict JSON."""
    rfm      = r["segment_rfm_context"]
    sent     = r["sentiment"]
    seg_prof = r["segment_profile"]
    seg_act  = r["segment_actions"]

    rfm_lines = "\n".join([
        f"  {k.replace('_', ' ').title()}: customer={v['customer']:.1f}, segment_avg={v['segment_avg']:.1f}"
        for k, v in rfm.items()
    ])

    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON, no other text, no markdown.

CUSTOMER PROFILE:
ID: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']})
Segment: {r['segment_label']} | Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}

CUSTOMER vs SEGMENT AVERAGES:
{rfm_lines}

SEGMENT PROFILE:
Customers: {seg_prof.get('count', 'N/A')} | Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk: {seg_prof.get('pct_high_risk', 'N/A')}% | Avg tenure: {seg_prof.get('avg_tenure_days', 'N/A')} days
Description: {seg_act.get('description', '')}

SENTIMENT:
{sent['label']} (VADER: {sent['vader_compound']:+.3f}) | Topic: {sent['dominant_topic']} | Urgency: {sent['urgency_level']}

Do not use asterisks, bold, or markdown in any value.

Reply JSON:
{{"segment_reason":"1-2 plain sentences why this customer belongs here","characteristics":["trait 1","trait 2","trait 3"],"watch_out":"1 plain sentence about the main concern","strategy":"1 plain sentence about the best action"}}"""


def build_segment_cohort_prompt(seg_label: str, seg_prof: dict, seg_desc: str,
                                 retain_actions: list, offer_actions: list,
                                 total_customers: int) -> str:
    """Segment cohort narrative prompt — one call per segment, not per customer."""
    share_pct = round(seg_prof.get("count", 0) / max(total_customers, 1) * 100, 1)

    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON, no other text, no markdown.

SEGMENT DATA: {seg_label}
Total customers: {seg_prof.get('count', 'N/A')} ({share_pct}% of all customers)
Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk: {seg_prof.get('pct_high_risk', 'N/A')}%
Avg revenue: ${seg_prof.get('avg_revenue', 0):,.0f}/month
Avg usage hrs: {seg_prof.get('avg_usage_hrs', 0):.0f}h/month
Avg NPS: {seg_prof.get('avg_nps', 0):.1f}/10
Avg tenure: {seg_prof.get('avg_tenure_days', 0):.0f} days
Churn rate: {seg_prof.get('churn_rate', 0)*100:.1f}%
Description: {seg_desc}
Retention actions available: {retain_actions}
Offers available: {offer_actions}

Do not use asterisks, bold, or markdown in any value.

Reply with this exact JSON:
{{
  "narrative": "3-4 plain sentences describing this segment: key characteristics, behavior, and business situation",
  "defining_traits": ["short trait 1 (max 4 words)", "short trait 2", "short trait 3", "short trait 4"],
  "top_priority_action": "1 plain sentence with the single highest-priority action for this segment",
  "risk_summary": "1 plain sentence summarizing the churn risk for this segment"
}}"""


async def _call_llm_xai(prompt: str) -> str:
    """
    Call OpenAI-compatible chat API for XAI/predict endpoints (single-prompt, JSON mode).
    Supports both native Ollama (/api/chat) and OpenAI-compatible (/chat/completions).
    """
    _raw_url = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
    if not _raw_url.startswith("http"):
        _raw_url = f"https://{_raw_url}"
    base_url = _raw_url.rstrip("/")

    is_native_ollama = (
        "localhost" in base_url
        or "127.0.0.1" in base_url
        or base_url.rstrip("/").endswith("/api")
    )

    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"

    if is_native_ollama:
        endpoint = base_url.replace("/v1", "").rstrip("/") + "/api/chat"
        payload  = {
            "model":    LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "format":   "json",
            "options":  {"temperature": 0.2, "num_predict": 1200},
        }
    else:
        endpoint = f"{base_url}/chat/completions"
        payload  = {
            "model":       LLM_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  1200,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        elif "message" in data:
            content = data["message"]["content"]
        else:
            return json.dumps({"error": f"unexpected response shape: {list(data.keys())}"})

        # Strip <think>…</think> blocks emitted by some reasoning models
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        content = re.sub(r"^```(?:json)?\s*", "", content).rstrip("```").strip()
        content = _strip_markdown(content)
        return content

    except httpx.ConnectError as e:
        return json.dumps({"error": f"cannot connect to {endpoint}: {str(e)[:100]}"})
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code} from {endpoint}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:100]}"})


# Endpoints
class SegmentCohortRequest(BaseModel):
    segment_label: str
    total_customers: int
    avg_churn_score: float
    pct_high_risk: float
    avg_revenue: float
    avg_usage_hrs: float
    avg_nps: float
    avg_tenure_days: float = 0.0
    churn_rate: float = 0.0
    segment_description: str = ""
    retain_actions: list = []
    offer_actions: list = []
    total_all_customers: int = 0


@app.post("/generate-cohort-xai")
async def generate_cohort_xai(segments: list[SegmentCohortRequest]):
    """
    Generate LLM cohort narratives from pre-aggregated segment stats.
    Does NOT re-run the ML pipeline — only calls the LLM per segment.
    Used by the regenerate button on the Segmentation page.
    """
    results: dict[str, str] = {}
    total = sum(s.total_customers for s in segments) or 1
    for seg in segments:
        prompt = build_segment_cohort_prompt(
            seg_label       = seg.segment_label,
            seg_prof        = {
                "count":            seg.total_customers,
                "avg_churn_score":  seg.avg_churn_score,
                "pct_high_risk":    seg.pct_high_risk,
                "avg_revenue":      seg.avg_revenue,
                "avg_usage_hrs":    seg.avg_usage_hrs,
                "avg_nps":          seg.avg_nps,
                "avg_tenure_days":  seg.avg_tenure_days,
                "churn_rate":       seg.churn_rate,
            },
            seg_desc        = seg.segment_description,
            retain_actions  = seg.retain_actions,
            offer_actions   = seg.offer_actions,
            total_customers = seg.total_all_customers or total,
        )
        results[seg.segment_label] = await _call_llm_xai(prompt)
    return sanitize_floats({"status": "success", "cohort_xai": results})


@app.get("/health")
def health():
    model_type = type(MODEL).__name__
    has_calibrated = "calibrated_model" in A
    return {
        "status":          "ok",
        "model_version":   A.get("model_version", "v2.1"),
        "model_type":      model_type,
        "calibrated":      has_calibrated,
        "model_name":      A.get("best_model_name", "unknown"),
        "risk_low":        RISK_LOW,
        "risk_high":       RISK_HIGH,
        "fusion_alpha":    FUSION_ALPHA,
        "n_features":      len(FEATURES),
        "n_topics":        N_TOPICS,
        "stability":       A.get("stability", {}),
    }


@app.post("/predict")
async def predict(
    customer_accounts:         UploadFile = File(...),
    monthly_usage_metrics:     UploadFile = File(...),
    billing_data:              UploadFile = File(...),
    support_tickets:           UploadFile = File(...),
    nps_surveys_with_feedback: UploadFile = File(...),
    generate_xai:              bool = True,
):
    """
    Predict all customers.
    Returns per-customer churn + segmentation XAI.
    Also generates per-segment cohort narratives (xai_segment_cohort) when generate_xai=True.
    """
    try:
        ca_df  = pd.read_csv(io.BytesIO(await customer_accounts.read()))
        um_df  = pd.read_csv(io.BytesIO(await monthly_usage_metrics.read()))
        bd_df  = pd.read_csv(io.BytesIO(await billing_data.read()))
        st_df  = pd.read_csv(io.BytesIO(await support_tickets.read()))
        nps_df = pd.read_csv(io.BytesIO(await nps_surveys_with_feedback.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    try:
        results = run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    if generate_xai:
        for r in results:
            r["xai_churn_explanation"]   = await _call_llm_xai(build_churn_xai_prompt(r))
            r["xai_segment_explanation"] = await _call_llm_xai(build_segment_xai_prompt(r))

        # Per-segment cohort narratives
        # Aggregate segment stats from results, then call LLM once per segment
        total_customers = len(results)
        seen_segments: set[str] = set()
        segment_cohort_xai: dict[str, str] = {}

        for r in results:
            seg = r["segment_label"]
            if seg in seen_segments:
                continue
            seen_segments.add(seg)
            seg_prof = r["segment_profile"]
            seg_act  = r["segment_actions"]
            cohort_prompt = build_segment_cohort_prompt(
                seg_label       = seg,
                seg_prof        = seg_prof,
                seg_desc        = seg_act.get("description", ""),
                retain_actions  = seg_act.get("retain", []),
                offer_actions   = seg_act.get("offer", []),
                total_customers = total_customers,
            )
            segment_cohort_xai[seg] = await _call_llm_xai(cohort_prompt)

        # Attach cohort XAI to every customer row in that segment
        for r in results:
            r["xai_segment_cohort"] = segment_cohort_xai.get(r["segment_label"])

    else:
        for r in results:
            r["xai_churn_explanation"]   = None
            r["xai_segment_explanation"] = None
            r["xai_segment_cohort"]      = None

    return sanitize_floats({"status": "success", "total_customers": len(results), "predictions": results})


@app.post("/predict/single")
async def predict_single(
    customer_id:               str,
    customer_accounts:         UploadFile = File(...),
    monthly_usage_metrics:     UploadFile = File(...),
    billing_data:              UploadFile = File(...),
    support_tickets:           UploadFile = File(...),
    nps_surveys_with_feedback: UploadFile = File(...),
):
    """Single customer — full XAI for both churn prediction and segmentation."""
    try:
        ca_df  = pd.read_csv(io.BytesIO(await customer_accounts.read()))
        um_df  = pd.read_csv(io.BytesIO(await monthly_usage_metrics.read()))
        bd_df  = pd.read_csv(io.BytesIO(await billing_data.read()))
        st_df  = pd.read_csv(io.BytesIO(await support_tickets.read()))
        nps_df = pd.read_csv(io.BytesIO(await nps_surveys_with_feedback.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")

    if customer_id not in ca_df["customer_id"].values:
        raise HTTPException(status_code=404, detail=f"{customer_id} not found")

    for df in [ca_df, um_df, bd_df, st_df, nps_df]:
        mask = df["customer_id"] == customer_id
        df.drop(df[~mask].index, inplace=True)

    results = run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df)
    r = results[0]
    r["xai_churn_explanation"]   = await _call_llm_xai(build_churn_xai_prompt(r))
    r["xai_segment_explanation"] = await _call_llm_xai(build_segment_xai_prompt(r))
    r["xai_segment_cohort"]      = await _call_llm_xai(
        build_segment_cohort_prompt(
            seg_label       = r["segment_label"],
            seg_prof        = r["segment_profile"],
            seg_desc        = r["segment_actions"].get("description", ""),
            retain_actions  = r["segment_actions"].get("retain", []),
            offer_actions   = r["segment_actions"].get("offer", []),
            total_customers = 1,
        )
    )
    return sanitize_floats(r)


# Simulation: Churn Trajectory

from fastapi.responses import StreamingResponse  # noqa: E402

async def stream_llm(system: str, user_msg: str, max_tokens: int = 600):
    """Stream tokens from LLM via OpenAI-compatible SSE."""
    _raw_url = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
    if not _raw_url.startswith("http"):
        _raw_url = f"https://{_raw_url}"
    base_url = _raw_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"

    payload = {
        "model":       LLM_MODEL,
        "messages":    [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        "stream":      True,
        "temperature": 0.7,
        "max_tokens":  max_tokens,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", endpoint, headers=headers, json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        obj   = json.loads(data)
                        token = obj["choices"][0]["delta"].get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
    except Exception as exc:
        yield f"[Error: {str(exc)[:80]}]"


async def stream_llm_no_think(system: str, user_msg: str, max_tokens: int = 600):
    """Stream tokens from LLM, filtering <think>...</think> blocks in real-time.

    Fix: when buf is too short to safely emit (could be start of a <think> tag),
    hold it in the buffer without emitting — do NOT emit-and-keep (causes repetition).
    """
    OPEN = "<think>"
    CLOSE = "</think>"
    buf = ""
    in_think = False

    async for tok in stream_llm(system, user_msg, max_tokens):
        buf += tok

        while True:
            if in_think:
                pos = buf.find(CLOSE)
                if pos >= 0:
                    buf = buf[pos + len(CLOSE):]
                    in_think = False
                    # continue loop — there may be normal content after </think>
                else:
                    # Still inside <think>; discard everything except a trailing
                    # window that could be a split closing tag.
                    keep = len(CLOSE) - 1
                    buf = buf[-keep:] if len(buf) > keep else buf
                    break
            else:
                pos = buf.find(OPEN)
                if pos >= 0:
                    out = buf[:pos]
                    buf = buf[pos + len(OPEN):]
                    in_think = True
                    if out:
                        yield out
                    # continue loop — process remainder after <think>
                else:
                    # No <think> found; emit all but a trailing window that could
                    # be a partial opening tag starting at the end of buf.
                    keep = len(OPEN) - 1
                    if len(buf) > keep:
                        out = buf[:-keep]
                        buf = buf[-keep:]
                        if out:
                            yield out
                    # else: buf is too short — hold everything, wait for more tokens
                    break

    # Flush whatever remains outside a think block
    if buf and not in_think:
        yield buf


class _CustomerDataSim(BaseModel):
    customer_id:         str
    churn_score:         float
    risk_level:          str
    plan_type:           str
    contract_type:       str
    segment_label:       str
    shap_top5:           list
    sentiment:           dict
    segment_rfm_context: dict


class SimulateRequest(BaseModel):
    customer_data:  _CustomerDataSim
    scenario:       str        = ""   # optional: intervention scenario for projection
    chat_history:   list[dict] = []   # [{question, narrative}] — prior turns
    horizon_weeks:  int        = 12   # forecast horizon in weeks (4/8/12/24/52)
    segment_labels: list[str]  = []   # actual ML segment names from DB (for migration labels)


def _build_ctx(c: _CustomerDataSim, scenario: str) -> str:
    """Build a compact customer context string for LLM prompts."""
    rfm      = c.segment_rfm_context
    shap_txt = "\n".join(
        f"  - {f['feature_label']}: {f['shap_value']:+.3f} "
        f"({'increases' if f['direction'] == 'increases_churn' else 'decreases'} churn risk)"
        for f in c.shap_top5
    )
    rev      = rfm.get("total_revenue",       {}).get("customer", 0)
    usage    = rfm.get("monthly_usage_hrs",    {}).get("customer", 0)
    adoption = rfm.get("feature_adoption_pct", {}).get("customer", 0)
    nps      = rfm.get("avg_nps_score",        {}).get("customer", 0)
    dsl      = rfm.get("days_since_login",     {}).get("customer", 0)

    feedback_items = c.sentiment.get("feedback_texts", [])
    feedback_preview = feedback_items[0][:150] if feedback_items else "No feedback"

    ctx = (
        f"CUSTOMER: {c.customer_id} | Plan: {c.plan_type} ({c.contract_type}) | Segment: {c.segment_label}\n"
        f"Churn Score: {c.churn_score:.1f}/100 | Risk Level: {c.risk_level}\n"
        f"Revenue: ${rev:,.0f}/mo | Usage: {usage:.0f}h/mo | Feature Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10\n"
        f"Days since last login: {dsl:.0f}\n"
        f"Sentiment: {c.sentiment.get('label', 'N/A')} "
        f"(VADER: {c.sentiment.get('vader_compound', 0):+.3f}) | "
        f"Urgency: {c.sentiment.get('urgency_level', 'N/A')}\n"
        f"Feedback: \"{feedback_preview}\"\n\n"
        f"TOP CHURN DRIVERS (SHAP):\n{shap_txt}"
    )
    if scenario.strip():
        ctx += f"\n\nINTERVENTION SCENARIO BEING EVALUATED:\n{scenario}"
    return ctx


async def call_llm(system: str, user_msg: str, max_tokens: int = 1200) -> str:
    """Non-streaming LLM call; returns the full response text."""
    _raw_url = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
    if not _raw_url.startswith("http"):
        _raw_url = f"https://{_raw_url}"
    base_url = _raw_url.rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = f"Bearer {LLM_KEY}"

    payload = {
        "model":       LLM_MODEL,
        "messages":    [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        "stream":      False,
        "temperature": 0.4,
        "max_tokens":  max_tokens,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _build_sim_json_system(horizon_weeks: int = 12, segment_labels: list = None) -> str:
    """Build dynamic simulation JSON system prompt for given forecast horizon."""
    # Determine time step — keep ~7-14 points regardless of horizon
    if horizon_weeks <= 8:
        step = 1
    elif horizon_weeks <= 16:
        step = 2
    else:
        step = 4  # monthly steps (4 weeks per month)

    time_points = list(range(0, horizon_weeks + 1, step))
    if time_points[-1] != horizon_weeks:
        time_points.append(horizon_weeks)

    # Build the baseline array schema
    baseline_items = ",\n    ".join(
        f'{{"week": {w}, "prob": <float 0-100>}}' for w in time_points
    )
    # First point must pin to actual churn_score
    baseline_items = baseline_items.replace(
        '{"week": 0, "prob": <float 0-100>}',
        '{"week": 0, "prob": <ACTUAL_CHURN_SCORE_0_TO_100>}',
        1,
    )

    # Build segment migration labels
    if segment_labels and len(segment_labels) >= 2:
        seg_items = ",\n    ".join(
            f'{{"label": "{label}", "prob": <float 0.0-1.0>}}' for label in segment_labels
        )
        seg_note = f"Use these exact segment labels from the system: {segment_labels}"
    else:
        seg_items = (
            '{"label": "Churned", "prob": <float 0.0-1.0>},\n'
            '    {"label": "High Risk", "prob": <float 0.0-1.0>},\n'
            '    {"label": "At Risk", "prob": <float 0.0-1.0>},\n'
            '    {"label": "Retained", "prob": <float 0.0-1.0>}'
        )
        seg_note = "Use these standard risk migration labels"

    mid_week   = time_points[len(time_points) // 2]
    never_val  = horizon_weeks + 1

    return f"""\
You are a customer retention analytics engine. Given a customer's churn data, \
produce a precise JSON churn trajectory forecast for a {horizon_weeks}-week horizon. \
No prose — output ONLY valid JSON.

The JSON schema (all fields required):
{{
  "baseline": [
    {baseline_items}
  ],
  "projection": null,
  "retention_window_weeks": <int: weeks before churn probability exceeds 80>,
  "revenue_at_risk": <float: monthly_revenue * (churn_prob/100) * 3>,
  "confidence": <float 0.0-1.0>,
  "intervention_impact_pct": null,
  "segment_migration": [
    {seg_items}
  ]
}}

Rules:
- baseline[0].prob MUST equal the actual churn_score from the customer data exactly.
- The forecast spans {horizon_weeks} weeks ({horizon_weeks // 4} months). \
  Generate realistic churn evolution based on the customer's specific risk factors (SHAP values, \
  usage, NPS, sentiment). High-risk drivers cause upward pressure; positive signals allow decay.
- CRITICAL — THE TRAJECTORY MUST NOT BE FLAT. You MUST generate meaningful variation:
  * If churn_score >= 85: baseline should oscillate ±5-15 points but generally remain high. \
    Show some natural fluctuation (e.g. 98→95→97→92→94 over the period). NEVER keep all values at 100.
  * If churn_score 60-84: show gradual increase of 5-20 points over the horizon with variation.
  * If churn_score < 60: show moderate increase or plateau with slight variation.
  * Final value should differ from week-0 by at least 5 percentage points.
- segment_migration probs must sum to 1.0. {seg_note}.
- retention_window_weeks: weeks until baseline churn probability crosses 80. \
  If already above 80 at week 0, set to 0. \
  If it never crosses 80 within {horizon_weeks} weeks, set to {never_val}.
- revenue_at_risk: monthly_revenue × (churn_prob at week {mid_week} / 100) × 3.
"""

_SIM_NARRATIVE_SYSTEM = """\
You are a senior customer success analyst at a SaaS company.
Write EXACTLY 4 short paragraphs (2-3 sentences each) in plain English:

Paragraph 1: Current churn risk — state the score, the top SHAP factors with their values, and what this means for the account.
Paragraph 2: Trajectory forecast — where the churn probability is heading over this period and the key drivers.
Paragraph 3: Retention window and financial exposure — how much time is left and how much revenue is at risk.
Paragraph 4: The single most urgent action with a concrete timeline and expected outcome.

Separate each paragraph with one blank line.
Rules: No bullet points. No headers. No markdown. No asterisks. Write directly without an opening phrase.
"""

_SCENARIO_NARRATIVE_SYSTEM = """\
You are a senior customer success analyst at a SaaS company.
Write EXACTLY 4 short paragraphs (2-3 sentences each) in plain English:

Paragraph 1: What this scenario proposes and the projected change in churn probability (use specific numbers).
Paragraph 2: The strongest reason why this intervention could succeed based on the customer data.
Paragraph 3: The main risks or downsides — what could fail and what conditions must be met.
Paragraph 4: The concrete next steps with a specific timeline and measurable success metrics.

Separate each paragraph with one blank line.
Rules: No bullet points. No headers. No markdown. No asterisks. Write directly without an opening phrase.
"""

_ASK_SYSTEM = """\
You are a customer success analytics assistant at a SaaS company.
Answer the user's question based on the available customer data and simulation context.
Use plain, professional English that is easy for a business team to understand.
Format: 2-4 short paragraphs or concise bullet points, whichever fits better.
Include numbers from the data when relevant.
Do not use markdown bold (**), asterisks, or headers.
Answer directly without openers like "Sure" or "Based on the data".
"""

# Unified Agent Config
# One list of persona metadata. System prompts are selected per-mode below.
AGENT_PERSONAS = [
    {"name": "Risk Analyst",    "short": "RA", "color": "#ef4444"},
    {"name": "Customer Success","short": "CS", "color": "#3b82f6"},
    {"name": "Finance Analyst", "short": "FN", "color": "#f59e0b"},
    {"name": "Product Manager", "short": "PM", "color": "#8b5cf6"},
]

# Prompts for "initial" mode — agents examine baseline situation in English
AGENT_ANALYZE_SYSTEMS: dict[str, str] = {
    "Risk Analyst": (
        "You are a Churn Risk Analyst at a SaaS company. "
        "Review this customer's churn score, top SHAP risk factors, and trajectory. "
        "Give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: most critical risk factor with its SHAP value, what is driving churn up, "
        "one trend to watch, and one specific action to reduce risk. "
        "Use actual numbers from the data."
    ),
    "Customer Success": (
        "You are a Customer Success Manager at a SaaS company. "
        "Analyze this customer's last login, NPS score, feedback sentiment, and engagement signals. "
        "Give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: root cause of disengagement or dissatisfaction, key engagement signal, "
        "current sentiment summary, and one specific outreach action to take this week."
    ),
    "Finance Analyst": (
        "You are a Finance Analyst at a SaaS company. "
        "Use this customer's revenue data and churn probability. "
        "Give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: estimated revenue loss if customer churns, customer value vs segment average, "
        "economic case for intervention, and one cost-effective retention offer with a concrete value."
    ),
    "Product Manager": (
        "You are a Product Manager at a SaaS company. "
        "Review this customer's feature adoption rate and product usage hours. "
        "Give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: the biggest feature adoption gap driving churn risk, concerning usage pattern, "
        "most relevant unused feature, and one specific product action to close the gap."
    ),
}

# Prompts for "scenario" mode — agents debate a specific intervention in English
AGENT_SCENARIO_SYSTEMS: dict[str, str] = {
    "Risk Analyst": (
        "You are a Churn Risk Analyst at a SaaS company. "
        "Based on the customer data and the proposed intervention scenario, "
        "give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: estimated churn probability reduction in percentage points, "
        "which risk factor is most affected, residual risk if the intervention fails, "
        "and your confidence in this intervention succeeding."
    ),
    "Customer Success": (
        "You are a Customer Success Manager at a SaaS company. "
        "Based on the customer data and the proposed intervention, "
        "give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: whether this intervention addresses the real root cause of churn, "
        "what in the customer's situation supports or blocks success, "
        "what must accompany this intervention, and a realistic execution timeline."
    ),
    "Finance Analyst": (
        "You are a Finance Analyst at a SaaS company. "
        "Based on the customer data and the proposed intervention, "
        "give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: estimated intervention cost vs monthly revenue at risk, "
        "ROI if the intervention succeeds with concrete numbers, "
        "break-even point, and your financial recommendation."
    ),
    "Product Manager": (
        "You are a Product Manager at a SaaS company. "
        "Based on the customer data and the proposed intervention, "
        "give 4-5 bullet points using - (hyphen). Each point is one short, clear sentence. No asterisks or bold.\n"
        "Cover: impact on feature adoption and product engagement, "
        "most relevant feature or flow for this intervention, "
        "product change that would strengthen the outcome, "
        "and the key product metric to monitor."
    ),
}


def _build_contextual_fallback(
    risk_level: str,
    customer_profile: dict,
    scenario: str = "",
) -> list[str]:
    """
    Build 4 contextual fallback recommendations from actual customer attributes.
    Priority order: top SHAP factors → segment → plan/contract → risk urgency.
    """
    segment  = customer_profile.get("segment_label", "")
    plan     = customer_profile.get("plan_type", "")
    contract = customer_profile.get("contract_type", "")
    shap5    = customer_profile.get("shap_top5", [])

    _FACTOR_ACTIONS: dict[str, str] = {
        "days since login":     "Run a re-engagement campaign with a live feature demo",
        "monthly usage hrs":    "Schedule an intensive onboarding session to increase usage",
        "avg payment delay":    "Review payment options and enable auto-billing",
        "feature adoption":     "Run a 30-day 1-on-1 premium feature training with CSM",
        "adoption x usage":     "Launch a feature activation sprint with step-by-step guidance",
        "nps score":            "Conduct an NPS recovery call and resolve complaints within 24h",
        "avg nps score":        "Conduct an NPS recovery call and resolve complaints within 24h",
        "tenure days":          "Offer a loyalty appreciation program for long-term customers",
        "contract type":        "Convert to annual contract with a 25% discount",
        "plan type":            "Upgrade to a higher plan with a free 30-day trial",
        "support tickets":      "Fast-track all open tickets and assign a priority support agent",
        "billing issues":       "Audit billing, remove unclear charges, and offer account credit",
    }

    pool: list[str] = []
    seen: set[str]  = set()

    def add(rec: str) -> None:
        if rec not in seen:
            seen.add(rec)
            pool.append(rec)

    for factor in shap5[:3]:
        label  = str(factor.get("feature_label", "")).lower()
        shap_v = float(factor.get("shap_value", 0))
        if shap_v <= 0:
            continue
        for key, action in _FACTOR_ACTIONS.items():
            if key in label or label in key:
                add(action)
                break

    seg = segment.lower()
    if any(x in seg for x in ("critical", "at-risk", "at risk")):
        add("Escalate to senior team and create a 30-day emergency retention plan")
        add("Contact the decision maker directly for a win-back meeting")
    elif "champion" in seg:
        add("Offer early access to exclusive features as a loyalty reward")
        add("Launch an ambassador program with renewal discounts and extra benefits")
    elif "loyalist" in seg:
        add("Enroll in a VIP loyalty program with long-term exclusive benefits")
        add("Invite to the customer advisory board for exclusive feedback sessions")
    elif any(x in seg for x in ("potential", "prospect")):
        add("Schedule a free consultation to find the most relevant product value")
        add("Offer a plan better suited to current needs at a special price")

    p = plan.lower()
    if any(x in p for x in ("basic", "starter", "free")):
        add("Upgrade to Pro plan with a free 60-day trial, no commitment required")
    elif any(x in p for x in ("enterprise", "business", "corporate")):
        add("Assign a dedicated account manager and schedule a quarterly executive review")
    elif "pro" in p:
        add("Activate enterprise add-on features at a special rate for 3 months")

    ct = contract.lower()
    if "month" in ct:
        add("Convert to annual contract with a 30% discount and price lock-in")
    elif any(x in ct for x in ("annual", "year")):
        add("Extend to a 2-year contract with a price freeze and feature credits")

    if risk_level == "High":
        add("Freeze billing for 2 months and grant full premium feature access")
        add("Contact customer within 24 hours for emergency retention intervention")
    elif risk_level == "Medium":
        add("Schedule automated weekly CSM check-ins for the next 8 weeks")
        add("Offer a 20% discount for a 6-month contract extension")
    else:
        add("Send a satisfaction survey and activate an exclusive referral program")
        add("Give a usage credit bonus as a loyalty appreciation gesture")

    if scenario.strip():
        add("Combine the above intervention with a long-term loyalty incentive")
        add("A/B test: price discount vs increased CSM service intensity")

    _generics = [
        "Schedule a monthly business review and monitor account health metrics",
        "Grant free premium feature access for 60 days",
        "Assign a dedicated Customer Success Manager for 90 days",
        "Offer a 20% discount for a 6-month contract extension",
    ]
    for g in _generics:
        if len(pool) >= 4:
            break
        add(g)

    return pool[:4]


async def _extract_recommendations(
    ctx: str,
    agent_outputs: list,
    scenario: str = "",
    risk_level: str = "High",
    customer_profile: dict | None = None,
) -> list:
    """
    Extract 4 clickable scenario options.
    Primary: LLM generates them from agent debate output.
    Fallback: contextual recs built from real customer attributes (no LLM needed).
    """
    cp = customer_profile or {}

    debate_text = "\n".join(
        f"[{o['name']}]: {o['content'][:500]}" for o in agent_outputs
    )
    if scenario.strip():
        context_note = (
            f"The analyst team just discussed this intervention: \"{scenario}\".\n"
            f"Generate 4 FOLLOW-UP or ALTERNATIVE scenarios worth simulating next."
        )
    else:
        context_note = (
            "The analyst team just analyzed this customer's baseline churn situation.\n"
            "Generate 4 INTERVENTION scenarios worth simulating to reduce churn risk."
        )

    prompt = (
        f"{context_note}\n\n"
        f"Rules:\n"
        f"- Each option must be a short, specific action phrase (15-70 characters)\n"
        f"- Write in English\n"
        f"- Concrete enough to be used as a simulation scenario input\n"
        f"- Varied — do not repeat the same idea\n\n"
        f"Customer context:\n"
        f"{ctx[:200]}\n\n"
        f"Agent analysis outputs:\n{debate_text}\n\n"
        f"Return ONLY a valid JSON array with exactly 4 strings. "
        f"Example: [\"Offer 20% discount for 3 months\", \"Assign dedicated CSM\", ...]\n"
        f"No explanation, no markdown fence, no extra text."
    )

    # Pre-build contextual fallback (zero extra LLM calls, uses real customer data)
    _fallback_recs = _build_contextual_fallback(risk_level, cp, scenario)

    try:
        raw = await call_llm(
            "Generate 4 intervention scenario options. Return ONLY a JSON array of 4 strings.",
            prompt,
            max_tokens=450,
        )
        # Strip think blocks and fences
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE).strip()
        # Strategy 1: parse full JSON array
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start >= 0 and end > start:
            items = json.loads(raw[start:end])
            cleaned = [str(x).strip() for x in items if str(x).strip()]
            if len(cleaned) >= 2:
                return cleaned[:4]
        # Strategy 2: extract quoted strings
        quoted = re.findall(r'"([^"]{10,100})"', raw)
        if len(quoted) >= 2:
            return quoted[:4]
    except Exception:
        pass

    return _fallback_recs


# Scenario JSON system prompt
_SCENARIO_UPDATE_JSON_SYSTEM = """\
You are a customer retention analytics engine. A multi-agent team has debated an intervention \
scenario. Based on their analysis, produce a JSON update with ONLY these fields. \
No prose — output ONLY valid JSON.

Schema:
{
  "projection": [
    {"week": 0, "prob": <MUST EQUAL current churn_score exactly>},
    {"week": W, "prob": <float 0-100>},
    ...
  ],
  "intervention_impact_pct": <float: percentage-point reduction at final week vs baseline_final>,
  "confidence": <float 0.0-1.0>,
  "retention_window_weeks": <int: weeks until projection prob exceeds 80; use horizon+1 if never>,
  "revenue_at_risk": <float: monthly_revenue × (projection_midpoint_prob/100) × 3>,
  "segment_migration": [
    {"label": "...", "prob": <float 0.0-1.0>},
    ...
  ]
}

Critical rules:
- projection[0].prob MUST equal the customer's current churn_score exactly.
- projection values must generally be LOWER than baseline (intervention reduces churn).
- intervention_impact_pct = baseline_final_prob − projection_final_prob (positive number).
- segment_migration probs must sum to 1.0; use the same labels as provided.
- Match the same weekly time-points as the baseline provided.
- Output ONLY the JSON object. No explanation, no markdown fences.
"""


def _extract_json(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$",          "", cleaned, flags=re.MULTILINE).strip()
    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(cleaned[start:end])
    raise ValueError("no JSON object found")


def _clean_narrative(raw: str) -> str:
    if not raw:
        return raw
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned:
        return _strip_markdown(cleaned)
    result = re.sub(r"</?think>", "", raw, flags=re.IGNORECASE).strip()
    return _strip_markdown(result)


# /simulate endpoint
class SimulateRequest(BaseModel):
    customer_data:          _CustomerDataSim
    compare_customer_data:  _CustomerDataSim | None = None
    scenario:               str        = ""
    chat_history:           list[dict] = []
    horizon_weeks:          int        = 12
    segment_labels:         list[str]  = []
    mode:                   str        = "initial"  # "initial" | "scenario" | "ask"


@app.post("/simulate")
async def simulate(request: SimulateRequest):
    """
    Churn trajectory simulation SSE — three modes:

      initial  → LLM generates trajectory JSON + streams narrative (fast, no agents)
      analyze  → 4 agents analyze baseline + extract recommendations + brief narrative
      scenario → 4 agents debate intervention + synthesise projection JSON + narrative
    """
    c         = request.customer_data
    cc        = request.compare_customer_data  # may be None
    ctx       = _build_ctx(c, "")
    ctx_b     = _build_ctx(cc, "") if cc else ""
    horizon   = request.horizon_weeks
    seg_lbls  = request.segment_labels or []

    # Normalise mode (backwards compat: non-empty scenario → scenario mode)
    mode = request.mode
    if request.scenario.strip() and mode not in ("scenario", "ask"):
        mode = "scenario"
    if mode == "analyze":          # legacy compat
        mode = "initial"

    history_block = ""
    if request.chat_history:
        lines = [
            f"Q: {t.get('question', '')}\nA: {str(t.get('narrative', ''))[:250]}"
            for t in request.chat_history[-4:]
        ]
        history_block = "\n\nPREVIOUS TURNS:\n" + "\n\n".join(lines)

    # Shared helpers
    if horizon <= 8:    _step = 1
    elif horizon <= 16: _step = 2
    else:               _step = 4
    _fallback_points = list(range(0, horizon + 1, _step))
    if _fallback_points[-1] != horizon:
        _fallback_points.append(horizon)

    def _fallback_sim(exc_msg: str = "") -> dict:
        import math, random
        base_score = c.churn_score
        rng = random.Random(hash(c.customer_id) % (2**31))  # deterministic per customer
        baseline = []
        cur = base_score
        for i, w in enumerate(_fallback_points):
            if i == 0:
                baseline.append({"week": w, "prob": round(cur, 2)})
                continue
            if base_score >= 85:
                # High risk: slight oscillation but staying elevated
                delta = rng.uniform(-3, 5) - 1.5  # slight downward trend with noise
                cur = max(72, min(99, cur + delta))
            elif base_score >= 60:
                delta = rng.uniform(0.5, 2.5)
                cur = min(98, cur + delta)
            else:
                delta = rng.uniform(-0.5, 1.5)
                cur = max(5, min(80, cur + delta))
            baseline.append({"week": w, "prob": round(cur, 2)})
        mid_prob    = baseline[len(baseline) // 2]['prob'] if len(baseline) > 1 else base_score
        monthly_rev = c.segment_rfm_context.get("total_revenue", {}).get("customer", 0)
        if seg_lbls and len(seg_lbls) >= 2:
            n     = len(seg_lbls)
            probs = [
                round(base_score / 100 * (0.7 / max(n - 1, 1)) * i
                      + (1 - base_score / 100) * (0.3 / max(n - 1, 1)) * (n - 1 - i), 3)
                for i in range(n)
            ]
            tot = sum(probs) or 1
            seg_mig = [{"label": seg_lbls[i], "prob": round(probs[i] / tot, 3)} for i in range(n)]
        else:
            seg_mig = [
                {"label": "Churned",   "prob": round(base_score / 100 * 0.7,       3)},
                {"label": "High Risk", "prob": round(base_score / 100 * 0.2,       3)},
                {"label": "At Risk",   "prob": round((1 - base_score / 100) * 0.4, 3)},
                {"label": "Retained",  "prob": round((1 - base_score / 100) * 0.6, 3)},
            ]
        result = {
            "baseline":                baseline,
            "projection":              None,
            "retention_window_weeks":  next((pt['week'] for pt in baseline if pt['prob'] >= 80), 0),
            "revenue_at_risk":         round(monthly_rev * (mid_prob / 100) * 3, 2),
            "confidence":              0.6,
            "intervention_impact_pct": None,
            "segment_migration":       seg_mig,
        }
        if exc_msg:
            result["_error"] = exc_msg[:120]
        return result

    async def _run_agents(agent_ctx: str, agent_systems: dict[str, str]) -> list[dict]:
        """Stream 4 agents and collect their outputs. Yields SSE events."""
        outputs: list[dict] = []
        for persona in AGENT_PERSONAS:
            name   = persona["name"]
            system = agent_systems.get(name, "You are a customer success expert. Provide a brief analysis.")
            yield f"data: {json.dumps({'type': 'agent_start', 'agent': name, 'short': persona['short'], 'color': persona['color']})}\n\n"
            content = ""
            async for token in stream_llm_no_think(system, agent_ctx, max_tokens=400):
                content += token
                yield f"data: {json.dumps({'type': 'agent_token', 'agent': name, 'content': token})}\n\n"
            outputs.append({"name": name, "content": _strip_markdown(content)})
            yield f"data: {json.dumps({'type': 'agent_done', 'agent': name})}\n\n"
        # store outputs in a closure-accessible list
        _run_agents._last_outputs = outputs  # type: ignore[attr-defined]


    async def event_stream():
        # ══════════════════════════════════════════════════════════════════════
        # MODE: ask — Q&A about customer/results, no chart update
        # ══════════════════════════════════════════════════════════════════════
        if mode == "ask":
            question = request.scenario.strip() or "Provide a summary of this customer's situation."
            compare_note = ""
            if cc:
                compare_note = (
                    f"\n\nCOMPARE CUSTOMER B: {cc.customer_id} | Churn: {cc.churn_score:.1f}% | "
                    f"Risk: {cc.risk_level} | Segment: {cc.segment_label} | Plan: {cc.plan_type}"
                )
            ask_prompt = (
                f"{ctx}{compare_note}{history_block}"
                f"\n\nUSER QUESTION: {question}"
                + ("\n\nAnswer considering the context of BOTH customers where relevant." if cc else "")
            )
            full_answer = ""
            async for token in stream_llm_no_think(_ASK_SYSTEM, ask_prompt, max_tokens=650):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'narrative': _clean_narrative(full_answer)})}\n\n"
            return

        # ══════════════════════════════════════════════════════════════════════
        # MODE: initial — trajectory JSON → 4 agents → recommendations → narrative
        # ══════════════════════════════════════════════════════════════════════
        if mode == "initial":
            yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

            sim_data: dict = {}
            try:
                raw = await call_llm(
                    _build_sim_json_system(horizon, seg_lbls),
                    f"{ctx}{history_block}\n\nGenerate the churn trajectory JSON now.",
                    max_tokens=900,
                )
                sim_data = _extract_json(raw)
                if sim_data.get("baseline"):
                    sim_data["baseline"][0]["prob"] = round(c.churn_score, 2)
            except Exception as exc:
                sim_data = _fallback_sim(str(exc))

            # Natural trajectory delta (positive = improving / negative = worsening)
            baseline_pts = sim_data.get("baseline", [])
            if len(baseline_pts) >= 2:
                sim_data["intervention_impact_pct"] = round(
                    baseline_pts[0].get("prob", c.churn_score) -
                    baseline_pts[-1].get("prob", c.churn_score), 2
                )

            yield f"data: {json.dumps({'type': 'data', 'payload': sim_data})}\n\n"

            # Run 4 agents after trajectory is sent
            last_pt   = baseline_pts[-1] if baseline_pts else {}
            agent_ctx = (
                f"{ctx}{history_block}"
                f"\n\nFORECAST ({horizon} weeks): churn {c.churn_score:.1f}% → "
                f"{last_pt.get('prob', c.churn_score):.1f}% at week {horizon}. "
                f"Revenue at risk: ${sim_data.get('revenue_at_risk', 0):,.0f}."
                f"\n\nProvide your analysis."
            )
            async for evt in _run_agents(agent_ctx, AGENT_ANALYZE_SYSTEMS):
                yield evt
            agent_outputs = getattr(_run_agents, "_last_outputs", [])

            # Extract recommendations for primary customer
            recs = await _extract_recommendations(
                ctx, agent_outputs, scenario="",
                risk_level=c.risk_level,
                customer_profile={
                    "segment_label":  c.segment_label,
                    "plan_type":      c.plan_type,
                    "contract_type":  c.contract_type,
                    "churn_score":    c.churn_score,
                    "shap_top5":      c.shap_top5,
                },
            )
            # Extract recommendations for compare customer if present
            compare_recs: list[str] = []
            if cc:
                compare_recs = await _extract_recommendations(
                    ctx_b, agent_outputs, scenario="",
                    risk_level=cc.risk_level,
                    customer_profile={
                        "segment_label": cc.segment_label,
                        "plan_type":     cc.plan_type,
                        "contract_type": cc.contract_type,
                        "churn_score":   cc.churn_score,
                        "shap_top5":     cc.shap_top5,
                    },
                )
            if recs or compare_recs:
                evt_payload: dict = {"type": "agent_recommendations", "recommendations": recs}
                if compare_recs:
                    evt_payload["compare_recommendations"] = compare_recs
                yield f"data: {json.dumps(evt_payload)}\n\n"

            # Stream narrative summary
            debate_text = "\n".join(f"[{o['name']}]: {o['content'][:300]}" for o in agent_outputs)
            compare_ctx_note = ""
            if cc:
                compare_ctx_note = (
                    f"\n\nCOMPARE CUSTOMER B: {cc.customer_id} | Churn: {cc.churn_score:.1f}% | Risk: {cc.risk_level} | Segment: {cc.segment_label}"
                )
            narrative_prompt = (
                f"{ctx}{compare_ctx_note}{history_block}"
                f"\n\nChurn trajectory ({horizon} weeks): "
                f"week 0={c.churn_score:.1f}%, "
                f"week {last_pt.get('week', horizon)}={last_pt.get('prob', c.churn_score):.1f}%, "
                f"retention_window={sim_data.get('retention_window_weeks', '?')} weeks, "
                f"revenue_at_risk=${sim_data.get('revenue_at_risk', 0):,.0f}."
                + (f" | Compare B churn: {cc.churn_score:.1f}% Risk: {cc.risk_level}" if cc else "")
                + f"\n\nTEAM ANALYSIS SUMMARY:\n{debate_text[:600]}"
            )
            full_narrative = ""
            async for token in stream_llm(_SIM_NARRATIVE_SYSTEM, narrative_prompt, max_tokens=700):
                full_narrative += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'narrative': _clean_narrative(full_narrative)})}\n\n"
            return

        # ══════════════════════════════════════════════════════════════════════
        # MODE: scenario — 4 agents debate → projection JSON → narrative
        # ══════════════════════════════════════════════════════════════════════
        scenario_block = f"\n\nINTERVENTION SCENARIO: {request.scenario}"
        compare_note_s = ""
        if cc:
            compare_note_s = (
                f"\n\nCOMPARE CUSTOMER B: {cc.customer_id} | Churn: {cc.churn_score:.1f}% | "
                f"Risk: {cc.risk_level} | Segment: {cc.segment_label}"
            )
        agent_ctx = f"{ctx}{compare_note_s}{scenario_block}{history_block}\n\nProvide your analysis."

        agent_outputs_s: list[dict] = []
        async for evt in _run_agents(agent_ctx, AGENT_SCENARIO_SYSTEMS):
            yield evt
        agent_outputs_s = getattr(_run_agents, "_last_outputs", [])

        # Extract follow-up recommendations for primary customer
        recs = await _extract_recommendations(
            ctx, agent_outputs_s, scenario=request.scenario,
            risk_level=c.risk_level,
            customer_profile={
                "segment_label":  c.segment_label,
                "plan_type":      c.plan_type,
                "contract_type":  c.contract_type,
                "churn_score":    c.churn_score,
                "shap_top5":      c.shap_top5,
            },
        )
        compare_recs_s: list[str] = []
        if cc:
            compare_recs_s = await _extract_recommendations(
                ctx_b, agent_outputs_s, scenario=request.scenario,
                risk_level=cc.risk_level,
                customer_profile={
                    "segment_label": cc.segment_label,
                    "plan_type":     cc.plan_type,
                    "contract_type": cc.contract_type,
                    "churn_score":   cc.churn_score,
                    "shap_top5":     cc.shap_top5,
                },
            )
        if recs or compare_recs_s:
            evt_payload_s: dict = {"type": "agent_recommendations", "recommendations": recs}
            if compare_recs_s:
                evt_payload_s["compare_recommendations"] = compare_recs_s
            yield f"data: {json.dumps(evt_payload_s)}\n\n"

        # Synthesise projection JSON
        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        debate_text = "\n\n".join(f"[{o['name']}]: {o['content']}" for o in agent_outputs_s)
        monthly_rev = c.segment_rfm_context.get("total_revenue", {}).get("customer", 0)
        seg_note    = (f"Use segment labels: {seg_lbls}" if seg_lbls
                       else "Use standard labels: Churned, High Risk, At Risk, Retained")

        synth_user = (
            f"{ctx}{scenario_block}"
            f"\n\nBASELINE TIME POINTS (weeks): {_fallback_points}"
            f"\nMONTHLY REVENUE: ${monthly_rev:,.0f}"
            f"\n{seg_note}"
            f"\n\nMULTI-AGENT DEBATE:\n{debate_text}"
            f"\n\nGenerate the scenario update JSON now."
        )

        update_data: dict = {}
        try:
            raw = await call_llm(_SCENARIO_UPDATE_JSON_SYSTEM, synth_user, max_tokens=800)
            update_data = _extract_json(raw)
            if update_data.get("projection"):
                update_data["projection"][0]["prob"] = round(c.churn_score, 2)
        except Exception as exc:
            update_data = {
                "projection": [
                    {"week": w, "prob": round(min(100, c.churn_score * (0.70 ** (1 + i * 0.15))), 2)}
                    for i, w in enumerate(_fallback_points)
                ],
                "intervention_impact_pct": 30.0,
                "confidence":              0.55,
                "retention_window_weeks":  min(horizon + 1, 8),
                "revenue_at_risk":         round(monthly_rev * (c.churn_score * 0.65 / 100) * 3, 2),
                "_error": str(exc)[:120],
            }
            update_data["projection"][0]["prob"] = round(c.churn_score, 2)

        yield f"data: {json.dumps({'type': 'data', 'payload': update_data})}\n\n"

        proj_last = (update_data.get("projection") or [{}])[-1]
        compare_proj_note = ""
        if cc:
            compare_proj_note = f"\n\nCOMPARE B: {cc.customer_id} churn {cc.churn_score:.1f}% ({cc.risk_level}) — scenario applies to both customers."
        narrative_prompt = (
            f"{ctx}{scenario_block}{compare_proj_note}"
            f"\n\nDEBATE SUMMARY:\n{debate_text[:800]}"
            f"\n\nProjection result: "
            f"churn week {proj_last.get('week', horizon)}={proj_last.get('prob', c.churn_score):.1f}%, "
            f"intervention_impact={update_data.get('intervention_impact_pct', 0):.0f}pp reduction, "
            f"revenue_at_risk=${update_data.get('revenue_at_risk', 0):,.0f}."
        )
        full_narrative = ""
        async for token in stream_llm(_SCENARIO_NARRATIVE_SYSTEM, narrative_prompt, max_tokens=700):
            full_narrative += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'narrative': _clean_narrative(full_narrative)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )