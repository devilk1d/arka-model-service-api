import os, sys, json, io, warnings, re, asyncio
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import httpx
from scipy.sparse import hstack, csr_matrix

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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

# ─── Load artifacts ───────────────────────────────────────────────────────────
print("Loading artifacts...")

A   = joblib.load(os.getenv("ARTIFACTS_PATH",     "arka_model_artifacts_v6.pkl"))
NLP = joblib.load(os.getenv("NLP_ARTIFACTS_PATH", "arka_nlp_artifacts_v2.pkl"))

MODEL        = A.get("calibrated_model") or A["model"]
FEATURES     = A["production_features"]
LE_PLAN      = A["le_plan"]
LE_CONTRACT  = A["le_contract"]
SCALER_SEG   = A["scaler_seg"]
KMEANS       = A["kmeans"]
LABEL_MAP    = A["cluster_label_map"]
SEG_FEATURES = A["seg_features"]
SEG_PROFILES = A["seg_profiles"]
SEG_ACTIONS  = A["seg_actions"]
RISK_LOW     = A["risk_thresholds"]["low"]
RISK_HIGH    = A["risk_thresholds"]["high"]
REF          = pd.Timestamp(A["reference_date"])

CV_VEC      = NLP["cv_vec"]
LDA         = NLP["lda"]
TOPIC_NAMES = NLP["topic_names"]
URGENCY_LEX = NLP["urgency_lexicon"]
N_TOPICS    = NLP["n_topics"]

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

EXPLAINER = A.get("explainer")
if EXPLAINER is None:
    _base = MODEL.calibrated_classifiers_[0].estimator if hasattr(MODEL, "calibrated_classifiers_") else MODEL
    EXPLAINER = shap.TreeExplainer(_base)

ANALYZER = SentimentIntensityAnalyzer()

FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "1.0"))

# ─── LLM providers: Groq primary, Ollama fallback ──────────────────────────────
def _clean_base_url(raw: str) -> str:
    if not raw.startswith("http"):
        raw = f"https://{raw}"
    return raw.rstrip("/")

# Primary: Groq (OpenAI-compatible, very high tokens/sec)
GROQ_URL   = _clean_base_url(os.getenv("GROQ_URL", "https://api.groq.com/openai/v1"))
GROQ_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Fallback: Ollama Cloud (existing setup)
OLLAMA_URL   = _clean_base_url(os.getenv("OLLAMA_URL", "https://ollama.com/v1"))
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")

def _build_providers() -> list[dict]:
    """Ordered provider list. Groq first when a key is set, Ollama as fallback."""
    provs: list[dict] = []
    if GROQ_KEY:
        provs.append({
            "name": "groq", "base": GROQ_URL, "key": GROQ_KEY, "model": GROQ_MODEL,
            "native_ollama": False,
            "extra": {},
        })
    provs.append({
        "name": "ollama", "base": OLLAMA_URL, "key": OLLAMA_KEY, "model": OLLAMA_MODEL,
        "native_ollama": ("localhost" in OLLAMA_URL or "127.0.0.1" in OLLAMA_URL
                          or OLLAMA_URL.endswith("/api")),
        "extra": {},
    })
    return provs


PROVIDERS = _build_providers()


def _provider_headers(p: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if p.get("key"):
        headers["Authorization"] = f"Bearer {p['key']}"
    return headers

print("✅ Artifacts loaded (v6 — GradientBoosting, 38 features, no calibration)")
print(f"   model_version : {A.get('model_version', 'unknown')}")
print(f"   model_name    : {A.get('best_model_name', 'unknown')}")
print(f"   test_auc      : {A.get('test_auc', 'unknown')}")
print(f"   SEG_FEATURES  : {SEG_FEATURES}")
print(f"   FEATURES count: {len(FEATURES)}")
print(f"   RISK_LOW/HIGH : {RISK_LOW} / {RISK_HIGH}")
print(f"   N_TOPICS      : {N_TOPICS}")
print(f"✅ NLP Artifacts loaded (sentiment: TF-IDF + LightGBM, 4-tier)")
print(f"   nlp_auc_cv    : {NLP.get('nlp_auc_cv', 'unknown')}")
print("✅ LLM providers : " + " → ".join(f"{p['name']}({p['model']})" for p in PROVIDERS))

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_level(score: float) -> str:
    return "Low" if score <= RISK_LOW else ("Medium" if score <= RISK_HIGH else "High")


def sanitize_floats(obj):
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    return obj


def _parse_feedback(text: str) -> list:
    """Split aggregated feedback into a list of individual items."""
    if not text or text.strip() in ("", "nan"):
        return []
    return [p.strip() for p in text.split(" | ") if p.strip() and p.strip() != "nan"]


def _strip_markdown(text: str) -> str:
    """Remove markdown bold/italic asterisks from prose text.
    Never apply this to JSON strings — it will corrupt them.
    """
    if not text:
        return text
    return re.sub(r'\*{1,3}', '', text).strip()


_FEATURE_LABEL_MAP = {
    "vader_compound":            "Overall Sentiment Tone",
    "vader_neg":                 "Negative Feedback Intensity",
    "vader_pos":                 "Positive Feedback Intensity",
    "vader_neu":                 "Neutral Feedback Level",
    "vader_min_sent":            "Lowest Sentiment Moment",
    "vader_std_sent":            "Sentiment Consistency",
    "pct_neg_sent":              "Negative Feedback Percentage",
    "pct_negative_sent":         "Negative Feedback Percentage",
    "urgency_score":             "Urgency Level in Feedback",
    "avg_dissatisfaction_proba": "Customer Dissatisfaction Score",
    "dissatisfaction_proba":     "Customer Dissatisfaction Score",
    "ling_churn_intent":         "Churn Intent Signals in Feedback",
    "ling_contrast":             "Mixed or Contradictory Feedback",
    "ling_word_count":           "Feedback Engagement Volume",
    "log_total_revenue":         "Total Revenue",
    "log_monthly_usage_hrs":     "Monthly Usage Hours",
    "log_total_tickets":         "Total Support Tickets",
    "log_total_users":           "Total Users",
    "dunning_per_tenure":        "Late Payment Rate Over Time",
    "usage_per_user":            "Usage Hours Per User",
    "ticket_per_revenue":        "Support Ticket Burden per Revenue",
    "adoption_x_usage":          "Feature Adoption and Usage Combined",
    "nps_x_dunning":             "Satisfaction vs. Payment Risk",
    "revenue_per_month":         "Monthly Revenue",
    "payments_per_month":        "Payment Frequency",
    # Encoded categorical & raw features missing from original map
    "contract_enc":              "Contract Type",
    "plan_enc":                  "Subscription Plan",
    "tenure_days":               "Account Tenure",
    "days_since_last_payment":   "Days Since Last Payment",
    "days_since_login":          "Days Since Last Login",
    "monthly_usage_hrs":         "Monthly Usage Hours",
    "feature_adoption_pct":      "Feature Adoption Rate",
    "avg_nps_score":             "Average NPS Score",
    "total_users":               "Total Users",
    "total_tickets":             "Total Support Tickets",
    "dunning_count":             "Late / Missed Payments",
    "avg_payment_delay":         "Average Payment Delay",
}


def _resolve_feature_label(k: str) -> str:
    if k in _FEATURE_LABEL_MAP:
        return _FEATURE_LABEL_MAP[k]
    # topic_N → actual topic name from NLP model artifacts
    if k.startswith("topic_"):
        try:
            ti = int(k.split("_")[1])
            if ti < len(TOPIC_NAMES) and TOPIC_NAMES[ti]:
                return f"Feedback Theme: {TOPIC_NAMES[ti]}"
        except (ValueError, IndexError):
            pass
        return "Feedback Topic Signal"
    return k.replace("_", " ").title()


def get_top_shap(shap_row: pd.Series, top_n: int = 5) -> list:
    top = shap_row.abs().nlargest(top_n)
    return [
        {
            "feature":       k,
            "impact_score":  round(float(shap_row[k]), 4),
            "direction":     "raises_risk" if shap_row[k] > 0 else "lowers_risk",
            "importance":    round(abs(float(shap_row[k])), 4),
            "feature_label": _resolve_feature_label(k),
        }
        for k in top.index
    ]


# ─── NLP / Sentiment pipeline ─────────────────────────────────────────────────
def extract_sent_features(text: str) -> list:
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
        return "Slightly Dissatisfied"
    elif proba < 0.80:
        return "Dissatisfied"
    else:
        return "Highly Dissatisfied"


def predict_sentiment(text: str) -> dict:
    ling_feats = np.array([extract_sent_features(text)])
    X_tfidf    = TFIDF_SENT.transform([str(text)])
    X_extra    = csr_matrix(SCALER_SENT.transform(ling_feats))
    X_full     = hstack([X_tfidf, X_extra])
    proba      = float(SENT_LGBM.predict_proba(X_full)[0, 1])
    return {"label": map_sentiment_tier(proba), "dissatisfaction_score": proba}


def compute_vader_features(text: str) -> dict:
    # Empty template — includes both old names (display) and v6 model feature names
    empty = {k: 0.0 for k in [
        "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "vader_min_sent", "vader_std_sent",
        "pct_negative_sent",      # display alias
        "pct_neg_sent",           # v6 model feature name
        "urgency_score", "avg_words_per_sent",
        "dissatisfaction_proba",  # display alias
        "avg_dissatisfaction_proba",  # v6 model feature name
        "ling_churn_intent",      # v6: churn intent keyword count
        "ling_contrast",          # v6: contrast pattern count
        "ling_word_count",        # v6: feedback word count
    ]}
    if not text or pd.isna(text):
        return empty
    t   = str(text)
    tl  = t.lower()
    doc = ANALYZER.polarity_scores(t)
    sents = [s.strip() for s in re.split(r"[.!?|]+", t) if len(s.strip()) > 10]
    sc    = [ANALYZER.polarity_scores(s)["compound"] for s in sents] if sents else [0.0]
    sent_result = predict_sentiment(t)

    pct_neg      = sum(1 for s in sc if s < -0.05) / len(sc)
    dis_proba    = sent_result["dissatisfaction_score"]
    churn_intent = float(sum(1 for kw in CHURN_INTENT_LEXICON if kw in tl))
    contrast_cnt = float(len(re.findall(CONTRAST_PATTERN, tl)))
    word_count   = float(len(t.split()))

    return {
        "vader_compound":        doc["compound"],
        "vader_pos":             doc["pos"],
        "vader_neg":             doc["neg"],
        "vader_neu":             doc["neu"],
        "vader_min_sent":        min(sc),
        "vader_std_sent":        float(np.std(sc)),
        # Both names — old for display, v6 name for model
        "pct_negative_sent":         pct_neg,
        "pct_neg_sent":              pct_neg,
        "urgency_score":         float(sum(1 for w in URGENCY_LEX if w in tl)),
        "avg_words_per_sent":    float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
        # Both names — old for display, v6 name for model
        "dissatisfaction_proba":     dis_proba,
        "avg_dissatisfaction_proba": dis_proba,
        # v6 NLP features
        "ling_churn_intent": churn_intent,
        "ling_contrast":     contrast_cnt,
        "ling_word_count":   word_count,
    }


def compute_topic_features(texts: list[str]) -> dict:
    X_counts  = CV_VEC.transform(texts)
    X_topics  = LDA.transform(X_counts)       # shape (n, N_TOPICS)
    dom_idx   = X_topics.argmax(axis=1)
    dom_score = X_topics.max(axis=1)
    dom_label = [TOPIC_NAMES[i] for i in dom_idx]

    result = {
        "dominant_topic":       dom_idx,
        "dominant_topic_label": dom_label,
        "dominant_topic_score": dom_score,
        "topic_distribution":   X_topics,
    }
    # v6 model features: individual topic probability columns (topic_0 .. topic_{N-1})
    for ti in range(X_topics.shape[1]):
        result[f"topic_{ti}"] = X_topics[:, ti]
    return result


# ─── Core pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df):
    ca_df  = ca_df.copy()
    bd_df  = bd_df.copy()
    um_df  = um_df.copy()
    st_df  = st_df.copy()
    nps_df = nps_df.copy()

    ca_df["plan_type"]     = ca_df["plan_type"].str.capitalize().str.strip()
    ca_df["contract_type"] = ca_df["contract_type"].str.strip()
    nps_df["nps_score"]    = nps_df["nps_score"].clip(lower=0, upper=10)

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

    ca_df["obs_date"]    = ca_df["unsubscribed_date"].fillna(REF)
    ca_df["_tenure_raw"] = (ca_df["obs_date"] - ca_df["subscription_date"]).dt.days
    ca_df = ca_df[ca_df["_tenure_raw"] >= 0].drop(columns="_tenure_raw").reset_index(drop=True)
    bd_df = bd_df.drop_duplicates().reset_index(drop=True)

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

    payments = bd_df[bd_df["record_type"] == "payment"].copy()
    payments["delay_days"] = (payments["payment_date"] - payments["billing_date"]).dt.days
    bf = payments.groupby("customer_id").agg(
        total_revenue     =("payment_value", "sum"),
        avg_payment_value =("payment_value", "mean"),
        payment_count     =("payment_value", "count"),
        avg_payment_delay =("delay_days",    "mean"),
        max_payment_delay =("delay_days",    "max"),
    ).reset_index()
    # ── days_since_last_payment: SHAP #1 in v6 (recency of last payment) ──────
    last_pay = payments.groupby("customer_id")["payment_date"].max().reset_index()
    last_pay["days_since_last_payment"] = (REF - last_pay["payment_date"]).dt.days.clip(lower=0)
    bf = bf.merge(last_pay[["customer_id", "days_since_last_payment"]], on="customer_id", how="left")
    dun = (bd_df[bd_df["record_type"] == "dunning"]
           .groupby("customer_id").size()
           .reset_index(name="dunning_count"))
    bf  = bf.merge(dun, on="customer_id", how="left")
    bf["dunning_count"] = bf["dunning_count"].fillna(0)

    uf = um_df.copy()
    uf = uf.merge(ca_df[["customer_id", "obs_date"]], on="customer_id", how="left")
    uf["days_since_login"] = (REF - uf["last_login_date"]).dt.days.clip(lower=0)
    uf = uf.drop(columns=["last_login_date", "obs_date"], errors="ignore")

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

    nf = nps_df.groupby("customer_id").agg(
        avg_nps_score =("nps_score", "mean"),
        min_nps_score =("nps_score", "min"),
        survey_count  =("survey_id", "count"),
        pct_detractor =("segment",   lambda x: (x == "detractor").mean()),
    ).reset_index()
    nf["has_nps_data"] = 1

    text_per = (nps_df.groupby("customer_id")["feedback_text"]
                .apply(lambda x: " | ".join(x.dropna().astype(str)))
                .reset_index())
    text_per.columns = ["customer_id", "all_feedback"]

    master = ca_df[["customer_id", "plan_type", "contract_type", "total_users",
                     "subscription_date", "obs_date"]].copy()
    master["tenure_days"] = (master["obs_date"] - master["subscription_date"]).dt.days.clip(lower=1)

    master = (master
              .merge(uf, on="customer_id", how="left")
              .merge(bf, on="customer_id", how="left")
              .merge(tf, on="customer_id", how="left")
              .merge(nf, on="customer_id", how="left"))

    fill_zero = ["total_tickets", "open_tickets", "billing_tickets", "technical_tickets",
                 "critical_tickets", "high_tickets", "unresolved_ratio", "critical_ratio",
                 "dunning_count", "avg_payment_delay", "max_payment_delay"]
    master[fill_zero]      = master[fill_zero].fillna(0)
    master["has_nps_data"] = master["has_nps_data"].fillna(0)

    master["log_total_revenue"]     = np.log1p(master["total_revenue"])
    master["log_monthly_usage_hrs"] = np.log1p(master["monthly_usage_hrs"])
    master["log_total_tickets"]     = np.log1p(master["total_tickets"])
    master["log_total_users"]       = np.log1p(master["total_users"])

    # ── Rate features (v6) ─────────────────────────────────────────────────────
    tenure_m = (master["tenure_days"] / 30).clip(lower=1)
    master["revenue_per_month"]  = master["total_revenue"]  / tenure_m
    master["payments_per_month"] = master["payment_count"]  / tenure_m

    master["dunning_per_tenure"] = (
        master["dunning_count"] / (master["tenure_days"] / 30).replace(0, 1)
    )
    master["usage_per_user"] = (
        master["monthly_usage_hrs"] / master["total_users"].replace(0, 1)
    )
    master["ticket_per_revenue"] = (
        master["total_tickets"] / (master["total_revenue"].replace(0, 1) / 1000)
    )
    master["adoption_x_usage"] = (
        master["feature_adoption_pct"] * master["log_monthly_usage_hrs"]
    )
    master["nps_x_dunning"] = (
        master["avg_nps_score"].fillna(5) * (master["dunning_count"] + 1)
    )

    master["plan_enc"]     = LE_PLAN.transform(master["plan_type"])
    master["contract_enc"] = LE_CONTRACT.transform(master["contract_type"])

    DEFAULT_IMPUTE = {
        "avg_nps_score":            7.0,
        "min_nps_score":            7.0,
        "survey_count":             0.0,
        "pct_detractor":            0.0,
        "total_revenue":            0.0,
        "payment_count":            0.0,
        "monthly_usage_hrs":        0.0,
        "feature_adoption_pct":     0.0,
        "days_since_login":         0.0,
        "tenure_days":              30.0,
        "total_tickets":            0.0,
        "open_tickets":             0.0,
        "billing_tickets":          0.0,
        "technical_tickets":        0.0,
        "critical_tickets":         0.0,
        "high_tickets":             0.0,
        "unresolved_ratio":         0.0,
        "critical_ratio":           0.0,
        "dunning_count":            0.0,
        "avg_payment_delay":        0.0,
        "max_payment_delay":        0.0,
        "total_users":              1.0,
        "days_since_last_payment":  0.0,   # v6: filled with median below
        "revenue_per_month":        0.0,   # v6
        "payments_per_month":       0.0,   # v6
    }

    for col in ["avg_nps_score", "min_nps_score", "survey_count", "pct_detractor"]:
        med = master[col].median()
        if pd.isna(med):
            med = DEFAULT_IMPUTE.get(col, 7.0)
        master[col] = master[col].fillna(med)

    # ── v6: days_since_last_payment filled with median (per notebook cell 50) ──
    for col in ["days_since_last_payment"]:
        med = master[col].median() if col in master.columns else float("nan")
        if pd.isna(med):
            med = DEFAULT_IMPUTE.get(col, 0.0)
        master[col] = master[col].fillna(med)

    seg_data = master[SEG_FEATURES].copy()
    for c in SEG_FEATURES:
        med = seg_data[c].median()
        if pd.isna(med):
            med = DEFAULT_IMPUTE.get(c, 0.0)
        seg_data[c] = seg_data[c].fillna(med)
    X_seg = SCALER_SEG.transform(seg_data.values)
    master["segment_cluster"] = KMEANS.predict(X_seg)
    master["segment_label"]   = master["segment_cluster"].map(LABEL_MAP)

    master = master.merge(text_per, on="customer_id", how="left")
    master["all_feedback"] = master["all_feedback"].fillna("")

    vader_rows = master["all_feedback"].apply(compute_vader_features)
    vader_df   = pd.DataFrame(list(vader_rows))
    for col in vader_df.columns:
        master[col] = vader_df[col].values

    master["urgency_level"]   = master["urgency_score"].apply(
        lambda u: "high" if u >= 3 else ("medium" if u >= 1 else "low")
    )
    master["sentiment_label"] = master["dissatisfaction_proba"].apply(map_sentiment_tier)

    topic_feats = compute_topic_features(master["all_feedback"].tolist())
    master["dominant_topic_label"] = topic_feats["dominant_topic_label"]
    master["dominant_topic_score"] = topic_feats["dominant_topic_score"]
    # v6 model features: store individual topic probabilities
    for ti in range(N_TOPICS):
        master[f"topic_{ti}"] = topic_feats.get(f"topic_{ti}", 0.0)

    X_tab     = master[FEATURES].fillna(0).values
    tab_proba = MODEL.predict_proba(X_tab)[:, 1]

    shap_vals = EXPLAINER.shap_values(master[FEATURES].fillna(0))
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=FEATURES)

    churn_score = (tab_proba * 100).round(1)
    master["churn_proba"] = tab_proba.round(4)
    master["churn_score"] = churn_score
    master["risk_level"]  = [risk_level(s) for s in churn_score]

    centroids_raw = SCALER_SEG.inverse_transform(KMEANS.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_raw, columns=SEG_FEATURES)

    results = []
    for i in range(len(master)):
        row     = master.iloc[i]
        seg     = row["segment_label"]
        seg_cl  = int(row["segment_cluster"])
        seg_prof = next((p for p in SEG_PROFILES if p["segment_label"] == seg), {})
        centroid = centroid_df.iloc[seg_cl].to_dict()

        seg_action_data = SEG_ACTIONS.get(seg, {})
        seg_act = {
            "description": seg_action_data.get("description", ""),
            "retain":      seg_action_data.get("retain", []),
            "offer":       seg_action_data.get("offer",  []),
            "priority":    seg_action_data.get("priority", row["risk_level"]),
        }

        seg_rfm_context = {
            "days_since_login":     {"customer": round(float(row.get("days_since_login", 0) if pd.notna(row.get("days_since_login")) else 0), 1),
                                     "segment_avg": 0.0},
            "payment_count":        {"customer": round(float(row.get("payment_count", 0) if pd.notna(row.get("payment_count")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("payment_count", 0)), 1)},
            "total_revenue":        {"customer": round(float(row.get("total_revenue", 0) if pd.notna(row.get("total_revenue")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("total_revenue", 0)), 1)},
            "monthly_usage_hrs":    {"customer": round(float(row.get("monthly_usage_hrs", 0) if pd.notna(row.get("monthly_usage_hrs")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("monthly_usage_hrs", 0)), 1)},
            "feature_adoption_pct": {"customer": round(float(row.get("feature_adoption_pct", 0) if pd.notna(row.get("feature_adoption_pct")) else 0), 1),
                                     "segment_avg": round(float(centroid.get("feature_adoption_pct", 0)), 1)},
            "avg_nps_score":        {"customer": round(float(row.get("avg_nps_score", 0) if pd.notna(row.get("avg_nps_score")) else 0), 2),
                                     "segment_avg": round(float(centroid.get("avg_nps_score", 0)), 2)},
        }

        results.append({
            "customer_id":          row["customer_id"],
            "plan_type":            row["plan_type"],
            "contract_type":        row["contract_type"],
            "churn_score":          float(row["churn_score"]),
            "churn_proba":          round(float(tab_proba[i]), 4),
            "tabular_proba":        round(float(tab_proba[i]), 4),
            "nlp_proba":            round(float(tab_proba[i]), 4),
            "risk_level":           row["risk_level"],
            "shap_top5":            get_top_shap(shap_df.iloc[i]),
            "sentiment": {
                "label":                 row["sentiment_label"],
                "dissatisfaction_score": round(float(row["dissatisfaction_proba"]), 4),
                "tone_score":            round(float(row["vader_compound"]), 4),
                "negative_feedback_pct": round(float(row["pct_negative_sent"]) * 100, 1),
                "urgency_level":         row["urgency_level"],
                "urgency_score":         int(row["urgency_score"]),
                "dominant_topic":        row["dominant_topic_label"],
                "topic_strength":        round(float(row["dominant_topic_score"]), 3),
                "feedback_texts":        _parse_feedback(str(row["all_feedback"])),
            },
            "has_nps_data":         int(row["has_nps_data"]),
            "segment_label":        seg,
            "segment_cluster":      seg_cl,
            "segment_rfm_context":  seg_rfm_context,
            "segment_profile":      seg_prof,
            "segment_actions":      seg_act,
        })

    return results


# ─── LLM helpers ──────────────────────────────────────────────────────────────
# Each call tries providers in order (Groq → Ollama). Groq is the default; if a
# request to it fails (HTTP error, timeout, connection issue) we fall back to
# Ollama transparently so output is never lost.

async def call_llm(system: str, user_msg: str, max_tokens: int = 1200) -> str:
    """Non-streaming LLM call; returns the full response text. Groq → Ollama fallback."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    last_err = None
    for p in PROVIDERS:
        try:
            payload = {
                "model":       p["model"],
                "messages":    messages,
                "stream":      False,
                "temperature": 0.4,
                "max_tokens":  max_tokens,
                **p.get("extra", {}),
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{p['base']}/chat/completions",
                                         headers=_provider_headers(p), json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all LLM providers failed: {type(last_err).__name__}: {str(last_err)[:120]}")


async def _call_llm_xai(prompt: str, max_tokens: int = 1400) -> str:
    """XAI/JSON mode LLM call. Returns raw JSON string — do NOT strip markdown here.
    Tries each provider (Groq → Ollama) until one returns content."""
    last_err = None
    for p in PROVIDERS:
        try:
            headers = _provider_headers(p)
            if p["native_ollama"]:
                endpoint = p["base"].replace("/v1", "").rstrip("/") + "/api/chat"
                payload  = {
                    "model":    p["model"],
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "format":   "json",
                    "options":  {"temperature": 0.2, "num_predict": max_tokens},
                }
            else:
                endpoint = f"{p['base']}/chat/completions"
                payload  = {
                    "model":       p["model"],
                    "messages":    [{"role": "user", "content": prompt}],
                    "max_tokens":  max_tokens,
                    "temperature": 0.2,
                    "response_format": {"type": "json_object"},
                    **p.get("extra", {}),
                }

            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()

            if "choices" in data:
                content = data["choices"][0]["message"]["content"]
            elif "message" in data:
                content = data["message"]["content"]
            else:
                last_err = f"unexpected response shape: {list(data.keys())}"
                continue

            # Strip think blocks only — do NOT apply _strip_markdown (would corrupt JSON keys)
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            content = re.sub(r"^```(?:json)?\s*", "", content).rstrip("```").strip()
            return content

        except Exception as e:
            last_err = e
            continue

    return json.dumps({"error": f"all LLM providers failed: {str(last_err)[:100]}"})


async def stream_llm(system: str, user_msg: str, max_tokens: int = 600):
    """Stream tokens via OpenAI-compatible SSE. Falls back to the next provider
    only if the current one fails BEFORE emitting any token (mid-stream failures
    keep whatever was already streamed)."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_msg}]
    last_err = None
    for p in PROVIDERS:
        payload = {
            "model":       p["model"],
            "messages":    messages,
            "stream":      True,
            "temperature": 0.7,
            "max_tokens":  max_tokens,
            **p.get("extra", {}),
        }
        started = False
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", f"{p['base']}/chat/completions",
                                         headers=_provider_headers(p), json=payload) as resp:
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
                                started = True
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass
            return  # provider finished cleanly
        except Exception as exc:
            last_err = exc
            if started:
                # Already streamed partial output — cannot safely restart elsewhere.
                yield f"[Error: {str(exc)[:80]}]"
                return
            continue  # nothing emitted yet → try next provider
    yield f"[Error: {str(last_err)[:80]}]"


async def stream_llm_no_think(system: str, user_msg: str, max_tokens: int = 600):
    """Stream tokens, filtering <think>...</think> blocks in real-time."""
    OPEN  = "<think>"
    CLOSE = "</think>"
    buf      = ""
    in_think = False

    async for tok in stream_llm(system, user_msg, max_tokens):
        buf += tok
        while True:
            if in_think:
                pos = buf.find(CLOSE)
                if pos >= 0:
                    buf      = buf[pos + len(CLOSE):]
                    in_think = False
                else:
                    keep = len(CLOSE) - 1
                    buf  = buf[-keep:] if len(buf) > keep else buf
                    break
            else:
                pos = buf.find(OPEN)
                if pos >= 0:
                    out      = buf[:pos]
                    buf      = buf[pos + len(OPEN):]
                    in_think = True
                    if out:
                        yield out
                else:
                    keep = len(OPEN) - 1
                    if len(buf) > keep:
                        out = buf[:-keep]
                        buf = buf[-keep:]
                        if out:
                            yield out
                    break

    if buf and not in_think:
        yield buf


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
    """Remove think blocks and markdown asterisks from narrative prose."""
    if not raw:
        return raw
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = re.sub(r"</?think>", "", raw, flags=re.IGNORECASE).strip()
    return _strip_markdown(cleaned)


# ─── LLM prompt builders ──────────────────────────────────────────────────────
def build_churn_xai_prompt(r: dict) -> str:
    risk_lines = "\n".join([
        f"  {idx+1}. {f['feature_label']} "
        f"({'increases' if f.get('direction', '') in ('raises_risk', 'increases_churn') else 'decreases'} churn risk, "
        f"impact: {f.get('impact_score', f.get('shap_value', 0)):+.3f})"
        for idx, f in enumerate(r["shap_top5"])
    ])
    sent     = r["sentiment"]
    rfm      = r["segment_rfm_context"]
    revenue  = rfm.get("total_revenue",       {}).get("customer", 0) or 0
    usage    = rfm.get("monthly_usage_hrs",    {}).get("customer", 0) or 0
    adoption = rfm.get("feature_adoption_pct", {}).get("customer", 0) or 0
    nps      = rfm.get("avg_nps_score",        {}).get("customer", 0) or 0
    tenure   = rfm.get("days_since_login",     {}).get("customer", 0) or 0
    tone     = sent.get("tone_score", sent.get("vader_compound", 0)) or 0

    feedback_items = sent.get("feedback_texts", [])
    feedback_str   = " | ".join(feedback_items[:5]) if feedback_items else "No feedback available"

    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON — no prose, no markdown fences.

CUSTOMER: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']}) | Segment: {r['segment_label']}
Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}
Revenue: ${revenue:,.0f}/mo | Usage: {usage:.0f}h/mo | Feature Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10
Days since last login: {tenure:.0f}

TOP RISK FACTORS:
{risk_lines}

SENTIMENT: {sent['label']} | Tone score: {tone:+.3f} | Urgency: {sent['urgency_level']} | Topic: {sent['dominant_topic']}
Feedback: "{feedback_str[:400]}"

Rules for your JSON values:
- score_reason: Write 3–5 sentences as a coherent paragraph. Cover all of: (1) state the churn score and name the #1 risk factor with its measured value; (2) explain what that factor means in business terms and how it puts the account at risk; (3) bring in a second or third risk signal with its metric to show the pattern; (4) describe the customer's engagement or financial profile (revenue, usage, adoption, NPS) and what it reveals about the relationship; (5) conclude with the overall business exposure and what the situation calls for. Make every sentence specific — use actual numbers from the data, no vague generalizations.
- risk_factors: exactly 3 concise phrases (max 12 words each) citing actual business signals — each must contain a specific number, metric, or measurement from the data.
- feedback_signal: 1–2 sentences. Describe what the customer's feedback or sentiment reveals about their experience, citing the actual tone score, urgency level, or a quoted keyword. Write "No customer feedback recorded." if there is none.
- retain: exactly 3 action items (max 15 words each). Each must be specific to this customer — reference their plan ({r['plan_type']}), revenue level, or a metric from the data. No generic advice.
- offer: exactly 3 offer items (max 15 words each). Each must be tailored to this plan and revenue — include a specific value, discount %, or timeframe.
- reason: 1–2 sentences citing the most important data points that justify this retention approach and the financial stake.
No asterisks, no bold, no markdown, no technical terms (SHAP, VADER, model names, ML jargon) anywhere.

Reply with this exact JSON structure:
{{"score_reason":"...","risk_factors":["...","...","..."],"feedback_signal":"...","action":{{"retain":["...","...","..."],"offer":["...","...","..."],"reason":"..."}}}}"""


def build_segment_xai_prompt(r: dict) -> str:
    rfm      = r["segment_rfm_context"]
    sent     = r["sentiment"]
    seg_prof = r["segment_profile"]
    seg_act  = r["segment_actions"]
    rfm_lines = "\n".join([
        f"  {k.replace('_', ' ').title()}: customer={v['customer']:.1f}, segment_avg={v['segment_avg']:.1f}"
        for k, v in rfm.items()
    ])

    tone = sent.get("tone_score", sent.get("vader_compound", 0))
    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON — no prose, no markdown fences.

CUSTOMER: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']})
Segment: {r['segment_label']} | Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}

CUSTOMER vs SEGMENT AVERAGES:
{rfm_lines}

SEGMENT PROFILE:
Customers: {seg_prof.get('count', 'N/A')} | Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk: {seg_prof.get('pct_high_risk', 'N/A')}% | Avg tenure: {seg_prof.get('avg_tenure_days', 'N/A')} days
Description: {seg_act.get('description', '')}

SENTIMENT: {sent['label']} (Tone: {tone:+.3f}) | Topic: {sent['dominant_topic']} | Urgency: {sent['urgency_level']}

No asterisks, no bold, no markdown, no technical terms (SHAP, VADER, model names, ML jargon) in any value. Use plain business language only.

Reply with this exact JSON:
{{"segment_reason":"1-2 plain sentences why this customer belongs in this segment","characteristics":["short trait 1","short trait 2","short trait 3"],"watch_out":"1 plain sentence about the main concern","strategy":"1 plain sentence about the best action"}}"""


def build_segment_cohort_prompt(seg_label: str, seg_prof: dict, seg_desc: str,
                                 retain_actions: list, offer_actions: list,
                                 total_customers: int) -> str:
    share_pct = round(seg_prof.get("count", 0) / max(total_customers, 1) * 100, 1)
    return f"""You are a senior customer success analyst. Reply ONLY with valid JSON — no prose, no markdown fences.

SEGMENT: {seg_label}
Customers: {seg_prof.get('count', 'N/A')} ({share_pct}% of all)
Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100 | % High risk: {seg_prof.get('pct_high_risk', 'N/A')}%
Avg revenue: ${seg_prof.get('avg_revenue', 0):,.0f}/mo | Avg usage: {seg_prof.get('avg_usage_hrs', 0):.0f}h/mo
Avg NPS: {seg_prof.get('avg_nps', 0):.1f}/10 | Avg tenure: {seg_prof.get('avg_tenure_days', 0):.0f} days
Churn rate: {seg_prof.get('churn_rate', 0)*100:.1f}%
Description: {seg_desc}
Retention actions: {retain_actions}
Offers: {offer_actions}

No asterisks, no bold, no markdown in any value.

Reply with this exact JSON (all 4 fields required):
{{
  "narrative": "2 sentences with specific numbers from the data: key characteristic and behavior, then business risk or opportunity.",
  "defining_traits": ["trait with a metric", "trait with a metric", "trait with a metric"],
  "top_priority_action": "1 sentence: the highest-priority action with a specific target or timeline",
  "risk_summary": "1 sentence citing the churn rate and avg score from the data"
}}"""


# ─── Narrative / agent system prompts ─────────────────────────────────────────
_SIM_NARRATIVE_SYSTEM = """\
You are a senior customer success analyst. Write a compact, data-driven assessment in plain English.

Write 3-5 sentences total — exactly as many as the situation warrants, no more.
Every sentence must cite at least one specific number from the data (churn score, revenue, days, %, usage hours).
Cover: the churn risk level and its top drivers, the trajectory and financial exposure, and the single most urgent action with a concrete timeline.
If a customer has low risk or strong metrics, say so briefly and stop — do not pad.

No bullet points. No headers. No markdown. No asterisks. No filler. No technical jargon. Write directly without any opening phrase.
"""

_SCENARIO_NARRATIVE_SYSTEM = """\
You are a senior customer success analyst. Write a compact, data-driven scenario assessment in plain English.

Write 3-5 sentences total — exactly as many as the situation warrants, no more.
Every sentence must cite at least one specific number (probability change, revenue, timeline, %).
Cover: what this intervention proposes and its projected impact on churn (with numbers), the main condition for success, and the immediate next step with a concrete deadline and success metric.
If the impact is minor or risk is low, reflect that briefly — do not overstate.

No bullet points. No headers. No markdown. No asterisks. No filler. Write directly without any opening phrase.
"""

_ASK_SYSTEM = """\
You are the in-app AI analyst for Arkanalytics, a customer churn analytics dashboard.
You have been given complete data for one specific customer — use it to answer any question about them.

THE CUSTOMER DATA YOU HAVE ACCESS TO:
- Customer ID, plan type, contract type, and customer segment
- Churn risk score (0-100) and risk level (Low / Medium / High)
- Monthly revenue, monthly usage hours, feature adoption percentage, and NPS score
- Days since last login and days since last payment
- Top business risk factors with their impact on churn probability
- Sentiment analysis: overall tone, dissatisfaction score, urgency level, feedback topic, and actual feedback texts
- Segment profile: average metrics, churn rate, and recommended retention actions

HOW TO ANSWER:
- When asked "who is this customer" or similar, describe their ID, plan, segment, churn score, and key metrics
- Always use actual numbers from the data (scores, revenue, usage, percentages)
- Be direct and specific — avoid generic answers
- Keep responses concise: 3-6 sentences or 3-5 bullet points
- For greetings (hi, hello, etc.), respond warmly in one sentence and offer to help with the customer data
- For questions clearly outside this customer's data (e.g. coding questions, general knowledge, personal topics),
  politely decline in one sentence and suggest a churn-related question instead
- Never mention technical terms like SHAP, VADER, dissatisfaction probability, or model names in your answer
- Do not invent data not in the provided context

FORMATTING:
- No bold, no asterisks, no markdown headers. Bullets use "- " prefix only.
- Answer directly. No openers like "Sure", "Great question", or "Based on the data".
"""

AGENT_PERSONAS = [
    {"name": "Risk Analyst",     "short": "RA", "color": "#ef4444"},
    {"name": "Customer Success", "short": "CS", "color": "#3b82f6"},
    {"name": "Finance Analyst",  "short": "FN", "color": "#f59e0b"},
    {"name": "Product Manager",  "short": "PM", "color": "#8b5cf6"},
]

AGENT_ANALYZE_SYSTEMS: dict[str, str] = {
    "Risk Analyst": (
        "You are a Churn Risk Analyst at a SaaS company. "
        "Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST cite a specific number from the data (score, days, %, revenue).\n"
        "Cover: (1) the single most critical risk signal with its value and business meaning, "
        "(2) the behavioral pattern driving it, "
        "(3) one specific action to reduce risk with a concrete timeline. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Customer Success": (
        "You are a Customer Success Manager at a SaaS company. "
        "Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST cite a specific number or fact from the data.\n"
        "Cover: (1) the root cause of disengagement with evidence from the data, "
        "(2) the sentiment or feedback signal that confirms it, "
        "(3) one specific outreach action this week with a measurable goal. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Finance Analyst": (
        "You are a Finance Analyst at a SaaS company. "
        "Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST include a dollar amount or percentage.\n"
        "Cover: (1) revenue at risk if churned (calculate from the data), "
        "(2) this customer's value vs segment average with numbers, "
        "(3) the most cost-effective retention offer with its estimated ROI. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Product Manager": (
        "You are a Product Manager at a SaaS company. "
        "Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST cite a specific metric from the data.\n"
        "Cover: (1) the feature adoption or usage gap with its measured value, "
        "(2) the usage pattern that signals disengagement, "
        "(3) one specific product intervention with a success metric to track. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
}

AGENT_SCENARIO_SYSTEMS: dict[str, str] = {
    "Risk Analyst": (
        "You are a Churn Risk Analyst at a SaaS company. "
        "Analyze the proposed intervention. Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST cite a specific number.\n"
        "Cover: (1) estimated churn reduction in percentage points if this works, "
        "(2) the specific risk factor it addresses and by how much, "
        "(3) residual risk if it fails, with a confidence assessment. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Customer Success": (
        "You are a Customer Success Manager at a SaaS company. "
        "Analyze the proposed intervention. Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST be grounded in the customer's actual data.\n"
        "Cover: (1) whether this addresses the real root cause and why, "
        "(2) the biggest obstacle to success based on this customer's profile, "
        "(3) execution timeline with one specific milestone. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Finance Analyst": (
        "You are a Finance Analyst at a SaaS company. "
        "Analyze the proposed intervention. Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST include a dollar amount or percentage.\n"
        "Cover: (1) cost of this intervention vs revenue at risk (with numbers), "
        "(2) expected ROI if retention succeeds, "
        "(3) financial recommendation — proceed, modify, or reject — with one-line reasoning. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
    "Product Manager": (
        "You are a Product Manager at a SaaS company. "
        "Analyze the proposed intervention. Give exactly 3 bullet points using - (hyphen). One sentence each. No asterisks or bold. "
        "Each point MUST cite a specific metric or feature.\n"
        "Cover: (1) how this changes feature adoption or usage (with target metric), "
        "(2) the product gap it closes for this customer specifically, "
        "(3) the one KPI to monitor in the first 30 days. "
        "No generic statements. No SHAP, VADER, or ML jargon."
    ),
}


# ─── Simulation helpers ────────────────────────────────────────────────────────
def _build_sim_json_system(horizon_weeks: int = 12, segment_labels: list = None) -> str:
    if horizon_weeks <= 8:    step = 1
    elif horizon_weeks <= 16: step = 2
    else:                     step = 4

    time_points = list(range(0, horizon_weeks + 1, step))
    if time_points[-1] != horizon_weeks:
        time_points.append(horizon_weeks)

    baseline_items = ",\n    ".join(
        f'{{"week": {w}, "prob": <float 0-100>}}' for w in time_points
    ).replace(
        '{"week": 0, "prob": <float 0-100>}',
        '{"week": 0, "prob": <ACTUAL_CHURN_SCORE>}', 1
    )

    if segment_labels and len(segment_labels) >= 2:
        seg_items = ",\n    ".join(
            f'{{"label": "{lbl}", "prob": <float 0.0-1.0>}}' for lbl in segment_labels
        )
        seg_note = f"Use these exact segment labels: {segment_labels}"
    else:
        seg_items = (
            '{"label": "Churned", "prob": <float>},\n'
            '    {"label": "High Risk", "prob": <float>},\n'
            '    {"label": "At Risk", "prob": <float>},\n'
            '    {"label": "Retained", "prob": <float>}'
        )
        seg_note = "Use these standard labels"

    mid_week  = time_points[len(time_points) // 2]
    never_val = horizon_weeks + 1

    return f"""\
You are a retention analytics engine. Produce a precise JSON churn trajectory for {horizon_weeks} weeks. Output ONLY valid JSON.

Schema:
{{
  "baseline": [{baseline_items}],
  "projection": null,
  "retention_window_weeks": <int>,
  "revenue_at_risk": <float>,
  "confidence": <float 0-1>,
  "intervention_impact_pct": null,
  "segment_migration": [{seg_items}]
}}

Rules:
- baseline[0].prob MUST equal the customer's actual churn_score exactly.
- Trajectory MUST NOT be flat. Show meaningful variation:
  * churn >= 85: oscillate ±5-15 pts, stay high (e.g. 98→95→97→92→94). Never all 100.
  * churn 60-84: gradual increase 5-20 pts over horizon with variation.
  * churn < 60: moderate increase or plateau with slight variation.
  * Final value must differ from week-0 by at least 5 pts.
- segment_migration probs must sum to 1.0. {seg_note}.
- retention_window_weeks: weeks until prob crosses 80. 0 if already above 80. {never_val} if never.
- revenue_at_risk: monthly_revenue × (prob at week {mid_week} / 100) × 3.
"""


_SCENARIO_UPDATE_JSON_SYSTEM = """\
You are a retention analytics engine. A multi-agent team has debated an intervention.
Produce a JSON update. Output ONLY valid JSON.

Schema:
{
  "projection": [{"week": 0, "prob": <MUST EQUAL current churn_score>}, {"week": W, "prob": <float>}, ...],
  "intervention_impact_pct": <float: pp reduction at final week vs baseline>,
  "confidence": <float 0-1>,
  "retention_window_weeks": <int>,
  "revenue_at_risk": <float: monthly_revenue × midpoint_prob/100 × 3>,
  "segment_migration": [{"label": "...", "prob": <float>}, ...]
}

Rules:
- projection[0].prob MUST equal the customer's current churn_score exactly.
- projection values must generally be LOWER than baseline (intervention reduces churn).
- intervention_impact_pct = baseline_final - projection_final (positive number).
- segment_migration MUST reflect the intervention effect: shift probability AWAY from high-risk/churned labels and TOWARD retained/loyal labels compared to what the baseline implied. The change must be visible (at least 5-15pp shift across labels). Use the SAME label names as provided in the context.
- segment_migration probs must sum to exactly 1.0.
- Match the same weekly time-points as the baseline.
- Output ONLY the JSON object.
"""


def _build_ctx(c, scenario: str) -> str:
    rfm      = c.segment_rfm_context
    risk_txt = "\n".join(
        f"  - {f['feature_label']}: {f.get('impact_score', f.get('shap_value', 0)):+.3f} "
        f"({'increases' if f.get('direction', '') in ('raises_risk', 'increases_churn') else 'decreases'} risk)"
        for f in c.shap_top5
    )
    rev      = rfm.get("total_revenue",       {}).get("customer", 0) or 0
    usage    = rfm.get("monthly_usage_hrs",    {}).get("customer", 0) or 0
    adoption = rfm.get("feature_adoption_pct", {}).get("customer", 0) or 0
    nps      = rfm.get("avg_nps_score",        {}).get("customer", 0) or 0
    dsl      = rfm.get("days_since_login",     {}).get("customer", 0) or 0
    tone     = c.sentiment.get("tone_score", c.sentiment.get("vader_compound", 0)) or 0

    feedback_items   = c.sentiment.get("feedback_texts", [])
    feedback_preview = feedback_items[0][:150] if feedback_items else "No feedback"

    ctx = (
        f"CUSTOMER: {c.customer_id} | Plan: {c.plan_type} ({c.contract_type}) | Segment: {c.segment_label}\n"
        f"Churn Score: {c.churn_score:.1f}/100 | Risk: {c.risk_level}\n"
        f"Revenue: ${rev:,.0f}/mo | Usage: {usage:.0f}h/mo | Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10\n"
        f"Days since last login: {dsl:.0f}\n"
        f"Sentiment: {c.sentiment.get('label', 'N/A')} "
        f"(Tone: {tone:+.3f}) | "
        f"Urgency: {c.sentiment.get('urgency_level', 'N/A')}\n"
        f"Feedback: \"{feedback_preview}\"\n\n"
        f"TOP RISK FACTORS:\n{risk_txt}"
    )
    if scenario.strip():
        ctx += f"\n\nINTERVENTION SCENARIO:\n{scenario}"
    return ctx


def _build_contextual_fallback(risk_lvl: str, customer_profile: dict, scenario: str = "") -> list[str]:
    segment  = customer_profile.get("segment_label", "")
    plan     = customer_profile.get("plan_type", "")
    contract = customer_profile.get("contract_type", "")
    shap5    = customer_profile.get("shap_top5", [])

    _FACTOR_ACTIONS: dict[str, str] = {
        "days since login":  "Run a re-engagement campaign with a live feature demo",
        "monthly usage hrs": "Schedule an intensive onboarding session to increase usage",
        "avg payment delay": "Review payment options and enable auto-billing",
        "feature adoption":  "Run a 30-day 1-on-1 premium feature training with CSM",
        "adoption x usage":  "Launch a feature activation sprint with step-by-step guidance",
        "nps score":         "Conduct an NPS recovery call and resolve complaints within 24h",
        "avg nps score":     "Conduct an NPS recovery call and resolve complaints within 24h",
        "tenure days":       "Offer a loyalty appreciation program for long-term customers",
        "contract type":     "Convert to annual contract with a 25% discount",
        "plan type":         "Upgrade to a higher plan with a free 30-day trial",
        "support tickets":   "Fast-track all open tickets and assign a priority support agent",
        "billing issues":    "Audit billing, remove unclear charges, and offer account credit",
    }

    pool: list[str] = []
    seen: set[str]  = set()

    def add(rec: str) -> None:
        if rec not in seen:
            seen.add(rec); pool.append(rec)

    for factor in shap5[:3]:
        label  = str(factor.get("feature_label", "")).lower()
        shap_v = float(factor.get("impact_score", factor.get("shap_value", 0)))
        if shap_v <= 0:
            continue
        for key, action in _FACTOR_ACTIONS.items():
            if key in label or label in key:
                add(action); break

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

    if risk_lvl == "High":
        add("Freeze billing for 2 months and grant full premium feature access")
        add("Contact customer within 24 hours for emergency retention intervention")
    elif risk_lvl == "Medium":
        add("Schedule automated weekly CSM check-ins for the next 8 weeks")
        add("Offer a 20% discount for a 6-month contract extension")
    else:
        add("Send a satisfaction survey and activate an exclusive referral program")
        add("Give a usage credit bonus as a loyalty appreciation gesture")

    if scenario.strip():
        add("Combine the above intervention with a long-term loyalty incentive")
        add("A/B test: price discount vs increased CSM service intensity")

    for g in [
        "Schedule a monthly business review and monitor account health metrics",
        "Grant free premium feature access for 60 days",
        "Assign a dedicated Customer Success Manager for 90 days",
        "Offer a 20% discount for a 6-month contract extension",
    ]:
        if len(pool) >= 4: break
        add(g)

    return pool[:4]


async def _extract_recommendations(
    ctx: str, agent_outputs: list, scenario: str = "",
    risk_level: str = "High", customer_profile: dict | None = None,
) -> list:
    cp = customer_profile or {}
    debate_text = "\n".join(f"[{o['name']}]: {o['content'][:400]}" for o in agent_outputs)

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
        f"- Each option is a concrete action phrase (15-60 chars), grounded in this customer's actual data.\n"
        f"- Reference the customer's plan type, churn score, revenue, usage level, or a specific metric — no generic advice.\n"
        f"- Each option must be distinct (different lever: price, engagement, product, support).\n\n"
        f"Customer context:\n{ctx[:300]}\n\n"
        f"Agent outputs:\n{debate_text}\n\n"
        f"Return ONLY a valid JSON array of exactly 4 strings. No explanation, no fences.\n"
        f"Example: [\"Offer 20% discount on Annual plan renewal\", \"Assign CSM for 90-day check-ins\", ...]"
    )

    _fallback = _build_contextual_fallback(risk_level, cp, scenario)

    try:
        raw = await call_llm(
            "Generate 4 intervention scenario options. Return ONLY a JSON array of 4 strings.",
            prompt, max_tokens=400,
        )
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\s*```\s*$",        "", raw, flags=re.MULTILINE).strip()
        start = raw.find("["); end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            items   = json.loads(raw[start:end])
            cleaned = [str(x).strip() for x in items if str(x).strip()]
            if len(cleaned) >= 2:
                return cleaned[:4]
        quoted = re.findall(r'"([^"]{10,100})"', raw)
        if len(quoted) >= 2:
            return quoted[:4]
    except Exception:
        pass
    return _fallback


# ─── Endpoints ────────────────────────────────────────────────────────────────
class SegmentCohortRequest(BaseModel):
    segment_label:       str
    total_customers:     int
    avg_churn_score:     float
    pct_high_risk:       float
    avg_revenue:         float
    avg_usage_hrs:       float
    avg_nps:             float
    avg_tenure_days:     float = 0.0
    churn_rate:          float = 0.0
    segment_description: str   = ""
    retain_actions:      list  = []
    offer_actions:       list  = []
    total_all_customers: int   = 0


@app.post("/generate-cohort-xai")
async def generate_cohort_xai(segments: list[SegmentCohortRequest]):
    total = sum(s.total_customers for s in segments) or 1

    async def _one(seg):
        prompt = build_segment_cohort_prompt(
            seg_label       = seg.segment_label,
            seg_prof        = {
                "count":           seg.total_customers,
                "avg_churn_score": seg.avg_churn_score,
                "pct_high_risk":   seg.pct_high_risk,
                "avg_revenue":     seg.avg_revenue,
                "avg_usage_hrs":   seg.avg_usage_hrs,
                "avg_nps":         seg.avg_nps,
                "avg_tenure_days": seg.avg_tenure_days,
                "churn_rate":      seg.churn_rate,
            },
            seg_desc        = seg.segment_description,
            retain_actions  = seg.retain_actions,
            offer_actions   = seg.offer_actions,
            total_customers = seg.total_all_customers or total,
        )
        return seg.segment_label, await _call_llm_xai(prompt)

    pairs   = await asyncio.gather(*[_one(s) for s in segments])
    results = dict(pairs)
    return sanitize_floats({"status": "success", "cohort_xai": results})


@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": A.get("model_version", "v2.1"),
        "model_type":    type(MODEL).__name__,
        "calibrated":    "calibrated_model" in A,
        "model_name":    A.get("best_model_name", "unknown"),
        "risk_low":      RISK_LOW,
        "risk_high":     RISK_HIGH,
        "fusion_alpha":  FUSION_ALPHA,
        "n_features":    len(FEATURES),
        "n_topics":      N_TOPICS,
        "stability":     A.get("stability", {}),
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
        # ── Run all per-customer XAI calls in parallel ──────────────────────
        async def _xai_one(r: dict) -> None:
            r["xai_churn_explanation"], r["xai_segment_explanation"] = await asyncio.gather(
                _call_llm_xai(build_churn_xai_prompt(r)),
                _call_llm_xai(build_segment_xai_prompt(r)),
            )

        await asyncio.gather(*[_xai_one(r) for r in results])

        # ── Per-segment cohort narratives — parallel across unique segments ──
        unique_seg_rows: dict[str, dict] = {}
        for r in results:
            seg = r["segment_label"]
            if seg not in unique_seg_rows:
                unique_seg_rows[seg] = r

        async def _cohort_for(seg: str, row: dict) -> tuple[str, str]:
            return seg, await _call_llm_xai(
                build_segment_cohort_prompt(
                    seg_label       = seg,
                    seg_prof        = row["segment_profile"],
                    seg_desc        = row["segment_actions"].get("description", ""),
                    retain_actions  = row["segment_actions"].get("retain", []),
                    offer_actions   = row["segment_actions"].get("offer", []),
                    total_customers = len(results),
                )
            )

        cohort_pairs = await asyncio.gather(*[_cohort_for(seg, row) for seg, row in unique_seg_rows.items()])
        seg_cohort   = dict(cohort_pairs)

        for r in results:
            r["xai_segment_cohort"] = seg_cohort.get(r["segment_label"])
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
        df.drop(df[df["customer_id"] != customer_id].index, inplace=True)

    try:
        results = run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")
    r = results[0]

    # ── All 3 XAI calls in parallel ─────────────────────────────────────────
    r["xai_churn_explanation"], r["xai_segment_explanation"], r["xai_segment_cohort"] = \
        await asyncio.gather(
            _call_llm_xai(build_churn_xai_prompt(r)),
            _call_llm_xai(build_segment_xai_prompt(r)),
            _call_llm_xai(build_segment_cohort_prompt(
                seg_label       = r["segment_label"],
                seg_prof        = r["segment_profile"],
                seg_desc        = r["segment_actions"].get("description", ""),
                retain_actions  = r["segment_actions"].get("retain", []),
                offer_actions   = r["segment_actions"].get("offer", []),
                total_customers = 1,
            )),
        )
    return sanitize_floats(r)


# ─── Simulation ───────────────────────────────────────────────────────────────
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
    customer_data:             _CustomerDataSim
    scenario:                  str        = ""
    chat_history:              list[dict] = []
    horizon_weeks:             int        = 12
    segment_labels:            list[str]  = []
    mode:                      str        = "initial"
    current_segment_migration: list | None = None


@app.post("/simulate")
async def simulate(request: SimulateRequest):
    c        = request.customer_data
    ctx      = _build_ctx(c, "")
    horizon  = request.horizon_weeks
    seg_lbls = request.segment_labels or []

    mode = request.mode
    if request.scenario.strip() and mode not in ("scenario", "ask"):
        mode = "scenario"
    if mode == "analyze":
        mode = "initial"

    history_block = ""
    if request.chat_history:
        lines = [
            f"Q: {t.get('question', '')}\nA: {str(t.get('narrative', ''))[:250]}"
            for t in request.chat_history[-4:]
        ]
        history_block = "\n\nPREVIOUS TURNS:\n" + "\n\n".join(lines)

    if horizon <= 8:    _step = 1
    elif horizon <= 16: _step = 2
    else:               _step = 4
    _fallback_points = list(range(0, horizon + 1, _step))
    if _fallback_points[-1] != horizon:
        _fallback_points.append(horizon)

    def _fallback_sim(exc_msg: str = "") -> dict:
        import random
        base  = c.churn_score
        rng   = random.Random(hash(c.customer_id) % (2**31))
        bline = []
        cur   = base
        for i, w in enumerate(_fallback_points):
            if i == 0:
                bline.append({"week": w, "prob": round(cur, 2)}); continue
            if base >= 85:
                cur = max(72, min(99, cur + rng.uniform(-3, 5) - 1.5))
            elif base >= 60:
                cur = min(98, cur + rng.uniform(0.5, 2.5))
            else:
                cur = max(5, min(80, cur + rng.uniform(-0.5, 1.5)))
            bline.append({"week": w, "prob": round(cur, 2)})
        mid_prob    = bline[len(bline) // 2]["prob"] if len(bline) > 1 else base
        monthly_rev = c.segment_rfm_context.get("total_revenue", {}).get("customer", 0)
        if seg_lbls and len(seg_lbls) >= 2:
            n     = len(seg_lbls)
            probs = [round(base / 100 * 0.7 / max(n-1,1) * i + (1-base/100) * 0.3 / max(n-1,1) * (n-1-i), 3) for i in range(n)]
            tot   = sum(probs) or 1
            seg_mig = [{"label": seg_lbls[i], "prob": round(probs[i]/tot, 3)} for i in range(n)]
        else:
            seg_mig = [
                {"label": "Churned",   "prob": round(base/100*0.7, 3)},
                {"label": "High Risk", "prob": round(base/100*0.2, 3)},
                {"label": "At Risk",   "prob": round((1-base/100)*0.4, 3)},
                {"label": "Retained",  "prob": round((1-base/100)*0.6, 3)},
            ]
        result = {
            "baseline": bline, "projection": None,
            "retention_window_weeks": next((pt["week"] for pt in bline if pt["prob"] >= 80), 0),
            "revenue_at_risk": round(monthly_rev * (mid_prob/100) * 3, 2),
            "confidence": 0.6, "intervention_impact_pct": None, "segment_migration": seg_mig,
        }
        if exc_msg:
            result["_error"] = exc_msg[:120]
        return result

    async def _run_agents_parallel(agent_ctx: str, agent_systems: dict[str, str]):
        """Run all 4 agents concurrently; stream tokens and done status in real-time using an asyncio.Queue."""
        queue = asyncio.Queue()
        active_tasks = 0

        async def _run_one(persona: dict):
            name = persona["name"]
            system = agent_systems.get(name, "You are a customer success expert. Give a brief analysis.")
            try:
                # 1. Put agent_start event into the queue
                await queue.put({"type": "agent_start", "agent": name, "short": persona["short"], "color": persona["color"]})
                
                content = ""
                # 2. Stream tokens into the queue
                async for tok in stream_llm_no_think(system, agent_ctx, max_tokens=450):
                    content += tok
                    await queue.put({"type": "agent_token", "agent": name, "content": tok})
                
                # 3. Put agent_done event into the queue
                await queue.put({"type": "agent_done", "agent": name, "content": _strip_markdown(content)})
            except Exception as e:
                # If an error happens, still mark agent as done or log
                await queue.put({"type": "agent_done", "agent": name, "content": f"Analysis unavailable: {str(e)}"})

        # Start all tasks
        for p in AGENT_PERSONAS:
            asyncio.create_task(_run_one(p))
            active_tasks += 1

        outputs: dict[str, str] = {p["name"]: "" for p in AGENT_PERSONAS}
        completed_agents = 0

        # Read from queue and yield events
        while completed_agents < active_tasks:
            evt = await queue.get()
            if evt["type"] == "agent_start":
                yield f"data: {json.dumps({'type': 'agent_start', 'agent': evt['agent'], 'short': evt['short'], 'color': evt['color']})}\n\n"
            elif evt["type"] == "agent_token":
                outputs[evt["agent"]] += evt["content"]
                yield f"data: {json.dumps({'type': 'agent_token', 'agent': evt['agent'], 'content': evt['content']})}\n\n"
            elif evt["type"] == "agent_done":
                # Ensure the content is stripped and saved
                outputs[evt["agent"]] = _strip_markdown(outputs[evt["agent"]])
                yield f"data: {json.dumps({'type': 'agent_done', 'agent': evt['agent']})}\n\n"
                completed_agents += 1
            queue.task_done()

        # Save the outputs back for the recommendations and narrative
        _run_agents_parallel._last_outputs = [
            {"name": name, "content": content} for name, content in outputs.items()
        ]

    async def event_stream():

        # ── MODE: ask ──────────────────────────────────────────────────────
        if mode == "ask":
            question = request.scenario.strip() or "Provide a summary of this customer's situation."
            ask_prompt = f"{ctx}{history_block}\n\nUSER QUESTION: {question}"
            full_answer = ""
            async for tok in stream_llm_no_think(_ASK_SYSTEM, ask_prompt, max_tokens=2500):
                full_answer += tok
                yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
            cleaned = _clean_narrative(full_answer)
            if not cleaned.strip():
                # Fallback: model habis token untuk thinking, tidak ada jawaban yang tersisa
                cleaned = "I can help you analyze this customer's churn risk, retention options, sentiment, or revenue at risk — what would you like to know?"
                yield f"data: {json.dumps({'type': 'token', 'content': cleaned})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'narrative': cleaned})}\n\n"
            return

        # ── MODE: initial ──────────────────────────────────────────────────
        if mode == "initial":
            yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

            sim_data: dict = {}
            try:
                raw      = await call_llm(
                    _build_sim_json_system(horizon, seg_lbls),
                    f"{ctx}{history_block}\n\nGenerate the churn trajectory JSON now.",
                    max_tokens=1200,
                )
                sim_data = _extract_json(raw)
                if sim_data.get("baseline"):
                    sim_data["baseline"][0]["prob"] = round(c.churn_score, 2)
            except Exception as exc:
                sim_data = _fallback_sim(str(exc))

            baseline_pts = sim_data.get("baseline", [])
            if len(baseline_pts) >= 2:
                sim_data["intervention_impact_pct"] = round(
                    baseline_pts[0].get("prob", c.churn_score) -
                    baseline_pts[-1].get("prob", c.churn_score), 2
                )

            yield f"data: {json.dumps({'type': 'data', 'payload': sim_data})}\n\n"

            last_pt   = baseline_pts[-1] if baseline_pts else {}
            agent_ctx = (
                f"{ctx}{history_block}"
                f"\n\nFORECAST ({horizon} weeks): churn {c.churn_score:.1f}% → "
                f"{last_pt.get('prob', c.churn_score):.1f}% at week {horizon}. "
                f"Revenue at risk: ${sim_data.get('revenue_at_risk', 0):,.0f}."
                f"\n\nProvide your analysis."
            )

            async for evt in _run_agents_parallel(agent_ctx, AGENT_ANALYZE_SYSTEMS):
                yield evt
            agent_outputs = getattr(_run_agents_parallel, "_last_outputs", [])

            recs = await _extract_recommendations(ctx, agent_outputs, scenario="",
                risk_level=c.risk_level, customer_profile={
                    "segment_label": c.segment_label, "plan_type": c.plan_type,
                    "contract_type": c.contract_type, "churn_score": c.churn_score,
                    "shap_top5": c.shap_top5,
                })
            if recs:
                yield f"data: {json.dumps({'type': 'agent_recommendations', 'recommendations': recs})}\n\n"

            debate_text = "\n".join(f"[{o['name']}]: {o['content'][:300]}" for o in agent_outputs)
            narrative_prompt = (
                f"{ctx}{history_block}"
                f"\n\nChurn trajectory ({horizon} weeks): "
                f"week 0={c.churn_score:.1f}%, week {last_pt.get('week', horizon)}="
                f"{last_pt.get('prob', c.churn_score):.1f}%, "
                f"retention_window={sim_data.get('retention_window_weeks', '?')} weeks, "
                f"revenue_at_risk=${sim_data.get('revenue_at_risk', 0):,.0f}."
                + f"\n\nTEAM ANALYSIS:\n{debate_text[:600]}"
            )
            full_narrative = ""
            async for tok in stream_llm_no_think(_SIM_NARRATIVE_SYSTEM, narrative_prompt, max_tokens=450):
                full_narrative += tok
                yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'narrative': _clean_narrative(full_narrative)})}\n\n"
            return

        # ── MODE: scenario ─────────────────────────────────────────────────
        scenario_block = f"\n\nINTERVENTION SCENARIO: {request.scenario}"
        agent_ctx = f"{ctx}{scenario_block}{history_block}\n\nProvide your analysis."

        async for evt in _run_agents_parallel(agent_ctx, AGENT_SCENARIO_SYSTEMS):
            yield evt
        agent_outputs_s = getattr(_run_agents_parallel, "_last_outputs", [])

        recs_s = await _extract_recommendations(ctx, agent_outputs_s, scenario=request.scenario,
            risk_level=c.risk_level, customer_profile={
                "segment_label": c.segment_label, "plan_type": c.plan_type,
                "contract_type": c.contract_type, "churn_score": c.churn_score,
                "shap_top5": c.shap_top5,
            })
        if recs_s:
            yield f"data: {json.dumps({'type': 'agent_recommendations', 'recommendations': recs_s})}\n\n"

        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        debate_text = "\n\n".join(f"[{o['name']}]: {o['content']}" for o in agent_outputs_s)
        monthly_rev = c.segment_rfm_context.get("total_revenue", {}).get("customer", 0)
        seg_note    = (f"Use segment labels: {seg_lbls}" if seg_lbls
                       else "Use standard labels: Churned, High Risk, At Risk, Retained")

        # Include current baseline segment migration so LLM knows what to shift from
        baseline_seg_note = ""
        if request.current_segment_migration:
            seg_lines = ", ".join(
                f"{s['label']}={round(s['prob']*100)}%"
                for s in request.current_segment_migration
                if isinstance(s, dict)
            )
            baseline_seg_note = (
                f"\nBASELINE SEGMENT MIGRATION (pre-intervention): {seg_lines}"
                f"\nYou MUST shift these probabilities — reduce churn/high-risk by 8-20pp, increase retained/loyal by same amount."
            )

        synth_user = (
            f"{ctx}{scenario_block}"
            f"\n\nBASELINE TIME POINTS (weeks): {_fallback_points}"
            f"\nMONTHLY REVENUE: ${monthly_rev:,.0f}"
            f"\n{seg_note}{baseline_seg_note}"
            f"\n\nMULTI-AGENT DEBATE:\n{debate_text}"
            f"\n\nGenerate the scenario update JSON now."
        )

        update_data: dict = {}
        try:
            raw         = await call_llm(_SCENARIO_UPDATE_JSON_SYSTEM, synth_user, max_tokens=1000)
            update_data = _extract_json(raw)
            if update_data.get("projection"):
                update_data["projection"][0]["prob"] = round(c.churn_score, 2)
        except Exception as exc:
            # Build a fallback segment_migration shifted toward retention
            _cur_seg = request.current_segment_migration or []
            if _cur_seg and len(_cur_seg) >= 2:
                # Shift 15pp from churn/high-risk labels toward retained/loyal labels
                _shift    = 0.15
                _churn_kw = {"churned", "risk", "critical", "danger", "high"}
                _keep_kw  = {"retained", "loyal", "champion", "stable", "active"}
                _new_seg  = []
                _total    = sum(s.get("prob", 0) for s in _cur_seg) or 1.0
                for s in _cur_seg:
                    lbl  = str(s.get("label", "")).lower()
                    prob = s.get("prob", 0) / _total
                    if any(k in lbl for k in _churn_kw):
                        prob = max(0.02, prob - _shift / len(_cur_seg) * 2)
                    elif any(k in lbl for k in _keep_kw):
                        prob = min(0.95, prob + _shift / len(_cur_seg) * 2)
                    _new_seg.append({"label": s["label"], "prob": round(prob, 3)})
                _norm = sum(x["prob"] for x in _new_seg) or 1
                _fallback_seg_mig = [{"label": x["label"], "prob": round(x["prob"] / _norm, 3)} for x in _new_seg]
            elif seg_lbls and len(seg_lbls) >= 2:
                n = len(seg_lbls)
                base = c.churn_score / 100
                probs = [max(0.02, base * 0.5 / max(n-1,1) * (n-1-i) + (1-base) * 0.4 / max(n-1,1) * i) for i in range(n)]
                tot = sum(probs) or 1
                _fallback_seg_mig = [{"label": seg_lbls[i], "prob": round(probs[i]/tot, 3)} for i in range(n)]
            else:
                base = c.churn_score / 100
                _fallback_seg_mig = [
                    {"label": "Churned",   "prob": round(max(0.02, base * 0.45), 3)},
                    {"label": "High Risk", "prob": round(base * 0.15, 3)},
                    {"label": "At Risk",   "prob": round((1-base) * 0.35, 3)},
                    {"label": "Retained",  "prob": round((1-base) * 0.60, 3)},
                ]
                _tot = sum(x["prob"] for x in _fallback_seg_mig) or 1
                _fallback_seg_mig = [{"label": x["label"], "prob": round(x["prob"]/_tot, 3)} for x in _fallback_seg_mig]

            update_data = {
                "projection": [
                    {"week": w, "prob": round(min(100, c.churn_score * (0.70 ** (1 + i * 0.15))), 2)}
                    for i, w in enumerate(_fallback_points)
                ],
                "intervention_impact_pct": 30.0,
                "confidence":              0.55,
                "retention_window_weeks":  min(horizon + 1, 8),
                "revenue_at_risk":         round(monthly_rev * (c.churn_score * 0.65 / 100) * 3, 2),
                "segment_migration":       _fallback_seg_mig,
                "_error": str(exc)[:120],
            }
            update_data["projection"][0]["prob"] = round(c.churn_score, 2)

        yield f"data: {json.dumps({'type': 'data', 'payload': update_data})}\n\n"

        proj_last = (update_data.get("projection") or [{}])[-1]
        narrative_prompt = (
            f"{ctx}{scenario_block}"
            f"\n\nDEBATE SUMMARY:\n{debate_text[:800]}"
            f"\n\nProjection: churn week {proj_last.get('week', horizon)}="
            f"{proj_last.get('prob', c.churn_score):.1f}%, "
            f"impact={update_data.get('intervention_impact_pct', 0):.0f}pp reduction, "
            f"revenue_at_risk=${update_data.get('revenue_at_risk', 0):,.0f}."
        )
        full_narrative = ""
        async for tok in stream_llm_no_think(_SCENARIO_NARRATIVE_SYSTEM, narrative_prompt, max_tokens=450):
            full_narrative += tok
            yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'narrative': _clean_narrative(full_narrative)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )