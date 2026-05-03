import os, json, io, warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import httpx
from scipy.sparse import hstack, csr_matrix

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
print("Loading artifacts...")

# Notebook 1: churn_pipeline_v3 artifacts
A = joblib.load(os.getenv("ARTIFACTS_PATH", "churn_artifacts_v1.pkl"))
MODEL        = A["model"]
FEATURES     = A["production_features"]
LE_PLAN      = A["le_plan"]
LE_CONTRACT  = A["le_contract"]
SCALER_SEG   = A["scaler_seg"]
KMEANS       = A["kmeans"]
LABEL_MAP    = A["cluster_label_map"]
SEG_FEATURES = A["seg_features"]
SEG_PROFILES = A["segment_profiles"]
SEG_ACTIONS  = A["segment_actions"]
RISK_LOW     = A["risk_thresholds"]["low"]
RISK_HIGH    = A["risk_thresholds"]["high"]
REF          = pd.Timestamp(A["reference_date"])
EXPLAINER    = shap.TreeExplainer(MODEL)

# Notebook 2: nlp_pipeline_complete artifacts
N = joblib.load(os.getenv("NLP_ARTIFACTS_PATH", "nlp_artifacts_v1.pkl"))
TFIDF        = N["tfidf"]
LDA          = N["lda"]
CV_VEC       = N["cv_vec"]
NLP_LR       = N["nlp_lr"]
SCALER_NLP   = N["scaler_nlp"]
TOPIC_NAMES  = N["topic_names"]
URGENCY_LEX  = N["urgency_lexicon"]
EXTRA_COLS   = N["extra_cols"]
N_TOPICS     = N["n_topics"]

ANALYZER     = SentimentIntensityAnalyzer()

_raw_url     = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
# Normalise: ensure scheme, strip trailing slash
if not _raw_url.startswith("http"):
    _raw_url = f"https://{_raw_url}"
OLLAMA_URL   = _raw_url.rstrip("/")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-4o-mini")
FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "0.8"))  # weight for tabular model

print("✅ Both artifacts loaded")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_level(score: float) -> str:
    return "Low" if score <= RISK_LOW else ("Medium" if score <= RISK_HIGH else "High")

def get_top_shap(shap_row: pd.Series, top_n: int = 5) -> list:
    top = shap_row.abs().nlargest(top_n)
    return [
        {
            "feature":     k,
            "shap_value":  round(float(shap_row[k]), 4),
            "direction":   "increases_churn" if shap_row[k] > 0 else "decreases_churn",
            "importance":  round(abs(float(shap_row[k])), 4),
            "feature_label": k.replace("_", " ").title(),
        }
        for k in top.index
    ]

# ─── NLP pipeline ─────────────────────────────────────────────────────────────
def compute_nlp_features(text_df: pd.DataFrame) -> dict:
    """
    Run full NLP pipeline on per-customer aggregated text.
    Returns dict of arrays, one value per customer row.
    """
    texts = text_df["all_feedback"].fillna("").tolist()

    # VADER document level
    vader_results = [ANALYZER.polarity_scores(str(t)) for t in texts]
    vader_compound   = np.array([r["compound"] for r in vader_results])
    vader_pos        = np.array([r["pos"]      for r in vader_results])
    vader_neg        = np.array([r["neg"]      for r in vader_results])
    vader_neu        = np.array([r["neu"]      for r in vader_results])

    # VADER sentence level
    def sent_vader(text):
        sentences = re.split(r"[.!?|]+", str(text))
        scores = [ANALYZER.polarity_scores(s.strip())["compound"]
                  for s in sentences if len(s.strip()) > 10]
        if not scores:
            return 0.0, 0.0, 0.0, 0.0
        return (min(scores), float(np.std(scores)),
                sum(1 for s in scores if s < -0.05) / len(scores),
                max(scores) - min(scores))

    sent_stats = [sent_vader(t) for t in texts]
    vader_min_sent     = np.array([s[0] for s in sent_stats])
    vader_std_sent     = np.array([s[1] for s in sent_stats])
    pct_negative_sent  = np.array([s[2] for s in sent_stats])
    vader_range        = np.array([s[3] for s in sent_stats])

    # Urgency score
    urgency_score = np.array([
        sum(1 for w in URGENCY_LEX if w in str(t).lower()) for t in texts
    ])

    # Sentiment label
    sentiment_labels = [
        "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        for c in vader_compound
    ]
    urgency_levels = [
        "high" if u >= 3 else ("medium" if u >= 1 else "low")
        for u in urgency_score
    ]

    # LDA topics
    X_counts = CV_VEC.transform(texts)
    X_topics = LDA.transform(X_counts)
    dominant_topic      = X_topics.argmax(axis=1)
    dominant_topic_score = X_topics.max(axis=1)
    dominant_topic_label = [TOPIC_NAMES[i] for i in dominant_topic]

    return {
        "vader_compound":        vader_compound,
        "vader_pos":             vader_pos,
        "vader_neg":             vader_neg,
        "vader_neu":             vader_neu,
        "vader_min_sent":        vader_min_sent,
        "vader_std_sent":        vader_std_sent,
        "pct_negative_sent":     pct_negative_sent,
        "vader_range":           vader_range,
        "urgency_score":         urgency_score,
        "urgency_level":         urgency_levels,
        "sentiment_label":       sentiment_labels,
        "dominant_topic":        dominant_topic,
        "dominant_topic_label":  dominant_topic_label,
        "dominant_topic_score":  dominant_topic_score,
        "topic_distribution":    X_topics,  # shape (n_customers, 8)
    }


def compute_nlp_proba(text_df: pd.DataFrame, nlp_feats: dict) -> np.ndarray:
    """Compute NLP model churn probability using the trained NLP LR."""
    texts = text_df["all_feedback"].fillna("").tolist()
    CAT_COLS = ["Account", "Billing", "Feature Request", "General", "Onboarding", "Technical"]

    # Reconstruct extra features in same order as training
    n = len(texts)
    extra = pd.DataFrame({
        "vader_compound":      nlp_feats["vader_compound"],
        "vader_pos":           nlp_feats["vader_pos"],
        "vader_neg":           nlp_feats["vader_neg"],
        "vader_neu":           nlp_feats["vader_neu"],
        "vader_min_sent":      nlp_feats["vader_min_sent"],
        "vader_std_sent":      nlp_feats["vader_std_sent"],
        "pct_negative_sent":   nlp_feats["pct_negative_sent"],
        "vader_range":         nlp_feats["vader_range"],
        "text_length":         text_df["all_feedback"].str.len().fillna(0).values,
        "word_count":          text_df["all_feedback"].str.split().str.len().fillna(0).values,
        "exclaim_count":       text_df["all_feedback"].str.count("!").fillna(0).values,
        "question_count":      text_df["all_feedback"].str.count(r"\?").fillna(0).values,
        "urgency_score":       nlp_feats["urgency_score"],
        "avg_words_per_sent":  np.zeros(n),  # approx
        "avg_nps_score":       text_df.get("avg_nps_score", pd.Series(np.zeros(n))).fillna(0).values,
        "min_nps_score":       text_df.get("min_nps_score", pd.Series(np.zeros(n))).fillna(0).values,
        "n_surveys":           text_df.get("n_surveys",     pd.Series(np.ones(n))).fillna(1).values,
        "pct_detractor":       text_df.get("pct_detractor", pd.Series(np.zeros(n))).fillna(0).values,
        "n_categories":        text_df.get("n_categories",  pd.Series(np.ones(n))).fillna(1).values,
    })
    for c in CAT_COLS:
        extra[c] = text_df.get(c, pd.Series(np.zeros(n))).fillna(0).values
    topic_df = pd.DataFrame(nlp_feats["topic_distribution"],
                            columns=[f"topic_{i}" for i in range(N_TOPICS)])
    extra = pd.concat([extra, topic_df.reset_index(drop=True)], axis=1)

    X_tfidf = TFIDF.transform(texts)
    X_extra = csr_matrix(SCALER_NLP.transform(extra.fillna(0).values))
    X_full  = hstack([X_tfidf, X_extra])
    return NLP_LR.predict_proba(X_full)[:, 1]


# ─── Core pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df):
    """
    Full pipeline combining both notebooks.
    Returns list of per-customer result dicts.
    """
    # Clean
    ca_df = ca_df.copy()
    ca_df["plan_type"]     = ca_df["plan_type"].str.capitalize().str.strip()
    ca_df["contract_type"] = ca_df["contract_type"].str.capitalize().str.strip()
    nps_df["nps_score"]    = nps_df["nps_score"].clip(lower=0)
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
    ca_df["tenure_days"] = (
        ca_df["unsubscribed_date"].fillna(REF) - ca_df["subscription_date"]
    ).dt.days.clip(lower=1)

    # Feature engineering
    payments = bd_df[bd_df["record_type"] == "payment"].copy()
    payments["delay_days"] = (payments["payment_date"] - payments["billing_date"]).dt.days
    bf = payments.groupby("customer_id").agg(
        total_revenue=("payment_value","sum"), avg_payment_value=("payment_value","mean"),
        payment_count=("payment_value","count"), avg_payment_delay=("delay_days","mean"),
        max_payment_delay=("delay_days","max"),
    ).reset_index()
    dun = bd_df[bd_df["record_type"]=="dunning"].groupby("customer_id").size().reset_index(name="dunning_count")
    bf  = bf.merge(dun, on="customer_id", how="left"); bf["dunning_count"] = bf["dunning_count"].fillna(0)

    uf = um_df.copy()
    uf["days_since_login"] = (REF - uf["last_login_date"]).dt.days
    uf = uf.drop(columns=["last_login_date"], errors="ignore")

    tf = st_df.groupby("customer_id").agg(
        total_tickets=("ticket_id","count"),
        open_tickets=("status",lambda x:(x=="Open").sum()),
        billing_tickets=("category",lambda x:(x=="Billing").sum()),
        technical_tickets=("category",lambda x:(x=="Technical").sum()),
        critical_tickets=("priority",lambda x:(x=="Critical").sum()),
        high_tickets=("priority",lambda x:(x=="High").sum()),
    ).reset_index()
    tf["unresolved_ratio"] = tf["open_tickets"]    / tf["total_tickets"].replace(0,1)
    tf["critical_ratio"]   = tf["critical_tickets"] / tf["total_tickets"].replace(0,1)

    # NPS tabular
    nf = nps_df.groupby("customer_id").agg(
        avg_nps_score=("nps_score","mean"), min_nps_score=("nps_score","min"),
        survey_count=("survey_id","count"),
        pct_detractor=("segment",lambda x:(x=="detractor").mean()),
    ).reset_index(); nf["has_nps_data"] = 1

    # NPS NLP features per customer
    CAT_COLS = ["Account","Billing","Feature Request","General","Onboarding","Technical"]
    cat_dummies = pd.get_dummies(nps_df["feedback_category"])
    for c in CAT_COLS:
        if c not in cat_dummies.columns: cat_dummies[c] = 0
    cat_dummies["customer_id"] = nps_df["customer_id"]
    cat_per = cat_dummies.groupby("customer_id")[CAT_COLS].sum().reset_index()

    text_per = nps_df.groupby("customer_id")["feedback_text"].apply(
        lambda x: " | ".join(x.dropna().astype(str))
    ).reset_index(); text_per.columns = ["customer_id","all_feedback"]

    # Master merge
    master = ca_df[["customer_id","plan_type","contract_type","total_users","tenure_days"]].copy()
    master = (master
        .merge(uf,       on="customer_id", how="left")
        .merge(bf,       on="customer_id", how="left")
        .merge(tf,       on="customer_id", how="left")
        .merge(nf,       on="customer_id", how="left")
    )

    # Segmentation (before imputation)
    seg_raw = master[SEG_FEATURES].copy()
    for c in SEG_FEATURES: seg_raw[c] = seg_raw[c].fillna(seg_raw[c].median())
    X_seg = SCALER_SEG.transform(seg_raw.values)
    master["segment_cluster"] = KMEANS.predict(X_seg)
    master["segment_label"]   = master["segment_cluster"].map(LABEL_MAP)

    # NPS cluster imputation
    for col in ["avg_nps_score","min_nps_score","survey_count","pct_detractor"]:
        med = master.groupby("segment_cluster")[col].transform("median")
        master[col] = master[col].fillna(med).fillna(master[col].median())
    master["has_nps_data"] = master["has_nps_data"].fillna(0).astype(int)
    for col in ["total_tickets","open_tickets","billing_tickets","technical_tickets",
                "critical_tickets","high_tickets","unresolved_ratio","critical_ratio",
                "avg_payment_delay","max_payment_delay"]:
        master[col] = master[col].fillna(0)

    master["plan_enc"]     = LE_PLAN.transform(master["plan_type"])
    master["contract_enc"] = LE_CONTRACT.transform(master["contract_type"])
    for col in ["total_users","monthly_usage_hrs","total_revenue","total_tickets"]:
        master[f"log_{col}"] = np.log1p(master[col])
    master["dunning_per_tenure"]  = master["dunning_count"]      / (master["tenure_days"] / 30).replace(0,1)
    master["usage_per_user"]      = master["monthly_usage_hrs"]   / master["total_users"].replace(0,1)
    master["ticket_per_revenue"]  = master["total_tickets"]       / (master["total_revenue"].replace(0,1)/1000)
    master["adoption_x_usage"]    = master["feature_adoption_pct"] * master["log_monthly_usage_hrs"]
    master["nps_x_dunning"]       = master["avg_nps_score"]       * (master["dunning_count"] + 1)

    # Tabular prediction
    X_tab       = master[FEATURES].values
    tab_proba   = MODEL.predict_proba(X_tab)[:, 1]
    shap_vals   = EXPLAINER.shap_values(master[FEATURES])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=FEATURES)

    # NLP pipeline
    nlp_input = master[["customer_id"]].copy()
    nlp_input = nlp_input.merge(text_per,   on="customer_id", how="left")
    nlp_input = nlp_input.merge(cat_per,    on="customer_id", how="left")
    nlp_input = nlp_input.merge(
        master[["customer_id","avg_nps_score","min_nps_score","survey_count","pct_detractor"]],
        on="customer_id", how="left"
    )
    nlp_input["all_feedback"] = nlp_input["all_feedback"].fillna("")
    nlp_input[CAT_COLS]       = nlp_input[CAT_COLS].fillna(0)
    nlp_input["n_categories"] = (nlp_input[CAT_COLS] > 0).sum(axis=1)
    nlp_input["n_surveys"]    = master["survey_count"].values

    nlp_feats = compute_nlp_features(nlp_input)
    nlp_proba = compute_nlp_proba(nlp_input, nlp_feats)

    # Late fusion: 80% tabular + 20% NLP
    fused_proba = FUSION_ALPHA * tab_proba + (1 - FUSION_ALPHA) * nlp_proba
    fused_score = (fused_proba * 100).round(1)

    # Get segment centroid values for XAI
    centroids_raw = SCALER_SEG.inverse_transform(KMEANS.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_raw, columns=SEG_FEATURES)

    # Build output
    results = []
    for i in range(len(master)):
        row       = master.iloc[i]
        score_val = float(fused_score[i])
        risk_val  = risk_level(score_val)
        seg       = row["segment_label"]
        seg_cl    = int(row["segment_cluster"])
        seg_prof  = next((p for p in SEG_PROFILES if p["segment_label"] == seg), {})
        seg_act   = SEG_ACTIONS.get(seg, {})
        centroid  = centroid_df.iloc[seg_cl].to_dict()

        # Segment RFM context (actual vs segment average)
        seg_rfm_context = {
            "days_since_login":      {"customer": round(float(row.get("days_since_login", 0)), 1),
                                      "segment_avg": round(centroid["days_since_login"], 1)},
            "payment_count":         {"customer": round(float(row.get("payment_count", 0)), 1),
                                      "segment_avg": round(centroid["payment_count"], 1)},
            "total_revenue":         {"customer": round(float(row.get("total_revenue", 0)), 1),
                                      "segment_avg": round(centroid["total_revenue"], 1)},
            "monthly_usage_hrs":     {"customer": round(float(row.get("monthly_usage_hrs", 0)), 1),
                                      "segment_avg": round(centroid["monthly_usage_hrs"], 1)},
            "feature_adoption_pct":  {"customer": round(float(row.get("feature_adoption_pct", 0)), 1),
                                      "segment_avg": round(centroid["feature_adoption_pct"], 1)},
            "avg_nps_score":         {"customer": round(float(row.get("avg_nps_score", 0)), 2),
                                      "segment_avg": round(centroid["avg_nps_score"], 2)},
        }

        results.append({
            # ── Identity ──────────────────────────────────────────────────────
            "customer_id":           row["customer_id"],
            "plan_type":             row["plan_type"],
            "contract_type":         row["contract_type"],

            # ── Churn Score (fused) ───────────────────────────────────────────
            "churn_score":           score_val,
            "churn_proba":           round(float(fused_proba[i]), 4),
            "tabular_proba":         round(float(tab_proba[i]), 4),
            "nlp_proba":             round(float(nlp_proba[i]), 4),
            "risk_level":            risk_val,

            # ── SHAP (tabular model) ──────────────────────────────────────────
            "shap_top5":             get_top_shap(shap_df.iloc[i], top_n=5),

            # ── NLP / Sentiment ───────────────────────────────────────────────
            "sentiment": {
                "label":              nlp_feats["sentiment_label"][i],
                "vader_compound":     round(float(nlp_feats["vader_compound"][i]), 4),
                "vader_neg":          round(float(nlp_feats["vader_neg"][i]), 4),
                "pct_negative_sent":  round(float(nlp_feats["pct_negative_sent"][i]) * 100, 1),
                "urgency_level":      nlp_feats["urgency_level"][i],
                "urgency_score":      int(nlp_feats["urgency_score"][i]),
                "dominant_topic":     nlp_feats["dominant_topic_label"][i],
                "topic_strength":     round(float(nlp_feats["dominant_topic_score"][i]), 3),
                "feedback_preview":   str(nlp_input["all_feedback"].iloc[i])[:300],
            },

            # ── Segmentation ──────────────────────────────────────────────────
            "segment_label":         seg,
            "segment_cluster":       seg_cl,
            "segment_rfm_context":   seg_rfm_context,
            "segment_profile":       seg_prof,
            "segment_actions":       seg_act,
        })

    return results


# ─── XAI Narrative builders ────────────────────────────────────────────────────
def build_churn_xai_prompt(r: dict) -> str:
    """
    Prompt untuk Churn Prediction XAI.
    Menggabungkan: churn score + SHAP factors + sentiment analysis + urgency.
    """
    shap_lines = "\n".join([
        f"  {idx+1}. {f['feature_label']} "
        f"({'meningkatkan' if f['direction'] == 'increases_churn' else 'menurunkan'} risiko, "
        f"SHAP: {f['shap_value']:+.3f})"
        for idx, f in enumerate(r["shap_top5"])
    ])
    sent = r["sentiment"]
    actions = r["segment_actions"]
    retain_opts  = ", ".join(actions.get("retain", []))
    offer_opts   = ", ".join(actions.get("offer", []))

    return f"""Anda adalah analis customer success senior. Tulis analisis dalam Bahasa Indonesia yang jelas dan actionable.

═══ DATA CUSTOMER ═══
ID          : {r['customer_id']}
Plan        : {r['plan_type']} ({r['contract_type']})
Churn Score : {r['churn_score']}/100  →  Risk Level: {r['risk_level']}
Tabular Model Probability : {r['tabular_proba']*100:.1f}%
NLP Model Probability     : {r['nlp_proba']*100:.1f}%

═══ FAKTOR UTAMA DARI MODEL ML (SHAP) ═══
{shap_lines}

═══ ANALISIS SENTIMEN FEEDBACK ═══
Sentimen keseluruhan : {sent['label'].upper()} (VADER: {sent['vader_compound']:+.3f})
Negatif per kalimat  : {sent['pct_negative_sent']:.1f}% kalimat bernada negatif
Urgency level        : {sent['urgency_level'].upper()} (score: {sent['urgency_score']})
Topik utama feedback : {sent['dominant_topic']} (strength: {sent['topic_strength']:.2f})
Preview feedback     : "{sent['feedback_preview'][:200]}"

═══ TULIS ANALISIS DALAM FORMAT INI ═══

**MENGAPA CHURN SCORE {r['churn_score']}/100?**
[2-3 kalimat: jelaskan kombinasi faktor model + sentimen yang menghasilkan score ini]

**FAKTOR RISIKO UTAMA**
[3 poin singkat, masing-masing 1 kalimat, langsung dari data SHAP di atas]

**SINYAL DARI FEEDBACK PELANGGAN**
[1-2 kalimat: apa yang dikatakan feedback mereka, hubungkan dengan sentimen dan topik]

**REKOMENDASI TINDAKAN**
Pilih 1 dari retain: [{retain_opts}]
Pilih 1 dari offer : [{offer_opts}]
[1 kalimat: tindakan spesifik yang paling tepat dan alasannya]

Maksimal 150 kata total. Langsung ke poin, tanpa pembuka."""


def build_segment_xai_prompt(r: dict) -> str:
    """
    Prompt untuk Customer Segmentation XAI.
    Menggabungkan: RFM values vs segment average + NLP topic + sentiment.
    """
    rfm = r["segment_rfm_context"]
    sent = r["sentiment"]
    seg_prof = r["segment_profile"]
    seg_act  = r["segment_actions"]

    rfm_lines = "\n".join([
        f"  {k.replace('_',' ').title():25s}: {v['customer']:>8.1f}  (rata-rata segment: {v['segment_avg']:.1f})"
        for k, v in rfm.items()
    ])

    return f"""Anda adalah analis customer success senior. Tulis analisis dalam Bahasa Indonesia.

═══ PROFIL CUSTOMER ═══
ID          : {r['customer_id']}
Plan        : {r['plan_type']} ({r['contract_type']})
Segment     : {r['segment_label']}
Churn Score : {r['churn_score']}/100

═══ NILAI RFM CUSTOMER vs RATA-RATA SEGMENTNYA ═══
{"Metrik":<25}  {"Customer":>8}  {"Avg Segment":>12}
{rfm_lines}

═══ PROFIL SEGMENT ═══
Jumlah customer  : {seg_prof.get('count', 'N/A')}
Avg churn score  : {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk      : {seg_prof.get('pct_high_risk', 'N/A')}%
Avg revenue      : {seg_prof.get('avg_revenue', 'N/A')}
Deskripsi        : {seg_act.get('description', '')}
Prioritas        : {seg_act.get('priority', '')}

═══ SENTIMEN & TOPIK FEEDBACK ═══
Sentimen    : {sent['label']} ({sent['vader_compound']:+.3f})
Topik utama : {sent['dominant_topic']}
Urgency     : {sent['urgency_level']}

═══ TULIS ANALISIS DALAM FORMAT INI ═══

**MENGAPA MASUK SEGMENT "{r['segment_label']}"?**
[2-3 kalimat: jelaskan kombinasi nilai RFM yang menempatkan customer ini di segment ini vs customer lain]

**KARAKTERISTIK DIBANDING SEGMENT RATA-RATA**
[2-3 poin: di mana customer ini lebih tinggi atau lebih rendah dari rata-rata segmentnya, dan apa artinya]

**APA YANG PERLU DIPERHATIKAN**
[1-2 kalimat: insight khusus dari kombinasi segment + sentimen + topik feedback]

**STRATEGI UNTUK SEGMENT INI**
[1-2 kalimat: pendekatan yang tepat untuk customer di segment ini secara umum]

Maksimal 150 kata total."""


async def call_qwen(prompt: str) -> str:
    """
    Call an OpenAI-compatible chat API (Ollama, OpenRouter, Together, OpenAI, etc.)
    Supports two payload formats:
      - OpenAI / OpenRouter / Together:  top-level max_tokens + temperature
      - Native Ollama (/api/chat):       options.{temperature, num_predict}
    Auto-detects Ollama by checking if OLLAMA_URL contains 'localhost' or
    the path ends with /api, and falls back to /api/chat in that case.
    """
    is_native_ollama = (
        "localhost" in OLLAMA_URL
        or "127.0.0.1" in OLLAMA_URL
        or OLLAMA_URL.rstrip("/").endswith("/api")
    )

    headers = {"Content-Type": "application/json"}
    if OLLAMA_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_KEY}"

    if is_native_ollama:
        # Native Ollama format → POST /api/chat
        endpoint = OLLAMA_URL.rstrip("/")
        if not endpoint.endswith("/api/chat"):
            # strip /v1 if present, append /api/chat
            endpoint = endpoint.replace("/v1", "").rstrip("/") + "/api/chat"
        payload = {
            "model":    OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream":   False,
            "options":  {"temperature": 0.3, "num_predict": 350},
        }
    else:
        # OpenAI-compatible format → POST /chat/completions
        endpoint = f"{OLLAMA_URL}/chat/completions"
        payload = {
            "model":       OLLAMA_MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "max_tokens":  400,
            "temperature": 0.3,
        }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Parse response — handle both OpenAI and native Ollama shapes
        if "choices" in data:
            # OpenAI-compatible shape
            content = data["choices"][0]["message"]["content"]
        elif "message" in data:
            # Native Ollama shape
            content = data["message"]["content"]
        else:
            return f"[XAI unavailable: unexpected response shape: {list(data.keys())}]"

        # Strip <think>…</think> blocks some Qwen models emit
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    except httpx.ConnectError as e:
        return f"[XAI unavailable: cannot connect to {endpoint} — {str(e)[:120]}]"
    except httpx.HTTPStatusError as e:
        return f"[XAI unavailable: HTTP {e.response.status_code} from {endpoint}]"
    except Exception as e:
        return f"[XAI unavailable: {type(e).__name__}: {str(e)[:120]}]"


# ─── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":        "ok",
        "model_version": A.get("model_version", "v1.0"),
        "fusion_alpha":  FUSION_ALPHA,
        "nlp_auc":       N.get("nlp_auc_cv"),
    }


@app.post("/predict")
async def predict(
    customer_accounts:          UploadFile = File(...),
    monthly_usage_metrics:      UploadFile = File(...),
    billing_data:               UploadFile = File(...),
    support_tickets:            UploadFile = File(...),
    nps_surveys_with_feedback:  UploadFile = File(...),
    generate_xai:               bool = True,
):
    """Predict all customers. Returns list with churn + segment XAI per customer."""
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
            # Two separate XAI narratives
            churn_prompt   = build_churn_xai_prompt(r)
            segment_prompt = build_segment_xai_prompt(r)
            r["xai_churn_explanation"]   = await call_qwen(churn_prompt)
            r["xai_segment_explanation"] = await call_qwen(segment_prompt)
    else:
        for r in results:
            r["xai_churn_explanation"]   = None
            r["xai_segment_explanation"] = None

    return {"status": "success", "total_customers": len(results), "predictions": results}


@app.post("/predict/single")
async def predict_single(
    customer_id:                str,
    customer_accounts:          UploadFile = File(...),
    monthly_usage_metrics:      UploadFile = File(...),
    billing_data:               UploadFile = File(...),
    support_tickets:            UploadFile = File(...),
    nps_surveys_with_feedback:  UploadFile = File(...),
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

    # Filter to single customer across all datasets
    for df in [ca_df, um_df, bd_df, st_df, nps_df]:
        mask = df["customer_id"] == customer_id
        df.drop(df[~mask].index, inplace=True)

    results = run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df)
    r = results[0]
    r["xai_churn_explanation"]   = await call_qwen(build_churn_xai_prompt(r))
    r["xai_segment_explanation"] = await call_qwen(build_segment_xai_prompt(r))
    return r