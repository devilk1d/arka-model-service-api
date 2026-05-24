import os, json, io, warnings, re
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

# ─── Load artifacts ───────────────────────────────────────────────────────────
print("Loading artifacts...")

A = joblib.load(os.getenv("ARTIFACTS_PATH", "churn_artifacts_v1.pkl"))

# v10 saves both best model and calibrated model — prefer calibrated
# The calibrated model gives better probability estimates (lower Brier score)
# and is fitted on the full dataset via CalibratedClassifierCV(cv=5, method='isotonic')
MODEL            = A.get("calibrated_model") or A["model"]
FEATURES         = A["production_features"]
LE_PLAN          = A["le_plan"]
LE_CONTRACT      = A["le_contract"]
SCALER_SEG       = A["scaler_seg"]
KMEANS           = A["kmeans"]
LABEL_MAP        = A["cluster_label_map"]
SEG_FEATURES     = A["seg_features"]               # ['days_since_login','payment_count','log_revenue','log_usage','feature_adoption_pct','avg_nps_score']
SEG_PROFILES     = A["segment_profiles"]
SEG_DESCRIPTIONS = A.get("segment_descriptions", {})  # new in v10
PLAN_ACTIONS     = A.get("plan_actions", {})           # new in v10: per-plan retain/offer options
RISK_LOW         = A["risk_thresholds"]["low"]
RISK_HIGH        = A["risk_thresholds"]["high"]
REF              = pd.Timestamp(A["reference_date"])
CV_VEC           = A["cv_vec"]
LDA              = A["lda"]
TOPIC_NAMES      = A["topic_names"]
URGENCY_LEX      = A["urgency_lexicon"]
N_TOPICS         = A["n_topics"]

# SHAP explainer — v10 stores it directly in the artifact bundle
# For CalibratedClassifierCV we need the underlying base estimator
EXPLAINER = A.get("explainer")
if EXPLAINER is None:
    _base = MODEL.calibrated_classifiers_[0].estimator if hasattr(MODEL, "calibrated_classifiers_") else MODEL
    EXPLAINER = shap.TreeExplainer(_base)

ANALYZER = SentimentIntensityAnalyzer()

LLM_URL   = os.getenv("OLLAMA_URL",      "https://api.ollama.ai/v1")
LLM_KEY   = os.getenv("OLLAMA_API_KEY",  "")
LLM_MODEL = os.getenv("OLLAMA_MODEL",    "qwen3.5:397b-cloud")
FUSION_ALPHA = float(os.getenv("FUSION_ALPHA", "1.0"))  # v10: single calibrated model, no NLP fusion

print("✅ Artifacts loaded (v10 — calibrated model active)")
print(f"   model_version : {A.get('model_version', 'unknown')}")
print(f"   model_name    : {A.get('model_name', 'unknown')}")
print(f"   has_calibrated: {'calibrated_model' in A}")
print(f"   SEG_FEATURES  : {SEG_FEATURES}")
print(f"   FEATURES count: {len(FEATURES)}")
print(f"   RISK_LOW/HIGH : {RISK_LOW} / {RISK_HIGH}")
print(f"   N_TOPICS      : {N_TOPICS}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def risk_level(score: float) -> str:
    return "Low" if score <= RISK_LOW else ("Medium" if score <= RISK_HIGH else "High")


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


# ─── NLP / VADER pipeline (v10: VADER features only, no NLP-LR fusion) ────────
def compute_vader_features(text: str) -> dict:
    """Compute VADER + urgency features for a single customer text (v10 logic)."""
    empty = {k: 0.0 for k in [
        "vader_compound", "vader_pos", "vader_neg", "vader_neu",
        "vader_min_sent", "vader_std_sent", "pct_negative_sent",
        "urgency_score", "avg_words_per_sent",
    ]}
    if not text or pd.isna(text):
        return empty

    doc   = ANALYZER.polarity_scores(str(text))
    sents = [s.strip() for s in re.split(r"[.!?|]+", str(text)) if len(s.strip()) > 10]
    sc    = [ANALYZER.polarity_scores(s)["compound"] for s in sents] if sents else [0.0]

    return {
        "vader_compound":    doc["compound"],
        "vader_pos":         doc["pos"],
        "vader_neg":         doc["neg"],
        "vader_neu":         doc["neu"],
        "vader_min_sent":    min(sc),
        "vader_std_sent":    float(np.std(sc)),
        "pct_negative_sent": sum(1 for s in sc if s < -0.05) / len(sc),
        "urgency_score":     float(sum(1 for w in URGENCY_LEX if w in str(text).lower())),
        "avg_words_per_sent": float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
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
    Compute nlp_red_flag and loyalty_risk_flag exactly as in v10 notebook (Cell 91).
    Requires: vader_compound, urgency_score, churn_score, segment_label, tenure_days
    """
    master = master.copy()
    master["nlp_red_flag"] = (
        (master["vader_compound"] < -0.2) & (master["urgency_score"] >= 1)
    ).astype(int)
    master["loyalty_risk_flag"] = (
        (master["churn_score"] <= RISK_LOW) &
        (master["segment_label"] == "Critical") &
        (master["tenure_days"] > 365)
    ).astype(int)
    return master


# ─── Core pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df):
    """
    Full pipeline aligned with v10 notebook.
    Key differences vs v9:
      - Uses calibrated_model (not raw best model)
      - tenure_capped (<=365) used for dunning_per_tenure interaction feature
      - ticket_per_revenue uses log_total_revenue (+1e-3) denominator (not raw revenue)
      - nps_x_dunning multiplied by has_nps_data mask
      - SEG_FEATURES uses log_revenue / log_usage (pre-computed)
      - avg_words_per_sent included in VADER features
      - nlp_red_flag and loyalty_risk_flag added to output
      - segment_descriptions and plan_actions from artifact
    """
    # ── Clean ──────────────────────────────────────────────────────────────────
    ca_df  = ca_df.copy()
    ca_df["plan_type"]     = ca_df["plan_type"].str.lower().str.strip().str.title()
    ca_df["contract_type"] = ca_df["contract_type"].str.lower().str.strip().str.title()
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

    ca_df["tenure_days"]   = (
        ca_df["unsubscribed_date"].fillna(REF) - ca_df["subscription_date"]
    ).dt.days.clip(lower=1)
    ca_df["tenure_capped"] = ca_df["tenure_days"].clip(upper=365)  # v10: capped tenure for interaction

    # ── Billing features ───────────────────────────────────────────────────────
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

    # ── Usage features ─────────────────────────────────────────────────────────
    uf = um_df.copy()
    uf["days_since_login"] = (REF - uf["last_login_date"]).dt.days
    uf = uf.drop(columns=["last_login_date"], errors="ignore")

    # ── Ticket features ────────────────────────────────────────────────────────
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

    # ── NPS tabular features ───────────────────────────────────────────────────
    nf = nps_df.groupby("customer_id").agg(
        avg_nps_score =("nps_score", "mean"),
        min_nps_score =("nps_score", "min"),
        survey_count  =("survey_id", "count"),
        pct_detractor =("segment",   lambda x: (x == "detractor").mean()),
    ).reset_index()
    nf["has_nps_data"] = 1

    # ── NPS text: per-customer aggregated feedback ─────────────────────────────
    text_per = (nps_df.groupby("customer_id")["feedback_text"]
                .apply(lambda x: " | ".join(x.dropna().astype(str)))
                .reset_index())
    text_per.columns = ["customer_id", "all_feedback"]

    # ── Master merge ───────────────────────────────────────────────────────────
    master = ca_df[["customer_id", "plan_type", "contract_type", "total_users",
                     "tenure_days", "tenure_capped"]].copy()
    master = (master
              .merge(uf, on="customer_id", how="left")
              .merge(bf, on="customer_id", how="left")
              .merge(tf, on="customer_id", how="left")
              .merge(nf, on="customer_id", how="left"))

    # ── Fillna(0) semua kolom numerik sebelum pipeline lanjut ───────────────────
    # Ini mencegah KeyError/NaN error jika sub-table (billing/usage/ticket) kosong
    # setelah left-merge — kolom tetap ada tapi NaN, bukan missing sama sekali.
    # Khusus kolom string/text tidak tersentuh karena select_dtypes(include=number).
    _numeric_cols = master.select_dtypes(include=[np.number]).columns
    master[_numeric_cols] = master[_numeric_cols].fillna(0)

    # ── Segmentation (before imputation, using log_revenue/log_usage) ──────────
    master["log_revenue"] = np.log1p(master["total_revenue"].fillna(0))
    master["log_usage"]   = np.log1p(master["monthly_usage_hrs"].fillna(0))

    seg_raw = master[SEG_FEATURES].copy()   # ['days_since_login','payment_count','log_revenue','log_usage','feature_adoption_pct','avg_nps_score']
    for c in SEG_FEATURES:
        seg_raw[c] = seg_raw[c].fillna(seg_raw[c].median())
    X_seg = SCALER_SEG.transform(seg_raw.values)
    master["segment_cluster"] = KMEANS.predict(X_seg)
    master["segment_label"]   = master["segment_cluster"].map(LABEL_MAP)

    # ── NPS imputation (cluster-median) ────────────────────────────────────────
    for col in ["avg_nps_score", "min_nps_score", "survey_count", "pct_detractor"]:
        med = master.groupby("segment_cluster")[col].transform("median")
        master[col] = master[col].fillna(med).fillna(master[col].median())
    master["has_nps_data"] = master["has_nps_data"].fillna(0).astype(int)

    # ── Zero-fill ticket / payment delay ───────────────────────────────────────
    for col in ["total_tickets", "open_tickets", "billing_tickets", "technical_tickets",
                "critical_tickets", "high_tickets", "unresolved_ratio", "critical_ratio",
                "avg_payment_delay", "max_payment_delay"]:
        master[col] = master[col].fillna(0)

    # ── Encodings ─────────────────────────────────────────────────────────────
    master["plan_enc"]     = LE_PLAN.transform(master["plan_type"])
    master["contract_enc"] = LE_CONTRACT.transform(master["contract_type"])

    # ── Log transforms (fillna before log to avoid NaN propagation) ─────────────
    for col in ["total_users", "monthly_usage_hrs", "total_revenue", "total_tickets"]:
        master[f"log_{col}"] = np.log1p(master[col].fillna(0))

    # ── Interaction features (v10: tenure_capped, log denom, has_nps_data mask) ─
    master["dunning_per_tenure"] = (
        master["dunning_count"].fillna(0) /
        np.where(master["tenure_capped"].fillna(1) / 30 == 0, 1, master["tenure_capped"].fillna(1) / 30)
    )
    master["usage_per_user"] = (
        master["monthly_usage_hrs"].fillna(0) /
        np.where(master["total_users"].fillna(1) == 0, 1, master["total_users"].fillna(1))
    )
    master["ticket_per_revenue"] = (
        master["total_tickets"].fillna(0) / (master["log_total_revenue"].fillna(0) + 1e-3)
    )
    master["adoption_x_usage"] = (
        master["feature_adoption_pct"].fillna(0) * master["log_monthly_usage_hrs"].fillna(0)
    )
    master["nps_x_dunning"] = (
        master["avg_nps_score"].fillna(0) *
        (master["dunning_count"].fillna(0) + 1) *
        master["has_nps_data"].fillna(0)
    )

    # ── VADER / NLP features (v10: per-customer, merged back) ─────────────────
    master = master.merge(text_per, on="customer_id", how="left")
    master["all_feedback"] = master["all_feedback"].fillna("")

    vader_rows = master["all_feedback"].apply(compute_vader_features)
    vader_df   = pd.DataFrame(list(vader_rows))
    for col in vader_df.columns:
        master[col] = vader_df[col].values

    master["urgency_level"]   = master["urgency_score"].apply(
        lambda u: "high" if u >= 3 else ("medium" if u >= 1 else "low")
    )
    master["sentiment_label"] = master["vader_compound"].apply(
        lambda c: "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
    )

    # ── Topic features (LDA) ──────────────────────────────────────────────────
    topic_feats = compute_topic_features(master["all_feedback"].tolist())
    master["dominant_topic_label"] = topic_feats["dominant_topic_label"]
    master["dominant_topic_score"] = topic_feats["dominant_topic_score"]
    X_topics = topic_feats["topic_distribution"]

    # ── Model prediction (calibrated model → better probabilities) ────────────
    X_tab     = master[FEATURES].values
    tab_proba = MODEL.predict_proba(X_tab)[:, 1]

    # ── SHAP (v10: uses the stored explainer which wraps the base estimator) ───
    shap_vals = EXPLAINER.shap_values(master[FEATURES])
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=FEATURES)

    # ── Score & risk ──────────────────────────────────────────────────────────
    churn_score = (tab_proba * 100).round(1)          # numpy ndarray
    master["churn_proba"] = tab_proba.round(4)
    master["churn_score"] = churn_score
    master["risk_level"]  = [risk_level(s) for s in churn_score]  # list comprehension, not .apply()

    # ── NLP flags (v10: Cell 91) ───────────────────────────────────────────────
    master = compute_nlp_flags(master)

    # ── Segment centroid RFM context ──────────────────────────────────────────
    centroids_raw = SCALER_SEG.inverse_transform(KMEANS.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_raw, columns=SEG_FEATURES)

    # ── Build output ──────────────────────────────────────────────────────────
    results = []
    for i in range(len(master)):
        row      = master.iloc[i]
        seg      = row["segment_label"]
        seg_cl   = int(row["segment_cluster"])
        seg_prof = next((p for p in SEG_PROFILES if p["segment_label"] == seg), {})
        centroid = centroid_df.iloc[seg_cl].to_dict()

        # Plan-based retain/offer actions (v10: PLAN_ACTIONS keyed by plan_type)
        plan_act = PLAN_ACTIONS.get(row["plan_type"], {})
        seg_act  = {
            "description": SEG_DESCRIPTIONS.get(seg, ""),
            "retain":      plan_act.get("retain", []),
            "offer":       plan_act.get("offer",  []),
            "priority":    row["risk_level"],
        }

        # centroid keys = SEG_FEATURES = ['days_since_login','payment_count',
        # 'log_revenue','log_usage','feature_adoption_pct','avg_nps_score']
        # total_revenue & monthly_usage_hrs exist in master row but NOT in centroid
        seg_rfm_context = {
            "days_since_login":      {"customer": round(float(row.get("days_since_login", 0)), 1),
                                      "segment_avg": round(float(centroid.get("days_since_login", 0)), 1)},
            "payment_count":         {"customer": round(float(row.get("payment_count", 0)), 1),
                                      "segment_avg": round(float(centroid.get("payment_count", 0)), 1)},
            "total_revenue":         {"customer": round(float(row.get("total_revenue", 0)), 1),
                                      "segment_avg": round(float(np.expm1(centroid.get("log_revenue", 0))), 1)},
            "monthly_usage_hrs":     {"customer": round(float(row.get("monthly_usage_hrs", 0)), 1),
                                      "segment_avg": round(float(np.expm1(centroid.get("log_usage", 0))), 1)},
            "feature_adoption_pct":  {"customer": round(float(row.get("feature_adoption_pct", 0)), 1),
                                      "segment_avg": round(float(centroid.get("feature_adoption_pct", 0)), 1)},
            "avg_nps_score":         {"customer": round(float(row.get("avg_nps_score", 0)), 2),
                                      "segment_avg": round(float(centroid.get("avg_nps_score", 0)), 2)},
        }

        results.append({
            # ── Identity ──────────────────────────────────────────────────────
            "customer_id":          row["customer_id"],
            "plan_type":            row["plan_type"],
            "contract_type":        row["contract_type"],

            # ── Churn Score ────────────────────────────────────────────────────
            "churn_score":          float(row["churn_score"]),
            "churn_proba":          round(float(tab_proba[i]), 4),
            "tabular_proba":        round(float(tab_proba[i]), 4),
            "nlp_proba":            round(float(tab_proba[i]), 4),  # v10: single model, no NLP fusion
            "risk_level":           row["risk_level"],

            # ── SHAP (tabular) ─────────────────────────────────────────────────
            "shap_top5":            get_top_shap(shap_df.iloc[i]),

            # ── NLP / Sentiment ────────────────────────────────────────────────
            "sentiment": {
                "label":             row["sentiment_label"],
                "vader_compound":    round(float(row["vader_compound"]), 4),
                "vader_neg":         round(float(row["vader_neg"]), 4),
                "pct_negative_sent": round(float(row["pct_negative_sent"]) * 100, 1),
                "urgency_level":     row["urgency_level"],
                "urgency_score":     int(row["urgency_score"]),
                "dominant_topic":    row["dominant_topic_label"],
                "topic_strength":    round(float(row["dominant_topic_score"]), 3),
                "feedback_preview":  str(row["all_feedback"])[:300],
            },

            # ── NLP flags (v10 new) ────────────────────────────────────────────
            "nlp_red_flag":         int(row["nlp_red_flag"]),
            "loyalty_risk_flag":    int(row["loyalty_risk_flag"]),

            # ── Segmentation ───────────────────────────────────────────────────
            "segment_label":        seg,
            "segment_cluster":      seg_cl,
            "segment_rfm_context":  seg_rfm_context,
            "segment_profile":      seg_prof,
            "segment_actions":      seg_act,
        })

    return results


# ─── LLM narrative builders ────────────────────────────────────────────────────
def build_churn_xai_prompt(r: dict) -> str:
    """Churn Prediction XAI prompt. Output: strict JSON."""
    shap_lines = "\n".join([
        f"  {idx+1}. {f['feature_label']} "
        f"({'meningkatkan' if f['direction'] == 'increases_churn' else 'menurunkan'} risiko, "
        f"SHAP: {f['shap_value']:+.3f})"
        for idx, f in enumerate(r["shap_top5"])
    ])
    sent    = r["sentiment"]
    rfm     = r["segment_rfm_context"]

    flags = []
    if r.get("nlp_red_flag"):
        flags.append("NLP Red Flag: sentimen sangat negatif + urgency tinggi")
    if r.get("loyalty_risk_flag"):
        flags.append("Loyalty Risk Flag: score rendah tapi segment Critical + tenure >1 tahun")
    flag_section = ", ".join(flags) if flags else "Tidak ada"

    # Customer context for personalized action recommendation
    tenure    = rfm.get("days_since_login", {}).get("customer", 0)
    revenue   = rfm.get("total_revenue", {}).get("customer", 0)
    usage     = rfm.get("monthly_usage_hrs", {}).get("customer", 0)
    adoption  = rfm.get("feature_adoption_pct", {}).get("customer", 0)
    nps       = rfm.get("avg_nps_score", {}).get("customer", 0)

    return f"""Anda adalah analis customer success senior. Balas HANYA dengan JSON valid, tanpa teks lain, tanpa markdown.

CUSTOMER: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']}) | Segment: {r['segment_label']}
Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']} | Probability: {r['churn_proba']*100:.1f}%
Revenue: ${revenue:,.0f}/mo | Usage: {usage:.0f}h/mo | Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10
Days since login: {tenure:.0f} | Flags: {flag_section}

FAKTOR SHAP:
{shap_lines}

SENTIMEN: {sent['label'].upper()} | VADER: {sent['vader_compound']:+.3f} | Urgency: {sent['urgency_level']} ({sent['urgency_score']}) | Topik: {sent['dominant_topic']}
Feedback: "{sent['feedback_preview'][:200]}"

Berikan rekomendasi retain dan offer yang SPESIFIK untuk customer ini berdasarkan plan, revenue, usage, dan sentiment-nya. Jangan gunakan opsi generik. Sesuaikan dengan kondisi aktual customer.

Balas JSON (bahasa Indonesia, singkat dan actionable):
{{"score_reason":"...","risk_factors":["...","...","..."],"feedback_signal":"...","action":{{"retain":"aksi retensi spesifik untuk customer ini","offer":"penawaran spesifik untuk customer ini","reason":"..."}}}}"""


def build_segment_xai_prompt(r: dict) -> str:
    """
    Segment XAI prompt — replaces hardcoded narasi di SegmentationPage.tsx.
    Output: strict JSON dengan data yang bisa langsung ditampilkan di frontend.
    """
    rfm     = r["segment_rfm_context"]
    sent    = r["sentiment"]
    seg_prof = r["segment_profile"]
    seg_act  = r["segment_actions"]

    rfm_lines = "\n".join([
        f"  {k.replace('_', ' ').title()}: customer={v['customer']:.1f}, segment_avg={v['segment_avg']:.1f}"
        for k, v in rfm.items()
    ])

    flags = []
    if r.get("nlp_red_flag"):
        flags.append("NLP Red Flag aktif (sentimen negatif + urgency)")
    if r.get("loyalty_risk_flag"):
        flags.append("Loyalty Risk Flag aktif (tenure lama tapi di segmen kritis)")
    flag_str = ", ".join(flags) if flags else "Tidak ada"

    return f"""Anda adalah analis customer success senior. Balas HANYA dengan JSON valid, tanpa teks lain, tanpa markdown, tanpa komentar.

PROFIL CUSTOMER:
ID: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']})
Segment: {r['segment_label']} | Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}

NILAI RFM vs RATA-RATA SEGMENT:
{rfm_lines}

PROFIL SEGMENT:
Jumlah customer: {seg_prof.get('count', 'N/A')} | Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk: {seg_prof.get('pct_high_risk', 'N/A')}% | Avg tenure: {seg_prof.get('avg_tenure_days', 'N/A')} hari
Deskripsi segmen: {seg_act.get('description', '')}

SENTIMEN & FLAGS:
{sent['label']} (VADER: {sent['vader_compound']:+.3f}) | Topik: {sent['dominant_topic']} | Urgency: {sent['urgency_level']}
Flags aktif: {flag_str}

RECOMMENDED ACTIONS (plan-based):
Retain: {seg_act.get('retain', [])}
Offer: {seg_act.get('offer', [])}

Balas JSON (bahasa Indonesia, singkat):
{{"segment_reason":"...","characteristics":["...","...","..."],"watch_out":"...","strategy":"..."}}"""


def build_segment_cohort_prompt(seg_label: str, seg_prof: dict, seg_desc: str,
                                 retain_actions: list, offer_actions: list,
                                 total_customers: int) -> str:
    """
    Prompt untuk narasi-level segment (bukan per-customer) — digunakan di halaman Segmentation
    untuk menggantikan teks hardcoded di getSegmentDescriptionAndTraits().
    Dipanggil sekali per segment setelah analyze all selesai.
    """
    share_pct = round(seg_prof.get("count", 0) / max(total_customers, 1) * 100, 1)

    return f"""Anda adalah analis customer success senior. Balas HANYA dengan JSON valid, tanpa teks lain, tanpa markdown.

DATA SEGMENT: {seg_label}
Total customers: {seg_prof.get('count', 'N/A')} ({share_pct}% dari keseluruhan)
Avg churn score: {seg_prof.get('avg_churn_score', 'N/A')}/100
% High risk: {seg_prof.get('pct_high_risk', 'N/A')}%
Avg revenue: ${seg_prof.get('avg_revenue', 0):,.0f}/month
Avg usage hrs: {seg_prof.get('avg_usage_hrs', 0):.0f}h/month
Avg NPS: {seg_prof.get('avg_nps', 0):.1f}/10
Avg tenure: {seg_prof.get('avg_tenure_days', 0):.0f} hari
Churn rate: {seg_prof.get('churn_rate', 0)*100:.1f}%
Deskripsi internal: {seg_desc}
Aksi retensi tersedia: {retain_actions}
Penawaran tersedia: {offer_actions}

Balas dengan JSON persis seperti ini:
{{
  "narrative": "3-4 kalimat deskripsi komprehensif kohort ini: karakteristik utama, perilaku, dan situasi bisnis mereka",
  "defining_traits": ["trait singkat 1 (maks 4 kata)", "trait singkat 2", "trait singkat 3", "trait singkat 4"],
  "top_priority_action": "1 kalimat rekomendasi aksi prioritas tertinggi untuk segment ini",
  "risk_summary": "1 kalimat ringkasan risiko churn untuk segment ini"
}}"""


async def call_llm(prompt: str) -> str:
    """
    Call OpenAI-compatible chat API (LLM inference).
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
        return content

    except httpx.ConnectError as e:
        return json.dumps({"error": f"cannot connect to {endpoint}: {str(e)[:100]}"})
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"HTTP {e.response.status_code} from {endpoint}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)[:100]}"})


# ─── Endpoints ─────────────────────────────────────────────────────────────────
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
        results[seg.segment_label] = await call_llm(prompt)
    return {"status": "success", "cohort_xai": results}


@app.get("/health")
def health():
    model_type = type(MODEL).__name__
    has_calibrated = "calibrated_model" in A
    return {
        "status":          "ok",
        "model_version":   A.get("model_version", "v10.0"),
        "model_type":      model_type,
        "calibrated":      has_calibrated,
        "model_name":      A.get("model_name", "unknown"),
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
            r["xai_churn_explanation"]   = await call_llm(build_churn_xai_prompt(r))
            r["xai_segment_explanation"] = await call_llm(build_segment_xai_prompt(r))

        # ── Per-segment cohort narratives (new in v10) ─────────────────────────
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
            segment_cohort_xai[seg] = await call_llm(cohort_prompt)

        # Attach cohort XAI to every customer row in that segment
        for r in results:
            r["xai_segment_cohort"] = segment_cohort_xai.get(r["segment_label"])

    else:
        for r in results:
            r["xai_churn_explanation"]   = None
            r["xai_segment_explanation"] = None
            r["xai_segment_cohort"]      = None

    return {"status": "success", "total_customers": len(results), "predictions": results}


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
    r["xai_churn_explanation"]   = await call_llm(build_churn_xai_prompt(r))
    r["xai_segment_explanation"] = await call_llm(build_segment_xai_prompt(r))
    r["xai_segment_cohort"]      = await call_llm(
        build_segment_cohort_prompt(
            seg_label       = r["segment_label"],
            seg_prof        = r["segment_profile"],
            seg_desc        = r["segment_actions"].get("description", ""),
            retain_actions  = r["segment_actions"].get("retain", []),
            offer_actions   = r["segment_actions"].get("offer", []),
            total_customers = 1,
        )
    )
    return r


# ─── Simulation: Churn Trajectory ────────────────────────────────────────────

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
    nlp_red_flag:        int = 0
    loyalty_risk_flag:   int = 0


class SimulateRequest(BaseModel):
    customer_data: _CustomerDataSim
    scenario:      str        = ""   # optional: intervention scenario for projection
    chat_history:  list[dict] = []   # [{question, narrative}] — prior turns


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
    ctx = (
        f"CUSTOMER: {c.customer_id} | Plan: {c.plan_type} ({c.contract_type}) | Segment: {c.segment_label}\n"
        f"Churn Score: {c.churn_score:.1f}/100 | Risk Level: {c.risk_level}\n"
        f"Revenue: ${rev:,.0f}/mo | Usage: {usage:.0f}h/mo | Feature Adoption: {adoption:.0f}% | NPS: {nps:.1f}/10\n"
        f"Days since last login: {dsl:.0f}\n"
        f"Sentiment: {c.sentiment.get('label', 'N/A')} "
        f"(VADER: {c.sentiment.get('vader_compound', 0):+.3f}) | "
        f"Urgency: {c.sentiment.get('urgency_level', 'N/A')}\n"
        f"Feedback preview: \"{c.sentiment.get('feedback_preview', '')[:200]}\"\n"
        f"NLP Red Flag: {'Yes' if c.nlp_red_flag else 'No'} | "
        f"Loyalty Risk: {'Yes' if c.loyalty_risk_flag else 'No'}\n\n"
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


_SIM_JSON_SYSTEM = """\
You are a customer retention analytics engine. Given a customer's churn data, \
produce a precise JSON churn trajectory forecast. No prose — output ONLY valid JSON.

The JSON schema (all fields required):
{
  "baseline": [
    {"week": 0, "prob": <ACTUAL_CHURN_SCORE_0_TO_100>},
    {"week": 2, "prob": <float 0-100>},
    {"week": 4, "prob": <float 0-100>},
    {"week": 6, "prob": <float 0-100>},
    {"week": 8, "prob": <float 0-100>},
    {"week": 10, "prob": <float 0-100>},
    {"week": 12, "prob": <float 0-100>}
  ],
  "projection": null,
  "retention_window_weeks": <int: estimated weeks before churn probability exceeds 80>,
  "revenue_at_risk": <float: monthly_revenue * (churn_prob/100) * 3>,
  "confidence": <float 0.0-1.0>,
  "intervention_impact_pct": null,
  "segment_migration": [
    {"label": "Churned", "prob": <float 0.0-1.0>},
    {"label": "High Risk", "prob": <float 0.0-1.0>},
    {"label": "At Risk", "prob": <float 0.0-1.0>},
    {"label": "Retained", "prob": <float 0.0-1.0>}
  ]
}

Rules:
- baseline[0].prob MUST equal the actual churn_score from the customer data exactly.
- If a scenario is provided, also fill "projection" with the same 7-point structure \
  (week 0 same as baseline[0]), and fill "intervention_impact_pct" with the estimated \
  churn reduction percentage (positive number).
- segment_migration probs must sum to 1.0.
- retention_window_weeks: how many weeks until baseline churn probability crosses 80. \
  If already above 80, set to 0. If it never crosses 80 in 12 weeks, set to 13.
- revenue_at_risk: monthly revenue × (week-6 churn_prob / 100) × 3.
"""

_SIM_NARRATIVE_SYSTEM = """\
You are a senior customer success analyst writing a concise retention briefing. \
Write 3-4 sentences in Indonesian (Bahasa Indonesia) that summarise:
1. The customer's current churn risk situation
2. The key drivers behind this trajectory
3. The most important recommended action

Be direct and specific. Reference actual numbers from the data. No bullet points — \
flowing prose only. Max 120 words.
"""


@app.post("/simulate")
async def simulate(request: SimulateRequest):
    """
    Churn trajectory simulation SSE.
    Step 1: call_llm → emit 'data' event with chart/metrics JSON.
    Step 2: stream_llm → emit 'token' events for narrative.
    Step 3: emit 'done'.
    """
    c   = request.customer_data
    ctx = _build_ctx(c, request.scenario)

    # Append prior chat turns for follow-up scenarios
    history_block = ""
    if request.chat_history:
        lines = [
            f"Q: {turn.get('question', '')}\nA (summary): {str(turn.get('narrative', ''))[:300]}"
            for turn in request.chat_history[-4:]
        ]
        history_block = "\n\nPREVIOUS CONVERSATION TURNS:\n" + "\n\n".join(lines)

    async def event_stream():
        # ── Step 1: Structured JSON data ─────────────────────────────────────
        yield f"data: {json.dumps({'type': 'thinking'})}\n\n"

        sim_data: dict = {}
        try:
            raw = await call_llm(
                _SIM_JSON_SYSTEM,
                f"{ctx}{history_block}\n\nGenerate the churn trajectory JSON now.",
                max_tokens=900,
            )
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                sim_data = json.loads(cleaned[start:end])
            else:
                raise ValueError("no JSON object found in LLM response")

            # Ensure week-0 is pinned to actual churn_score
            if sim_data.get("baseline") and len(sim_data["baseline"]) > 0:
                sim_data["baseline"][0]["prob"] = round(c.churn_score, 2)

        except Exception as exc:
            # Fallback: synthetic trajectory based on churn_score
            base_score = c.churn_score
            decay = 1.05 if base_score > 60 else 1.02
            sim_data = {
                "baseline": [
                    {"week": 0,  "prob": round(base_score, 2)},
                    {"week": 2,  "prob": round(min(100, base_score * decay),       2)},
                    {"week": 4,  "prob": round(min(100, base_score * decay**2),    2)},
                    {"week": 6,  "prob": round(min(100, base_score * decay**3),    2)},
                    {"week": 8,  "prob": round(min(100, base_score * decay**4),    2)},
                    {"week": 10, "prob": round(min(100, base_score * decay**5),    2)},
                    {"week": 12, "prob": round(min(100, base_score * decay**6),    2)},
                ],
                "projection": None,
                "retention_window_weeks": max(0, int((80 - base_score) / (base_score * (decay - 1) + 0.5))),
                "revenue_at_risk": round(
                    c.segment_rfm_context.get("total_revenue", {}).get("customer", 0)
                    * (min(100, base_score * decay**3) / 100) * 3, 2
                ),
                "confidence": 0.6,
                "intervention_impact_pct": None,
                "segment_migration": [
                    {"label": "Churned",   "prob": round(base_score / 100 * 0.7,        3)},
                    {"label": "High Risk", "prob": round(base_score / 100 * 0.2,        3)},
                    {"label": "At Risk",   "prob": round((1 - base_score / 100) * 0.4,  3)},
                    {"label": "Retained",  "prob": round((1 - base_score / 100) * 0.6,  3)},
                ],
                "_error": str(exc)[:120],
            }

        yield f"data: {json.dumps({'type': 'data', 'payload': sim_data})}\n\n"

        # ── Step 2: Streaming narrative ───────────────────────────────────────
        scenario_note = (
            f"\n\nScenario being evaluated: {request.scenario}"
            if request.scenario.strip() else ""
        )
        narrative_prompt = (
            f"{ctx}{scenario_note}{history_block}"
            f"\n\nChurn trajectory summary: week-0={c.churn_score:.1f}, "
            f"week-12={sim_data.get('baseline', [{}])[-1].get('prob', c.churn_score):.1f}, "
            f"retention_window={sim_data.get('retention_window_weeks', '?')} weeks, "
            f"revenue_at_risk=${sim_data.get('revenue_at_risk', 0):,.0f}."
        )

        full_narrative = ""
        async for token in stream_llm(_SIM_NARRATIVE_SYSTEM, narrative_prompt, max_tokens=250):
            full_narrative += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'narrative': full_narrative})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )

