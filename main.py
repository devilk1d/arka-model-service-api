import os, json, io, warnings, re
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import shap
import httpx

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Churn Prediction API", version="2.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ─── Load artifacts ───────────────────────────────────────────────────────────
print("Loading artifacts")
A = joblib.load(os.getenv("ARTIFACTS_PATH", "model/churn_artifacts_v61.pkl"))

MODEL        = A["model"]
FEATURES     = A["production_features"]
LE_PLAN      = A["le_plan"]
LE_CONTRACT  = A["le_contract"]
SCALER_SEG   = A["scaler_seg"]
KMEANS       = A["kmeans"]
LABEL_MAP    = A["cluster_label_map"]
SEG_FEATURES = A["seg_features"]
SEG_PROFILES = A["segment_profiles"]
SEG_DESCS    = A["segment_descriptions"]
RISK_LOW     = A["risk_thresholds"]["low"]
RISK_HIGH    = A["risk_thresholds"]["high"]
REF          = pd.Timestamp(A["reference_date"])
EXPLAINER    = shap.TreeExplainer(MODEL)

CV_VEC       = A["cv_vec"]
LDA          = A["lda"]
TOPIC_NAMES  = A["topic_names"]
URGENCY_LEX  = A["urgency_lexicon"]

ANALYZER     = SentimentIntensityAnalyzer()

OLLAMA_URL   = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
OLLAMA_KEY   = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gpt-4o-mini")

# Plan-aware action options untuk Qwen (v2.2+)
# PLAN_ACTIONS & RISK_HIGH_BIZ dimuat dari artifact (source of truth = notebook v6.1)
PLAN_ACTIONS  = A.get("plan_actions", {})
RISK_HIGH_BIZ = A["risk_thresholds"].get("high_biz", RISK_HIGH)

print(f"Artifacts loaded | model v{A.get('model_version','?')} | "
      f"k={KMEANS.n_clusters} clusters | RISK_LOW={RISK_LOW} | RISK_HIGH={RISK_HIGH} | RISK_HIGH_BIZ={RISK_HIGH_BIZ}")

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

def compute_vader_features(text: str) -> dict:
    """Hitung VADER multi-level per customer. Handles empty/None text safely."""
    empty = {k: 0.0 for k in ["vader_compound","vader_pos","vader_neg","vader_neu",
                                "vader_min_sent","vader_std_sent","pct_negative_sent",
                                "vader_range","urgency_score","avg_words_per_sent"]}
    # FIX: robust empty check — jangan proses string kosong atau "0"
    if not text or not isinstance(text, str) or text.strip() == "" or text.strip() == "0":
        return empty
    doc   = ANALYZER.polarity_scores(str(text))
    sents = [s.strip() for s in re.split(r"[.!?|]+", str(text)) if len(s.strip()) > 10]
    sc    = [ANALYZER.polarity_scores(s)["compound"] for s in sents] if sents else [0.0]
    return {
        "vader_compound":     doc["compound"],
        "vader_pos":          doc["pos"],
        "vader_neg":          doc["neg"],
        "vader_neu":          doc["neu"],
        "vader_min_sent":     min(sc),
        "vader_std_sent":     float(np.std(sc)),
        "pct_negative_sent":  sum(1 for s in sc if s < -0.05) / len(sc),
        "vader_range":        max(sc) - min(sc),
        "urgency_score":      sum(1 for w in URGENCY_LEX if w in str(text).lower()),
        "avg_words_per_sent": float(np.mean([len(s.split()) for s in sents])) if sents else 0.0,
    }

# ─── Core pipeline ─────────────────────────────────────────────────────────────
def run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df):
    """
    Full production pipeline — tabular LightGBM + NLP XAI/flag.
    v2.3 fixes:
      - FIX NO-FEEDBACK: all_feedback NaN tidak di-fillna(0) lagi → string "0" hilang
      - FIX NO-FEEDBACK: customer tanpa NPS dapat fallback sentiment yang tepat
      - FIX LOYALTY FLAG: loyalty_risk_flag ditambahkan di output
    """
    # ── Normalisasi casing ────────────────────────────────────────────────────
    ca_df = ca_df.copy()
    ca_df["plan_type"]     = ca_df["plan_type"].str.capitalize().str.strip()
    ca_df["contract_type"] = ca_df["contract_type"].str.capitalize().str.strip()
    nps_df = nps_df.copy()
    nps_df["nps_score"] = nps_df["nps_score"].clip(lower=0)

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

    ca_df["tenure_days"]  = (
        ca_df["unsubscribed_date"].fillna(REF) - ca_df["subscription_date"]
    ).dt.days.clip(lower=1)
    ca_df["tenure_capped"] = ca_df["tenure_days"].clip(upper=365)

    # ── Feature engineering ────────────────────────────────────────────────
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

    nf = nps_df.groupby("customer_id").agg(
        avg_nps_score=("nps_score","mean"), min_nps_score=("nps_score","min"),
        survey_count=("survey_id","count"),
        pct_detractor=("segment",lambda x:(x=="detractor").mean()),
    ).reset_index(); nf["has_nps_data"] = 1

    master = ca_df[["customer_id","plan_type","contract_type","total_users","tenure_days","tenure_capped"]].copy()
    master = (master
        .merge(uf, on="customer_id", how="left")
        .merge(bf, on="customer_id", how="left")
        .merge(tf, on="customer_id", how="left")
        .merge(nf, on="customer_id", how="left")
    )

    # Log-transform sebelum segmentasi
    master["log_revenue"] = np.log1p(master["total_revenue"].fillna(0))
    master["log_usage"]   = np.log1p(master["monthly_usage_hrs"].fillna(0))

    # ── Segmentation (sebelum imputation) ─────────────────────────────────────
    seg_raw = master[SEG_FEATURES].copy()
    for c in SEG_FEATURES:
        # FIX KMEANS NaN: fillna(median) lalu fillna(0) sebagai safety net.
        # Ketika predict/single dipanggil untuk customer tanpa NPS, nps_df kosong
        # sehingga avg_nps_score = NaN. median() dari series yang ALL NaN = NaN juga,
        # sehingga fillna(median) tidak membantu. fillna(0) sebagai fallback akhir
        # memastikan tidak ada NaN yang masuk ke KMEANS.predict().
        col_median = seg_raw[c].median()
        seg_raw[c] = seg_raw[c].fillna(col_median if pd.notna(col_median) else 0.0)
    # Pastikan tidak ada NaN tersisa sebelum transform (defence-in-depth)
    seg_raw = seg_raw.fillna(0.0)
    X_seg = SCALER_SEG.transform(seg_raw.values)
    master["segment_cluster"] = KMEANS.predict(X_seg)
    master["segment_label"]   = master["segment_cluster"].map(LABEL_MAP)

    # ── Imputation ────────────────────────────────────────────────────────────
    for col in ["avg_nps_score","min_nps_score","survey_count","pct_detractor"]:
        med = master.groupby("segment_cluster")[col].transform("median")
        master[col] = master[col].fillna(med).fillna(master[col].median())
    master["has_nps_data"] = master["has_nps_data"].fillna(0).astype(int)

    fill_0_cols = ["total_tickets","open_tickets","billing_tickets","technical_tickets",
                   "critical_tickets","high_tickets","unresolved_ratio","critical_ratio",
                   "avg_payment_delay","max_payment_delay"]
    master[fill_0_cols] = master[fill_0_cols].fillna(0)

    # FIX NO-FEEDBACK: hanya isi NaN untuk kolom NUMERIC — jangan sentuh string/text
    # Ini mencegah all_feedback NaN ter-fillna(0) menjadi integer 0
    numeric_cols = master.select_dtypes(include=[np.number]).columns
    master[numeric_cols] = master[numeric_cols].fillna(0)

    # ── Encode + transforms ──────────────────────────────────────────────────
    master["plan_enc"]     = LE_PLAN.transform(master["plan_type"])
    master["contract_enc"] = LE_CONTRACT.transform(master["contract_type"])
    for col in ["total_users","monthly_usage_hrs","total_revenue","total_tickets"]:
        master[f"log_{col}"] = np.log1p(master[col])
    master["dunning_per_tenure"]  = master["dunning_count"]      / (master["tenure_capped"] / 30).replace(0,1)
    master["usage_per_user"]      = master["monthly_usage_hrs"]   / master["total_users"].replace(0,1)
    master["ticket_per_revenue"]  = master["total_tickets"]       / (master["total_revenue"].replace(0,1)/1000)
    master["adoption_x_usage"]    = master["feature_adoption_pct"] * master["log_monthly_usage_hrs"]
    master["nps_x_dunning"]       = master["avg_nps_score"]       * (master["dunning_count"] + 1)

    # ── Tabular prediction ────────────────────────────────────────────────────
    master = master.reset_index(drop=True)
    X_tab     = master[FEATURES].values
    tab_proba = MODEL.predict_proba(X_tab)[:, 1]
    shap_vals = EXPLAINER.shap_values(master[FEATURES])
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    shap_df = pd.DataFrame(shap_vals, columns=FEATURES)

    # ── NLP pipeline — XAI context & risk flag only ──────────────────────────
    # FIX NO-FEEDBACK: hanya buat text_per untuk customer yang punya NPS
    # Customer tanpa NPS tidak masuk nlp_df → after merge hasilnya NaN (bukan 0)
    text_per = nps_df.groupby("customer_id")["feedback_text"].apply(
        lambda x: " | ".join(x.dropna().astype(str))
    ).reset_index()
    text_per.columns = ["customer_id", "all_feedback"]

    # Bersihkan feedback: string kosong atau hanya whitespace → None
    text_per["all_feedback"] = text_per["all_feedback"].apply(
        lambda t: t.strip() if isinstance(t, str) and t.strip() else None
    )
    # Hanya proses customer yang punya feedback
    text_per_valid = text_per[text_per["all_feedback"].notna()].copy()

    if len(text_per_valid) > 0:
        nlp_feats_series = text_per_valid["all_feedback"].apply(compute_vader_features)
        nlp_df = pd.DataFrame(list(nlp_feats_series))
        nlp_df["customer_id"] = text_per_valid["customer_id"].values
        nlp_df["all_feedback"] = text_per_valid["all_feedback"].values
        nlp_df["sentiment_label"] = nlp_df["vader_compound"].apply(
            lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral"))
        nlp_df["urgency_level"] = nlp_df["urgency_score"].apply(
            lambda u: "high" if u >= 3 else ("medium" if u >= 1 else "low"))

        X_counts = CV_VEC.transform(nlp_df["all_feedback"].fillna("").tolist())
        X_topics = LDA.transform(X_counts)
        nlp_df["dominant_topic"]       = X_topics.argmax(axis=1)
        nlp_df["dominant_topic_label"] = nlp_df["dominant_topic"].map(TOPIC_NAMES)
        nlp_df["dominant_topic_score"] = X_topics.max(axis=1)

        master = master.merge(nlp_df[[
            "customer_id","sentiment_label","urgency_level","urgency_score",
            "vader_compound","vader_neg","pct_negative_sent","vader_min_sent",
            "avg_words_per_sent","dominant_topic_label","dominant_topic_score","all_feedback",
        ]], on="customer_id", how="left")
    else:
        # Tidak ada feedback sama sekali — tambahkan kolom kosong
        for col in ["sentiment_label","urgency_level","dominant_topic_label","all_feedback"]:
            master[col] = None
        for col in ["urgency_score","vader_compound","vader_neg","pct_negative_sent",
                    "vader_min_sent","avg_words_per_sent","dominant_topic_score"]:
            master[col] = 0.0

    # FIX NO-FEEDBACK: fillna yang tepat per tipe kolom
    # Numeric NLP columns → 0
    for c in ["urgency_score","vader_compound","vader_neg","pct_negative_sent",
              "vader_min_sent","avg_words_per_sent","dominant_topic_score"]:
        master[c] = master[c].fillna(0)
    # String/label NLP columns → meaningful fallback (bukan 0)
    master["sentiment_label"]      = master["sentiment_label"].fillna("unknown")
    master["urgency_level"]        = master["urgency_level"].fillna("low")
    master["dominant_topic_label"] = master["dominant_topic_label"].fillna("No Feedback")
    # FIX NO-FEEDBACK: all_feedback tetap None/NaN untuk customer tanpa feedback
    # Jangan fillna("") atau fillna(0) — biarkan None sehingga feedback_preview bisa cek

    tab_proba   = np.nan_to_num(tab_proba, nan=0.0)
    fused_score = (tab_proba * 100).round(1)
    fused_score = np.nan_to_num(fused_score, nan=0.0)

    master["nlp_red_flag"] = (
        (fused_score <= RISK_HIGH) &
        (master["vader_compound"] < -0.2) &
        (master["urgency_score"] >= 1)
    ).astype(int)

    # LOYALTY RISK FLAG: Low churn score TAPI Critical segment + tenure panjang
    # Menangkap "silent at-risk" — customer loyal yang sudah tidak puas
    master["loyalty_risk_flag"] = (
        (fused_score <= RISK_LOW) &
        (master["segment_label"] == "Critical") &
        (master["tenure_days"] > 365)
    ).astype(int)

    centroids_raw = SCALER_SEG.inverse_transform(KMEANS.cluster_centers_)
    centroid_df   = pd.DataFrame(centroids_raw, columns=SEG_FEATURES)

    # Hanya replace inf, jangan fillna semua kolom — agar all_feedback tetap None
    master  = master.replace([np.inf, -np.inf], np.nan)
    shap_df = shap_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    # Numeric-only fillna di master
    num_cols = master.select_dtypes(include=[np.number]).columns
    master[num_cols] = master[num_cols].fillna(0)

    results = []
    for i in range(len(master)):
        row      = master.iloc[i]
        score_val= float(fused_score[i])
        risk_val = risk_level(score_val)
        seg      = row["segment_label"]
        seg_cl   = int(row["segment_cluster"])
        seg_prof = next((p for p in SEG_PROFILES if p["segment_label"] == seg), {})
        centroid = centroid_df.iloc[seg_cl].to_dict()

        # RFM context — log features di-expm1 untuk display UI
        seg_rfm_context = {}
        rfm_display_map = {
            "days_since_login":     "days_since_login",
            "payment_count":        "payment_count",
            "log_revenue":          "total_revenue",
            "total_revenue":        "total_revenue",
            "log_usage":            "monthly_usage_hrs",
            "monthly_usage_hrs":    "monthly_usage_hrs",
            "feature_adoption_pct": "feature_adoption_pct",
            "avg_nps_score":        "avg_nps_score",
        }
        for feat, centroid_val in centroid.items():
            display_key = rfm_display_map.get(feat, feat)
            if feat == "log_revenue":
                cust_val = round(float(np.expm1(row.get(feat, 0))), 1)
                seg_avg  = round(float(np.expm1(centroid_val)), 1)
            elif feat == "log_usage":
                cust_val = round(float(np.expm1(row.get(feat, 0))), 1)
                seg_avg  = round(float(np.expm1(centroid_val)), 1)
            else:
                cust_val = round(float(row.get(feat, 0)), 2 if "nps" in feat else 1)
                seg_avg  = round(float(centroid_val), 2 if "nps" in feat else 1)
            seg_rfm_context[display_key] = {"customer": cust_val, "segment_avg": seg_avg}

        # FIX NO-FEEDBACK: feedback_preview hanya diisi jika ada feedback nyata
        raw_feedback = row.get("all_feedback", None)
        has_feedback = isinstance(raw_feedback, str) and raw_feedback.strip() not in ("", "0")
        feedback_preview = raw_feedback[:300] if has_feedback else ""

        results.append({
            "customer_id":          row["customer_id"],
            "plan_type":            row["plan_type"],
            "contract_type":        row["contract_type"],
            "churn_score":          score_val,
            "churn_proba":          round(float(tab_proba[i]), 4),
            "tabular_proba":        round(float(tab_proba[i]), 4),
            "nlp_proba":            None,
            "risk_level":           risk_val,
            "shap_top5":            get_top_shap(shap_df.iloc[i], top_n=5),
            "nlp_red_flag":         int(row["nlp_red_flag"]),
            "loyalty_risk_flag":    int(row["loyalty_risk_flag"]),
            "has_nps_data":         int(row["has_nps_data"]),
            "tenure_days":          int(row.get("tenure_days", 0)),
            "sentiment": {
                "label":             row["sentiment_label"],
                "vader_compound":    round(float(row.get("vader_compound", 0)), 4),
                "vader_neg":         round(float(row.get("vader_neg", 0)), 4),
                "pct_negative_sent": round(float(row.get("pct_negative_sent", 0)) * 100, 1),
                "urgency_level":     row["urgency_level"],
                "urgency_score":     int(row.get("urgency_score", 0)),
                "dominant_topic":    row["dominant_topic_label"],
                "topic_strength":    round(float(row.get("dominant_topic_score", 0)), 3),
                "feedback_preview":  feedback_preview,
            },
            "segment_label":        seg,
            "segment_cluster":      seg_cl,
            "segment_rfm_context":  seg_rfm_context,
            "segment_profile":      seg_prof,
            "segment_description":  SEG_DESCS.get(seg, ""),
        })

    return results


# ─── XAI Prompt Builders ───────────────────────────────────────────────────────
def build_churn_xai_prompt(r: dict) -> str:
    """
    Prompt untuk XAI churn prediction.
    v2.3: Menangani customer tanpa feedback NPS dengan konteks yang tepat.
    """
    shap_lines = "\n".join([
        f"  {idx+1}. {f['feature_label']} "
        f"({'meningkatkan' if f['direction'] == 'increases_churn' else 'menurunkan'} risiko, "
        f"SHAP: {f['shap_value']:+.3f})"
        for idx, f in enumerate(r["shap_top5"])
    ])
    sent = r["sentiment"]
    ml_prob_str  = f"{r['tabular_proba']*100:.1f}%"
    red_flag_str = " [NLP RED FLAG - feedback sangat negatif]" if r.get("nlp_red_flag") else ""
    loyalty_str  = " [LOYALTY RISK - Low churn score tapi Critical segment + loyal lama]" if r.get("loyalty_risk_flag") else ""

    plan_key    = r["plan_type"] if r["plan_type"] in PLAN_ACTIONS else "Professional"
    plan_opts   = PLAN_ACTIONS[plan_key]
    retain_opts = plan_opts["retain"]
    offer_opts  = plan_opts["offer"]

    rfm_lines = "\n".join([
        f"  {k.replace('_',' ').title()}: customer={v['customer']}, avg={v['segment_avg']}"
        for k, v in r["segment_rfm_context"].items()
    ])

    # FIX NO-FEEDBACK: sentimen section disesuaikan jika customer tidak punya feedback
    has_feedback = bool(r.get("has_nps_data", 0)) and bool(sent.get("feedback_preview", "").strip())
    if has_feedback:
        sentiment_section = f"""SENTIMEN FEEDBACK:
Label: {sent['label'].upper()} | VADER: {sent['vader_compound']:+.3f} | Kalimat negatif: {sent['pct_negative_sent']:.1f}%
Urgency: {sent['urgency_level'].upper()} (score: {sent['urgency_score']}) | Topik utama: {sent['dominant_topic']}
Preview: "{sent['feedback_preview'][:200]}"
"""
    else:
        sentiment_section = """SENTIMEN FEEDBACK:
Tidak ada data feedback NPS untuk customer ini. Gunakan data SHAP dan profil RFM sebagai basis utama rekomendasi.
"""

    return f"""Anda adalah analis customer success senior yang berpengalaman. Balas HANYA dengan JSON valid, tanpa teks lain, tanpa markdown, tanpa komentar.

DATA CUSTOMER:
ID: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']}) | Churn Score: {r['churn_score']}/100 | Risk: {r['risk_level']}{red_flag_str}{loyalty_str}
ML Probability: {ml_prob_str}

FAKTOR CHURN (SHAP):
{shap_lines}

{sentiment_section}
PROFIL SEGMEN: {r.get('segment_description', '')}

KONTEKS RFM vs RATA-RATA SEGMEN:
{rfm_lines}

OPSI RETENSI (pilih atau adaptasi yang paling sesuai untuk plan {r['plan_type']}):
{chr(10).join(f'  - {o}' for o in retain_opts)}

OPSI PENAWARAN (pilih atau adaptasi yang paling sesuai untuk plan {r['plan_type']}):
{chr(10).join(f'  - {o}' for o in offer_opts)}

Berdasarkan SELURUH data di atas, pilih atau adaptasi tindakan yang PALING TEPAT untuk customer INI secara spesifik.

Balas dengan JSON persis seperti ini (dalam bahasa Indonesia, singkat dan actionable):
{{
  "score_reason": "2-3 kalimat mengapa churn score setinggi ini berdasarkan SHAP + konteks",
  "risk_factors": [
    "1 kalimat faktor risiko terbesar dari SHAP",
    "1 kalimat faktor risiko kedua",
    "1 kalimat insight dari sentimen atau konteks RFM"
  ],
  "feedback_signal": "1-2 kalimat tentang sinyal kepuasan atau risiko yang tersirat dari data yang tersedia",
  "action": {{
    "retain": "tindakan retensi terbaik untuk customer INI (boleh adaptasi dari opsi di atas)",
    "offer": "penawaran terbaik untuk customer INI (boleh adaptasi dari opsi di atas)",
    "reason": "1 kalimat mengapa kombinasi ini paling tepat untuk profil customer ini"
  }}
}}"""


def build_segment_xai_prompt(r: dict) -> str:
    """Prompt untuk XAI segmentasi customer. v2.3: loyalty_risk_flag aware."""
    rfm = r["segment_rfm_context"]
    sent = r["sentiment"]
    seg_prof = r["segment_profile"]
    loyalty_note = "\nCATATAN: Customer ini memiliki Loyalty Risk Flag — Low churn tapi masuk Critical segment dengan tenure panjang. Sertakan insight ini dalam strategi." if r.get("loyalty_risk_flag") else ""

    rfm_lines = "\n".join([
        f"  {k.replace('_',' ').title()}: customer={v['customer']:.1f}, segment_avg={v['segment_avg']:.1f}"
        for k, v in rfm.items()
    ])

    return f"""Anda adalah analis customer success senior. Balas HANYA dengan JSON valid, tanpa teks lain, tanpa markdown, tanpa komentar.

PROFIL CUSTOMER:
ID: {r['customer_id']} | Plan: {r['plan_type']} ({r['contract_type']})
Segment: {r['segment_label']} | Churn Score: {r['churn_score']}/100
Deskripsi segment: {r.get('segment_description', '')}{loyalty_note}

NILAI RFM vs RATA-RATA SEGMENT:
{rfm_lines}

PROFIL SEGMENT:
Jumlah customer: {seg_prof.get('count','N/A')} | Avg churn: {seg_prof.get('avg_churn_score','N/A')}/100
% High risk: {seg_prof.get('pct_high_risk','N/A')}%

SENTIMEN: {sent['label']} (VADER: {sent['vader_compound']:+.3f}) | Topik: {sent['dominant_topic']} | Urgency: {sent['urgency_level']}

Balas dengan JSON persis seperti ini (bahasa Indonesia):
{{
  "segment_reason": "2-3 kalimat mengapa customer masuk segment ini berdasarkan nilai RFM",
  "characteristics": [
    "1 kalimat perbedaan customer vs rata-rata segment #1",
    "1 kalimat perbedaan customer vs rata-rata segment #2",
    "1 kalimat perbedaan customer vs rata-rata segment #3"
  ],
  "watch_out": "1-2 kalimat insight khusus dari kombinasi segment + sentimen + topik",
  "strategy": "1-2 kalimat pendekatan terbaik untuk customer di segment ini"
}}"""


async def call_qwen(prompt: str) -> str:
    """Call OpenAI-compatible chat API."""
    _raw_url = os.getenv("OLLAMA_URL", "https://api.openai.com/v1")
    if not _raw_url.startswith("http"):
        _raw_url = f"https://{_raw_url}"
    base_url = _raw_url.rstrip("/")

    is_native_ollama = (
        "localhost" in base_url or "127.0.0.1" in base_url
        or base_url.rstrip("/").endswith("/api")
    )

    headers = {"Content-Type": "application/json"}
    if OLLAMA_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_KEY}"

    if is_native_ollama:
        endpoint = base_url.replace("/v1", "").rstrip("/") + "/api/chat"
        payload  = {
            "model":   OLLAMA_MODEL,
            "messages":[{"role":"user","content":prompt}],
            "stream":  False, "format": "json",
            "options": {"temperature":0.2, "num_predict":600},
        }
    else:
        endpoint = f"{base_url}/chat/completions"
        payload  = {
            "model":    OLLAMA_MODEL,
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": 700,
            "temperature": 0.2,
            "response_format": {"type":"json_object"},
        }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(endpoint, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        if "choices" in data:
            content = data["choices"][0]["message"]["content"]
        elif "message" in data:
            content = data["message"]["content"]
        else:
            return json.dumps({"error": f"unexpected shape: {list(data.keys())}"})
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
@app.get("/health")
def health():
    return {
        "status":           "ok",
        "model_version":    A.get("model_version", "v2.1"),
        "pipeline_version": "v2.4",
        "nlp_role":         A.get("nlp_role", "xai_context_and_flag"),
        "seg_features":     SEG_FEATURES,
        "n_clusters":       int(KMEANS.n_clusters),
        "risk_thresholds":  {"low": RISK_LOW, "high": RISK_HIGH},
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
            r["xai_churn_explanation"]   = await call_qwen(build_churn_xai_prompt(r))
            r["xai_segment_explanation"] = await call_qwen(build_segment_xai_prompt(r))
    else:
        for r in results:
            r["xai_churn_explanation"]   = None
            r["xai_segment_explanation"] = None

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
    """Single customer — full XAI untuk churn prediction dan segmentasi."""
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

    # Filter semua dataframe ke satu customer saja
    ca_df  = ca_df[ca_df["customer_id"] == customer_id].copy()
    um_df  = um_df[um_df["customer_id"] == customer_id].copy()
    bd_df  = bd_df[bd_df["customer_id"] == customer_id].copy()
    st_df  = st_df[st_df["customer_id"] == customer_id].copy()
    nps_df = nps_df[nps_df["customer_id"] == customer_id].copy()

    results = run_full_pipeline(ca_df, um_df, bd_df, st_df, nps_df)
    r = results[0]
    r["xai_churn_explanation"]   = await call_qwen(build_churn_xai_prompt(r))
    r["xai_segment_explanation"] = await call_qwen(build_segment_xai_prompt(r))
    return r