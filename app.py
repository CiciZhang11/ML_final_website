"""
Streamlit app: Resume Screening with XGBoost.
- Option 1: Upload CSV for batch predictions
- Option 2: Fill form for single resume, get Accept/Reject + improvement suggestion
"""
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

EDUCATION_OPTIONS = ["B.Sc", "B.Tech", "M.Tech", "MBA", "PhD"]
JOB_ROLE_OPTIONS = ["AI Researcher", "Cybersecurity Analyst", "Data Scientist", "Software Engineer"]


def load_artifacts():
    """Load model and preprocessors."""
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    encoder = joblib.load(os.path.join(MODELS_DIR, "onehot_encoder.joblib"))
    tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    category_importance = {}
    train_stats = {}
    if os.path.exists(os.path.join(MODELS_DIR, "category_importance.json")):
        with open(os.path.join(MODELS_DIR, "category_importance.json")) as f:
            category_importance = json.load(f)
    if os.path.exists(os.path.join(MODELS_DIR, "train_stats.json")):
        with open(os.path.join(MODELS_DIR, "train_stats.json")) as f:
            train_stats = json.load(f)
    return model, scaler, encoder, tfidf, feature_columns, category_importance, train_stats


def preprocess_new_data(df: pd.DataFrame, encoder, tfidf, feature_columns: list) -> pd.DataFrame:
    """Preprocess new CSV or single row to match training pipeline."""
    X = df.copy()
    for col in ["Resume_ID", "Name", "Recruiter Decision", "AI Score (0-100)"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    required = ["Skills", "Experience (Years)", "Education", "Job Role",
                "Salary Expectation ($)", "Projects Count"]
    missing = [c for c in required if c not in X.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if "Certifications" not in X.columns:
        X["Certifications"] = ""
    X["Certifications"] = X["Certifications"].fillna("").astype(str)
    X["Skills"] = X["Skills"].fillna("").astype(str)

    # Outlier clipping (skip for single row - use values as-is; clipping needs distribution)
    for col in ["Experience (Years)", "Projects Count", "Salary Expectation ($)"]:
        if col in X.columns and len(X) > 1 and X[col].dtype in [np.float64, np.int64]:
            Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            IQR = max(Q3 - Q1, 1e-6)
            lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            X[col] = X[col].clip(lb, ub)

    # OneHot
    cat_cols = ["Education", "Job Role"]
    enc_arr = encoder.transform(X[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)

    # TF-IDF
    X["text_all"] = (X["Skills"].fillna("") + " " + X["Certifications"].fillna("")).str.strip()
    tfidf_arr = tfidf.transform(X["text_all"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_arr, columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()], index=X.index)
    X = pd.concat([X.drop(columns=["Skills", "Certifications", "text_all"]), tfidf_df], axis=1)

    # Feature engineering
    X["experience_per_project"] = X["Experience (Years)"] / (X["Projects Count"] + 1)
    X["projects_per_year"] = X["Projects Count"] / (X["Experience (Years)"] + 1)
    X["log_salary_expectation"] = np.log1p(X["Salary Expectation ($)"])

    for c in feature_columns:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_columns]
    return X


def get_improvement_suggestion(pred: int, values: dict, category_importance: dict, train_stats: dict) -> str:
    """Suggest which feature to improve when rejected. Prioritizes high-importance features where user is weak."""
    if pred == 1:
        return ""

    exp = values.get("Experience (Years)", 0)
    projects = values.get("Projects Count", 0)
    salary = values.get("Salary Expectation ($)", 0)
    skills = str(values.get("Skills", "")).strip()
    certs = str(values.get("Certifications", "")).strip()

    # Build (priority, message) - lower priority number = suggest first
    candidates = []
    if category_importance:
        for cat, imp in category_importance.items():
            if imp < 0.01:
                continue
            if cat == "Experience" and exp < train_stats.get("Experience_median", 4):
                candidates.append((2 - imp, f"**Experience** ({exp} yrs) — Median for accepted is ~{train_stats.get('Experience_median', 4):.0f} yrs. Consider internships or project-based experience."))
            elif cat == "Projects Count" and projects < train_stats.get("Projects_median", 5):
                candidates.append((3 - imp, f"**Projects Count** ({projects}) — Median for accepted is ~{train_stats.get('Projects_median', 5):.0f}. Add more portfolio projects."))
            elif cat == "Skills & Certifications" and (len(skills) < 20 or len(certs) < 5):
                candidates.append((4 - imp, "**Skills & Certifications** — Add more relevant skills and industry certifications (e.g., AWS Certified, Google ML, Deep Learning)."))
            elif cat == "Education":
                candidates.append((5 - imp, "**Education** — Higher degrees (e.g., M.Tech, PhD) tend to perform better for technical roles."))
            elif cat == "Job Role":
                candidates.append((6 - imp, "**Job Role fit** — Ensure skills align closely with the target role."))
            elif cat == "Salary" and salary > train_stats.get("Salary_median", 80000) * 1.2:
                candidates.append((7 - imp, "**Salary expectation** — May be above typical range; consider aligning with market rates."))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # Fallback when no category stats
    if exp < 2:
        return "**Experience** — Gain more work experience or showcase project-based experience."
    return "**Skills & Certifications** — Add industry-recognized skills (Python, ML, etc.) and certs (AWS, Google ML)."


def run_csv_mode(model, scaler, encoder, tfidf, feature_columns):
    """Option 1: CSV upload for batch predictions."""
    st.subheader("Option 1: Upload CSV")
    st.markdown("Upload a CSV with resume data. Required columns: Skills, Experience (Years), Education, "
                "Certifications, Job Role, Salary Expectation ($), Projects Count.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="csv_upload")
    if uploaded is None:
        st.info("Upload a CSV to start.")
        return

    df = pd.read_csv(uploaded)
    st.dataframe(df.head(20), use_container_width=True)

    try:
        X = preprocess_new_data(df, encoder, tfidf, feature_columns)
        X_scaled = scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return

    result = df[["Name"]].copy() if "Name" in df.columns else pd.DataFrame(index=df.index)
    if "Name" not in df.columns:
        result["Name"] = [f"Resume {i+1}" for i in range(len(df))]
    result["Prediction"] = ["Accept" if p == 1 else "Reject" for p in preds]
    result["P(Hire)"] = np.round(probs, 3)
    if "Recruiter Decision" in df.columns:
        result["Actual"] = df["Recruiter Decision"].values

    st.subheader("Predictions")
    st.dataframe(result, use_container_width=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Accept", (preds == 1).sum())
    col2.metric("Reject", (preds == 0).sum())
    col3.metric("Total", len(df))
    st.download_button("Download predictions (CSV)", result.to_csv(index=False), "predictions.csv", "text/csv", key="dl_csv")


def run_form_mode(model, scaler, encoder, tfidf, feature_columns, category_importance, train_stats):
    """Option 2: Form inputs for single resume + improvement suggestion."""
    st.subheader("Option 2: Single Resume Form")
    st.markdown("Fill in the fields below to get an Accept/Reject prediction and a suggestion on which feature to improve.")

    with st.form("resume_form"):
        skills = st.text_area("Skills (comma-separated)", placeholder="e.g., Python, Machine Learning, SQL")
        certs = st.text_input("Certifications (optional)", placeholder="e.g., AWS Certified, Google ML")
        exp = st.slider("Experience (Years)", 0, 20, 3)
        education = st.selectbox("Education", EDUCATION_OPTIONS)
        job_role = st.selectbox("Job Role", JOB_ROLE_OPTIONS)
        salary = st.number_input("Salary Expectation ($)", min_value=30000, max_value=200000, value=80000, step=5000)
        projects = st.slider("Projects Count", 0, 15, 5)
        submitted = st.form_submit_button("Get Prediction")

    if not submitted:
        return

    values = {
        "Skills": skills or "",
        "Certifications": certs or "",
        "Experience (Years)": exp,
        "Education": education,
        "Job Role": job_role,
        "Salary Expectation ($)": salary,
        "Projects Count": projects,
    }
    df = pd.DataFrame([values])

    try:
        X = preprocess_new_data(df, encoder, tfidf, feature_columns)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0, 1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    st.divider()
    result_text = "**Accept**" if pred == 1 else "**Reject**"
    st.metric("Prediction", result_text)
    st.caption(f"P(Hire) = {prob:.2%}")

    if pred == 0:
        suggestion = get_improvement_suggestion(pred, values, category_importance, train_stats)
        st.info(f"💡 **Suggestion:** {suggestion}")


def main():
    st.set_page_config(page_title="Resume Screening", page_icon="📋", layout="wide")
    st.title("📋 Resume Screening with XGBoost")

    if not os.path.exists(os.path.join(MODELS_DIR, "xgboost_model.joblib")):
        st.error("Model not found. Run `python train_and_export.py` first.")
        return

    try:
        model, scaler, encoder, tfidf, feature_columns, category_importance, train_stats = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    mode = st.radio(
        "Choose mode",
        ["Option 1: Upload CSV", "Option 2: Fill Form (Single Resume)"],
        horizontal=True,
    )

    if "CSV" in mode:
        run_csv_mode(model, scaler, encoder, tfidf, feature_columns)
    else:
        run_form_mode(model, scaler, encoder, tfidf, feature_columns, category_importance, train_stats)


if __name__ == "__main__":
    main()
