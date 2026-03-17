"""
Train XGBoost on resume screening data and export model + preprocessors.
Run this once before using the Streamlit app.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DATA_PATH = os.path.join(SCRIPT_DIR, "AI_Resume_Screening.csv")


def load_and_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same preprocessing as Data_Cleaning notebook."""
    X = df.copy()

    # Drop identifiers, target, and AI Score
    for col in ["Resume_ID", "Name", "Recruiter Decision", "AI Score (0-100)"]:
        if col in X.columns:
            X = X.drop(columns=[col])

    X["Certifications"] = X["Certifications"].fillna("").astype(str)
    for col in ["Skills"]:
        if col in X.columns:
            X[col] = X[col].fillna("").astype(str)

    # Outlier clipping on numeric columns
    numeric_cols = ["Experience (Years)", "Projects Count", "Salary Expectation ($)"]
    numeric_cols = [c for c in numeric_cols if c in X.columns]
    for col in numeric_cols:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        X[col] = X[col].clip(lb, ub)

    return X


def fit_preprocessors_and_transform(X: pd.DataFrame, fit: bool = True, 
                                    encoder=None, tfidf=None) -> tuple:
    """OneHot encode + TF-IDF + feature engineering."""
    X = X.copy()

    # OneHot: Education, Job Role
    cat_cols = ["Education", "Job Role"]
    cat_cols = [c for c in cat_cols if c in X.columns]
    if fit:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        enc_arr = encoder.fit_transform(X[cat_cols])
    else:
        enc_arr = encoder.transform(X[cat_cols])
    enc_df = pd.DataFrame(enc_arr, columns=encoder.get_feature_names_out(cat_cols), index=X.index)
    X = pd.concat([X.drop(columns=cat_cols), enc_df], axis=1)

    # TF-IDF on Skills + Certifications
    text_cols = ["Skills", "Certifications"]
    text_cols = [c for c in text_cols if c in X.columns]
    for col in text_cols:
        X[col] = X[col].fillna("").astype(str)
    X["text_all"] = X[text_cols].agg(" ".join, axis=1)
    if fit:
        tfidf = TfidfVectorizer(lowercase=True, stop_words="english", max_features=5000, ngram_range=(1, 2))
        tfidf_arr = tfidf.fit_transform(X["text_all"]).toarray()
    else:
        tfidf_arr = tfidf.transform(X["text_all"]).toarray()
    tfidf_df = pd.DataFrame(tfidf_arr, columns=[f"tfidf_{t}" for t in tfidf.get_feature_names_out()], index=X.index)
    X = pd.concat([X.drop(columns=text_cols + ["text_all"]), tfidf_df], axis=1)

    # Feature engineering
    if "Experience (Years)" in X.columns and "Projects Count" in X.columns:
        X["experience_per_project"] = X["Experience (Years)"] / (X["Projects Count"] + 1)
        X["projects_per_year"] = X["Projects Count"] / (X["Experience (Years)"] + 1)
    if "Salary Expectation ($)" in X.columns:
        X["log_salary_expectation"] = np.log1p(X["Salary Expectation ($)"])

    return X, encoder, tfidf


def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # Target
    y = df["Recruiter Decision"].map({"Hire": 1, "Reject": 0})
    X_raw = load_and_preprocess(df.drop(columns=["Recruiter Decision"]))

    # Fit preprocessors on full data
    X_processed, encoder, tfidf = fit_preprocessors_and_transform(X_raw, fit=True)

    # Ensure column order is consistent
    feature_columns = list(X_processed.columns)

    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler = MinMaxScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

    # Train XGBoost
    print("Training XGBoost...")
    model = XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train_scaled, y_train)

    # Feature importance by user-facing category (for suggestions)
    imp = model.feature_importances_
    feat_names = feature_columns
    category_map = {
        "Experience": [f for f in feat_names if "Experience" in f or "experience_per_project" in f or "projects_per_year" in f],
        "Education": [f for f in feat_names if "Education" in f],
        "Job Role": [f for f in feat_names if "Job Role" in f],
        "Skills & Certifications": [f for f in feat_names if f.startswith("tfidf_")],
        "Salary": [f for f in feat_names if "Salary" in f or "log_salary" in f],
        "Projects Count": [f for f in feat_names if f == "Projects Count"],
    }
    category_importance = {}
    for cat, cols in category_map.items():
        idx = [feat_names.index(c) for c in cols if c in feat_names]
        category_importance[cat] = float(imp[idx].sum()) if idx else 0.0
    category_importance = dict(sorted(category_importance.items(), key=lambda x: -x[1]))

    # Stats from accepted candidates (for comparison)
    accepted = X_train.loc[y_train[y_train == 1].index]
    train_stats = {
        "Experience_median": float(accepted["Experience (Years)"].median()),
        "Projects_median": float(accepted["Projects Count"].median()),
        "Salary_median": float(accepted["Salary Expectation ($)"].median()),
    }

    # Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(encoder, os.path.join(MODELS_DIR, "onehot_encoder.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    with open(os.path.join(MODELS_DIR, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    with open(os.path.join(MODELS_DIR, "category_importance.json"), "w") as f:
        json.dump(category_importance, f, indent=2)
    with open(os.path.join(MODELS_DIR, "train_stats.json"), "w") as f:
        json.dump(train_stats, f, indent=2)

    # Quick eval
    acc = (model.predict(X_test_scaled) == y_test).mean()
    print(f"Test accuracy: {acc:.4f}")
    print(f"Saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
