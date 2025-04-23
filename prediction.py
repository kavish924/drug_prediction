"""
Streamlit Drug Prediction Analysis App (Cirrhosis Dataset)
---------------------------------------------------------
This Streamlit app is pre‑wired to the **cirrhosis.csv** dataset you just
provided and walks through a complete drug / survival prediction workflow that
can run locally or in the cloud (Streamlit Community Cloud, AWS, GCP…).

Key Adjustments
===============
* Auto‑loads the uploaded *cirrhosis.csv* (Mayo Clinic Primary Biliary Cirrhosis
  dataset).
* Exposes target‑column selector so you can predict **Drug** (if present) or
  any other outcome column (e.g., *Status* for survival).
* Handles the dataset’s mix of numeric, categorical and missing values.
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Optional models
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

# ----------------------------- Load Cirrhosis dataset ----------------------------- #

def load_cirrhosis() -> pd.DataFrame:
    csv_path = Path(r"C:\Users\kavis\Downloads\cirrhosis.csv")
    if not csv_path.exists():
        st.error("cirrhosis.csv not found. Please upload the file.")
        st.stop()
    df = pd.read_csv(csv_path)
    return df


def preprocess_df(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]

    # Simple missing‑value fill: numeric→median, categorical→mode
    X = X.copy()
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
        else:
            X[c] = X[c].fillna(X[c].mode()[0])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return X, y, preprocessor


def train_model(model_name: str, X, y, pre, params):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier()

    pipe = Pipeline([("prep", pre), ("model", models[model_name])])

    if params:
        grid = GridSearchCV(pipe, params, cv=3, scoring="accuracy", n_jobs=-1)
        grid.fit(X, y)
        return grid.best_estimator_
    pipe.fit(X, y)
    return pipe


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted"),
    }
    try:
        y_prob = model.predict_proba(X_test)
        scores["roc_auc"] = roc_auc_score(y_test, y_prob, multi_class="ovo", average="weighted")
    except Exception:
        pass
    cm = confusion_matrix(y_test, y_pred)
    return scores, cm

# --------------------------------- Streamlit UI --------------------------------- #

st.set_page_config("Cirrhosis Prediction", "💊", layout="wide")
st.title("💊 Cirrhosis Drug/Outcome Prediction")

st.sidebar.header("Configuration")

# Load data automatically
with st.spinner("Loading cirrhosis.csv …"):
    df = load_cirrhosis()

# Target column
target = st.sidebar.selectbox("Target column", options=df.columns, index=len(df.columns)-1)

X, y, pre = preprocess_df(df, target)

# Train/test split params
st.sidebar.subheader("Train/Test Split")
size = st.sidebar.slider("Test size %", 10, 50, 20) / 100
state = st.sidebar.number_input("Random state", 0, 10000, 42)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=size, random_state=42)

# Model choice
st.sidebar.subheader("Model")
models_list = ["Logistic Regression", "Random Forest"] + (["XGBoost"] if XGBClassifier else []) + (["LightGBM"] if LGBMClassifier else [])
model_name = st.sidebar.selectbox("Model", options=models_list)

param_json = st.sidebar.text_area("GridSearch params (JSON)")
params = None
if param_json.strip():
    try:
        params = json.loads(param_json)
    except json.JSONDecodeError as e:
        st.sidebar.error(e)

if st.sidebar.button("Train"):
    with st.spinner("Training model …"):
        model = train_model(model_name, X_tr, y_tr, pre, params)
        scores, cm = evaluate(model, X_te, y_te)
    st.success("Done!")

    # Metrics
    cols = st.columns(len(scores))
    for i, (k, v) in enumerate(scores.items()):
        cols[i].metric(k, f"{v:.3f}")

    # Confusion matrix plot
    # Confusion matrix (suppress tick labels if too many)
    fig, ax = plt.subplots(figsize=(4, 4))
    if len(model.classes_) > 50:
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(ax=ax, colorbar=False, include_values=True)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Confusion matrix (labels hidden, {len(model.classes_)} classes)")
    else:
        disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        disp.plot(ax=ax, colorbar=False, xticks_rotation=90)
        st.pyplot(fig)

    # Download model
    buf = io.BytesIO()
    import joblib
    joblib.dump(model, buf)
    st.download_button("Download model", buf.getvalue(), "model.pkl")

    # Prediction form
    st.subheader("Predict New Case")
    with st.form("predict"):
        inp = {}
        for c in X.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                inp[c] = st.number_input(c, value=float(df[c].median()))
            else:
                inp[c] = st.selectbox(c, sorted(df[c].dropna().unique()))
        if st.form_submit_button("Predict"):
            pred = model.predict(pd.DataFrame([inp]))[0]
            st.success(f"Predicted {target}: **{pred}**")