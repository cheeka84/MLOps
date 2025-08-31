import os, json
from pathlib import Path

import joblib
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="Tourism Package Predictor", page_icon="🧳", layout="centered")

# --------- defaults (edit as you like) ---------
DEFAULT_MODEL_REPO = "cheeka84/tourism-package-pred"   # HF model repo with joblib pipeline
DEFAULT_DATASET_REPO = "cheeka84/tourism-package-pred" # HF dataset repo with test CSVs
CANDIDATE_MODEL_FILES = ["best_pipeline.joblib", "rf_pipeline.joblib", "xgb_pipeline.joblib"]

# --------- caching helpers ---------
@st.cache_resource(show_spinner=False)
def load_model_from_hf(model_repo: str, candidates: list[str]):
    last_err = None
    for fname in candidates:
        try:
            path = hf_hub_download(repo_id=model_repo, repo_type="model", filename=fname)
            m = joblib.load(path)
            return m, fname, None
        except Exception as e:
            last_err = e
    return None, None, last_err

@st.cache_data(show_spinner=False)
def list_repo_csvs(repo_id: str, repo_type: str = "dataset"):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    except RepositoryNotFoundError as e:
        return [], str(e)
    csvs = [f for f in files if f.lower().endswith(".csv")]
    return csvs, None

@st.cache_data(show_spinner=False)
def download_csv(repo_id: str, filename: str, repo_type: str = "dataset") -> pd.DataFrame:
    p = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
    return pd.read_csv(p)

# --------- model feature + class helpers ---------
def find_column_transformer(pipeline) -> ColumnTransformer | None:
    try:
        for obj in getattr(pipeline, "named_steps", {}).values():
            if isinstance(obj, ColumnTransformer):
                return obj
    except Exception:
        pass
    if isinstance(pipeline, ColumnTransformer):
        return pipeline
    return None

def get_expected_cols_from_model(pipeline) -> list[str] | None:
    ct = find_column_transformer(pipeline)
    if ct is not None:
        cols = getattr(ct, "feature_names_in_", None)
        if cols is not None:
            return list(cols)
    cols = getattr(pipeline, "feature_names_in_", None)
    return list(cols) if cols is not None else None

def get_classes(pipeline) -> np.ndarray:
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        try:
            _, last = list(pipeline.named_steps.items())[-1]
            classes = getattr(last, "classes_", None)
        except Exception:
            pass
    if classes is None:
        raise RuntimeError("Model has no classes_. Cannot map predict_proba.")
    return np.asarray(classes)

def detect_positive_class_index(classes: np.ndarray) -> int:
    try:
        if 1 in classes:
            return int(np.where(classes == 1)[0][0])
    except Exception:
        pass
    for label in ("Yes", "Purchased", "WillPurchase", "Positive", "True"):
        try:
            if label in classes:
                return int(np.where(classes == label)[0][0])
        except Exception:
            pass
    try:
        return int(np.argmax(classes))
    except Exception:
        return 1 if len(classes) > 1 else 0

def smart_find_target_col(df: pd.DataFrame, user_target: str | None):
    if user_target and user_target in df.columns:
        return user_target
    for c in ["ProdTaken", "ProductTaken", "Purchase", "Purchased", "WillPurchase", "target", "label", "y"]:
        if c in df.columns:
            return c
    raise ValueError("Target column not found. Provide it explicitly.")

# ========================= UI =========================
st.title("🧳 Tourism Package Purchase — Online & Batch")
with st.sidebar:
    st.subheader("Repositories")
    model_repo = st.text_input("HF Model Repo", value=DEFAULT_MODEL_REPO)
    dataset_repo = st.text_input("HF Dataset Repo", value=DEFAULT_DATASET_REPO)
    st.caption("If your model/data are private, set HF_TOKEN in environment or Secrets.")

# Load model once
model, loaded_model_file, last_err = load_model_from_hf(model_repo, CANDIDATE_MODEL_FILES)
if model is None:
    st.error(f"Could not load a model from {model_repo}. Last error: {last_err}")
    st.stop()

expected_cols = get_expected_cols_from_model(model)
classes = get_classes(model)

tab_online, tab_batch = st.tabs(["🔮 Online Prediction", "📦 Batch Evaluation"])

# ========================= ONLINE PREDICTION =========================
with tab_online:
    st.caption(f"Loaded model: **{loaded_model_file}** from **{model_repo}**")
    if expected_cols:
        with st.expander("debug: expected input columns (from training schema)"):
            st.write(expected_cols)

    threshold = st.slider(
        "decision threshold (probability ≥ threshold ⇒ purchase)",
        min_value=0.05, max_value=0.95, value=0.50, step=0.01
    )

    with st.form("online_form"):
        st.subheader("Enter customer details")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=30, step=1)
            citytier = st.selectbox("CityTier", [1, 2, 3], index=0)
            durationofpitch = st.number_input("DurationOfPitch (minutes)", min_value=0, max_value=500, value=30, step=1)
            occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Small Business", "Large Business", "Student", "Unemployed"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            typeofcontact = st.selectbox("TypeofContact", ["Self Enquiry", "Company Invited"])
            designation = st.selectbox("Designation", ["Executive", "Senior Manager", "Manager", "AVP", "VP"])
            monthlyincome = st.number_input("MonthlyIncome", min_value=0.0, value=50000.0, step=100.0)
        with col2:
            numberofpersonvisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=20, value=2, step=1)
            numberoffollowups = st.number_input("NumberOfFollowups", min_value=0, max_value=20, value=2, step=1)
            productpitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
            preferredpropertystar = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5], index=2)
            maritalstatus = st.selectbox("MaritalStatus", ["Married", "Single", "Divorced", "Unmarried"])
            numberoftrips = st.number_input("NumberOfTrips", min_value=0, max_value=50, value=2, step=1)
            passport = st.checkbox("Passport (has passport)")
            pitchsatisfactionscore = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5], index=3)
            owncar = st.checkbox("OwnCar (has car)")
            numberofchildrenvisiting = st.number_input("NumberOfChildrenVisiting", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = {c: np.nan for c in expected_cols} if expected_cols else {}
        row.update({
            "Age": age, "CityTier": citytier, "DurationOfPitch": durationofpitch,
            "Occupation": occupation, "Gender": gender, "TypeofContact": typeofcontact,
            "Designation": designation, "MonthlyIncome": monthlyincome,
            "NumberOfPersonVisiting": numberofpersonvisiting, "NumberOfFollowups": numberoffollowups,
            "ProductPitched": productpitched, "PreferredPropertyStar": preferredpropertystar,
            "MaritalStatus": maritalstatus, "NumberOfTrips": numberoftrips,
            "Passport": 1 if passport else 0, "PitchSatisfactionScore": pitchsatisfactionscore,
            "OwnCar": 1 if owncar else 0, "NumberOfChildrenVisiting": numberofchildrenvisiting,
        })
        X = pd.DataFrame([row])
        if expected_cols:
            for c in expected_cols:
                if c not in X.columns:
                    X[c] = np.nan
            X = X[expected_cols]

        try:
            if not hasattr(model, "predict_proba"):
                raise RuntimeError("Model does not support predict_proba().")
            proba = model.predict_proba(X)[0]  # (n_classes,)
            if len(classes) > 2:
                order = np.argsort(proba)[::-1]
                topk = [(str(classes[i]), float(proba[i])) for i in order[:3]]
                pred_idx = int(order[0])
                pred_label = str(classes[pred_idx])
                st.subheader("Prediction")
                st.write(f"predicted class: **{pred_label}**")
                st.write("top probabilities:")
                for cls, p in topk:
                    st.write(f"- {cls}: {p:.3f}")
            else:
                pos_idx = detect_positive_class_index(classes)
                pos_label = classes[pos_idx]
                pos_proba = float(proba[pos_idx])
                pred = int(pos_proba >= threshold)
                label_text = "Will Purchase" if pred == 1 else "Won't Purchase"
                st.subheader("Prediction")
                st.metric(f"P({pos_label})", f"{pos_proba:.3f}")
                st.write(f"threshold = **{threshold:.2f}** → predicted: **{label_text}**")
                neg_idx = 1 - pos_idx
                st.caption(f"P({classes[neg_idx]}): {float(proba[neg_idx]):.3f}")

            with st.expander("debug: raw class probabilities"):
                st.dataframe(pd.DataFrame({"class": classes, "probability": proba}))
        except Exception as e:
            st.error(f"prediction failed: {e}")

# ========================= BATCH EVALUATION =========================
with tab_batch:
    st.caption("Fetch test data from HF Dataset repo, predict in batch, and compare with ground truth.")
    csvs, err = list_repo_csvs(dataset_repo, repo_type="dataset")
    if err:
        st.error(err)
        st.stop()

    mode = st.radio("Select input format", ["X/y files", "Single test.csv"], index=0, horizontal=True)

    if mode == "X/y files":
        # Defaults if present
        default_x = csvs.index("X_test.csv") if "X_test.csv" in csvs else 0
        default_y = csvs.index("y_test.csv") if "y_test.csv" in csvs else 0
        x_file = st.selectbox("X_test file", csvs, index=default_x if csvs else 0)
        y_file = st.selectbox("y_test file", csvs, index=default_y if csvs else 0)
        target_col = None
    else:
        default_test = csvs.index("test.csv") if "test.csv" in csvs else 0
        test_file = st.selectbox("test.csv (features + target)", csvs, index=default_test if csvs else 0)
        target_col = st.text_input("Target column (if auto-detect fails)", value="ProdTaken")
        x_file = y_file = None

    use_threshold = False
    if len(classes) == 2:
        use_threshold = st.checkbox("Use threshold for binary prediction", value=False,
                                    help="If checked, predictions use P(pos) ≥ threshold instead of model.predict().")
        if use_threshold:
            threshold_batch = st.slider("Batch decision threshold", 0.05, 0.95, 0.5, 0.01)
        else:
            threshold_batch = 0.5

    upload_back = st.checkbox("Upload results to model repo under eval/", value=False)
    run = st.button("Run Batch Evaluation", type="primary")

    if run:
        try:
            if mode == "X/y files":
                X = download_csv(dataset_repo, x_file, repo_type="dataset")
                y_df = download_csv(dataset_repo, y_file, repo_type="dataset")
                if y_df.shape[1] == 1:
                    y = y_df.iloc[:, 0]
                elif "y" in y_df.columns:
                    y = y_df["y"]
                else:
                    raise ValueError("y_test must have a single column or a column named 'y'.")
            else:
                df = download_csv(dataset_repo, test_file, repo_type="dataset")
                tgt = smart_find_target_col(df, target_col)
                y = df[tgt]
                X = df.drop(columns=[tgt])

            # align to training schema
            if expected_cols:
                for c in expected_cols:
                    if c not in X.columns:
                        X[c] = np.nan
                X = X[expected_cols]

            # predictions
            if use_threshold and len(classes) == 2 and hasattr(model, "predict_proba"):
                proba_mat = model.predict_proba(X)
                pos_idx = detect_positive_class_index(classes)
                y_proba = proba_mat[:, pos_idx]
                y_pred = (y_proba >= threshold_batch).astype(int)
            else:
                y_pred = model.predict(X)
                # proba for reporting
                y_proba = None
                if hasattr(model, "predict_proba"):
                    proba_mat = model.predict_proba(X)
                    if len(classes) == 2:
                        pos_idx = detect_positive_class_index(classes)
                        y_proba = proba_mat[:, pos_idx]
                    else:
                        y_proba = proba_mat

            # metrics
            metrics = {}
            metrics["accuracy"] = float(accuracy_score(y, y_pred))
            if len(classes) == 2:
                metrics["f1_binary"] = float(f1_score(y, y_pred))
                if isinstance(y_proba, np.ndarray) and y_proba.ndim == 1:
                    try:
                        metrics["roc_auc"] = float(roc_auc_score(y, y_proba))
                    except Exception:
                        pass
            else:
                metrics["f1_weighted"] = float(f1_score(y, y_pred, average="weighted"))
                if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2:
                    try:
                        metrics["roc_auc_ovr"] = float(roc_auc_score(y, y_proba, multi_class="ovr"))
                    except Exception:
                        pass

            metrics["classification_report"] = classification_report(y, y_pred, output_dict=True, zero_division=0)
            cm = confusion_matrix(y, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            metrics["model_file"] = loaded_model_file
            metrics["classes"] = classes.tolist()

            # show results
            st.subheader("Metrics")
            st.json(metrics)

            st.subheader("Confusion Matrix")
            st.dataframe(pd.DataFrame(cm, index=classes, columns=classes))

            st.subheader("Sample Predictions")
            pred_df = pd.DataFrame({"y_true": y})
            pred_df["y_pred"] = y_pred
            if isinstance(y_proba, np.ndarray):
                if y_proba.ndim == 1:
                    pred_df["proba_pos"] = y_proba
                else:
                    for i, cls in enumerate(classes):
                        pred_df[f"proba_{cls}"] = y_proba[:, i]
            st.dataframe(pred_df.head(25))

            # downloads
            st.download_button("Download predictions CSV", pred_df.to_csv(index=False).encode("utf-8"),
                               file_name="batch_predictions.csv", mime="text/csv")
            st.download_button("Download metrics JSON", json.dumps(metrics, indent=2).encode("utf-8"),
                               file_name="batch_metrics.json", mime="application/json")

            # optional upload back to model repo/eval
            if upload_back:
                api = HfApi(token=os.getenv("HF_TOKEN"))
                try:
                    # save temp
                    out_dir = Path("artifacts/eval"); out_dir.mkdir(parents=True, exist_ok=True)
                    preds_path = out_dir / "batch_predictions.csv"
                    metrics_path = out_dir / "batch_metrics.json"
                    pred_df.to_csv(preds_path, index=False)
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)

                    api.upload_file(repo_id=model_repo, repo_type="model",
                                    path_or_fileobj=str(preds_path), path_in_repo=f"eval/{preds_path.name}")
                    api.upload_file(repo_id=model_repo, repo_type="model",
                                    path_or_fileobj=str(metrics_path), path_in_repo=f"eval/{metrics_path.name}")
                    st.success(f"Uploaded to {model_repo}/eval/")
                except HfHubHTTPError as e:
                    st.warning(f"Upload failed (is repo private? need HF_TOKEN?). Error: {e}")

        except Exception as e:
            st.error(f"Batch evaluation failed: {e}")
