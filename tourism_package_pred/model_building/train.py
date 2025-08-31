# trains rf and xgb, compares on test, uploads the best model to HF
import os
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import xgboost as xgb
import joblib

from huggingface_hub import hf_hub_download, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier

# ---------------- config (all lowercase) ----------------
data_repo_id = "cheeka84/tourism-package-pred"   # HF dataset repo
data_repo_type = "dataset"
prefix = "prepared"                              # folder in dataset repo

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise SystemExit("please set HF_TOKEN in the environment (do not hardcode).")



tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)
else:
    mlruns_dir = Path(os.getenv("MLFLOW_DIR", "mlruns")).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())

mlflow.set_experiment("tourism-rf-vs-xgb")


output_dir = Path(os.getenv("OUTPUT_DIR", "outputs")).resolve()
output_dir.mkdir(parents=True, exist_ok=True)

rf_model_outpath   = output_dir / "rf_pipeline.joblib"
xgb_model_outpath  = output_dir / "xgb_pipeline.joblib"
best_model_outpath = output_dir / "best_pipeline.joblib"

classification_threshold = 0.50
# --------------------------------------------------------

def _dl(name: str) -> str:
    return hf_hub_download(
        repo_id=data_repo_id,
        repo_type=data_repo_type,
        filename=f"{prefix}/{name}",
        token=hf_token
    )

# ---- probability helpers (class-safe) ----
def _get_classes(fitted_estimator):
    """
    Works for sklearn Pipelines or bare estimators.
    Returns numpy array of class labels in the order of predict_proba columns.
    """
    classes = getattr(fitted_estimator, "classes_", None)
    if classes is None and hasattr(fitted_estimator, "named_steps"):
        # final estimator is last step in a Pipeline
        _, last = list(fitted_estimator.named_steps.items())[-1]
        classes = getattr(last, "classes_", None)
    if classes is None:
        raise RuntimeError("Estimator has no classes_; cannot map probabilities.")
    return np.asarray(classes)

def _pos_index(classes: np.ndarray) -> int:
    """
    Heuristic: prefer numeric 1, else common positive labels, else numeric max, else index 1/0.
    """
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

# ---- load splits from HF ----
xtrain = pd.read_csv(_dl("Xtrain.csv"))
xtest  = pd.read_csv(_dl("Xtest.csv"))
ytrain = pd.read_csv(_dl("ytrain.csv")).iloc[:, 0].astype(int)
ytest  = pd.read_csv(_dl("ytest.csv")).iloc[:, 0].astype(int)

print("loaded from HF:",
      "xtrain", xtrain.shape, "| xtest", xtest.shape,
      "| ytrain", ytrain.shape, "| ytest", ytest.shape)

# ---- shared preprocessing ----
num_sel = make_column_selector(dtype_include=np.number)
cat_sel = make_column_selector(dtype_exclude=np.number)

numeric_pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
categorical_pipe = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_sel),
        ("cat", categorical_pipe, cat_sel),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# ---- scorer (class-safe) ----
def auc_scorer(estimator, xv, yv):
    proba = estimator.predict_proba(xv)
    classes = _get_classes(estimator)
    if len(classes) == 2:
        pos_idx = _pos_index(classes)
        return roc_auc_score(yv, proba[:, pos_idx])
    else:
        return roc_auc_score(yv, proba, multi_class="ovr")

with mlflow.start_run(run_name="rf_vs_xgb"):
    # ======================= random forest =======================
    rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
    rf_pipe = make_pipeline(preprocessor, rf)

    rf_grid = {
        "randomforestclassifier__n_estimators": [300, 600, 900],
        "randomforestclassifier__max_depth": [10, 12, 15],
        "randomforestclassifier__min_samples_split": [5, 8, 10]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    with mlflow.start_run(run_name="rf", nested=True):
        rf_gs = GridSearchCV(rf_pipe, rf_grid, cv=cv, n_jobs=-1, scoring=auc_scorer, error_score="raise")
        rf_gs.fit(xtrain, ytrain)

        rf_best = rf_gs.best_estimator_
        rf_classes = _get_classes(rf_best)
        rf_pos = _pos_index(rf_classes)

        proba_tr = rf_best.predict_proba(xtrain)[:, rf_pos]
        proba_te = rf_best.predict_proba(xtest)[:, rf_pos]
        yhat_tr = (proba_tr >= classification_threshold).astype(int)
        yhat_te = (proba_te >= classification_threshold).astype(int)

        rep_te = classification_report(ytest, yhat_te, output_dict=True, zero_division=0)
        auc_tr = roc_auc_score(ytrain, proba_tr)
        auc_te = roc_auc_score(ytest,  proba_te)
        f1_te  = f1_score(ytest, yhat_te, zero_division=0)

        mlflow.log_params(rf_gs.best_params_)
        mlflow.log_param("threshold", classification_threshold)
        mlflow.log_metrics({
            "train_auc": float(auc_tr), "test_auc": float(auc_te),
            "test_f1": float(f1_te),
            "test_precision_pos": float(rep_te[str(rf_classes[rf_pos])]["precision"]),
            "test_recall_pos": float(rep_te[str(rf_classes[rf_pos])]["recall"]),
            "test_accuracy": float(rep_te["accuracy"]),
        })

        cm = confusion_matrix(ytest, yhat_te, labels=rf_classes)
        print("rf test cm (rows=true, cols=pred):\n", cm)

        joblib.dump(rf_best, rf_model_outpath)
        mlflow.log_artifact(str(rf_model_outpath), artifact_path="models")
        print("saved rf to:", rf_model_outpath)

    # ======================= xgboost =======================
    pos = ytrain.value_counts().get(1, 0)
    neg = ytrain.value_counts().get(0, 0)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    xgb_pipe = make_pipeline(preprocessor, xgb_clf)

    xgb_grid = {
        "xgbclassifier__n_estimators": [200, 400, 1000],
        "xgbclassifier__max_depth": [7, 10, 15],
        "xgbclassifier__learning_rate": [0.05, 0.1],
        "xgbclassifier__subsample": [0.8, 1.0],
        "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
    }

    with mlflow.start_run(run_name="xgb", nested=True):
        xgb_gs = GridSearchCV(xgb_pipe, xgb_grid, cv=cv, n_jobs=-1, scoring=auc_scorer, error_score="raise")
        xgb_gs.fit(xtrain, ytrain)

        xgb_best = xgb_gs.best_estimator_
        xgb_classes = _get_classes(xgb_best)
        xgb_pos = _pos_index(xgb_classes)

        proba_tr = xgb_best.predict_proba(xtrain)[:, xgb_pos]
        proba_te = xgb_best.predict_proba(xtest)[:, xgb_pos]
        yhat_tr = (proba_tr >= classification_threshold).astype(int)
        yhat_te = (proba_te >= classification_threshold).astype(int)

        rep_te = classification_report(ytest,  yhat_te, output_dict=True, zero_division=0)
        auc_tr = roc_auc_score(ytrain, proba_tr)
        auc_te = roc_auc_score(ytest,  proba_te)
        f1_te  = f1_score(ytest, yhat_te, zero_division=0)

        mlflow.log_params(xgb_gs.best_params_)
        mlflow.log_param("threshold", classification_threshold)
        mlflow.log_metrics({
            "train_auc": float(auc_tr), "test_auc": float(auc_te),
            "test_f1": float(f1_te),
            "test_precision_pos": float(rep_te[str(xgb_classes[xgb_pos])]["precision"]),
            "test_recall_pos": float(rep_te[str(xgb_classes[xgb_pos])]["recall"]),
            "test_accuracy": float(rep_te["accuracy"]),
        })

        cm = confusion_matrix(ytest, yhat_te, labels=xgb_classes)
        print("xgb test cm (rows=true, cols=pred):\n", cm)

        joblib.dump(xgb_best, xgb_model_outpath)
        mlflow.log_artifact(str(xgb_model_outpath), artifact_path="models")
        print("saved xgb to:", xgb_model_outpath)

    # ======================= choose best & upload =======================
    rf_loaded  = joblib.load(rf_model_outpath)
    xgb_loaded = joblib.load(xgb_model_outpath)

    # RF AUC
    rf_classes = _get_classes(rf_loaded)
    rf_pos = _pos_index(rf_classes)
    rf_auc  = roc_auc_score(ytest, rf_loaded.predict_proba(xtest)[:, rf_pos])

    # XGB AUC
    xgb_classes = _get_classes(xgb_loaded)
    xgb_pos = _pos_index(xgb_classes)
    xgb_auc = roc_auc_score(ytest, xgb_loaded.predict_proba(xtest)[:, xgb_pos])

    if abs(rf_auc - xgb_auc) < 1e-6:
        rf_f1  = f1_score(ytest, (rf_loaded.predict_proba(xtest)[:, rf_pos]  >= classification_threshold).astype(int), zero_division=0)
        xgb_f1 = f1_score(ytest, (xgb_loaded.predict_proba(xtest)[:, xgb_pos] >= classification_threshold).astype(int), zero_division=0)
        best_name = "rf" if rf_f1 >= xgb_f1 else "xgb"
    else:
        best_name = "rf" if rf_auc > xgb_auc else "xgb"

    best_model_path = rf_model_outpath if best_name == "rf" else xgb_model_outpath
    best_model = rf_loaded if best_name == "rf" else xgb_loaded

    joblib.dump(best_model, best_model_outpath)
    mlflow.log_artifact(str(best_model_outpath), artifact_path="models")
    mlflow.log_param("best_model", best_name)
    mlflow.log_metric("best_model_test_auc", float(max(rf_auc, xgb_auc)))
    print(f"selected best model: {best_name} (auc={max(rf_auc, xgb_auc):.4f})")
    print("saved best to:", best_model_outpath)

    # upload to HF model repo
    model_repo_id = "cheeka84/tourism-package-pred"  # HF model repo
    model_repo_type = "model"
    api = HfApi(token=hf_token)
    try:
        api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
        print(f"hf model repo exists: {model_repo_id}")
    except RepositoryNotFoundError:
        print(f"creating hf model repo: {model_repo_id}")
        create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False, exist_ok=True, token=hf_token)

    api.upload_file(
        path_or_fileobj=str(best_model_outpath),
        path_in_repo=os.path.basename(best_model_outpath),
        repo_id=model_repo_id,
        repo_type=model_repo_type,
        commit_message=f"upload best model ({best_name}) for tourism package prediction"
    )
    print("✅ uploaded best model to hf:", f"{model_repo_id}/{os.path.basename(best_model_outpath)}")
