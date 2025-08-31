import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="tourism package purchase", page_icon="🧳", layout="centered")

# ---------- model loader ----------
model_repo = "cheeka84/tourism-package-pred"
candidate_filenames = [
    "rf_pipeline.joblib",
    "xgb_pipeline.joblib",
    "best_pipeline.joblib",
]

model = None
loaded_filename = None
last_err = None
for fname in candidate_filenames:
    try:
        path = hf_hub_download(repo_id=model_repo, repo_type="model", filename=fname)
        model = joblib.load(path)
        loaded_filename = fname
        break
    except Exception as e:
        last_err = e

if model is None:
    st.error(f"could not load a model from {model_repo}. last error: {last_err}")
    st.stop()

# try to discover columns the pipeline was fitted on
expected_cols = None
try:
    expected_cols = list(model.named_steps["columntransformer"].feature_names_in_)
except Exception:
    try:
        expected_cols = list(model.feature_names_in_)
    except Exception:
        expected_cols = None

st.title("Tourism Package Purchase Predictor")
st.caption(f"loaded model: {loaded_filename} from {model_repo}")

threshold = st.slider(
    "decision threshold (probability ≥ threshold ⇒ purchase)",
    min_value=0.05, max_value=0.95, value=0.50, step=0.01
)

with st.form("input_form"):
    st.subheader("enter customer details")

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

    submitted = st.form_submit_button("predict")

if expected_cols:
    with st.expander("debug: expected input columns (from the fitted pipeline)"):
        st.write(expected_cols)

if submitted:
    # base row with all expected cols as NaN so imputers can work
    if expected_cols:
        row = {c: np.nan for c in expected_cols}
    else:
        row = {}

    # update with user inputs using the *exact* training column names
    row.update({
        "Age": age,
        "CityTier": citytier,
        "DurationOfPitch": durationofpitch,
        "Occupation": occupation,
        "Gender": gender,
        "TypeofContact": typeofcontact,
        "Designation": designation,
        "MonthlyIncome": monthlyincome,
        "NumberOfPersonVisiting": numberofpersonvisiting,
        "NumberOfFollowups": numberoffollowups,
        "ProductPitched": productpitched,
        "PreferredPropertyStar": preferredpropertystar,
        "MaritalStatus": maritalstatus,
        "NumberOfTrips": numberoftrips,
        "Passport": 1 if passport else 0,
        "PitchSatisfactionScore": pitchsatisfactionscore,
        "OwnCar": 1 if owncar else 0,
        "NumberOfChildrenVisiting": numberofchildrenvisiting,
    })

    # build input frame; if expected_cols is known, reindex to match training order
    input_df = pd.DataFrame([row])
    if expected_cols:
        # drop unexpected, add missing
        for c in expected_cols:
            if c not in input_df.columns:
                input_df[c] = np.nan
        input_df = input_df[expected_cols]

    # predict
    try:
        proba = float(model.predict_proba(input_df)[:, 1][0])
        pred = int(proba >= threshold)
        label = "Will Purchase" if pred == 1 else "Won't Purchase"

        st.subheader("prediction")
        st.metric("probability of purchase", f"{proba:.3f}")
        st.write(f"threshold = **{threshold:.2f}** → predicted: **{label}**")
    except Exception as e:
        st.error(f"prediction failed: {e}")
