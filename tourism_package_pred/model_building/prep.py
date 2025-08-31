import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi, hf_hub_download

# ---- config you can tweak ----
repo_id = "cheeka84/tourism-package-pred"     # HF dataset repo (no trailing /data)
repo_type = "dataset"
source_file = "tourism.csv"                   # if None, pick first .csv in repo
target_col = "ProdTaken"

test_size = 0.20
random_state = 42
local_out_dir = "tourism_package_pred/data/cleaned"
upload_prefix = "prepared"                    # uploads to prepared/Xtrain.csv etc.
# --------------------------------

# need a write token in env (do not hardcode)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("Please set HF_TOKEN in the environment before running.")

api = HfApi(token=HF_TOKEN)
os.makedirs(local_out_dir, exist_ok=True)

# 1) pick the csv in the dataset
if source_file is None:
    files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
    csvs = [f for f in files if f.lower().endswith(".csv")]
    if not csvs:
        raise SystemExit("No CSV files found in the dataset repo.")
    source_file = csvs[0]

print("Downloading from HF:", source_file)
local_csv = hf_hub_download(
    repo_id=repo_id,
    repo_type=repo_type,
    filename=source_file,
    token=HF_TOKEN
)

# 2) load + simple cleaning
data = pd.read_csv(local_csv)
print("Loaded shape:", data.shape)

# drop obvious id/noise columns if they exist
for col in ["CustomerID", "Unnamed: 0"]:
    if col in data.columns:
        data.drop(columns=[col], inplace=True)

# coerce target to 0/1 integers (very simple mapping)
y_raw = data[target_col]
if y_raw.dtype == object:
    m = {"yes":1, "no":0, "true":1, "false":0, "1":1, "0":0, "y":1, "n":0}
    data[target_col] = y_raw.astype(str).str.lower().map(m)
else:
    data[target_col] = pd.to_numeric(y_raw, errors="coerce")

# drop rows where target is missing
before = len(data)
data = data.dropna(subset=[target_col])
after = len(data)
if after != before:
    print("Dropped rows with missing target:", before - after)


# 3) split
X = data.drop(columns=[target_col])
y = data[target_col].astype(int)

stratify = y if set(y.unique()) == {0, 1} else None
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=stratify
)

# 4) save four files locally
xtrain_path = os.path.join(local_out_dir, "Xtrain.csv")
xtest_path = os.path.join(local_out_dir, "Xtest.csv")
ytrain_path = os.path.join(local_out_dir, "ytrain.csv")
ytest_path = os.path.join(local_out_dir, "ytest.csv")

Xtrain.to_csv(xtrain_path, index=False)
Xtest.to_csv(xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("Saved:")
print(" -", xtrain_path)
print(" -", xtest_path)
print(" -", ytrain_path)
print(" -", ytest_path)

# 5) upload the four files back to the HF dataset
uploads = {
    f"{upload_prefix}/Xtrain.csv": xtrain_path,
    f"{upload_prefix}/Xtest.csv":  xtest_path,
    f"{upload_prefix}/ytrain.csv": ytrain_path,
    f"{upload_prefix}/ytest.csv":  ytest_path,
}
for path_in_repo, local_path in uploads.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Add {os.path.basename(local_path)}"
    )

print("Done uploading to:", repo_id, "under", upload_prefix)
