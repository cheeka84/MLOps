# tourism_package_pred/model_building/data_register.py
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise SystemExit("HF_TOKEN not set. In GitHub Actions, add it as a repo secret.")

repo_id = "cheeka84/tourism-package-pred"
repo_type = "dataset"

# --- resolve paths relative to this file; allow overrides via env ---
# this file is .../tourism_package_pred/model_building/data_register.py
project_root = Path(__file__).resolve().parents[1]   # .../tourism_package_pred
data_dir = Path(os.getenv("DATA_DIR", project_root / "data"))
data_file = Path(os.getenv("DATA_FILE", data_dir / "tourism.csv"))

api = HfApi(token=hf_token)

# ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"dataset repo exists: {repo_id}")
except RepositoryNotFoundError:
    print(f"creating dataset repo: {repo_id}")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False, exist_ok=True, token=hf_token)

# upload folder if it exists; otherwise fall back to single file
if data_dir.is_dir():
    print(f"uploading folder: {data_dir}")
    api.upload_folder(
        folder_path=str(data_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="sync data folder",
        allow_patterns=["*.csv", "*.parquet", "*.json"]
    )
    print("✅ uploaded folder")
elif data_file.is_file():
    print(f"uploading file: {data_file}")
    api.upload_file(
        path_or_fileobj=str(data_file),
        path_in_repo=data_file.name,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message="add/update dataset file"
    )
    print("✅ uploaded file")
else:
    print("❌ neither folder nor file found")
    print(" checked data_dir:", data_dir)
    print(" checked data_file:", data_file)
    print(" cwd:", Path.cwd())
    raise SystemExit(1)
