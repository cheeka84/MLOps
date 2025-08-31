from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo, login
import os

repo_id = "cheeka84/tourism-package-pred"
repo_type = "dataset"

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise SystemExit("HF_TOKEN is not set. Provide it via environment/Actions secret.")

# Initialize API client
api = HfApi(token=hf_token)

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="/content/tourism_package_pred/data",  # Corrected path
    repo_id=repo_id,
    repo_type=repo_type,
)
