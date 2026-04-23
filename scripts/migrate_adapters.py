import os
import json
import subprocess
from pathlib import Path
from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path("/home/learner/Desktop/mewtwo")
STAGING_DIR = PROJECT_ROOT / "hf_publish"
NEW_TOKEN = os.getenv("HF_TOKEN")
NEW_BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
OLD_BASE_MODEL = "nvidia/Nemotron-3-Nano-30B-A3B"

ADAPTERS = {
    "math": "nemotron-30b-math-reasoner-peft",
    "code": "nemotron-30b-code-hyper-reasoner-peft",
    "science": "nemotron-30b-science-expert-peft",
    "merged": "nemotron-30b-multi-domain-merged-peft"
}

def update_metadata(adapter_dir):
    print(f"  📝 Updating metadata in {adapter_dir.name}...")
    
    # 1. Update adapter_config.json
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        
        config["base_model_name_or_path"] = NEW_BASE_MODEL
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        print(f"  ⚠️ Warning: No adapter_config.json found in {adapter_dir.name}")

    # 2. Update README.md
    readme_path = adapter_dir / "README.md"
    if readme_path.exists():
        with open(readme_path, "r") as f:
            content = f.read()
        
        content = content.replace(OLD_BASE_MODEL, NEW_BASE_MODEL)
        content = content.replace("udit6969/", "uditjain/")
        
        with open(readme_path, "w") as f:
            f.write(content)

    # 3. Update Kaggle dataset-metadata.json
    kaggle_path = adapter_dir / "dataset-metadata.json"
    if kaggle_path.exists():
        with open(kaggle_path, "r") as f:
            k_config = json.load(f)
        
        k_config["isPrivate"] = False
        
        with open(kaggle_path, "w") as f:
            json.dump(k_config, f, indent=2)

def main():
    print("🚀 Starting Migration Script")
    
    # Authenticate APIs
    old_api = HfApi() # Uses ~/.cache/huggingface/token
    new_api = HfApi(token=NEW_TOKEN)
    
    for key, repo_slug in ADAPTERS.items():
        print(f"\n--- Processing {key} ({repo_slug}) ---")
        
        # 1. DELETE FROM OLD NAMESPACE
        old_repo = f"udit6969/{repo_slug}"
        try:
            print(f"  🗑️ Deleting old repo: {old_repo}")
            old_api.delete_repo(repo_id=old_repo, repo_type="model")
            print(f"  ✅ Deleted {old_repo}")
        except Exception as e:
            print(f"  ⚠️ Could not delete {old_repo}: {e}")

        # 2. UPDATE METADATA LOCALLY
        adapter_dir = STAGING_DIR / key
        update_metadata(adapter_dir)
        
        # 3. UPLOAD TO NEW NAMESPACE
        new_repo = f"uditjain/{repo_slug}"
        try:
            print(f"  ☁️ Creating new repo: {new_repo}")
            new_api.create_repo(repo_id=new_repo, repo_type="model", exist_ok=True, private=False)
            
            print(f"  ☁️ Uploading to {new_repo}")
            new_api.upload_folder(
                folder_path=str(adapter_dir),
                repo_id=new_repo,
                repo_type="model"
            )
            print(f"  ✅ Uploaded to {new_repo}")
        except Exception as e:
            print(f"  ❌ Failed to upload to {new_repo}: {e}")
            
        # 4. KAGGLE: MAKE PUBLIC
        kaggle_cmd = [str(PROJECT_ROOT / ".venv/bin/kaggle"), "datasets", "version", "-p", str(adapter_dir), "-m", "Settings updated to public."]
        print(f"  ☁️ Making Kaggle dataset public via version update...")
        result = subprocess.run(kaggle_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Kaggle dataset version created successfully.")
        else:
            print(f"  ❌ Kaggle update failed: {result.stderr}")

    print("\n🎉 Migration complete!")

if __name__ == "__main__":
    main()
