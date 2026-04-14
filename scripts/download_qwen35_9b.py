from huggingface_hub import snapshot_download
import sys

def download_model(repo_id):
    print(f"Starting download for {repo_id}...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            resume_download=True
        )
        print(f"Successfully downloaded {repo_id} to {path}")
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # We assume 'qwen 3.5:9b' maps to Qwen/Qwen3.5-9B confirmed in cache
    model_id = "Qwen/Qwen3.5-9B" 
    download_model(model_id)
