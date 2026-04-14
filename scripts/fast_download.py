from huggingface_hub import snapshot_download
import sys
import threading

def download(repo):
    print(f"📥 Downloading {repo}...")
    try:
        snapshot_download(repo_id=repo, resume_download=True)
        print(f"✅ {repo} downloaded.")
    except Exception as e:
        print(f"❌ Error downloading {repo}: {e}")

models = ["Qwen/Qwen2.5-14B-Instruct", "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B"]
threads = []
for m in models:
    t = threading.Thread(target=download, args=(m,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
print("🚀 All models pre-downloaded!")
