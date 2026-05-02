import os
import time
import json
from pathlib import Path

# Config
MODELS = ["qwen_0.8b", "nemotron_4b"]
DOMAINS = ["math", "code", "science"]
TECHNIQUES = ["SFT", "DPO"]
RANKS = [1, 2, 8, 128, 1024, 3072]
OUTPUT_DIR = Path("/home/learner/Desktop/mewtwo/hf_kaggle_opensource/outputs")
LOG_FILE = Path("/home/learner/Desktop/mewtwo/hf_kaggle_opensource/pipeline.log")

def get_status():
    status = {}
    times = {}
    completed = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                if (d / "adapter_config.json").exists() or (d / "model.safetensors").exists() or (d / "merged_sft" / "adapter_config.json").exists():
                    completed.append(d.name)
                    timing_file = d / "timing.json"
                    if timing_file.exists():
                        try:
                            with open(timing_file, "r") as f:
                                data = json.load(f)
                                times[d.name] = data.get("duration_seconds")
                        except: pass
    
    current = None
    if LOG_FILE.exists():
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
            for line in reversed(lines):
                if "STARTING:" in line:
                    current = line.split("STARTING:")[1].split("(")[0].strip()
                    break
    
    for m in MODELS:
        for r in RANKS:
            # SFT domains
            for d in DOMAINS:
                name = f"{m}_{d}_SFT_rank{r}"
                if name in completed:
                    time_str = ""
                    if name in times:
                        t_sec = times[name]
                        time_str = f" ({t_sec/60:.1f}m)" if t_sec > 60 else f" ({t_sec:.0f}s)"
                    status[name] = f"✅ DONE{time_str}"
                elif name == current:
                    status[name] = "🔥 LIVE"
                else:
                    status[name] = "⏳ QUEUE"
            
            # DARE Merge
            dare_name = f"{m}_merged_DARE_rank{r}"
            if dare_name in completed:
                status[dare_name] = "✅ DONE"
            elif current and "MERGING" in current and str(r) in current and m in current:
                status[dare_name] = "🔥 LIVE"
            else:
                status[dare_name] = "⏳ QUEUE"
                
            # Final DPO
            dpo_name = f"{m}_math_DPO_rank{r}"
            if dpo_name in completed:
                status[dpo_name] = "✅ DONE"
            elif current == dpo_name:
                status[dpo_name] = "🔥 LIVE"
            else:
                status[dpo_name] = "⏳ QUEUE"
    return status

def print_dashboard():
    os.system('clear')
    st = get_status()
    print("="*60)
    print("      🚀 SYNAPTA ADAPTER PIPELINE DASHBOARD 🚀")
    print("="*60)
    
    for m in MODELS:
        print(f"\nMODEL: {m.upper()}")
        print("-" * 80)
        print(f"{'Rank':<8} | {'Math SFT':<12} | {'Code SFT':<12} | {'Science':<12} | {'DARE':<8} | {'DPO':<8}")
        print("-" * 80)
        for r in RANKS:
            row = f"r={r:<6} | "
            for d in DOMAINS:
                name = f"{m}_{d}_SFT_rank{r}"
                val = st.get(name, "⏳")
                row += f"{val:<12} | "
            
            # DARE
            val_dare = st.get(f"{m}_merged_DARE_rank{r}", "⏳")
            row += f"{val_dare:<8} | "
            
            # DPO
            val_dpo = st.get(f"{m}_math_DPO_rank{r}", "⏳")
            row += f"{val_dpo:<8} | "
            print(row.rstrip(" | "))
    print("\n" + "="*60)
    print(f"Update Time: {time.strftime('%H:%M:%S')}")
    print("Use: tail -f pipeline.log for raw telemetry")
    print("="*60)

if __name__ == "__main__":
    while True:
        try:
            print_dashboard()
            time.sleep(10)
        except KeyboardInterrupt: break
