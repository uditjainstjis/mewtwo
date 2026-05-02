import os
import json
import glob
import re
import subprocess
import signal
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

PROJECT_ROOT = "/home/learner/Desktop/mewtwo"
LOG_DIR = f"{PROJECT_ROOT}/logs/lori_moe"
STATE_FILES = [
    f"{PROJECT_ROOT}/adapters/lori_moe/pipeline_state.json",
    f"{PROJECT_ROOT}/adapters/lori_moe/shadow_state.json"
]
CONFIG_FILE = f"{PROJECT_ROOT}/adapters/lori_moe/queue_config.json"

QUEUE_MODELS = [
    {"name": "Qwen/Qwen2.5-14B-Instruct", "short": "qwen2.5_14b_instruct", "log_name": "qwen2.5_14b_instruct"},
    {"name": "Qwen/Qwen3.5-0.8B", "short": "qwen3.5_0.8b", "log_name": "qwen3.5_0.8b"},
    {"name": "Qwen/Qwen3.5-9B", "short": "qwen3.5_9b", "log_name": "qwen3.5_9b"},
    {"name": "Qwen/Qwen3.5-27B", "short": "qwen3.5_27b", "log_name": "qwen3.5_27b"}
]

DOMAINS = ["math", "code", "science", "legal", "medical"]

def get_pipeline_state():
    combined = {"completed": {}, "failed": {}}
    for sf in STATE_FILES:
        if os.path.exists(sf):
            try:
                with open(sf, "r") as f:
                    s = json.load(f)
                    for k, v in s.get("completed", {}).items():
                        combined["completed"][k] = list(set(combined["completed"].get(k, []) + v))
                    for k, v in s.get("failed", {}).items():
                        combined["failed"][k] = list(set(combined["failed"].get(k, []) + v))
            except Exception:
                pass
    return combined

def parse_live_stats():
    active_runs = []
    logs = glob.glob(f"{LOG_DIR}/train_*.log")
    for log_path in logs:
        try:
            mtime = os.path.getmtime(log_path)
            from time import time
            if time() - mtime > 120: # 2 mins stale
                continue
        except Exception:
            continue
            
        stats = {
            "model_log_name": "Unknown",
            "current_domain": "Waiting...",
            "step": "N/A", "total_steps": "N/A",
            "epoch": "N/A", "loss": "N/A",
            "gpu": "N/A", "eta": "N/A", "lr": "N/A",
            "example": "Waiting..."
        }
        
        filename = os.path.basename(log_path).replace(".log", "")
        parts = filename.split("_")
        if len(parts) >= 2:
            stats["current_domain"] = parts[-1].upper()
            stats["model_log_name"] = "_".join(parts[1:-1])
            
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                for line in reversed(lines[-100:]):
                        step_match = re.search(r"Step (\d+)/(\d+)", line)
                        if step_match and stats["step"] == "N/A":
                            stats["step"] = step_match.group(1)
                            stats["total_steps"] = step_match.group(2)
                        
                        loss_match = re.search(r"Loss:\s*([\d.]+)", line)
                        if loss_match and stats["loss"] == "N/A": stats["loss"] = loss_match.group(1)
                        
                        lr_match = re.search(r"LR:\s*([\d.e-]+)", line)
                        if lr_match and stats["lr"] == "N/A": stats["lr"] = lr_match.group(1)
                        
                        gpu_match = re.search(r"GPU:\s*([\d.]+GB)", line)
                        if gpu_match and stats["gpu"] == "N/A": stats["gpu"] = gpu_match.group(1)
                        
                        eta_match = re.search(r"ETA:\s*([\w.]+)", line)
                        if eta_match and stats["eta"] == "N/A": stats["eta"] = eta_match.group(1)
                        
                        epoch_match = re.search(r"Epoch (\d+/\d+)", line)
                        if epoch_match and stats["epoch"] == "N/A": stats["epoch"] = epoch_match.group(1)
                        
                        example_match = re.search(r"Example:\s*(.*)", line)
                        if example_match and stats["example"] == "Waiting...": stats["example"] = example_match.group(1)

                        if stats["loss"] != "N/A" and stats["example"] != "Waiting...":
                            break
        except Exception:
            pass
        
        active_runs.append(stats)
    return active_runs

@app.route("/api/state")
def get_state():
    pipeline = get_pipeline_state()
    live_runs = parse_live_stats()
    
    is_running = False
    pid_file = "/tmp/lori_pipeline.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
                os.kill(pid, 0)
                is_running = True
        except:
            pass
            
    # Load config to see enabled domains
    config_data = QUEUE_MODELS
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config_data = json.load(f)
        except:
            pass

    models_status = []
    for m in QUEUE_MODELS:
        c_item = next((ci for ci in config_data if ci.get("short_name") == m["short"] or ci.get("short") == m["short"]), {})
        enabled = c_item.get("enabled_domains", DOMAINS)
        
        completed_doms = pipeline["completed"].get(m["short"], [])
        failed_doms = pipeline["failed"].get(m["short"], [])
        alive_stats = next((l for l in live_runs if l["model_log_name"] == m["log_name"]), None)
        
        status = "pending"
        if len(completed_doms) >= len(enabled):
            status = "completed"
        elif alive_stats:
            status = "active"
            
        models_status.append({
            "name": m["name"],
            "short": m["short"],
            "completed": completed_doms,
            "failed": failed_doms,
            "status": status,
            "live_stats": alive_stats,
            "enabled_domains": enabled
        })
        
    return jsonify({
        "models": models_status,
        "pipeline_running": is_running
    })

@app.route("/api/config", methods=["GET", "POST"])
def manage_config():
    if request.method == "POST":
        new_config = request.json
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(new_config, f, indent=2)
        return jsonify({"status": "success"})
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return jsonify(json.load(f))
    return jsonify(QUEUE_MODELS)

@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    stop_signal = f"{PROJECT_ROOT}/adapters/lori_moe/STOP_PIPELINE"
    if os.path.exists(stop_signal):
        os.remove(stop_signal)
    
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && nohup python scripts/autonomous_pipeline.py >> {LOG_DIR}/autonomous_pipeline.log 2>&1 &"
    subprocess.Popen(cmd, shell=True, executable="/bin/bash")
    return jsonify({"status": "starting"})

@app.route("/api/pipeline/stop", methods=["POST"])
def stop_pipeline():
    stop_signal = f"{PROJECT_ROOT}/adapters/lori_moe/STOP_PIPELINE"
    with open(stop_signal, "w") as f:
        f.write("STOP")
    
    subprocess.run("pkill -f train_lori_adapter.py", shell=True)
    
    pid_file = "/tmp/lori_pipeline.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = f.read().strip()
                subprocess.run(f"kill {pid}", shell=True)
        except:
            pass
    return jsonify({"status": "stopping"})

@app.route("/api/pipeline/reset", methods=["POST"])
def reset_pipeline():
    state_file = f"{PROJECT_ROOT}/adapters/lori_moe/pipeline_state.json"
    if os.path.exists(state_file):
        os.remove(state_file)
    return jsonify({"status": "reset"})

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mewtwo | Control v2</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #09090b;
            --surface: rgba(24, 24, 27, 0.7);
            --border: rgba(255, 255, 255, 0.1);
            --primary: #8b5cf6;
            --primary-glow: rgba(139, 92, 246, 0.5);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-cyan: #06b6d4;
        }

        body {
            margin: 0; padding: 0;
            background: var(--bg);
            background-image: radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-main);
            font-family: 'Outfit', sans-serif;
        }

        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }
        header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 1.5rem; margin-bottom: 2rem;}
        
        .grid { display: grid; grid-template-columns: 1fr 2fr; gap: 2rem; }
        .glass-panel { background: var(--surface); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;}
        
        .stat-box { background: rgba(0,0,0,0.3); border-radius: 12px; padding: 1rem; text-align: center; }
        .stat-value { font-size: 1.8rem; font-weight: 800; }
        .stat-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; }

        .queue-item { padding: 1rem; border-bottom: 1px solid var(--border); border-left: 2px solid transparent; }
        .queue-item.active { border-left-color: var(--primary); background: rgba(139, 92, 246, 0.05); }
        
        .pill { font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px; background: rgba(255,255,255,0.05); color: var(--text-muted); cursor: pointer;}
        .pill.done { background: var(--accent-green); color: #000; font-weight: 800; }
        .pill.active { background: var(--primary); color: #fff; animation: pulse 1s infinite alternate; }
        
        @keyframes pulse { from { opacity: 0.6; } to { opacity: 1; } }
        
        .progress-bar { height: 8px; background: rgba(255,255,255,0.1); border-radius: 4px; overflow: hidden; margin: 1rem 0; }
        .progress-fill { height: 100%; background: var(--primary); transition: width 0.5s; }

        button { border: none; padding: 0.8rem 1.5rem; border-radius: 8px; font-weight: 700; cursor: pointer; transition: 0.2s; }
        button:hover { filter: brightness(1.2); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1 style="margin:0">Mewtwo Mission Control <span style="font-size: 1rem; color: var(--primary);">[RTX 5090]</span></h1>
                <p style="color: grey; margin: 0;">Interactive Deep VRAM Saturation Engine</p>
            </div>
            <div id="global-status" class="pill">○ OFFLINE</div>
        </header>

        <div class="grid">
            <div class="glass-panel" style="padding: 0">
                <h3 style="padding: 1.5rem 1.5rem 0.5rem 1.5rem; margin: 0;">Resource Matrix</h3>
                <div id="queue-container"></div>
                <div style="padding: 1.5rem;">
                    <button onclick="controlPipeline('start')" id="start-btn" style="background: var(--accent-green); width: 100%; margin-bottom: 1rem;">START MISSION</button>
                    <button onclick="controlPipeline('stop')" style="background: #ef4444; width: 100%;">ABORT MISSION</button>
                </div>
            </div>

            <div id="active-zone">
                <!-- Large Active Stats will go here -->
            </div>
        </div>
    </div>

    <script>
        const DOMAINS = ["math", "code", "science", "legal", "medical"];
        
        async function update() {
            const res = await fetch('/api/state');
            const data = await res.json();
            
            // Status Badge
            const status = document.getElementById('global-status');
            status.innerText = data.pipeline_running ? '● MISSION ACTIVE' : '○ IDLE';
            status.style.color = data.pipeline_running ? 'var(--accent-green)' : 'var(--text-muted)';
            document.getElementById('start-btn').disabled = data.pipeline_running;

            // Model List
            let qHtml = '';
            let activeHtml = '';
            
            data.models.forEach(m => {
                let pills = '';
                DOMAINS.forEach(d => {
                    const isDone = m.completed.includes(d);
                    const isActive = m.status === 'active' && m.live_stats && m.live_stats.current_domain.toLowerCase() === d;
                    const isEnabled = m.enabled_domains.includes(d);
                    
                    pills += `<span class="pill ${isDone ? 'done' : (isActive ? 'active' : '')}" 
                                   style="opacity: ${isEnabled ? 1 : 0.2}"
                                   onclick="toggleDomain('${m.short}', '${d}')">${d}</span> `;
                });

                qHtml += `
                    <div class="queue-item ${m.status === 'active' ? 'active' : ''}">
                        <div style="font-weight: 800;">${m.name.split('/')[1]}</div>
                        <div style="display:flex; gap: 4px; margin-top: 0.5rem;">${pills}</div>
                    </div>
                `;

                if (m.status === 'active' && m.live_stats) {
                    const ls = m.live_stats;
                    const pct = (parseInt(ls.step) / parseInt(ls.total_steps) * 100) || 0;
                    activeHtml += `
                        <div class="glass-panel" style="border-left: 4px solid var(--primary)">
                            <h2 style="margin:0">${m.name}</h2>
                            <p style="color: var(--primary)">Current Objective: <b>${ls.current_domain}</b></p>
                            
                            <div class="progress-bar"><div class="progress-fill" style="width: ${pct}%"></div></div>
                            <div style="display:flex; justify-content: space-between; font-size: 0.8rem; color: grey;">
                                <span>STEP ${ls.step} / ${ls.total_steps}</span>
                                <span>EPOCH ${ls.epoch}</span>
                            </div>

                            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;">
                                <div class="stat-box"><div class="stat-value" style="color: var(--accent-yellow)">${ls.loss}</div><div class="stat-label">Loss</div></div>
                                <div class="stat-box"><div class="stat-value">${ls.gpu}</div><div class="stat-label">VRAM</div></div>
                                <div class="stat-box"><div class="stat-value">${ls.eta}</div><div class="stat-label">ETA</div></div>
                            </div>
                            
                            <div style="margin-top: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.4); border-radius: 8px; border: 1px solid var(--border);">
                                <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem; text-transform: uppercase;">▶ CURRENT EXAMPLE (Tokenized)</div>
                                <div style="font-family: monospace; font-size: 0.9rem; color: var(--accent-cyan); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                                    ${ls.example}
                                </div>
                            </div>
                        </div>
                    `;
                }
            });

            document.getElementById('queue-container').innerHTML = qHtml;
            document.getElementById('active-zone').innerHTML = activeHtml || '<div class="glass-panel" style="text-align:center; color:grey; padding: 5rem;">NO ACTIVE MISSION</div>';
        }

        async function toggleDomain(mShort, domain) {
            const res = await fetch('/api/state');
            const data = await res.json();
            const models = data.models;
            const model = models.find(mo => mo.short === mShort);
            
            if (model.enabled_domains.includes(domain)) {
                model.enabled_domains = model.enabled_domains.filter(d => d !== domain);
            } else {
                model.enabled_domains.push(domain);
            }
            
            const configPayload = models.map(mo => ({
                short_name: mo.short,
                model_id: mo.name,
                enabled_domains: mo.enabled_domains
            }));

            await fetch('/api/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(configPayload)
            });
            update();
        }

        async function controlPipeline(action) {
            await fetch('/api/pipeline/' + action, { method: 'POST' });
            setTimeout(update, 1000);
        }

        setInterval(update, 2000);
        update();
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
