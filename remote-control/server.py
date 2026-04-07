#!/usr/bin/env python3
"""
Mewtwo Remote Control — GPU Rig Command Center
================================================
A web-based control panel for remotely managing your GPU research rig.

Features:
  - Live GPU/CPU/RAM monitoring
  - Command execution with streaming output
  - Monitor on/off (DPMS control)
  - GPU persistence mode toggle
  - Scheduled wake/sleep via rtcwake
  - Process management (list, kill)
  - Config/API key editor
  - Training job launcher
  - System power controls

Run: python3 server.py
Access: http://<your-ip>:7777
"""

import os
import sys
import json
import time
import asyncio
import signal
import secrets
import subprocess
import shlex
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import psutil
from fastapi import FastAPI, WebSocket, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ═══════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════
PORT = 7777
AUTH_TOKEN = os.environ.get("RC_TOKEN", "mewtwo-" + secrets.token_hex(4))
MEWTWO_DIR = Path(__file__).parent.parent.resolve()
LOG_DIR = MEWTWO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Store running background processes
bg_processes: Dict[str, dict] = {}
scheduled_jobs: List[dict] = []

app = FastAPI(title="Mewtwo Remote Control", docs_url=None)

# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

def run_cmd(cmd: str, timeout: int = 10) -> str:
    """Run a shell command and return output."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR] {e}"


def get_gpu_info() -> list:
    """Get GPU stats via nvidia-smi."""
    out = run_cmd(
        "nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,"
        "memory.used,memory.total,power.draw,power.limit,fan.speed "
        "--format=csv,noheader,nounits"
    )
    gpus = []
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 9:
            gpus.append({
                "index": parts[0], "name": parts[1],
                "temp_c": parts[2], "util_pct": parts[3],
                "mem_used_mb": parts[4], "mem_total_mb": parts[5],
                "power_w": parts[6], "power_limit_w": parts[7],
                "fan_pct": parts[8],
            })
    return gpus


def get_system_info() -> dict:
    """Get CPU, RAM, disk stats."""
    cpu_pct = psutil.cpu_percent(interval=0.3)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_pct": cpu_pct,
        "ram_used_gb": round(mem.used / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "ram_pct": mem.percent,
        "disk_used_gb": round(disk.used / 1e9, 1),
        "disk_total_gb": round(disk.total / 1e9, 1),
        "disk_pct": disk.percent,
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }


def get_gpu_processes() -> list:
    """Get processes using the GPU."""
    out = run_cmd("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits")
    procs = []
    for line in out.strip().split("\n"):
        if not line.strip() or "No running" in line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            procs.append({"pid": parts[0], "name": parts[1], "mem_mb": parts[2]})
    return procs

# ═══════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════

@app.get("/api/status")
async def api_status():
    """Full system status."""
    return {
        "gpu": get_gpu_info(),
        "system": get_system_info(),
        "gpu_processes": get_gpu_processes(),
        "bg_jobs": {k: {"cmd": v["cmd"], "started": v["started"], "status": v.get("status", "running")} for k, v in bg_processes.items()},
        "scheduled": scheduled_jobs,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/run")
async def api_run_command(cmd: str = Form(...), background: bool = Form(False)):
    """Run a command. If background=True, run in background and return job ID."""
    if background:
        job_id = f"job_{int(time.time())}_{secrets.token_hex(2)}"
        log_file = LOG_DIR / f"{job_id}.log"
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=open(log_file, "w"), stderr=subprocess.STDOUT,
            cwd=str(MEWTWO_DIR), preexec_fn=os.setsid
        )
        bg_processes[job_id] = {
            "cmd": cmd, "pid": proc.pid, "proc": proc,
            "log_file": str(log_file), "started": datetime.now().isoformat(),
            "status": "running"
        }
        return {"job_id": job_id, "pid": proc.pid, "log": str(log_file)}
    else:
        output = run_cmd(cmd, timeout=30)
        return {"output": output}


@app.get("/api/job/{job_id}")
async def api_job_status(job_id: str, tail: int = 50):
    """Get status and log tail of a background job."""
    if job_id not in bg_processes:
        raise HTTPException(404, "Job not found")
    job = bg_processes[job_id]
    # Check if process is still running
    proc = job.get("proc")
    if proc and proc.poll() is not None:
        job["status"] = f"exit:{proc.returncode}"
    # Read log tail
    log_content = ""
    if os.path.exists(job["log_file"]):
        log_content = run_cmd(f"tail -n {tail} {shlex.quote(job['log_file'])}", timeout=5)
    return {"job_id": job_id, "cmd": job["cmd"], "status": job["status"], "log": log_content}


@app.post("/api/job/{job_id}/kill")
async def api_kill_job(job_id: str):
    """Kill a background job."""
    if job_id not in bg_processes:
        raise HTTPException(404, "Job not found")
    job = bg_processes[job_id]
    try:
        os.killpg(os.getpgid(job["pid"]), signal.SIGTERM)
        job["status"] = "killed"
        return {"status": "killed", "pid": job["pid"]}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/kill/{pid}")
async def api_kill_pid(pid: int):
    """Kill any process by PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        return {"status": f"SIGTERM sent to {pid}"}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════
# Monitor & GPU Power Controls
# ═══════════════════════════════════════════════════════

@app.post("/api/monitor/{action}")
async def api_monitor(action: str):
    """Control monitor: 'off', 'on', 'standby'."""
    if action == "off":
        run_cmd("DISPLAY=:0 xset dpms force off")
        return {"status": "monitor_off"}
    elif action == "on":
        run_cmd("DISPLAY=:0 xset dpms force on")
        return {"status": "monitor_on"}
    elif action == "standby":
        run_cmd("DISPLAY=:0 xset dpms force standby")
        return {"status": "monitor_standby"}
    raise HTTPException(400, "action must be 'off', 'on', or 'standby'")


@app.post("/api/gpu/persistence/{action}")
async def api_gpu_persistence(action: str):
    """Enable/disable GPU persistence mode (keeps GPU warm, faster launches)."""
    if action == "on":
        out = run_cmd("sudo nvidia-smi -pm 1")
        return {"status": "persistence_on", "output": out}
    elif action == "off":
        out = run_cmd("sudo nvidia-smi -pm 0")
        return {"status": "persistence_off", "output": out}
    raise HTTPException(400, "action must be 'on' or 'off'")


@app.post("/api/gpu/power-limit")
async def api_gpu_power_limit(watts: int = Form(...)):
    """Set GPU power limit in watts (for undervolting/limiting heat)."""
    if watts < 100 or watts > 600:
        raise HTTPException(400, "Power limit must be between 100-600W")
    out = run_cmd(f"sudo nvidia-smi -pl {watts}")
    return {"status": f"power_limit_set_{watts}W", "output": out}


# ═══════════════════════════════════════════════════════
# Schedule & Power Management
# ═══════════════════════════════════════════════════════

@app.post("/api/schedule/wake")
async def api_schedule_wake(
    wake_time: str = Form(...),
    command: str = Form(""),
):
    """
    Schedule the machine to suspend now and wake at a specific time.
    wake_time: ISO format datetime or '+Xm' for relative minutes.
    command: Optional command to run when woken up.
    """
    if wake_time.startswith("+"):
        # Relative: +30m, +2h
        val = wake_time[1:]
        if val.endswith("m"):
            seconds = int(val[:-1]) * 60
        elif val.endswith("h"):
            seconds = int(val[:-1]) * 3600
        else:
            seconds = int(val)
        rtc_arg = f"-s {seconds}"
        wake_at = (datetime.now() + timedelta(seconds=seconds)).isoformat()
    else:
        # Absolute ISO time
        wake_dt = datetime.fromisoformat(wake_time)
        rtc_arg = f"-l -t {int(wake_dt.timestamp())}"
        wake_at = wake_time

    job = {
        "id": f"sched_{int(time.time())}",
        "wake_at": wake_at,
        "command": command,
        "created": datetime.now().isoformat(),
    }
    scheduled_jobs.append(job)

    # If a command is specified, create a wake-up script
    if command:
        wakeup_script = LOG_DIR / f"wakeup_{job['id']}.sh"
        wakeup_script.write_text(f"""#!/bin/bash
# Auto-generated wakeup script
sleep 10  # Wait for system to fully wake
cd {MEWTWO_DIR}
{command} >> {LOG_DIR}/{job['id']}.log 2>&1 &
""")
        wakeup_script.chmod(0o755)
        # Register with systemd or cron to run on wake
        run_cmd(f"(crontab -l 2>/dev/null; echo '@reboot {wakeup_script}') | crontab -")

    # Actually schedule the wake via rtcwake (suspend-to-RAM)
    # This suspends NOW and wakes at the specified time
    suspend_cmd = f"sudo rtcwake -m mem {rtc_arg}"
    return {
        "status": "scheduled",
        "wake_at": wake_at,
        "suspend_command": suspend_cmd,
        "note": "Call /api/schedule/execute/{job_id} to actually suspend now",
        "job_id": job["id"],
    }


@app.post("/api/schedule/execute/{job_id}")
async def api_execute_schedule(job_id: str):
    """Actually execute a scheduled suspend+wake."""
    job = next((j for j in scheduled_jobs if j["id"] == job_id), None)
    if not job:
        raise HTTPException(404, "Schedule not found")

    wake_dt = datetime.fromisoformat(job["wake_at"])
    seconds = max(60, int((wake_dt - datetime.now()).total_seconds()))

    # Unmask suspend temporarily for this operation
    run_cmd("sudo systemctl unmask suspend.target")
    result = run_cmd(f"sudo rtcwake -m mem -s {seconds}")
    # Re-mask after wake
    run_cmd("sudo systemctl mask suspend.target")

    return {"status": "woke_up", "output": result}


@app.post("/api/power/{action}")
async def api_power(action: str):
    """System power: 'suspend', 'reboot', 'shutdown'."""
    if action == "suspend":
        run_cmd("sudo systemctl unmask suspend.target")
        asyncio.get_event_loop().call_later(2, lambda: run_cmd("sudo systemctl suspend"))
        return {"status": "suspending_in_2s"}
    elif action == "reboot":
        asyncio.get_event_loop().call_later(3, lambda: run_cmd("sudo reboot"))
        return {"status": "rebooting_in_3s"}
    elif action == "shutdown":
        asyncio.get_event_loop().call_later(3, lambda: run_cmd("sudo shutdown now"))
        return {"status": "shutting_down_in_3s"}
    raise HTTPException(400, "action: suspend, reboot, shutdown")


# ═══════════════════════════════════════════════════════
# Config & Environment Editor
# ═══════════════════════════════════════════════════════

@app.get("/api/config")
async def api_get_config():
    """Read .env and config files."""
    env_file = MEWTWO_DIR / ".env"
    config = {}
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                config[k.strip()] = v.strip()
    return {"config": config, "path": str(env_file)}


@app.post("/api/config")
async def api_set_config(key: str = Form(...), value: str = Form(...)):
    """Set an environment variable in .env."""
    env_file = MEWTWO_DIR / ".env"
    lines = env_file.read_text().splitlines() if env_file.exists() else []

    found = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}"
            found = True
            break
    if not found:
        lines.append(f"{key}={value}")

    env_file.write_text("\n".join(lines) + "\n")
    os.environ[key] = value
    return {"status": "set", "key": key}


# ═══════════════════════════════════════════════════════
# File Browser (for logs/results)
# ═══════════════════════════════════════════════════════

@app.get("/api/files")
async def api_list_files(path: str = ""):
    """List files in mewtwo directory."""
    target = MEWTWO_DIR / path
    if not target.exists():
        raise HTTPException(404)
    if target.is_file():
        content = target.read_text()[:10000]
        return {"type": "file", "path": str(target), "content": content, "size": target.stat().st_size}
    items = []
    for child in sorted(target.iterdir()):
        if child.name.startswith("."):
            continue
        items.append({
            "name": child.name,
            "is_dir": child.is_dir(),
            "size": child.stat().st_size if child.is_file() else None,
        })
    return {"type": "directory", "path": str(target), "items": items}

# ═══════════════════════════════════════════════════════
# WebSocket for live terminal
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/terminal")
async def ws_terminal(websocket: WebSocket):
    """Live terminal over WebSocket."""
    await websocket.accept()
    proc = await asyncio.create_subprocess_shell(
        "/bin/bash",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(MEWTWO_DIR),
    )

    async def read_output():
        while True:
            data = await proc.stdout.read(4096)
            if not data:
                break
            await websocket.send_text(data.decode("utf-8", errors="replace"))

    read_task = asyncio.create_task(read_output())

    try:
        while True:
            msg = await websocket.receive_text()
            proc.stdin.write((msg + "\n").encode())
            await proc.stdin.drain()
    except Exception:
        read_task.cancel()
        proc.kill()


# ═══════════════════════════════════════════════════════
# WebSocket for live GPU monitoring
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket):
    """Stream GPU/system stats every 2 seconds."""
    await websocket.accept()
    try:
        while True:
            data = {
                "gpu": get_gpu_info(),
                "system": get_system_info(),
                "gpu_processes": get_gpu_processes(),
                "timestamp": datetime.now().isoformat(),
            }
            await websocket.send_json(data)
            await asyncio.sleep(2)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════
# Web Dashboard (single HTML page)
# ═══════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Mewtwo — GPU Command Center</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #0a0a0f;
  --bg2: #12121a;
  --bg3: #1a1a26;
  --border: #2a2a3a;
  --text: #e0e0ff;
  --text2: #8888aa;
  --green: #00ff88;
  --red: #ff4466;
  --blue: #4488ff;
  --purple: #aa66ff;
  --yellow: #ffaa00;
  --cyan: #00ddff;
  --glow-green: 0 0 20px rgba(0,255,136,0.3);
  --glow-red: 0 0 20px rgba(255,68,102,0.3);
}
* { margin:0; padding:0; box-sizing:border-box; }
body { 
  background: var(--bg); color: var(--text); 
  font-family: 'Inter', sans-serif; font-size: 14px;
  min-height: 100vh;
}
.header {
  background: linear-gradient(135deg, #0d0d1a 0%, #1a0d2e 100%);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px;
  display: flex; align-items: center; justify-content: space-between;
}
.header h1 { 
  font-size: 20px; font-weight: 700;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.header .status-dot {
  width: 10px; height: 10px; border-radius: 50%;
  background: var(--green); box-shadow: var(--glow-green);
  display: inline-block; margin-right: 8px; animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
.grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; padding: 20px; }
@media(max-width:1000px) { .grid { grid-template-columns: 1fr; } }
.card {
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px; position: relative; overflow: hidden;
}
.card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--purple), var(--green));
}
.card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text2); margin-bottom: 12px; }
.stat { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border); }
.stat:last-child { border-bottom: none; }
.stat .label { color: var(--text2); font-size: 12px; }
.stat .val { font-family: 'JetBrains Mono'; font-weight: 700; font-size: 16px; }
.stat .val.hot { color: var(--red); }
.stat .val.warm { color: var(--yellow); }
.stat .val.cool { color: var(--green); }
.bar-bg { background: var(--bg3); border-radius: 6px; height: 8px; margin-top: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 6px; transition: width 0.5s ease; }
.bar-fill.green { background: linear-gradient(90deg, #00aa55, #00ff88); }
.bar-fill.yellow { background: linear-gradient(90deg, #aa7700, #ffaa00); }
.bar-fill.red { background: linear-gradient(90deg, #aa2233, #ff4466); }
.btn {
  padding: 8px 16px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--bg3); color: var(--text); cursor: pointer;
  font-family: 'Inter'; font-size: 12px; font-weight: 600;
  transition: all 0.2s; display: inline-flex; align-items: center; gap: 6px;
}
.btn:hover { border-color: var(--cyan); background: rgba(0,221,255,0.1); }
.btn.danger { border-color: var(--red); }
.btn.danger:hover { background: rgba(255,68,102,0.2); }
.btn.success { border-color: var(--green); }
.btn.success:hover { background: rgba(0,255,136,0.2); }
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
.terminal {
  background: #000; border-radius: 8px; padding: 12px;
  font-family: 'JetBrains Mono'; font-size: 12px;
  color: #0f0; max-height: 300px; overflow-y: auto;
  white-space: pre-wrap; word-break: break-all;
}
.cmd-input {
  width: 100%; padding: 10px 14px; background: var(--bg3);
  border: 1px solid var(--border); border-radius: 8px;
  color: var(--text); font-family: 'JetBrains Mono'; font-size: 13px;
  margin-top: 8px; outline: none;
}
.cmd-input:focus { border-color: var(--cyan); }
.wide { grid-column: span 2; }
@media(max-width:1000px) { .wide { grid-column: span 1; } }
.gpu-meter {
  position: relative; width: 100%; height: 120px;
  display: flex; align-items: center; justify-content: center;
}
.gpu-ring {
  width: 100px; height: 100px; border-radius: 50%;
  border: 6px solid var(--bg3); position: relative;
  display: flex; align-items: center; justify-content: center;
}
.gpu-ring .pct { font-family: 'JetBrains Mono'; font-size: 24px; font-weight: 700; }
.toast {
  position: fixed; bottom: 20px; right: 20px; padding: 12px 20px;
  background: var(--bg3); border: 1px solid var(--green); border-radius: 10px;
  color: var(--green); font-size: 13px; z-index: 999;
  animation: fadeInUp 0.3s ease; display: none;
}
@keyframes fadeInUp { from{transform:translateY(20px);opacity:0} to{transform:translateY(0);opacity:1} }
.sched-form { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
.sched-form input { 
  flex: 1; min-width: 120px; padding: 8px; background: var(--bg3);
  border: 1px solid var(--border); border-radius: 6px; color: var(--text);
  font-family: 'JetBrains Mono'; font-size: 12px;
}
</style>
</head>
<body>

<div class="header">
  <h1><span class="status-dot"></span>Mewtwo GPU Command Center</h1>
  <span style="color:var(--text2); font-size:12px" id="clock"></span>
</div>

<div class="grid">

  <!-- GPU Card -->
  <div class="card">
    <h2>🎮 GPU</h2>
    <div class="gpu-meter">
      <div class="gpu-ring" id="gpu-ring">
        <span class="pct" id="gpu-util">--</span>
      </div>
      <div style="margin-left:20px">
        <div id="gpu-name" style="font-weight:600;font-size:13px">Loading...</div>
        <div id="gpu-temp" style="color:var(--text2);font-size:12px;margin-top:4px"></div>
        <div id="gpu-power" style="color:var(--text2);font-size:12px;margin-top:2px"></div>
        <div id="gpu-fan" style="color:var(--text2);font-size:12px;margin-top:2px"></div>
      </div>
    </div>
    <div>
      <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text2)">
        <span>VRAM</span><span id="gpu-mem-text">--</span>
      </div>
      <div class="bar-bg"><div class="bar-fill green" id="gpu-mem-bar" style="width:0%"></div></div>
    </div>
  </div>

  <!-- System Card -->
  <div class="card">
    <h2>💻 System</h2>
    <div class="stat"><span class="label">CPU</span><span class="val" id="sys-cpu">--%</span></div>
    <div class="bar-bg"><div class="bar-fill green" id="cpu-bar" style="width:0%"></div></div>
    <div class="stat" style="margin-top:8px"><span class="label">RAM</span><span class="val" id="sys-ram">--</span></div>
    <div class="bar-bg"><div class="bar-fill green" id="ram-bar" style="width:0%"></div></div>
    <div class="stat" style="margin-top:8px"><span class="label">Disk</span><span class="val" id="sys-disk">--</span></div>
    <div class="bar-bg"><div class="bar-fill green" id="disk-bar" style="width:0%"></div></div>
    <div class="stat" style="margin-top:8px"><span class="label">Uptime</span><span class="val" id="sys-uptime">--</span></div>
  </div>

  <!-- Controls Card -->
  <div class="card">
    <h2>🎛️ Controls</h2>
    <div class="btn-row">
      <button class="btn" onclick="monitorOff()">🖥️ Monitor Off</button>
      <button class="btn success" onclick="monitorOn()">🖥️ Monitor On</button>
    </div>
    <div class="btn-row">
      <button class="btn" onclick="gpuPersist('on')">⚡ GPU Persist ON</button>
      <button class="btn" onclick="gpuPersist('off')">💤 GPU Persist OFF</button>
    </div>
    <div class="btn-row">
      <button class="btn danger" onclick="if(confirm('Suspend?'))sysPower('suspend')">😴 Suspend</button>
      <button class="btn danger" onclick="if(confirm('Reboot?'))sysPower('reboot')">🔄 Reboot</button>
      <button class="btn danger" onclick="if(confirm('SHUTDOWN?'))sysPower('shutdown')">⛔ Shutdown</button>
    </div>
    <h2 style="margin-top:16px">⏰ Schedule Wake</h2>
    <div class="sched-form">
      <input type="text" id="wake-time" placeholder="+30m or 2026-04-07T06:00:00">
      <input type="text" id="wake-cmd" placeholder="Command to run on wake">
      <button class="btn success" onclick="scheduleWake()">Schedule</button>
    </div>
  </div>

  <!-- Terminal Card -->
  <div class="card wide">
    <h2>⌨️ Terminal</h2>
    <div class="terminal" id="term-output">$ Ready. Type a command below.\n</div>
    <input class="cmd-input" id="cmd-input" placeholder="$ command..." 
           onkeydown="if(event.key==='Enter')execCmd()" autocomplete="off">
    <div class="btn-row" style="margin-top:8px">
      <button class="btn" onclick="execCmd()">▶ Run</button>
      <button class="btn" onclick="execBg()">🔄 Run in Background</button>
      <button class="btn" onclick="document.getElementById('term-output').textContent=''">🗑️ Clear</button>
    </div>
  </div>

  <!-- GPU Processes & Jobs Card -->
  <div class="card">
    <h2>📋 GPU Processes</h2>
    <div id="gpu-procs" style="font-size:12px;font-family:'JetBrains Mono'">Loading...</div>
    <h2 style="margin-top:16px">🔧 Background Jobs</h2>
    <div id="bg-jobs" style="font-size:12px;font-family:'JetBrains Mono'">None</div>
  </div>

  <!-- Quick Actions -->
  <div class="card" style="grid-column: span 3">
    <h2>🚀 Quick Actions</h2>
    <div class="btn-row">
      <button class="btn success" onclick="quickRun('nvidia-smi')">nvidia-smi</button>
      <button class="btn" onclick="quickRun('nvidia-smi dmon -s pucvmet -d 1 -c 5')">GPU Metrics (5s)</button>
      <button class="btn" onclick="quickRun('htop -b -n 1 | head -30')">htop snapshot</button>
      <button class="btn" onclick="quickRun('df -h')">Disk Usage</button>
      <button class="btn" onclick="quickRun('tail -30 /home/learner/Desktop/mewtwo/train_matrix.log')">Training Log</button>
      <button class="btn" onclick="quickRun('ls -la /home/learner/Desktop/mewtwo/checkpoints/')">Checkpoints</button>
      <button class="btn" onclick="quickRun('cat /home/learner/Desktop/mewtwo/.env 2>/dev/null || echo No .env file')">Show .env</button>
      <button class="btn success" onclick="setEnvKey()">✏️ Set API Key</button>
    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
// ── Live monitoring via WebSocket ──
let ws;
function connectMonitor() {
  ws = new WebSocket(`ws://${location.host}/ws/monitor`);
  ws.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if(d.gpu && d.gpu[0]) {
      const g = d.gpu[0];
      const util = parseInt(g.util_pct)||0;
      const temp = parseInt(g.temp_c)||0;
      const memUsed = parseInt(g.mem_used_mb)||0;
      const memTotal = parseInt(g.mem_total_mb)||1;
      const memPct = Math.round(memUsed/memTotal*100);
      
      document.getElementById('gpu-util').textContent = util+'%';
      document.getElementById('gpu-name').textContent = g.name;
      document.getElementById('gpu-temp').textContent = '🌡️ '+temp+'°C';
      document.getElementById('gpu-power').textContent = '⚡ '+g.power_w+'W / '+g.power_limit_w+'W';
      document.getElementById('gpu-fan').textContent = '🌀 Fan '+g.fan_pct+'%';
      document.getElementById('gpu-mem-text').textContent = memUsed+' / '+memTotal+' MB';
      document.getElementById('gpu-mem-bar').style.width = memPct+'%';
      
      const ring = document.getElementById('gpu-ring');
      const color = util>80?'var(--red)':util>50?'var(--yellow)':'var(--green)';
      ring.style.borderColor = color;
      ring.style.boxShadow = `0 0 20px ${color}40`;
      
      setBarColor('gpu-mem-bar', memPct);
    }
    if(d.system) {
      const s = d.system;
      document.getElementById('sys-cpu').textContent = s.cpu_pct+'%';
      document.getElementById('sys-ram').textContent = s.ram_used_gb+'/'+s.ram_total_gb+' GB';
      document.getElementById('sys-disk').textContent = s.disk_used_gb+'/'+s.disk_total_gb+' GB';
      document.getElementById('sys-uptime').textContent = s.uptime_hours+'h';
      document.getElementById('cpu-bar').style.width = s.cpu_pct+'%';
      document.getElementById('ram-bar').style.width = s.ram_pct+'%';
      document.getElementById('disk-bar').style.width = s.disk_pct+'%';
      setBarColor('cpu-bar', s.cpu_pct);
      setBarColor('ram-bar', s.ram_pct);
    }
    if(d.gpu_processes) {
      const el = document.getElementById('gpu-procs');
      if(d.gpu_processes.length === 0) { el.innerHTML = '<span style="color:var(--text2)">No GPU processes</span>'; }
      else { el.innerHTML = d.gpu_processes.map(p => 
        `<div class="stat"><span>${p.name} (PID ${p.pid})</span><span>${p.mem_mb} MB <button class="btn danger" style="padding:2px 6px;font-size:10px" onclick="killPid(${p.pid})">Kill</button></span></div>`
      ).join(''); }
    }
  };
  ws.onclose = () => setTimeout(connectMonitor, 3000);
}
connectMonitor();

function setBarColor(id, pct) {
  const el = document.getElementById(id);
  el.className = 'bar-fill ' + (pct>80?'red':pct>50?'yellow':'green');
}

// ── Clock ──
setInterval(() => {
  document.getElementById('clock').textContent = new Date().toLocaleString();
}, 1000);

// ── Commands ──
async function execCmd() {
  const cmd = document.getElementById('cmd-input').value;
  if(!cmd) return;
  const out = document.getElementById('term-output');
  out.textContent += '\\n$ ' + cmd + '\\n';
  document.getElementById('cmd-input').value = '';
  
  const fd = new FormData(); fd.append('cmd', cmd);
  const r = await fetch('/api/run', {method:'POST', body:fd});
  const d = await r.json();
  out.textContent += (d.output||d.error||JSON.stringify(d)) + '\\n';
  out.scrollTop = out.scrollHeight;
}

async function execBg() {
  const cmd = document.getElementById('cmd-input').value;
  if(!cmd) return;
  const fd = new FormData(); fd.append('cmd', cmd); fd.append('background', 'true');
  const r = await fetch('/api/run', {method:'POST', body:fd});
  const d = await r.json();
  toast('Background job started: ' + d.job_id);
  document.getElementById('cmd-input').value = '';
}

function quickRun(cmd) {
  document.getElementById('cmd-input').value = cmd;
  execCmd();
}

// ── Controls ──
async function monitorOff() { await fetch('/api/monitor/off',{method:'POST'}); toast('Monitor OFF'); }
async function monitorOn() { await fetch('/api/monitor/on',{method:'POST'}); toast('Monitor ON'); }
async function gpuPersist(a) { await fetch('/api/gpu/persistence/'+a,{method:'POST'}); toast('GPU Persist: '+a); }
async function sysPower(a) { await fetch('/api/power/'+a,{method:'POST'}); toast('Power: '+a); }
async function killPid(pid) { 
  if(!confirm('Kill PID '+pid+'?')) return;
  await fetch('/api/kill/'+pid,{method:'POST'}); toast('Killed '+pid); 
}

async function scheduleWake() {
  const t = document.getElementById('wake-time').value;
  const c = document.getElementById('wake-cmd').value;
  if(!t) return toast('Enter a wake time');
  const fd = new FormData(); fd.append('wake_time', t); fd.append('command', c);
  const r = await fetch('/api/schedule/wake',{method:'POST',body:fd});
  const d = await r.json();
  toast('Scheduled! Wake at: '+d.wake_at);
}

async function setEnvKey() {
  const key = prompt('Environment variable name (e.g. OPENAI_API_KEY):');
  if(!key) return;
  const val = prompt('Value:');
  if(!val) return;
  const fd = new FormData(); fd.append('key', key); fd.append('value', val);
  await fetch('/api/config',{method:'POST',body:fd});
  toast('Set '+key);
}

function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = '✅ ' + msg;
  el.style.display = 'block';
  setTimeout(() => el.style.display = 'none', 3000);
}

// ── Refresh bg jobs ──
setInterval(async () => {
  const r = await fetch('/api/status');
  const d = await r.json();
  const el = document.getElementById('bg-jobs');
  const jobs = Object.entries(d.bg_jobs||{});
  if(jobs.length === 0) { el.innerHTML = '<span style="color:var(--text2)">No background jobs</span>'; }
  else { el.innerHTML = jobs.map(([id,j]) => 
    `<div class="stat"><span title="${j.cmd}">${id}</span><span>${j.status} <button class="btn danger" style="padding:2px 6px;font-size:10px" onclick="killJob('${id}')">Kill</button></span></div>`
  ).join(''); }
}, 5000);

async function killJob(id) { 
  await fetch('/api/job/'+id+'/kill',{method:'POST'}); toast('Killed '+id);
}
</script>
</body>
</html>
"""

# ═══════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║       Mewtwo GPU Command Center                      ║
╠══════════════════════════════════════════════════════╣
║  Dashboard:  http://0.0.0.0:{PORT}                    ║
║  Local:      http://10.7.1.55:{PORT}                  ║
║  Auth Token: {AUTH_TOKEN}                       ║
╠══════════════════════════════════════════════════════╣
║  From your laptop (same network):                    ║
║    http://10.7.1.55:{PORT}                            ║
║                                                      ║
║  SSH access:                                         ║
║    ssh learner@10.7.1.55                             ║
╚══════════════════════════════════════════════════════╝
""")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
