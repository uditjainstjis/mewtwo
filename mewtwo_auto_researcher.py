import time
import subprocess
import os
import sys
from datetime import datetime

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("pip install rich not finished yet. Waiting 10 seconds...", flush=True)
    time.sys.exit(1)

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.research.hypothesis_generator import HypothesisGenerator

console = Console()
researcher = HypothesisGenerator()

class HardwareTelemetry:
    def get_gpu_status(self):
        try:
            res = subprocess.check_output(
                "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits",
                shell=True, text=True
            ).strip().split(',')
            return {
                "util": float(res[0]),
                "mem_used": float(res[1]),
                "mem_total": float(res[2]),
                "power": float(res[3])
            }
        except Exception:
            return {"util": 0, "mem_used": 0, "mem_total": 32607, "power": 0}

telemetry = HardwareTelemetry()

def get_latest_logs():
    try:
        log_path = os.path.join(os.path.dirname(__file__), 'train_matrix.log')
        if os.path.exists(log_path):
            lines = subprocess.check_output(['tail', '-n', '25', log_path], text=True).splitlines()
            return "\n".join(lines)
        return "No active training logs found."
    except Exception as e:
        return f"Error reading logs: {e}"

def generate_dashboard(current_job=None, gpu_status=None):
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main")
    )
    layout["main"].split_row(
        Layout(name="logs", ratio=3),
        Layout(name="sidebar", ratio=2)
    )
    
    # Header Construction
    gpu_pct = gpu_status["util"] if gpu_status else 0
    mem_used = gpu_status["mem_used"] if gpu_status else 0
    power = gpu_status["power"] if gpu_status else 0
    
    color = "green" if gpu_pct > 80 else ("yellow" if gpu_pct > 30 else "red")
    
    header_text = f"🧪 MEWTWO AUTONOMOUS LABORATORY 🧪\nVolatile GPU: [{color}]{gpu_pct}%[/{color}] | VRAM: {mem_used:.0f}/32607 MB | Power: {power} W"
    layout["header"].update(Panel(Text.from_markup(header_text), style="bold cyan", border_style="cyan"))
    
    # Logs Construction
    log_text = get_latest_logs()
    layout["logs"].update(Panel(log_text, title="[green]Live 7B BF16 Training Stream", border_style="green"))
    
    # Sidebar Construction
    table = Table(title="Procedural Target Queue", header_style="bold magenta")
    table.add_column("EXP ID", style="cyan", no_wrap=True)
    table.add_column("Rank", justify="center")
    table.add_column("Lambda", justify="center")
    
    for i, hyp in enumerate(researcher.queue[:13]):
        table.add_row(hyp.id, str(hyp.rank), str(hyp.lambda_ortho))
    
    layout["sidebar"].split_column(
        Layout(Panel(table, border_style="magenta", padding=(0,0))),
        Layout(Panel(Text(f"Running:\n{current_job.description if current_job else 'Waiting for VRAM < 15GB'}", style="bold red"), title="[red]Active GPU Lock", border_style="red", size=8))
    )
    
    return layout

def auto_research_loop():
    active_job = None
    
    with Live(generate_dashboard(gpu_status=telemetry.get_gpu_status()), refresh_per_second=2, screen=True) as live:
        while True:
            gpu = telemetry.get_gpu_status()
            
            # Autonomous Orchestration Logic
            if gpu["mem_used"] < 15000 and active_job is None:
                # GPU is free enough to run another PEFT model!
                active_job = researcher.get_next()
                
                # In full implementation, we'd fire the subprocess here
                # subprocess.Popen(f"python3 run_experiment.py --rank {active_job.rank}", shell=True)
            
            # If our active job triggers VRAM to go up, we know it's running. 
            # When VRAM drops back down, the job finished.
            if active_job is not None and gpu["mem_used"] < 15000:
                # We assume the job finished or crashed. We cycle to next.
                pass 
                
            live.update(generate_dashboard(current_job=active_job, gpu_status=gpu))
            time.sleep(1)

if __name__ == "__main__":
    try:
        auto_research_loop()
    except KeyboardInterrupt:
        pass
