import os
import json
import time
import sys
from collections import defaultdict
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text

# Configuration
# Path is relative to the PROJECT_ROOT (which we'll assume is the CWD where this is run)
RESULTS_FILE = Path("results/injection_track_a.jsonl")
TOTAL_ITEMS = 440  # 40 items * 11 methods

console = Console()

def load_data():
    if not RESULTS_FILE.exists():
        return []
    data = []
    try:
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return data

def make_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3)
    )
    return layout

def generate_table(data):
    # Metrics breakdown
    metrics = defaultdict(lambda: {"sim": [], "ppl": [], "lat": [], "count": 0})
    
    for item in data:
        m = item["method"]
        metrics[m]["sim"].append(item["semantic_sim"])
        metrics[m]["ppl"].append(item["perplexity"])
        metrics[m]["lat"].append(item["latency_s"])
        metrics[m]["count"] += 1
    
    table = Table(title="Live Evaluation Metrics", expand=True)
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Avg Sim", justify="right", style="green")
    table.add_column("Avg PPL", justify="right", style="magenta")
    table.add_column("Avg Lat (s)", justify="right", style="yellow")
    
    # Sort methods by sim or alphabetically
    sorted_methods = sorted(metrics.keys())
    
    for m in sorted_methods:
        stats = metrics[m]
        count = stats["count"]
        if count == 0: continue
        avg_sim = sum(stats["sim"]) / count
        avg_ppl = sum(stats["ppl"]) / count
        avg_lat = sum(stats["lat"]) / count
        
        table.add_row(
            m,
            str(count),
            f"{avg_sim:.4f}",
            f"{avg_ppl:.2f}",
            f"{avg_lat:.2f}"
        )
    
    return table

def generate_content(data):
    if not data:
        return Panel(Text("Waiting for data in results/injection_track_a.jsonl...", justify="center", style="bold yellow"))
    
    table = generate_table(data)
    last_item = data[-1]
    last_info = Text.assemble(
        ("Last Item: ", "bold"),
        (f"{last_item.get('item_id', '???')} ", "cyan"),
        ("Method: ", "bold"),
        (f"{last_item.get('method', '???')} ", "yellow"),
        ("Sim: ", "bold"),
        (f"{last_item.get('semantic_sim', 0):.4f}", "green")
    )
    
    return Panel(table, subtitle=last_info)

def run_monitor():
    layout = make_layout()
    header = Text("Mewtwo Hypotheses Pipeline Monitor", justify="center", style="bold white on blue")
    layout["header"].update(Panel(header))
    
    with Live(layout, refresh_per_second=1, screen=True) as live:
        while True:
            data = load_data()
            progress = len(data)
            
            # Update body
            layout["body"].update(generate_content(data))
            
            # Update footer
            progress_pct = progress/TOTAL_ITEMS if TOTAL_ITEMS > 0 else 0
            progress_text = Text(f"Progress: {progress}/{TOTAL_ITEMS} ({progress_pct:.1%})", justify="center", style="bold")
            layout["footer"].update(Panel(progress_text))
            
            time.sleep(2)

if __name__ == "__main__":
    try:
        run_monitor()
    except KeyboardInterrupt:
        console.print("[bold red]Monitor stopped.[/bold red]")
