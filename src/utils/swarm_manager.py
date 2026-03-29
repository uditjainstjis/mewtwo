import subprocess
import time
import os
import signal
import sys
from pathlib import Path

class AgentSwarm:
    def __init__(self, tasks):
        """
        tasks: list of dicts with {'name': str, 'command': str, 'cwd': str}
        """
        self.tasks = tasks
        self.processes = {}
        self.running = True

    def start_task(self, task):
        print(f"[Swarm] Starting agent for task: {task['name']}")
        proc = subprocess.Popen(
            task['command'],
            shell=True,
            cwd=task['cwd'],
            stdout=open(f"logs/swarm_{task['name']}.log", "a"),
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
        self.processes[task['name']] = {
            'proc': proc,
            'task': task,
            'started_at': time.time()
        }

    def monitor(self):
        os.makedirs("logs", exist_ok=True)
        for task in self.tasks:
            self.start_task(task)

        try:
            while self.running:
                for name, info in list(self.processes.items()):
                    ret = info['proc'].poll()
                    if ret is not None:
                        print(f"[Swarm] Agent '{name}' exited with code {ret}. Restarting...")
                        self.start_task(info['task'])
                time.sleep(5)
        except KeyboardInterrupt:
            self.stop_all()

    def stop_all(self):
        self.running = False
        print("\n[Swarm] Stopping all agents...")
        for name, info in self.processes.items():
            os.killpg(os.getpgid(info['proc'].pid), signal.SIGTERM)
        print("[Swarm] Shutdown complete.")

if __name__ == "__main__":
    # Example tasks for the swarm
    evaluation_swarm = [
        {
            "name": "Mistral-SD-Bench",
            "command": "python3 backend/mistral_comparison.py", # This one does SD
            "cwd": str(Path(__file__).resolve().parent.parent.parent)
        },
        {
            "name": "Gated-Router-Live",
            "command": "python3 src/eval/run_eval_gated.py --router embedding",
            "cwd": str(Path(__file__).resolve().parent.parent.parent)
        }
    ]
    
    swarm = AgentSwarm(evaluation_swarm)
    swarm.monitor()
