import asyncio
import json
import websockets
import sys
import re
import time

PROMPTS = [
    {
        "title": "Prompt 1 (The Code/Systems Flex)",
        "prompt": "Write a heavily optimized lock-free concurrent queue in C++ using std::atomic and memory orders (memory_order_acquire, memory_order_release). Explain the ABA problem and how your implementation avoids it."
    },
    {
        "title": "Prompt 2 (The Science/Hardware Flex)",
        "prompt": "Calculate the theoretical steady-state junction temperature of a 3D-stacked silicon die using Through-Silicon Vias (TSVs). Assume a total power dissipation of 150W, an ambient temperature of 25°C, and a total package thermal resistance ($\\theta_{JA}$) of 0.4 °C/W. Show the thermodynamic derivation."
    },
    {
        "title": "Prompt 3 (The Pure Math/Algorithmic Flex)",
        "prompt": "Derive the closed-form solution for the Fibonacci sequence (Binet's formula) using generating functions and partial fraction decomposition. Show every algebraic step."
    }
]

URI = "ws://localhost:8765/ws/generate"

# ANSI Colors for terminal telemetry
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"
RESET = "\033[0m"
BOLD = "\033[1m"

def process_output(text: str) -> str:
    """Removes thinking blocks and standardizes the final answer wrapper."""
    # Strip <think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Optional: If the model uses 'The answer is:' or 'The final answer is:', 
    # we can format it explicitly, but here we just ensure clean spacing.
    text = text.replace("The answer is:", f"\n{BOLD}🎯 The answer is:{RESET}").strip()
    return text

async def run_pitch():
    print(f"\n{BOLD}{MAGENTA}=======================================================")
    print(f"🚀 SYNAPTA ENTERPRISE ROUTING ENGINE — INVESTOR DEMO")
    print(f"======================================================={RESET}\n")

    for p in PROMPTS:
        title = p["title"]
        prompt = p["prompt"]

        print(f"{BOLD}{CYAN}▶ {title}{RESET}")
        print(f"{YELLOW}Input:{RESET} {prompt}\n")
        print(f"{YELLOW}Status:{RESET} {BOLD}Routing...{RESET}", end="", flush=True)

        try:
            async with websockets.connect(URI, ping_interval=None) as ws:
                payload = {
                    "prompt": prompt,
                    "mode": "routed",
                    "adapter": None,
                    "max_tokens": 2048,
                    "thinking": True
                }
                await ws.send(json.dumps(payload))
                
                full_text = ""
                swaps = 0
                
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    
                    if data["type"] == "token":
                        full_text += data["text"]
                        # We don't print tokens mid-fight to keep the terminal clean 
                        # for the final impact, or we can print dots.
                        print(".", end="", flush=True)
                    elif data["type"] == "swap":
                        swaps += 1
                        val = data.get("to", "unknown").upper()
                        print(f"[{MAGENTA}SWAP ⚡ {val}{RESET}]", end="", flush=True)
                    elif data["type"] == "done":
                        print(f"\n\n{BOLD}{GREEN}✅ GENERATION COMPLETE{RESET}")
                        print(f"-------------------------------------------------------")
                        
                        clean_text = process_output(full_text)
                        print(f"{clean_text}\n")
                        
                        elapsed = data.get("elapsed_s", 0)
                        speed = data.get("speed", 0)
                        total_tokens = data.get("total_tokens", 0)
                        final_swaps = data.get("swaps", swaps)
                        
                        print(f"{BOLD}📊 TELEMETRY REPORT:{RESET}")
                        print(f"  • {CYAN}Total Tokens:{RESET}   {total_tokens}")
                        print(f"  • {CYAN}Throughput:{RESET}     {speed} Tok/sec")
                        print(f"  • {CYAN}Context Latency:{RESET} {elapsed}s")
                        print(f"  • {MAGENTA}Adapter Swaps:{RESET}  {final_swaps}")
                        print(f"-------------------------------------------------------\n")
                        break
                    elif data["type"] == "error":
                        print(f"\n{BOLD}\033[91m❌ ERROR: {data.get('message')}{RESET}\n")
                        break
                        
        except Exception as e:
            print(f"\n{BOLD}\033[91m❌ CONNECTION ERROR: {e}{RESET}\n")

if __name__ == "__main__":
    asyncio.run(run_pitch())
