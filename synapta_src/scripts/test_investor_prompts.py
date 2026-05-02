import asyncio
import json
import websockets

PROMPTS = [
    "Write a heavily optimized C++ function that uses Euler's method to simulate the trajectory of a projectile subject to quadratic air drag, and calculate the maximum theoretical altitude.",
    "Design a SQL database schema for storing genomic sequencing data that optimizes for querying specific allele frequencies, and write the query to find the variance.",
    "Provide a mathematical proof for the worst-case time complexity of the A* search algorithm, then write the Python implementation.",
    "If I have 5 apples and eat 2, and then the French Revolution happens in 1789, what is the square root of the year the Bastille fell?",
    "Translate the concept of quantum entanglement into a Python class structure. Do not explain the physics, just give me the code.",
    "Derive the Taylor series expansion for $f(x) = e^x \\cos(x)$ centered at $a = 0$ up to the $x^4$ term.",
    "Explain the exact mechanism by which speculative decoding failures impact memory bandwidth on ARM-based unified memory architectures.",
    "Analyze this sequence of operations: eval(base64.b64decode(user_input)). Explain the specific security vulnerability and rewrite it to be safe for a production environment."
]

URI = "ws://localhost:8765/ws/generate"

async def test_prompts():
    print("# Investor Pitch Prompts: Routing Evaluation\n")
    
    with open("/home/learner/Desktop/mewtwo/results/investor_prompt_results.md", "w") as f:
        f.write("# Investor Pitch Prompts: Routing Evaluation\n\n")

        for idx, prompt in enumerate(PROMPTS, 1):
            print(f"Testing Prompt {idx}/10...")
            f.write(f"## Prompt {idx}\n> {prompt}\n\n")
            
            try:
                # Add ping timeout configs for long generation
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
                        elif data["type"] == "swap":
                            swaps += 1
                        elif data["type"] == "done":
                            f.write(f"**Output (Swaps: {data.get('swaps', swaps)}, Time: {data.get('elapsed_s', 0)}s, Tokens: {data.get('total_tokens', 0)}):**\n")
                            f.write("```\n")
                            f.write(full_text)
                            f.write("\n```\n\n---\n\n")
                            f.flush()
                            print(f"  -> Done! Swaps: {data.get('swaps', swaps)}")
                            break
                        elif data["type"] == "error":
                            print(f"  -> Error: {data.get('message')}")
                            f.write(f"**Error:** {data.get('message')}\n\n---\n\n")
                            f.flush()
                            break
            except Exception as e:
                print(f"  -> Connection error: {e}")
                f.write(f"**Connection Error:** {e}\n\n---\n\n")

if __name__ == "__main__":
    import os
    os.makedirs("/home/learner/Desktop/mewtwo/results", exist_ok=True)
    asyncio.run(test_prompts())
