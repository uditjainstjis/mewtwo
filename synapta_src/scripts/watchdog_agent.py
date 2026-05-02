import time
import subprocess
import requests
import json
import os
from pathlib import Path

# User provided key
API_KEY = "AIzaSyBM-HNrnUQ6XbiyLrzc92eWhOVOz8V8Fls"
LOG_FILE = "/home/learner/Desktop/mewtwo/unattended_pipeline.log"
EXPLOIT_LOG = "/home/learner/Desktop/mewtwo/exploit.log"

def call_gemini(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code != 200:
            # Fallback to 1.5 flash
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
            response = requests.post(url, headers=headers, json=data)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"ALL_OK (Error hitting API: {e})"

def monitor_loop():
    print(f"🤖 Autonomous Watchdog Agent Started. Monitoring for 20 hours.")
    
    # 20 hours = 40 iterations of 30 mins
    for iteration in range(40):
        # Sleep for 30 minutes (1800 seconds)
        time.sleep(1800)
        
        # Read the latest logs
        log_snippet = ""
        try:
            with open(LOG_FILE, "r") as f:
                log_snippet += "MAIN PIPELINE LOG:\n" + "".join(f.readlines()[-150:])
        except:
            pass
            
        try:
            with open(EXPLOIT_LOG, "r") as f:
                log_snippet += "\n\nEXPLOIT SCRIPT LOG:\n" + "".join(f.readlines()[-50:])
        except:
            pass
            
        prompt = f"""
        You are an autonomous DevOps watchdog for an advanced Machine Learning pipeline running while the researcher is asleep.
        Your goal is to monitor these logs and fix any fatal crashes.
        
        LOGS:
        {log_snippet[-3000:]}
        
        RULES:
        1. If the pipeline is running normally (downloading, generating data, simulating), output strictly: ALL_OK
        2. If you detect a fatal crash (like out of memory, missing dependency, or script failure) that stopped the process, output a single Bash command that will attempt to unblock or restart the script. DO NOT include markdown backticks. Output just the raw command.
        """
        
        response = call_gemini(prompt).strip()
        print(f"[{time.ctime()}] Check {iteration+1}/40. Watchdog says: {response[:50]}")
        
        # If Gemini detected an error and returned a command
        if "ALL_OK" not in response and len(response) > 2 and '\n' not in response:
            print(f"⚠️ WATCHDOG INJECTING COMMAND: {response}")
            try:
                # Log the intervention
                with open(LOG_FILE, "a") as f:
                    f.write(f"\n\n=================================\n")
                    f.write(f"🤖 WATCHDOG INTERVENTION AT {time.ctime()}\n")
                    f.write(f"Executed: {response}\n")
                    f.write(f"=================================\n\n")
                
                # Execute the fix
                subprocess.run(response, shell=True, check=False)
            except Exception as e:
                print(f"Failed to execute watchdog command: {e}")

if __name__ == "__main__":
    monitor_loop()
