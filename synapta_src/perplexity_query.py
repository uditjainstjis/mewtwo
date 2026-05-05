#!/usr/bin/env python3
"""Reusable perplexity search wrapper.
Usage: python synapta_src/perplexity_query.py "your search query here"
       python synapta_src/perplexity_query.py "query" --mode pro
       python synapta_src/perplexity_query.py "query" --reason  # uses reasoning mode
"""
import sys, os, json, argparse
sys.path.insert(0, '/home/learner/Downloads/perplexity-proxy')
from perplexity import Client
from dotenv import dotenv_values

def _load_cookies():
    raw = os.environ.get('PERPLEXITY_COOKIES', '')
    if not raw:
        env = dotenv_values('/home/learner/Downloads/perplexity-proxy/.env')
        raw = env.get('PERPLEXITY_COOKIES', '')
    if not raw:
        raise RuntimeError("No PERPLEXITY_COOKIES in env or /home/learner/Downloads/perplexity-proxy/.env")
    try:
        return json.loads("".join(raw.splitlines()).strip())
    except json.JSONDecodeError:
        return {c.split('=')[0].strip(): '='.join(c.split('=')[1:]).strip() for c in raw.split(';') if '=' in c}

VALID_MODES = ("auto", "pro", "reasoning", "deep_research", "deep research")

def search(query: str, mode: str = "auto", reason: bool = False) -> dict:
    cookies = _load_cookies()
    client = Client(cookies)
    api_mode = "deep research" if mode == "deep_research" else mode
    if reason:
        return client.search(query, mode='reasoning')
    return client.search(query, mode=api_mode)

def extract_text(result):
    """Pull just the human-readable text from the perplexity response."""
    parts = []
    for step in result.get('text', []):
        if step.get('step_type') == 'FINAL':
            content = step.get('content', {})
            if 'answer' in content:
                parts.append(content['answer'])
        elif step.get('step_type') == 'SEARCH_RESULTS':
            results = step.get('content', {}).get('web_results', [])
            for r in results[:5]:
                parts.append(f"\n• {r.get('name', '')}: {r.get('snippet', '')[:300]} ({r.get('url', '')})")
    return '\n'.join(parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--mode", default="auto", choices=["auto", "pro", "reasoning", "deep_research"])
    parser.add_argument("--reason", action="store_true", help="Use reasoning focus")
    parser.add_argument("--raw", action="store_true", help="Output raw JSON")
    args = parser.parse_args()
    
    result = search(args.query, mode=args.mode, reason=args.reason)
    if args.raw:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(extract_text(result))
