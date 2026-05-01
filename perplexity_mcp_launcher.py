#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from mcp.server.fastmcp import FastMCP
from perplexity import Client


def _load_cookies() -> dict:
    if "PERPLEXITY_COOKIES" in os.environ:
        raw = os.environ["PERPLEXITY_COOKIES"]
    else:
        repo_root = Path("/home/learner/Downloads/perplexity-proxy")
        env_file = Path(
            os.environ.get(
                "PERPLEXITY_PROXY_ENV_FILE",
                str(repo_root / ".env"),
            )
        )
        if not env_file.exists():
            return {}
        env_values = dotenv_values(env_file)
        raw = env_values.get("PERPLEXITY_COOKIES", "")

    if not raw:
        return {}

    try:
        return json.loads("".join(raw.splitlines()).strip())
    except json.JSONDecodeError as exc:
        sys.exit(f"ERROR: PERPLEXITY_COOKIES is not valid JSON: {exc}")


def _extract_text(response: dict) -> str:
    answer = response.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()

    text = response.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    final_msg = response.get("final_sse_message")
    if isinstance(final_msg, dict):
        content = final_msg.get("text") or final_msg.get("answer")
        if isinstance(content, str) and content.strip():
            return content.strip()

    status = response.get("status")
    error_code = response.get("error_code")
    if status or error_code:
        parts = [part for part in [status, error_code] if part]
        return f"Perplexity returned no answer ({', '.join(parts)})."

    return "Perplexity returned no answer."


cookies = _load_cookies()
client = Client(cookies)

mcp = FastMCP(
    "perplexityPro",
    host=os.environ.get("MCP_HOST", "127.0.0.1"),
    port=int(os.environ.get("MCP_PORT", "8000")),
)


def _search(query: str, mode: str, target_model: str | None = None) -> str:
    kwargs = {"mode": mode}
    if target_model:
        kwargs["model"] = target_model
    response = client.search(query, **kwargs)
    return _extract_text(response)


@mcp.tool()
def perplexity_ask(query: str, target_model: str | None = None) -> str:
    """General-purpose Perplexity answer in auto mode."""
    return _search(query, "auto", target_model)


if client.own:
    @mcp.tool()
    def perplexity_search(query: str, target_model: str | None = None) -> str:
        """Perplexity Pro web search with synthesized answer."""
        return _search(query, "pro", target_model)


    @mcp.tool()
    def perplexity_reason(query: str, target_model: str | None = None) -> str:
        """Perplexity reasoning mode for step-by-step analysis."""
        return _search(query, "reasoning", target_model)


    @mcp.tool()
    def perplexity_research(query: str, target_model: str | None = None) -> str:
        """Perplexity deep research mode for long-form topic research."""
        return _search(query, "deep research", target_model)


def main() -> None:
    os.environ.setdefault("MCP_TRANSPORT", "stdio")
    transport = os.environ["MCP_TRANSPORT"]
    if transport == "stdio":
        mcp.run()
    elif transport == "http":
        mcp.run(transport="streamable-http")
    else:
        sys.exit("ERROR: MCP_TRANSPORT must be 'stdio' or 'http'")


if __name__ == "__main__":
    main()
