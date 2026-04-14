from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path


class PerplexityReasoningClient:
    """Small sync wrapper around the local perplexity-proxy repo."""

    def __init__(self, proxy_repo: str | Path = "~/Desktop/perplexity-proxy") -> None:
        self.proxy_repo = Path(proxy_repo).expanduser().resolve()
        if not self.proxy_repo.is_dir():
            raise FileNotFoundError(f"Proxy repo not found: {self.proxy_repo}")
        if str(self.proxy_repo) not in sys.path:
            sys.path.insert(0, str(self.proxy_repo))
        import perplexity_async  # type: ignore

        self.perplexity_async = perplexity_async
        raw = os.environ.get("PERPLEXITY_COOKIES", "")
        if not raw.strip():
            raw = self._load_cookies_from_env_file()
        if not raw.strip():
            raise RuntimeError("PERPLEXITY_COOKIES is not set in the environment.")
        self.cookies = json.loads(raw.replace("\n", ""))

    def _load_cookies_from_env_file(self) -> str:
        env_path = self.proxy_repo / ".env"
        if not env_path.is_file():
            return ""
        text = env_path.read_text()
        match = re.search(r"PERPLEXITY_COOKIES='(.*?)'\n", text, re.S)
        if not match:
            return ""
        return match.group(1)

    async def _ask_async(
        self,
        prompt: str,
        *,
        mode: str,
        model: str,
        sources: list[str],
    ) -> str:
        client = await self.perplexity_async.Client(self.cookies)
        resp = await client.search(
            prompt,
            mode=mode,
            model=model,
            sources=sources,
            files={},
            stream=False,
            language="en-US",
            follow_up=None,
            incognito=False,
        )
        if isinstance(resp, dict):
            return str(resp.get("answer") or "")
        return str(resp)

    def ask(
        self,
        prompt: str,
        *,
        mode: str = "reasoning",
        model: str = "claude-4.6-sonnet-thinking",
        sources: list[str] | None = None,
        max_attempts: int = 3,
        retry_sleep_s: float = 3.0,
    ) -> str:
        last_err = None
        for attempt in range(1, max_attempts + 1):
            try:
                return asyncio.run(
                    self._ask_async(
                        prompt,
                        mode=mode,
                        model=model,
                        sources=sources or ["web"],
                    )
                )
            except Exception as e:
                last_err = e
                if attempt == max_attempts:
                    break
                time.sleep(retry_sleep_s * attempt)
        raise RuntimeError(f"Perplexity request failed after {max_attempts} attempts: {last_err}") from last_err
