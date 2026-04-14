import os
import sys
from typing import Optional


class ProxyBridge:
    """
    Optional bridge to ~/Desktop/perplexity-proxy.
    Enabled only when CLUSTER_USE_PERPLEXITY_PROXY=1.
    """

    def __init__(self):
        self.enabled = os.getenv("CLUSTER_USE_PERPLEXITY_PROXY", "0") == "1"
        self.proxy_path = os.path.expanduser("~/Desktop/perplexity-proxy")
        self._client = None

    def _ensure_client(self):
        if not self.enabled:
            return None
        if self._client is not None:
            return self._client
        if not os.path.isdir(self.proxy_path):
            return None
        if self.proxy_path not in sys.path:
            sys.path.insert(0, self.proxy_path)
        try:
            import perplexity  # type: ignore

            self._client = perplexity.Client()
            return self._client
        except Exception:
            return None

    def ask(self, query: str, mode: str = "reasoning", model: Optional[str] = None) -> Optional[str]:
        client = self._ensure_client()
        if client is None:
            return None
        try:
            response = client.search(
                query,
                mode=mode,
                model=model,
                sources=["web"],
                stream=False,
                files={},
                language="en-US",
                follow_up=None,
                incognito=False,
            )
            if isinstance(response, dict):
                return response.get("answer") or str(response)
            return str(response)
        except Exception:
            return None
