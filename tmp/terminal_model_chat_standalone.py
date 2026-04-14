#!/usr/bin/env python3
"""Local model picker and chat launcher for terminal use.

This tool is built to work well with models downloaded via:

    hf download <repo-id>

It scans both local model folders and the default Hugging Face cache, shows a
full-screen picker in the terminal, and launches a chat session for the
selected model using either Transformers or MLX when available.
"""

from __future__ import annotations

import argparse
import curses
import gc
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_MODEL_DIRS = [
    PROJECT_ROOT / "models",
    Path.cwd() / "models",
]

HELP_TEXT = """Commands:
  Enter      select model and start chat
  Up/Down    move
  j/k        move
  r          refresh model list
  q          quit
"""


def human_bytes(num_bytes: Optional[int]) -> str:
    if not num_bytes or num_bytes < 0:
        return "n/a"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{num_bytes}B"


def human_count(count: Optional[float]) -> str:
    if count is None:
        return "n/a"
    if count >= 1_000_000_000:
        return f"{count / 1_000_000_000:.2f}B"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    return f"{int(count)}"


def clip(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    return text[: width - 3] + "..."


def wrap_lines(text: str, width: int) -> List[str]:
    width = max(12, width)
    lines: List[str] = []
    for chunk in text.splitlines() or [""]:
        wrapped = textwrap.wrap(chunk, width=width) or [""]
        lines.extend(wrapped)
    return lines


def extract_param_hint(name: str) -> Optional[str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*([BMbm])", name)
    if not match:
        return None
    value = float(match.group(1))
    suffix = match.group(2).upper()
    if value.is_integer():
        value_text = str(int(value))
    else:
        value_text = str(value)
    return f"{value_text}{suffix}"


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def infer_weight_bytes(model_dir: Path) -> Optional[int]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        metadata = read_json(index_path).get("metadata", {})
        total_size = metadata.get("total_size")
        if isinstance(total_size, int):
            return total_size

    patterns = ("*.safetensors", "*.bin", "*.gguf", "*.ggml")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(model_dir.glob(pattern))
    if not files:
        return None
    return sum(path.stat().st_size for path in files if path.is_file())


def infer_disk_bytes(model_dir: Path) -> int:
    total = 0
    for path in model_dir.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def infer_summary(readme_path: Path) -> str:
    if not readme_path.exists():
        return "No README summary found."
    try:
        lines = [line.strip() for line in readme_path.read_text(errors="ignore").splitlines()]
    except Exception:
        return "README exists but could not be read."

    cleaned = [line for line in lines if line and not line.startswith("![](")]
    if not cleaned:
        return "README exists but has no summary text."

    for line in cleaned:
        if line.startswith("#"):
            continue
        if len(line) < 6:
            continue
        return line
    return cleaned[0]


def infer_repo_id_from_cache(model_dir: Path) -> Optional[str]:
    for parent in model_dir.parents:
        name = parent.name
        if name.startswith("models--"):
            return name[len("models--") :].replace("--", "/")
    return None


def find_active_hf_snapshot(base_dir: Path) -> Optional[Path]:
    snapshots_dir = base_dir / "snapshots"
    if not snapshots_dir.exists():
        return None

    refs_dir = base_dir / "refs"
    for ref_name in ("main", "master"):
        ref_path = refs_dir / ref_name
        if ref_path.exists():
            try:
                commit = ref_path.read_text().strip()
            except Exception:
                continue
            snapshot = snapshots_dir / commit
            if snapshot.exists():
                return snapshot

    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    snapshots.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return snapshots[0]


@dataclass
class ModelInfo:
    name: str
    path: Path
    source: str
    repo_id: Optional[str]
    architecture: str
    model_type: str
    dtype: str
    layers: Optional[int]
    hidden_size: Optional[int]
    max_position_embeddings: Optional[int]
    weight_bytes: Optional[int]
    disk_bytes: int
    params_hint: Optional[str]
    quantization_hint: str
    summary: str
    config_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    config: dict = field(default_factory=dict)
    backend_status: Dict[str, str] = field(default_factory=dict)

    @property
    def display_name(self) -> str:
        return self.repo_id or self.name

    @property
    def source_label(self) -> str:
        if self.source == "hf-cache":
            return "HF cache"
        if self.source == "local":
            return "Local dir"
        return self.source

    @property
    def likely_mlx_model(self) -> bool:
        repo_id = (self.repo_id or "").lower()
        name = self.name.lower()
        return repo_id.startswith("mlx-community/") or name.startswith("mlx-")

    def subtitle(self) -> str:
        parts = [self.params_hint or "unknown size", self.architecture]
        if self.quantization_hint != "unknown":
            parts.append(self.quantization_hint)
        return " | ".join(parts)

    @property
    def preferred_backend(self) -> Optional[str]:
        for name in candidate_backend_names("auto", model=self):
            if self.backend_status.get(name) == "ready":
                return name
        return None

    @property
    def is_runnable(self) -> bool:
        return self.preferred_backend is not None


class ScanError(RuntimeError):
    pass


class BackendError(RuntimeError):
    pass


class ChatBackend:
    name = "backend"

    def load(self, model: ModelInfo) -> None:
        raise NotImplementedError

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        return None

    @staticmethod
    def compose_prompt(messages: Sequence[dict], system_prompt: Optional[str]) -> str:
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.extend(messages)
        lines = []
        for item in prompt_messages:
            role = item["role"].upper()
            lines.append(f"{role}: {item['content']}")
        lines.append("ASSISTANT:")
        return "\n".join(lines)


class TransformersBackend(ChatBackend):
    name = "transformers"

    def __init__(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as exc:
            raise BackendError(
                "Transformers backend unavailable. Install `transformers` and `torch`."
            ) from exc

        self.torch = torch
        self.AutoModelForCausalLM = AutoModelForCausalLM
        self.AutoTokenizer = AutoTokenizer
        self.model = None
        self.tokenizer = None
        self.device = self._pick_device()

    def _pick_device(self) -> str:
        torch = self.torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def load(self, model: ModelInfo) -> None:
        self.tokenizer = self.AutoTokenizer.from_pretrained(
            str(model.path),
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = self.AutoModelForCausalLM.from_pretrained(
            str(model.path),
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, messages: Sequence[dict], system_prompt: Optional[str]) -> str:
        if self.tokenizer is None:
            raise BackendError("Tokenizer not loaded.")
        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.extend(messages)

        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.compose_prompt(messages, system_prompt)

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise BackendError("Model backend is not loaded.")

        prompt = self._build_prompt(messages, system_prompt)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature
        else:
            generate_kwargs["do_sample"] = False

        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **generate_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        new_tokens = output[0][prompt_length:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        if self.device == "mps" and getattr(self.torch, "mps", None):
            try:
                self.torch.mps.empty_cache()
            except Exception:
                pass
        if self.device == "cuda":
            try:
                self.torch.cuda.empty_cache()
            except Exception:
                pass


class MlxBackend(ChatBackend):
    name = "mlx"

    def __init__(self) -> None:
        try:
            from mlx_lm import generate, load  # type: ignore
        except Exception as exc:
            raise BackendError(
                "MLX backend unavailable. Install `mlx` and `mlx-lm`."
            ) from exc
        self.mlx_generate = generate
        self.mlx_load = load
        self.model = None
        self.tokenizer = None

    def load(self, model: ModelInfo) -> None:
        self.model, self.tokenizer = self.mlx_load(str(model.path))

    def _build_prompt(self, messages: Sequence[dict], system_prompt: Optional[str]) -> str:
        if self.tokenizer is None:
            raise BackendError("Tokenizer not loaded.")

        prompt_messages = []
        if system_prompt:
            prompt_messages.append({"role": "system", "content": system_prompt})
        prompt_messages.extend(messages)

        apply_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_template):
            return apply_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.compose_prompt(messages, system_prompt)

    def generate(
        self,
        messages: Sequence[dict],
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise BackendError("Model backend is not loaded.")
        prompt = self._build_prompt(messages, system_prompt)
        # Keep the MLX call conservative because API details vary by version.
        text = self.mlx_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        return text.strip()


def available_backends() -> dict:
    backends = {}
    for backend_cls in (TransformersBackend, MlxBackend):
        try:
            backend = backend_cls()
        except BackendError as exc:
            backends[backend_cls.name] = str(exc)
        else:
            backends[backend_cls.name] = backend
    return backends


def summarize_runtime_state(backends: dict) -> Dict[str, str]:
    state: Dict[str, str] = {}
    for name, backend in backends.items():
        state[name] = "available" if not isinstance(backend, str) else backend
    return state


def detect_transformers_support(model: ModelInfo, backends: dict) -> str:
    backend = backends.get("transformers")
    if isinstance(backend, str):
        return backend
    try:
        from transformers import AutoConfig  # type: ignore
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING  # type: ignore
    except Exception as exc:
        return f"runtime import failed: {exc}"

    model_type = model.model_type
    if model_type not in CONFIG_MAPPING:
        return f"unsupported model type in installed transformers: {model_type}"
    try:
        AutoConfig.from_pretrained(
            str(model.path),
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as exc:
        return f"config load failed: {exc}"
    return "ready"


def detect_mlx_support(model: ModelInfo, backends: dict) -> str:
    backend = backends.get("mlx")
    if isinstance(backend, str):
        return backend
    if model.likely_mlx_model:
        return "ready"
    return "not an MLX-packaged model"


def compute_backend_status(model: ModelInfo, backends: dict) -> Dict[str, str]:
    return {
        "transformers": detect_transformers_support(model, backends),
        "mlx": detect_mlx_support(model, backends),
    }


def model_sort_key(model: ModelInfo) -> Tuple[int, str]:
    support_rank = 0 if model.is_runnable else 1
    preferred = model.preferred_backend or "zzz"
    return (support_rank, preferred, model.display_name.lower())


def choose_backend(name: str, model: Optional[ModelInfo] = None) -> ChatBackend:
    backends = available_backends()
    if name != "auto":
        backend = backends.get(name)
        if isinstance(backend, str):
            raise BackendError(backend)
        if backend is None:
            raise BackendError(f"Unknown backend: {name}")
        return backend

    if model is not None and model.likely_mlx_model:
        candidate_order = ("mlx", "transformers")
    else:
        candidate_order = ("transformers", "mlx")

    for candidate in candidate_order:
        backend = backends.get(candidate)
        if not isinstance(backend, str):
            return backend

    errors = [value for value in backends.values() if isinstance(value, str)]
    raise BackendError("No runtime backend is available.\n" + "\n".join(errors))


def candidate_backend_names(name: str, model: Optional[ModelInfo] = None) -> List[str]:
    if name != "auto":
        return [name]
    if model is not None and model.likely_mlx_model:
        return ["mlx", "transformers"]
    return ["transformers", "mlx"]


def load_model_info(model_dir: Path, source: str) -> Optional[ModelInfo]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None

    config = read_json(config_path)
    text_config = config.get("text_config", {})
    architecture = (
        (config.get("architectures") or ["unknown"])[0]
        if isinstance(config.get("architectures"), list)
        else "unknown"
    )
    if not is_chat_capable_architecture(architecture):
        return None
    model_type = (
        config.get("model_type")
        or text_config.get("model_type")
        or "unknown"
    )
    dtype = (
        text_config.get("dtype")
        or config.get("torch_dtype")
        or config.get("dtype")
        or "unknown"
    )
    layers = text_config.get("num_hidden_layers") or config.get("num_hidden_layers")
    hidden_size = text_config.get("hidden_size") or config.get("hidden_size")
    max_positions = (
        text_config.get("max_position_embeddings")
        or config.get("max_position_embeddings")
    )
    repo_id = infer_repo_id_from_cache(model_dir) if source == "hf-cache" else None
    display_seed = repo_id or model_dir.name
    params_hint = extract_param_hint(display_seed)
    readme_path = model_dir / "README.md"
    summary = infer_summary(readme_path)
    weight_bytes = infer_weight_bytes(model_dir)
    disk_bytes = infer_disk_bytes(model_dir)

    quantization_hint = "unknown"
    lower_seed = display_seed.lower()
    if any(token in lower_seed for token in ("4bit", "int4", "q4", "4-bit")):
        quantization_hint = "4-bit"
    elif any(token in lower_seed for token in ("8bit", "int8", "q8", "8-bit")):
        quantization_hint = "8-bit"
    elif dtype != "unknown":
        quantization_hint = dtype

    return ModelInfo(
        name=model_dir.name,
        path=model_dir.resolve(),
        source=source,
        repo_id=repo_id,
        architecture=architecture,
        model_type=model_type,
        dtype=str(dtype),
        layers=layers if isinstance(layers, int) else None,
        hidden_size=hidden_size if isinstance(hidden_size, int) else None,
        max_position_embeddings=max_positions if isinstance(max_positions, int) else None,
        weight_bytes=weight_bytes,
        disk_bytes=disk_bytes,
        params_hint=params_hint,
        quantization_hint=quantization_hint,
        summary=summary,
        config_path=config_path,
        readme_path=readme_path if readme_path.exists() else None,
        config=config,
    )


def scan_local_model_dirs(paths: Iterable[Path]) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    seen: set[Path] = set()
    for base_dir in paths:
        if not base_dir.exists() or not base_dir.is_dir():
            continue
        for child in sorted(base_dir.iterdir()):
            if not child.is_dir():
                continue
            info = load_model_info(child, source="local")
            if info is None:
                continue
            if info.path in seen:
                continue
            seen.add(info.path)
            models.append(info)
    return models


def scan_hf_cache(cache_dir: Path) -> List[ModelInfo]:
    models: List[ModelInfo] = []
    seen: set[Path] = set()
    if not cache_dir.exists():
        return models

    for child in sorted(cache_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("models--"):
            continue
        snapshot = find_active_hf_snapshot(child)
        if snapshot is None:
            continue
        info = load_model_info(snapshot, source="hf-cache")
        if info is None:
            continue
        if info.path in seen:
            continue
        seen.add(info.path)
        models.append(info)
    return models


def discover_hf_cache_dir() -> Optional[Path]:
    explicit = os.getenv("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def discover_models(extra_dirs: Sequence[str]) -> List[ModelInfo]:
    backends = available_backends()
    local_dirs = list(DEFAULT_LOCAL_MODEL_DIRS)
    for item in extra_dirs:
        local_dirs.append(Path(item).expanduser())

    discovered = scan_local_model_dirs(local_dirs)
    hf_cache = discover_hf_cache_dir()
    if hf_cache is not None:
        discovered.extend(scan_hf_cache(hf_cache))

    deduped: dict[Path, ModelInfo] = {}
    for item in discovered:
        deduped[item.path] = item

    models = list(deduped.values())
    for model in models:
        model.backend_status = compute_backend_status(model, backends)
    models.sort(key=model_sort_key)
    return models


def render_model_list_item(model: ModelInfo, width: int) -> str:
    title = model.display_name
    suffix = model.params_hint or human_bytes(model.weight_bytes)
    status = model.preferred_backend or "blocked"
    base = f"{title} [{suffix}] ({status})"
    return clip(base, width)


def render_model_details(model: ModelInfo, width: int) -> List[str]:
    lines = [
        f"Name: {model.display_name}",
        f"Path: {model.path}",
        f"Source: {model.source_label}",
        f"Architecture: {model.architecture}",
        f"Model type: {model.model_type}",
        f"Params: {model.params_hint or 'unknown'}",
        f"Weights: {human_bytes(model.weight_bytes)}",
        f"On disk: {human_bytes(model.disk_bytes)}",
        f"Dtype: {model.dtype}",
        f"Quantization: {model.quantization_hint}",
        f"Layers: {model.layers or 'n/a'}",
        f"Hidden size: {model.hidden_size or 'n/a'}",
        f"Context: {model.max_position_embeddings or 'n/a'}",
        f"Transformers: {model.backend_status.get('transformers', 'unknown')}",
        f"MLX: {model.backend_status.get('mlx', 'unknown')}",
        "",
        "Summary:",
    ]
    lines.extend(wrap_lines(model.summary, width))
    return lines


def is_chat_capable_architecture(architecture: str) -> bool:
    markers = (
        "causallm",
        "conditionalgeneration",
        "lmheadmodel",
        "seq2seq",
        "vision2seq",
    )
    normalized = architecture.lower()
    return any(marker in normalized for marker in markers)


def run_picker(models: List[ModelInfo], extra_dirs: Sequence[str]) -> Optional[ModelInfo]:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return run_plain_selection(models)

    first_runnable = next((i for i, model in enumerate(models) if model.is_runnable), 0)
    state = {"selected": first_runnable, "top": 0, "result": None, "models": models}

    def inner(stdscr: "curses._CursesWindow") -> None:
        curses.curs_set(0)
        stdscr.keypad(True)

        while True:
            stdscr.erase()
            height, width = stdscr.getmaxyx()

            title = "AI Model Picker"
            stdscr.addnstr(0, 2, title, max(0, width - 4), curses.A_BOLD)
            stdscr.addnstr(
                1,
                2,
                clip(f"Found {len(state['models'])} models. Ready models are shown first. Enter to chat, r to refresh, q to quit.", width - 4),
                max(0, width - 4),
            )

            if not state["models"]:
                stdscr.addnstr(3, 2, "No local models found.", max(0, width - 4), curses.A_BOLD)
                stdscr.addnstr(
                    5,
                    2,
                    clip("Place models in ./models or use `hf download` so they land in the Hugging Face cache.", width - 4),
                    max(0, width - 4),
                )
                stdscr.refresh()
                key = stdscr.getch()
                if key in (ord("q"), 27):
                    return
                if key == ord("r"):
                    state["models"] = discover_models(extra_dirs)
                continue

            split = max(32, min(48, width // 3))
            detail_width = max(24, width - split - 4)
            visible_rows = max(6, height - 5)
            selected = state["selected"]

            if selected < state["top"]:
                state["top"] = selected
            elif selected >= state["top"] + visible_rows:
                state["top"] = selected - visible_rows + 1

            stdscr.vline(2, split + 1, curses.ACS_VLINE, max(0, height - 3))
            end = min(len(state["models"]), state["top"] + visible_rows)
            for row, index in enumerate(range(state["top"], end), start=3):
                item = state["models"][index]
                attr = curses.A_REVERSE if index == selected else curses.A_NORMAL
                stdscr.addnstr(
                    row,
                    2,
                    render_model_list_item(item, split - 3),
                    max(0, split - 3),
                    attr,
                )

            detail_lines = render_model_details(state["models"][selected], detail_width)
            for row, line in enumerate(detail_lines, start=3):
                if row >= height - 2:
                    break
                stdscr.addnstr(row, split + 3, line, max(0, detail_width), curses.A_NORMAL)

            help_line = clip(HELP_TEXT.replace("\n", "  "), width - 4)
            stdscr.addnstr(height - 1, 2, help_line, max(0, width - 4), curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (ord("q"), 27):
                return
            if key in (curses.KEY_DOWN, ord("j")):
                state["selected"] = min(len(state["models"]) - 1, selected + 1)
            elif key in (curses.KEY_UP, ord("k")):
                state["selected"] = max(0, selected - 1)
            elif key in (10, 13, curses.KEY_ENTER):
                state["result"] = state["models"][selected]
                return
            elif key == ord("r"):
                state["models"] = discover_models(extra_dirs)
                state["selected"] = 0
                state["top"] = 0

    curses.wrapper(inner)
    return state["result"]


def run_plain_selection(models: List[ModelInfo]) -> Optional[ModelInfo]:
    if not models:
        print("No models found.")
        return None

    print("Available models:")
    for index, model in enumerate(models, start=1):
        status = model.preferred_backend or "blocked"
        print(f"  {index:>2}. {model.display_name}  ({model.subtitle()} | {status})")

    while True:
        choice = input("Select a model number or `q`: ").strip().lower()
        if choice in {"q", "quit", "exit"}:
            return None
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(models):
                return models[index]
        print("Invalid selection.")


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def print_chat_header(model: ModelInfo, backend: ChatBackend, max_tokens: int) -> None:
    clear_screen()
    print(f"Model: {model.display_name}")
    print(f"Path:  {model.path}")
    print(f"Using: {backend.name}")
    print(f"Max new tokens: {max_tokens}")
    print("")
    print("Commands: /exit  /models  /clear  /info  /help")
    print("")


def print_model_info(model: ModelInfo) -> None:
    print(f"name: {model.display_name}")
    print(f"path: {model.path}")
    print(f"arch: {model.architecture}")
    print(f"params: {model.params_hint or 'unknown'}")
    print(f"weights: {human_bytes(model.weight_bytes)}")
    print(f"disk: {human_bytes(model.disk_bytes)}")
    print(f"dtype: {model.dtype}")
    print(f"context: {model.max_position_embeddings or 'n/a'}")
    print(f"summary: {model.summary}")


def run_chat_loop(
    model: ModelInfo,
    backend_name: str,
    max_tokens: int,
    temperature: float,
    system_prompt: Optional[str],
) -> str:
    backend = None
    load_errors: List[str] = []
    for candidate in candidate_backend_names(backend_name, model=model):
        try:
            backend = choose_backend(candidate, model=model)
            print(f"Loading {model.display_name} with {backend.name}...")
            backend.load(model)
            break
        except Exception as exc:
            load_errors.append(f"{candidate}: {exc}")
            if backend is not None:
                try:
                    backend.unload()
                except Exception:
                    pass
            backend = None

    if backend is None:
        raise BackendError(
            "Could not load the selected model with the available backends.\n"
            + "\n".join(load_errors)
        )

    messages: List[dict] = []
    print_chat_header(model, backend, max_tokens)

    try:
        while True:
            try:
                user_input = input("You > ").strip()
            except EOFError:
                print("")
                return "quit"
            except KeyboardInterrupt:
                print("")
                return "quit"

            if not user_input:
                continue
            if user_input in {"/exit", "/quit"}:
                return "quit"
            if user_input == "/models":
                return "picker"
            if user_input == "/clear":
                messages.clear()
                print_chat_header(model, backend, max_tokens)
                continue
            if user_input == "/info":
                print_model_info(model)
                continue
            if user_input == "/help":
                print("Commands: /exit  /models  /clear  /info  /help")
                continue

            messages.append({"role": "user", "content": user_input})
            try:
                answer = backend.generate(
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            except KeyboardInterrupt:
                print("\nGeneration interrupted.")
                continue

            print("")
            print("AI  >")
            for line in wrap_lines(answer or "(empty response)", 100):
                print(line)
            print("")
            messages.append({"role": "assistant", "content": answer})
    finally:
        backend.unload()


def resolve_model(models: Sequence[ModelInfo], selector: str) -> Optional[ModelInfo]:
    lowered = selector.lower()
    exact = [model for model in models if model.display_name.lower() == lowered]
    if exact:
        return exact[0]
    partial = [model for model in models if lowered in model.display_name.lower()]
    if partial:
        return partial[0]
    return None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse local Hugging Face models and chat from the terminal.")
    parser.add_argument(
        "--models-dir",
        action="append",
        default=[],
        help="Additional local directory to scan for models.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "transformers", "mlx"],
        help="Inference backend to use.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum new tokens per assistant turn.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Use 0 for greedy decoding.",
    )
    parser.add_argument(
        "--system",
        default=None,
        help="Optional system prompt applied to the chat session.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print discovered models and exit.",
    )
    parser.add_argument(
        "--chat",
        metavar="MODEL",
        help="Start chat directly with a specific model name or partial match.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    models = discover_models(args.models_dir)

    if args.list:
        if not models:
            print("No models found.")
            return 1
        for model in models:
            print(
                f"{model.display_name}\t{model.path}\t{model.params_hint or 'unknown'}\t{model.architecture}\t{model.preferred_backend or 'blocked'}"
            )
        return 0

    if args.chat:
        selected = resolve_model(models, args.chat)
        if selected is None:
            print(f"No model matched: {args.chat}", file=sys.stderr)
            return 1
    else:
        selected = run_picker(models, args.models_dir)
        if selected is None:
            return 0

    while True:
        try:
            action = run_chat_loop(
                model=selected,
                backend_name=args.backend,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                system_prompt=args.system,
            )
        except BackendError as exc:
            print(str(exc), file=sys.stderr)
            if args.chat:
                return 1
            input("\nPress Enter to return to the picker...")
            models = discover_models(args.models_dir)
            selected = run_picker(models, args.models_dir)
            if selected is None:
                return 0
            continue
        except Exception as exc:
            print(f"Failed to run chat: {exc}", file=sys.stderr)
            if args.chat:
                return 1
            input("\nPress Enter to return to the picker...")
            models = discover_models(args.models_dir)
            selected = run_picker(models, args.models_dir)
            if selected is None:
                return 0
            continue

        if action != "picker":
            return 0

        models = discover_models(args.models_dir)
        selected = run_picker(models, args.models_dir)
        if selected is None:
            return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
