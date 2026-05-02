import argparse
import concurrent.futures
import fcntl
import gc
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"
OFFLOAD_DIR = ROOT / "eval_offload_cache"
BENCHMARK_CACHE_DIR = ROOT / "data" / "benchmark_cache"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "post_dpo_benchmarks.json"
SUMMARY_MD = RESULTS_DIR / "post_dpo_benchmarks_summary.md"
RESULTS_LOCK = RESULTS_DIR / "post_dpo_benchmarks.lock"
GENERATIONS_DIR = RESULTS_DIR / "benchmark_generations"

os.environ["HF_HOME"] = str(BENCHMARK_CACHE_DIR / "hf_home")
os.environ["HF_DATASETS_CACHE"] = str(BENCHMARK_CACHE_DIR / "datasets")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(BENCHMARK_CACHE_DIR / "hub")

RANKS = [1, 2, 8, 128, 1024, 3072]
BASE_RANK = 0
ALL_EVAL_RANKS = [BASE_RANK] + RANKS
MODEL_ORDER = ["qwen_0.8b", "nemotron_4b"]
STAGE_ORDER = ["base", "math_sft", "science_sft", "code_sft", "merged_dare", "dpo"]

MODEL_SPECS = {
    "qwen_0.8b": {
        "local_path": PROJECT_ROOT / "models" / "Qwen3.5-0.8B",
        "hub_id": "Qwen/Qwen3.5-0.8B",
        "cache_dir_name": "models--Qwen--Qwen3.5-0.8B",
        "prompt_max_length": 1024,
        "eval_batch_size": 64,
        "code_batch_size": 32,
        "use_4bit_eval": False,
    },
    "nemotron_4b": {
        "local_path": PROJECT_ROOT / "models" / "nemotron",
        "hub_id": "nvidia/Nemotron-Mini-4B-Instruct",
        "cache_dir_name": "models--nvidia--Nemotron-Mini-4B-Instruct",
        "prompt_max_length": 1024,
        "eval_batch_size": 4,
        "code_batch_size": 2,
        "use_4bit_eval": True,
    },
}

BENCHMARK_MODES = {
    "autonomous": {
        "gsm8k": 40,
        "math500": 40,
        "arc": 40,
        "mmlu": 40,
        "mbpp": 20,
        "humaneval": 20,
        "truthfulqa_mc": 40,
        "mmlu_pro": 40,
        "gpqa": 40,
    },
    "quick": {
        "gsm8k": 100,
        "math500": 100,
        "arc": 100,
        "mmlu": 100,
        "mbpp": 50,
        "humaneval": 50,
        "truthfulqa_mc": 100,
        "mmlu_pro": 100,
        "gpqa": 100,
    },
    "paper": {
        "gsm8k": 250,
        "math500": 250,
        "arc": 500,
        "mmlu": 500,
        "mbpp": 200,
        "humaneval": 164,
        "truthfulqa_mc": 250,
        "mmlu_pro": 500,
        "gpqa": 250,
    },
    "research": {
        "gsm8k": 500,
        "math500": 500,
        "arc": 1000,
        "mmlu": 1000,
        "mbpp": 500,
        "humaneval": 164,
        "truthfulqa_mc": 684,
        "mmlu_pro": 1000,
        "gpqa": 448,
    },
}

BENCHMARK_ORDER = ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval", "truthfulqa_mc", "mmlu_pro", "gpqa"]
BENCHMARK_LABELS = {
    "gsm8k": "GSM8K",
    "math500": "MATH-500",
    "arc": "ARC",
    "mmlu": "MMLU",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
    "truthfulqa_mc": "TruthfulQA-MC",
    "mmlu_pro": "MMLU-Pro",
    "gpqa": "GPQA",
}
BENCHMARK_SUITES = {
    "tier1": ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval"],
    "tier2_supported": ["truthfulqa_mc", "mmlu_pro", "gpqa"],
    "paper_supported": ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval", "truthfulqa_mc", "mmlu_pro", "gpqa"],
}


def hf_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def hf_model_kwargs() -> dict:
    token = hf_token()
    return {"token": token} if token else {}


def load_dataset_with_auth(*args, **kwargs):
    token = hf_token()
    if token:
        kwargs.setdefault("token", token)
    return load_dataset(*args, **kwargs)


def is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("r") as handle:
            head = handle.read(128)
        return head.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False


def local_model_snapshot_is_usable(model_key: str) -> bool:
    spec = MODEL_SPECS[model_key]
    local_path = spec["local_path"]
    if not local_path.exists():
        return False

    tokenizer_json = local_path / "tokenizer.json"
    if tokenizer_json.exists() and is_lfs_pointer(tokenizer_json):
        return False

    weight_index = local_path / "model.safetensors.index.json"
    shard_files = list(local_path.glob("model-*.safetensors"))
    single_weight = local_path / "model.safetensors"
    if weight_index.exists() and not shard_files:
        return False
    if not weight_index.exists() and not single_weight.exists():
        return False

    return True


def cached_hf_snapshot(model_key: str) -> Path | None:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / MODEL_SPECS[model_key]["cache_dir_name"]
    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        return None

    refs_main = cache_root / "refs" / "main"
    if refs_main.exists():
        try:
            ref = refs_main.read_text().strip()
            candidate = snapshots_dir / ref
            if candidate.exists():
                return candidate
        except Exception:
            pass

    candidates = sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def model_name(model_key: str) -> str:
    spec = MODEL_SPECS[model_key]
    if local_model_snapshot_is_usable(model_key):
        return str(spec["local_path"])
    cached_snapshot = cached_hf_snapshot(model_key)
    if cached_snapshot is not None:
        return str(cached_snapshot)
    return spec["hub_id"]


def available_cpu_ram_gib(reserve_gib: int = 8) -> int:
    try:
        meminfo = {}
        with Path("/proc/meminfo").open("r") as handle:
            for line in handle:
                key, value = line.split(":", 1)
                meminfo[key] = value.strip()
        available_kib = int(meminfo.get("MemAvailable", "0 kB").split()[0])
        return max(8, int(available_kib / (1024 * 1024)) - reserve_gib)
    except Exception:
        return 64


def build_max_memory():
    if not torch.cuda.is_available():
        return None
    total_vram_gib = int(torch.cuda.get_device_properties(0).total_memory / (1024**3))
    gpu_budget_gib = max(4, int(total_vram_gib * 0.82))
    return {0: f"{gpu_budget_gib}GiB", "cpu": f"{available_cpu_ram_gib()}GiB"}


def load_results():
    if not RESULTS_JSON.exists():
        return {"benchmarks": {}}
    try:
        with RESULTS_JSON.open("r") as handle:
            return json.load(handle)
    except Exception:
        return {"benchmarks": {}}


def save_results(results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_JSON.open("w") as handle:
        json.dump(results, handle, indent=2)


def update_results_entry(key: str, outcome: dict) -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOCK.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        results = load_results()
        results.setdefault("benchmarks", {})[key] = outcome
        save_results(results)
        write_summary(results)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return results


def benchmark_key(model_key: str, rank: int, stage: str, benchmark: str) -> str:
    return f"{model_key}|rank={rank}|stage={stage}|benchmark={benchmark}"


def wilson_interval(correct: int, total: int, z: float = 1.96):
    if total <= 0:
        return None, None
    p = correct / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    margin = (z * ((p * (1.0 - p) / total) + (z * z) / (4.0 * total * total)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def generation_path(model_key: str, rank: int, stage: str, benchmark: str) -> Path:
    filename = f"{model_key}_rank{rank}_{stage}_{benchmark}.jsonl"
    return GENERATIONS_DIR / filename


def write_generations(model_key: str, rank: int, stage: str, benchmark: str, rows: list[dict]):
    GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = generation_path(model_key, rank, stage, benchmark)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def option_letters(count: int) -> list[str]:
    return [chr(ord("A") + idx) for idx in range(count)]


def extract_choice_label(response: str, allowed_letters: list[str]) -> str:
    upper = response.upper()
    for pattern in [r"\b([A-Z])\b", r"ANSWER\s*[:\-]?\s*([A-Z])", r"\(([A-Z])\)"]:
        for match in re.finditer(pattern, upper):
            label = match.group(1)
            if label in allowed_letters:
                return label
    for ch in upper[:120]:
        if ch in allowed_letters:
            return ch
    return ""


def deterministic_shuffle_options(question: str, options: list[str]) -> list[str]:
    seed = int(hashlib.md5(question.encode("utf-8")).hexdigest()[:8], 16)
    shuffled = list(options)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def should_run_benchmark(existing_result: dict | None, desired_samples: int, generation_target: Path | None = None) -> bool:
    if not existing_result:
        return True
    if existing_result.get("metric") == "error":
        return True
    completed_samples = int(existing_result.get("samples", 0) or 0)
    if completed_samples < desired_samples:
        return True
    if generation_target is not None:
        generation_file = existing_result.get("generation_file")
        if not generation_file:
            return True
        if not Path(generation_file).exists():
            return True
        if Path(generation_file) != generation_target:
            return True
    return False


def stage_paths(model_key: str, rank: int):
    merged_root = OUTPUT_DIR / f"{model_key}_merged_DARE_rank{rank}"
    dpo_root = OUTPUT_DIR / f"{model_key}_math_DPO_rank{rank}"
    mapping = {}
    for sft_root in sorted(OUTPUT_DIR.glob(f"{model_key}_*_SFT_rank{rank}")):
        match = re.match(fr"^{re.escape(model_key)}_(.+)_SFT_rank{rank}$", sft_root.name)
        if not match:
            continue
        domain = match.group(1)
        if (sft_root / "adapter_config.json").exists():
            mapping[f"{domain}_sft"] = sft_root
    if (merged_root / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root
    elif (merged_root / "merged_sft" / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root / "merged_sft"
    if (dpo_root / "adapter_config.json").exists():
        mapping["dpo"] = dpo_root
    return mapping


def eval_targets():
    targets = [(model_key, BASE_RANK, {"base": None}) for model_key in MODEL_ORDER]
    for model_key in MODEL_ORDER:
        for rank in RANKS:
            stages = stage_paths(model_key, rank)
            if stages:
                targets.append((model_key, rank, stages))
    return targets


def reset_offload_dir(path: Path):
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def load_model_and_tokenizer(model_key: str, adapter_path: Path | None, offload_name: str):
    tok = AutoTokenizer.from_pretrained(model_name(model_key), trust_remote_code=True, **hf_model_kwargs())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    spec = MODEL_SPECS[model_key]
    if spec.get("use_4bit_eval", True):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        offload_dir = OFFLOAD_DIR / offload_name
        reset_offload_dir(offload_dir)
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "quantization_config": quant_config,
            "attn_implementation": "eager",
            "offload_folder": str(offload_dir),
            "offload_state_dict": True,
            "low_cpu_mem_usage": True,
        }
        max_memory = build_max_memory()
        if max_memory is not None:
            kwargs["max_memory"] = max_memory
    else:
        kwargs = {
            "device_map": {"": 0} if torch.cuda.is_available() else None,
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
            "low_cpu_mem_usage": True,
        }

    base_model = AutoModelForCausalLM.from_pretrained(model_name(model_key), **kwargs, **hf_model_kwargs())
    base_model.config.use_cache = True
    if adapter_path is None:
        base_model.eval()
        return base_model, tok
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False)
    model.eval()
    return model, tok


def cleanup(model=None):
    if model is not None:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)


def format_chat_prompt(tok, system_msg: str, user_msg: str) -> str:
    try:
        return tok.apply_chat_template(
            [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"System: {system_msg}\nUser: {user_msg}\nAssistant:"


def extract_number(text: str):
    for pattern in [r"\\boxed\{([^}]+)\}", r"####\s*(.+?)(?:\n|$)", r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)"]:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip().replace(",", "")
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    return nums[-1] if nums else None


def normalize(text: str | None) -> str:
    if text is None:
        return ""
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        num = float(value)
        return str(int(num)) if num == int(num) else f"{num:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return value.lower()


def build_mc_prompt(question: str, options: list[str]) -> str:
    letters = option_letters(len(options))
    return f"{question}\n\n" + "\n".join(f"{letter}. {text}" for letter, text in zip(letters, options))


def generate_response(model, tok, prompt: str, max_new_tokens: int, prompt_max_length: int):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=prompt_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_responses(model, tok, prompts: list[str], max_new_tokens: int, prompt_max_length: int):
    if not prompts:
        return []
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=prompt_max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
    decoded = []
    for idx, input_len in enumerate(input_lengths):
        decoded.append(tok.decode(outputs[idx][input_len:], skip_special_tokens=True))
    return decoded


def eval_batch_size_for(model_key: str) -> int:
    return int(MODEL_SPECS[model_key].get("eval_batch_size", 8))


def code_batch_size_for(model_key: str) -> int:
    return int(MODEL_SPECS[model_key].get("code_batch_size", 4))


def evaluate_gsm8k(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds = load_dataset_with_auth("openai/gsm8k", "main", split="test").select(range(n))
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="gsm8k"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = [
            format_chat_prompt(tok, "You are a math expert. Solve step by step. Put final answer after ####.", ex["question"])
            for ex in batch
        ]
        responses = generate_responses(model, tok, prompts, max_new_tokens=512, prompt_max_length=prompt_max_length)
        for ex, response in zip(batch, responses):
            pred = normalize(extract_number(response))
            gold = normalize(extract_number(ex["answer"]))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex["question"],
                    "gold_answer": ex["answer"],
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "exact_match", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_math500(model, tok, n: int, prompt_max_length: int, model_key: str):
    try:
        ds = load_dataset_with_auth("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        ds = load_dataset_with_auth("lighteval/MATH", split="test")
    ds = ds.select(range(min(n, len(ds))))
    question_key = "problem" if "problem" in ds.column_names else "question"
    answer_key = "solution" if "solution" in ds.column_names else "answer"
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="math500"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = [
            format_chat_prompt(tok, "Solve this math problem. Put your final answer in \\boxed{}.", ex[question_key])
            for ex in batch
        ]
        responses = generate_responses(model, tok, prompts, max_new_tokens=512, prompt_max_length=prompt_max_length)
        for ex, response in zip(batch, responses):
            pred = normalize(extract_number(response))
            gold = normalize(extract_number(ex[answer_key]))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex[question_key],
                    "gold_answer": ex[answer_key],
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "exact_match", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_arc(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds = load_dataset_with_auth("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="arc"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = []
        choice_rows = []
        for ex in batch:
            choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
            user = f"{ex['question']}\n\n{choices}"
            prompts.append(format_chat_prompt(tok, "Answer the multiple choice question with ONLY the letter A, B, C, or D.", user))
            choice_rows.append(user)
        responses = generate_responses(model, tok, prompts, max_new_tokens=16, prompt_max_length=prompt_max_length)
        for ex, response, choice_text in zip(batch, responses, choice_rows):
            pred = next((ch.upper() for ch in response[:50] if ch.upper() in "ABCD"), "")
            gold = ex["answerKey"].strip().upper()
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex["question"],
                    "choices": choice_text,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_mmlu(model, tok, n: int, prompt_max_length: int, model_key: str):
    try:
        ds = load_dataset_with_auth("cais/mmlu", "all", split="test")
    except Exception:
        ds = load_dataset_with_auth("lukaemon/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    correct = total = 0
    letters = ["A", "B", "C", "D"]
    batch_size = eval_batch_size_for(model_key)
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="mmlu"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = []
        options_rows = []
        for ex in batch:
            if "choices" in ex:
                options = ex["choices"]
            else:
                options = [ex.get(letter.lower(), "") for letter in letters]
            options_text = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, options))
            options_rows.append(options_text)
            prompts.append(
                format_chat_prompt(
                    tok,
                    "Answer the multiple choice question with ONLY the letter A, B, C, or D.",
                    f"{ex['question']}\n\n{options_text}",
                )
            )
        responses = generate_responses(model, tok, prompts, max_new_tokens=8, prompt_max_length=prompt_max_length)
        for ex, response, options_text in zip(batch, responses, options_rows):
            pred = next((ch.upper() for ch in response[:20] if ch.upper() in "ABCD"), "")
            answer = ex["answer"]
            gold = letters[answer] if isinstance(answer, int) else str(answer).upper().strip()
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex["question"],
                    "choices": options_text,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def extract_code(response: str, entry_point: str | None = None, prompt_code: str | None = None) -> str:
    code = response
    block = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if block:
        code = block.group(1)
    if entry_point and f"def {entry_point}" in code:
        code = code[code.index(f"def {entry_point}"):]
    elif prompt_code and not code.strip().startswith("def "):
        code = prompt_code + "\n" + code
    return code


def run_python_check(script: str, timeout: int = 10) -> bool:
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as handle:
            handle.write(script)
            handle.flush()
            result = subprocess.run([sys.executable, handle.name], capture_output=True, timeout=timeout, text=True)
            return result.returncode == 0
    except Exception:
        return False


def run_python_checks_parallel(scripts: list[str], timeout: int = 10) -> list[bool]:
    if not scripts:
        return []
    max_workers = max(1, min(len(scripts), os.cpu_count() or 4, 8))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda script: run_python_check(script, timeout=timeout), scripts))


def evaluate_humaneval(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds = load_dataset_with_auth("openai/openai_humaneval", split="test").select(range(min(n, 164)))
    correct = total = 0
    batch_size = code_batch_size_for(model_key)
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="humaneval"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = [
            format_chat_prompt(tok, "Complete the Python function. Output ONLY the code.", f"Complete this function:\n```python\n{ex['prompt']}\n```")
            for ex in batch
        ]
        responses = generate_responses(model, tok, prompts, max_new_tokens=512, prompt_max_length=prompt_max_length)
        scripts = []
        batch_rows = []
        for ex, response in zip(batch, responses):
            code = extract_code(response, entry_point=ex["entry_point"], prompt_code=ex["prompt"])
            scripts.append(code + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n")
            batch_rows.append(
                {
                    "task_id": ex["task_id"],
                    "question": ex["prompt"],
                    "entry_point": ex["entry_point"],
                    "gold_test": ex["test"],
                    "generated_text": response,
                    "generated_code": code,
                }
            )
        for row, passed in zip(batch_rows, run_python_checks_parallel(scripts, timeout=10)):
            if passed:
                correct += 1
            total += 1
            row["correct"] = passed
            generations.append(row)
    return {"metric": "pass@1", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_mbpp(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds = load_dataset_with_auth("google-research-datasets/mbpp", "sanitized", split="test").select(range(n))
    correct = total = 0
    batch_size = code_batch_size_for(model_key)
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="mbpp"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = [
            format_chat_prompt(tok, "Write a Python function to solve the task. Output ONLY the code.", ex["prompt"])
            for ex in batch
        ]
        responses = generate_responses(model, tok, prompts, max_new_tokens=512, prompt_max_length=prompt_max_length)
        scripts = []
        batch_rows = []
        for ex, response in zip(batch, responses):
            code = extract_code(response)
            scripts.append(code + "\n\n" + "\n".join(ex["test_list"]) + "\n")
            batch_rows.append(
                {
                    "task_id": ex["task_id"],
                    "question": ex["prompt"],
                    "gold_tests": ex["test_list"],
                    "generated_text": response,
                    "generated_code": code,
                }
            )
        for row, passed in zip(batch_rows, run_python_checks_parallel(scripts, timeout=10)):
            if passed:
                correct += 1
            total += 1
            row["correct"] = passed
            generations.append(row)
    return {"metric": "pass@1", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_truthfulqa_mc(model, tok, n: int, prompt_max_length: int, model_key: str):
    truth_candidates = [
        ("EleutherAI/truthful_qa_mc", "multiple_choice"),
        ("truthfulqa/truthful_qa", "multiple_choice"),
        ("EleutherAI/truthful_qa_mc", None),
    ]
    last_exc = None
    ds = None
    for dataset_name, config_name in truth_candidates:
        try:
            if config_name is None:
                ds = load_dataset_with_auth(dataset_name, split="validation")
            else:
                ds = load_dataset_with_auth(dataset_name, config_name, split="validation")
            break
        except Exception as exc:
            last_exc = exc
    if ds is None:
        raise last_exc if last_exc is not None else RuntimeError("Unable to load TruthfulQA-MC dataset")
    ds = ds.select(range(min(n, len(ds))))
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="truthfulqa_mc"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = []
        choice_rows = []
        gold_rows = []
        for ex in batch:
            if "choices" in ex:
                options = list(ex["choices"])
            elif "mc1_targets" in ex:
                options = list(ex["mc1_targets"]["choices"])
            elif "mc2_targets" in ex:
                options = list(ex["mc2_targets"]["choices"])
            else:
                raise KeyError("TruthfulQA choices not found")
            letters = option_letters(len(options))
            prompt_text = build_mc_prompt(ex["question"], options)
            prompts.append(format_chat_prompt(tok, "Choose the most truthful answer. Respond with ONLY the letter.", prompt_text))
            choice_rows.append(options)
            if "label" in ex:
                gold_rows.append(letters[int(ex["label"])])
            elif "mc1_targets" in ex:
                labels = list(ex["mc1_targets"]["labels"])
                gold_rows.append(letters[labels.index(1)] if 1 in labels else letters[int(max(range(len(labels)), key=lambda i: labels[i]))])
            elif "mc2_targets" in ex:
                labels = list(ex["mc2_targets"]["labels"])
                gold_rows.append(letters[labels.index(1)] if 1 in labels else letters[int(max(range(len(labels)), key=lambda i: labels[i]))])
            else:
                raise KeyError("TruthfulQA label not found")
        responses = generate_responses(model, tok, prompts, max_new_tokens=8, prompt_max_length=prompt_max_length)
        for ex, options, gold, response in zip(batch, choice_rows, gold_rows, responses):
            pred = extract_choice_label(response, option_letters(len(options)))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex["question"],
                    "choices": options,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def evaluate_mmlu_pro(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds_obj = load_dataset_with_auth("TIGER-Lab/MMLU-Pro")
    if isinstance(ds_obj, dict):
        if "test" in ds_obj:
            ds = ds_obj["test"]
        elif "validation" in ds_obj:
            ds = ds_obj["validation"]
        else:
            ds = ds_obj[next(iter(ds_obj.keys()))]
    else:
        ds = ds_obj
    ds = ds.select(range(min(n, len(ds))))
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="mmlu_pro"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = []
        option_rows = []
        gold_rows = []
        for ex in batch:
            options = list(ex["options"])
            letters = option_letters(len(options))
            prompt_text = build_mc_prompt(ex["question"], options)
            prompts.append(format_chat_prompt(tok, "Answer the multiple choice question with ONLY the correct letter.", prompt_text))
            option_rows.append(options)
            gold_rows.append(str(ex.get("answer") or letters[int(ex["answer_index"])]).strip().upper())
        responses = generate_responses(model, tok, prompts, max_new_tokens=8, prompt_max_length=prompt_max_length)
        for ex, options, gold, response in zip(batch, option_rows, gold_rows, responses):
            pred = extract_choice_label(response, option_letters(len(options)))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": ex["question"],
                    "choices": options,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


def load_gpqa_dataset():
    loaders = [
        ("Idavidrein/gpqa", {"name": "gpqa_diamond"}),
        ("Idavidrein/gpqa", {"name": "gpqa_main"}),
        ("Wanfq/gpqa", {"name": "gpqa_diamond"}),
        ("natong19/gpqa", {"name": "gpqa_diamond"}),
        ("natong19/gpqa", {"name": "gpqa_extended"}),
    ]
    last_exc = None
    for dataset_name, kwargs in loaders:
        try:
            return load_dataset_with_auth(dataset_name, **kwargs)
        except Exception as exc:
            last_exc = exc
    raise last_exc if last_exc is not None else RuntimeError("Unable to load GPQA dataset")


def canonicalize_gpqa_example(ex: dict):
    question = ex.get("question") or ex.get("Question")
    if question is None:
        raise KeyError("GPQA question field not found")

    if "options" in ex:
        options = list(ex["options"])
        letters = option_letters(len(options))
        gold = str(ex.get("answer") or letters[int(ex["answer_index"]) if "answer_index" in ex else 0]).strip().upper()
        return question, options, gold
    if "choices" in ex:
        options = list(ex["choices"])
        letters = option_letters(len(options))
        label = ex.get("label")
        if isinstance(label, int):
            gold = letters[label]
        else:
            gold = str(label).strip().upper()
        return question, options, gold

    correct = ex.get("Correct Answer") or ex.get("correct_answer")
    incorrect = []
    for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "incorrect_answer_1", "incorrect_answer_2", "incorrect_answer_3"):
        if ex.get(key):
            incorrect.append(ex[key])
    if correct is None or len(incorrect) < 3:
        raise KeyError("GPQA options not found in known field layout")

    options = deterministic_shuffle_options(question, [correct] + incorrect[:3])
    letters = option_letters(len(options))
    gold = letters[options.index(correct)]
    return question, options, gold


def evaluate_gpqa(model, tok, n: int, prompt_max_length: int, model_key: str):
    ds_obj = load_gpqa_dataset()
    if isinstance(ds_obj, dict):
        split_name = "train" if "train" in ds_obj else ("validation" if "validation" in ds_obj else next(iter(ds_obj.keys())))
        ds = ds_obj[split_name]
    else:
        ds = ds_obj
    ds = ds.select(range(min(n, len(ds))))
    batch_size = eval_batch_size_for(model_key)
    correct = total = 0
    generations = []
    for start in tqdm(range(0, len(ds), batch_size), desc="gpqa"):
        batch = ds.select(range(start, min(start + batch_size, len(ds))))
        prompts = []
        option_rows = []
        gold_rows = []
        question_rows = []
        for ex in batch:
            question, options, gold = canonicalize_gpqa_example(ex)
            prompt_text = build_mc_prompt(question, options)
            prompts.append(format_chat_prompt(tok, "Answer the multiple choice question with ONLY the correct letter.", prompt_text))
            option_rows.append(options)
            gold_rows.append(gold)
            question_rows.append(question)
        responses = generate_responses(model, tok, prompts, max_new_tokens=8, prompt_max_length=prompt_max_length)
        for question, options, gold, response in zip(question_rows, option_rows, gold_rows, responses):
            pred = extract_choice_label(response, option_letters(len(options)))
            is_correct = pred == gold
            if is_correct:
                correct += 1
            total += 1
            generations.append(
                {
                    "question": question,
                    "choices": options,
                    "gold_answer": gold,
                    "predicted_answer": pred,
                    "generated_text": response,
                    "correct": is_correct,
                }
            )
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total, "generations": generations}


BENCHMARK_FUNCTIONS = {
    "gsm8k": evaluate_gsm8k,
    "math500": evaluate_math500,
    "arc": evaluate_arc,
    "mmlu": evaluate_mmlu,
    "humaneval": evaluate_humaneval,
    "mbpp": evaluate_mbpp,
    "truthfulqa_mc": evaluate_truthfulqa_mc,
    "mmlu_pro": evaluate_mmlu_pro,
    "gpqa": evaluate_gpqa,
}


def write_summary(results: dict):
    lines = [
        "# Post-DPO Benchmark Summary",
        "",
        "| Model | Rank | Stage | " + " | ".join(BENCHMARK_LABELS[name] for name in BENCHMARK_ORDER) + " |",
        "|-------|------|-------|" + "|".join("-" * max(3, len(BENCHMARK_LABELS[name])) for name in BENCHMARK_ORDER) + "|",
    ]
    grouped = {}
    for key, value in results.get("benchmarks", {}).items():
        model_key, rank_part, stage_part, bench_part = key.split("|")
        rank = rank_part.split("=")[1]
        stage = stage_part.split("=")[1]
        bench = bench_part.split("=")[1]
        grouped.setdefault((model_key, rank, stage), {})[bench] = value.get("score")

    for model_key in MODEL_ORDER:
        for rank in ALL_EVAL_RANKS:
            for stage in STAGE_ORDER:
                row = grouped.get((model_key, str(rank), stage))
                if not row:
                    continue
                vals = [row.get(name) for name in BENCHMARK_ORDER]
                fmt = ["-" if value is None else f"{value:.3f}" for value in vals]
                lines.append(f"| {model_key} | {rank} | {stage} | " + " | ".join(fmt) + " |")

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark merged-vs-DPO adapters for hf_kaggle_opensource.")
    parser.add_argument("--mode", choices=sorted(BENCHMARK_MODES.keys()), default="paper")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--ranks", nargs="+", type=int, default=ALL_EVAL_RANKS)
    parser.add_argument("--suite", choices=sorted(BENCHMARK_SUITES.keys()), default="tier1")
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARK_ORDER)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    selected_benchmarks = args.benchmarks if args.benchmarks else BENCHMARK_SUITES[args.suite]

    targets = [(model_key, rank, stages) for model_key, rank, stages in eval_targets() if model_key in args.models and rank in args.ranks]
    if not targets:
        print("No benchmark targets found.", flush=True)
        return 0

    for model_key, rank, stages in targets:
        for stage_name in STAGE_ORDER:
            if stage_name not in stages:
                continue
            adapter_path = stages.get(stage_name)
            model = None
            try:
                model, tok = load_model_and_tokenizer(
                    model_key=model_key,
                    adapter_path=adapter_path,
                    offload_name=f"{model_key}_rank{rank}_{stage_name}",
                )
                for benchmark in selected_benchmarks:
                    sample_count = BENCHMARK_MODES[args.mode][benchmark]
                    key = benchmark_key(model_key, rank, stage_name, benchmark)
                    target_generation_file = generation_path(model_key, rank, stage_name, benchmark)
                    if not should_run_benchmark(
                        results.get("benchmarks", {}).get(key),
                        sample_count,
                        generation_target=target_generation_file,
                    ):
                        continue
                    print(f"Running {key} with n={sample_count}", flush=True)
                    started = time.time()
                    try:
                        outcome = BENCHMARK_FUNCTIONS[benchmark](
                            model,
                            tok,
                            sample_count,
                            MODEL_SPECS[model_key]["prompt_max_length"],
                            model_key,
                        )
                    except Exception as exc:
                        outcome = {
                            "metric": "error",
                            "score": None,
                            "correct": 0,
                            "total": 0,
                            "error": str(exc),
                        }
                    generation_rows = outcome.pop("generations", None)
                    generation_file = None
                    if generation_rows:
                        generation_file = write_generations(model_key, rank, stage_name, benchmark, generation_rows)
                    ci95_low, ci95_high = wilson_interval(int(outcome.get("correct", 0) or 0), int(outcome.get("total", 0) or 0))
                    outcome.update(
                        {
                            "model": model_key,
                            "rank": rank,
                            "stage": stage_name,
                            "benchmark": benchmark,
                            "samples": sample_count,
                            "adapter_path": None if adapter_path is None else str(adapter_path),
                            "duration_seconds": time.time() - started,
                            "generation_file": str(generation_file) if generation_file else None,
                            "ci95_low": ci95_low,
                            "ci95_high": ci95_high,
                        }
                    )
                    results = update_results_entry(key, outcome)
            finally:
                cleanup(model=model)

    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
