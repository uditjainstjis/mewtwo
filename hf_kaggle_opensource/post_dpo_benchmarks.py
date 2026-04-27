import argparse
import gc
import json
import os
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

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTPUT_DIR = ROOT / "outputs"
OFFLOAD_DIR = ROOT / "eval_offload_cache"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "post_dpo_benchmarks.json"
SUMMARY_MD = RESULTS_DIR / "post_dpo_benchmarks_summary.md"

RANKS = [1, 2, 8, 128, 1024, 3072]
MODEL_ORDER = ["qwen_0.8b", "nemotron_4b"]

MODEL_SPECS = {
    "qwen_0.8b": {
        "local_path": PROJECT_ROOT / "models" / "Qwen3.5-0.8B",
        "hub_id": "Qwen/Qwen3.5-0.8B",
        "cache_dir_name": "models--Qwen--Qwen3.5-0.8B",
        "prompt_max_length": 1024,
    },
    "nemotron_4b": {
        "local_path": PROJECT_ROOT / "models" / "nemotron",
        "hub_id": "nvidia/Nemotron-Mini-4B-Instruct",
        "cache_dir_name": "models--nvidia--Nemotron-Mini-4B-Instruct",
        "prompt_max_length": 1024,
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
    },
    "quick": {
        "gsm8k": 50,
        "math500": 50,
        "arc": 50,
        "mmlu": 50,
        "mbpp": 50,
        "humaneval": 50,
    },
    "paper": {
        "gsm8k": 100,
        "math500": 100,
        "arc": 100,
        "mmlu": 100,
        "mbpp": 100,
        "humaneval": 100,
    },
}

BENCHMARK_ORDER = ["gsm8k", "math500", "arc", "mmlu", "mbpp", "humaneval"]


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


def benchmark_key(model_key: str, rank: int, stage: str, benchmark: str) -> str:
    return f"{model_key}|rank={rank}|stage={stage}|benchmark={benchmark}"


def should_run_benchmark(existing_result: dict | None, desired_samples: int) -> bool:
    if not existing_result:
        return True
    if existing_result.get("metric") == "error":
        return True
    completed_samples = int(existing_result.get("samples", 0) or 0)
    return completed_samples < desired_samples


def stage_paths(model_key: str, rank: int):
    merged_root = OUTPUT_DIR / f"{model_key}_merged_DARE_rank{rank}"
    dpo_root = OUTPUT_DIR / f"{model_key}_math_DPO_rank{rank}"
    mapping = {}
    if (merged_root / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root
    elif (merged_root / "merged_sft" / "adapter_config.json").exists():
        mapping["merged_dare"] = merged_root / "merged_sft"
    if (dpo_root / "adapter_config.json").exists():
        mapping["dpo"] = dpo_root
    return mapping


def eval_targets():
    targets = []
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
    tok = AutoTokenizer.from_pretrained(model_name(model_key), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

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

    base_model = AutoModelForCausalLM.from_pretrained(model_name(model_key), **kwargs)
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


def generate_response(model, tok, prompt: str, max_new_tokens: int, prompt_max_length: int):
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=prompt_max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def evaluate_gsm8k(model, tok, n: int, prompt_max_length: int):
    ds = load_dataset("openai/gsm8k", "main", split="test").select(range(n))
    correct = total = 0
    for ex in tqdm(ds, desc="gsm8k"):
        prompt = format_chat_prompt(tok, "You are a math expert. Solve step by step. Put final answer after ####.", ex["question"])
        response = generate_response(model, tok, prompt, max_new_tokens=512, prompt_max_length=prompt_max_length)
        if normalize(extract_number(response)) == normalize(extract_number(ex["answer"])):
            correct += 1
        total += 1
    return {"metric": "exact_match", "score": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_math500(model, tok, n: int, prompt_max_length: int):
    try:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    except Exception:
        ds = load_dataset("lighteval/MATH", split="test")
    ds = ds.select(range(min(n, len(ds))))
    question_key = "problem" if "problem" in ds.column_names else "question"
    answer_key = "solution" if "solution" in ds.column_names else "answer"
    correct = total = 0
    for ex in tqdm(ds, desc="math500"):
        prompt = format_chat_prompt(tok, "Solve this math problem. Put your final answer in \\boxed{}.", ex[question_key])
        response = generate_response(model, tok, prompt, max_new_tokens=512, prompt_max_length=prompt_max_length)
        if normalize(extract_number(response)) == normalize(extract_number(ex[answer_key])):
            correct += 1
        total += 1
    return {"metric": "exact_match", "score": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_arc(model, tok, n: int, prompt_max_length: int):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test").select(range(n))
    correct = total = 0
    for ex in tqdm(ds, desc="arc"):
        choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
        user = f"{ex['question']}\n\n{choices}"
        prompt = format_chat_prompt(tok, "Answer the multiple choice question with ONLY the letter A, B, C, or D.", user)
        response = generate_response(model, tok, prompt, max_new_tokens=16, prompt_max_length=prompt_max_length).strip()
        pred = next((ch.upper() for ch in response[:50] if ch.upper() in "ABCD"), "")
        if pred == ex["answerKey"].strip().upper():
            correct += 1
        total += 1
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_mmlu(model, tok, n: int, prompt_max_length: int):
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
    except Exception:
        ds = load_dataset("lukaemon/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    correct = total = 0
    letters = ["A", "B", "C", "D"]
    for ex in tqdm(ds, desc="mmlu"):
        if "choices" in ex:
            options = ex["choices"]
        else:
            options = [ex.get(letter.lower(), "") for letter in letters]
        options_text = "\n".join(f"{letter}. {text}" for letter, text in zip(letters, options))
        prompt = format_chat_prompt(
            tok,
            "Answer the multiple choice question with ONLY the letter A, B, C, or D.",
            f"{ex['question']}\n\n{options_text}",
        )
        response = generate_response(model, tok, prompt, max_new_tokens=8, prompt_max_length=prompt_max_length).strip()
        pred = next((ch.upper() for ch in response[:20] if ch.upper() in "ABCD"), "")
        answer = ex["answer"]
        gold = letters[answer] if isinstance(answer, int) else str(answer).upper().strip()
        if pred == gold:
            correct += 1
        total += 1
    return {"metric": "accuracy", "score": correct / max(total, 1), "correct": correct, "total": total}


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


def evaluate_humaneval(model, tok, n: int, prompt_max_length: int):
    ds = load_dataset("openai/openai_humaneval", split="test").select(range(min(n, 164)))
    correct = total = 0
    for ex in tqdm(ds, desc="humaneval"):
        prompt = format_chat_prompt(tok, "Complete the Python function. Output ONLY the code.", f"Complete this function:\n```python\n{ex['prompt']}\n```")
        response = generate_response(model, tok, prompt, max_new_tokens=512, prompt_max_length=prompt_max_length)
        code = extract_code(response, entry_point=ex["entry_point"], prompt_code=ex["prompt"])
        full = code + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
        passed = False
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as handle:
                handle.write(full)
                handle.flush()
                result = subprocess.run([sys.executable, handle.name], capture_output=True, timeout=10, text=True)
                passed = result.returncode == 0
        except Exception:
            passed = False
        if passed:
            correct += 1
        total += 1
    return {"metric": "pass@1", "score": correct / max(total, 1), "correct": correct, "total": total}


def evaluate_mbpp(model, tok, n: int, prompt_max_length: int):
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test").select(range(n))
    correct = total = 0
    for ex in tqdm(ds, desc="mbpp"):
        prompt = format_chat_prompt(tok, "Write a Python function to solve the task. Output ONLY the code.", ex["prompt"])
        response = generate_response(model, tok, prompt, max_new_tokens=512, prompt_max_length=prompt_max_length)
        code = extract_code(response)
        full = code + "\n\n" + "\n".join(ex["test_list"]) + "\n"
        passed = False
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", dir="/tmp", delete=True) as handle:
                handle.write(full)
                handle.flush()
                result = subprocess.run([sys.executable, handle.name], capture_output=True, timeout=10, text=True)
                passed = result.returncode == 0
        except Exception:
            passed = False
        if passed:
            correct += 1
        total += 1
    return {"metric": "pass@1", "score": correct / max(total, 1), "correct": correct, "total": total}


BENCHMARK_FUNCTIONS = {
    "gsm8k": evaluate_gsm8k,
    "math500": evaluate_math500,
    "arc": evaluate_arc,
    "mmlu": evaluate_mmlu,
    "humaneval": evaluate_humaneval,
    "mbpp": evaluate_mbpp,
}


def write_summary(results: dict):
    lines = [
        "# Post-DPO Benchmark Summary",
        "",
        "| Model | Rank | Stage | GSM8K | MATH-500 | ARC | MMLU | HumanEval | MBPP |",
        "|-------|------|-------|-------|----------|-----|------|-----------|------|",
    ]
    grouped = {}
    for key, value in results.get("benchmarks", {}).items():
        model_key, rank_part, stage_part, bench_part = key.split("|")
        rank = rank_part.split("=")[1]
        stage = stage_part.split("=")[1]
        bench = bench_part.split("=")[1]
        grouped.setdefault((model_key, rank, stage), {})[bench] = value.get("score")

    for model_key in MODEL_ORDER:
        for rank in RANKS:
            for stage in ("merged_dare", "dpo"):
                row = grouped.get((model_key, str(rank), stage))
                if not row:
                    continue
                vals = [row.get(name) for name in ("gsm8k", "math500", "arc", "mmlu", "humaneval", "mbpp")]
                fmt = ["-" if value is None else f"{value:.3f}" for value in vals]
                lines.append(f"| {model_key} | {rank} | {stage} | " + " | ".join(fmt) + " |")

    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark merged-vs-DPO adapters for hf_kaggle_opensource.")
    parser.add_argument("--mode", choices=sorted(BENCHMARK_MODES.keys()), default="paper")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--ranks", nargs="+", type=int, default=RANKS)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    targets = [(model_key, rank, stages) for model_key, rank, stages in eval_targets() if model_key in args.models and rank in args.ranks]
    if not targets:
        print("No benchmark targets found.", flush=True)
        return 0

    for model_key, rank, stages in targets:
        for stage_name in ("merged_dare", "dpo"):
            adapter_path = stages.get(stage_name)
            if adapter_path is None:
                continue

            model = None
            try:
                model, tok = load_model_and_tokenizer(
                    model_key=model_key,
                    adapter_path=adapter_path,
                    offload_name=f"{model_key}_rank{rank}_{stage_name}",
                )
                for benchmark in BENCHMARK_ORDER:
                    sample_count = BENCHMARK_MODES[args.mode][benchmark]
                    key = benchmark_key(model_key, rank, stage_name, benchmark)
                    if not should_run_benchmark(results.get("benchmarks", {}).get(key), sample_count):
                        continue
                    print(f"Running {key} with n={sample_count}", flush=True)
                    started = time.time()
                    try:
                        outcome = BENCHMARK_FUNCTIONS[benchmark](
                            model,
                            tok,
                            sample_count,
                            MODEL_SPECS[model_key]["prompt_max_length"],
                        )
                    except Exception as exc:
                        outcome = {
                            "metric": "error",
                            "score": None,
                            "correct": 0,
                            "total": 0,
                            "error": str(exc),
                        }
                    outcome.update(
                        {
                            "model": model_key,
                            "rank": rank,
                            "stage": stage_name,
                            "benchmark": benchmark,
                            "samples": sample_count,
                            "adapter_path": str(adapter_path),
                            "duration_seconds": time.time() - started,
                        }
                    )
                    results.setdefault("benchmarks", {})[key] = outcome
                    save_results(results)
                    write_summary(results)
            finally:
                cleanup(model=model)

    write_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
