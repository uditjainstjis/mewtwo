import argparse
import fcntl
import json
import math
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

from post_dpo_benchmarks import (
    BASE_RANK,
    MODEL_ORDER,
    MODEL_SPECS,
    RANKS,
    STAGE_ORDER,
    cleanup,
    eval_targets,
    format_chat_prompt,
    generate_response,
    hf_token,
    load_model_and_tokenizer,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results" / "agentic_eval"
RESULTS_JSON = RESULTS_DIR / "tau_bench_results.json"
SUMMARY_MD = RESULTS_DIR / "tau_bench_summary.md"
RESULTS_LOCK = RESULTS_DIR / "tau_bench_results.lock"
RUNS_DIR = RESULTS_DIR / "tau_bench_runs"
TAU_BENCH_ROOT = ROOT / "benchmarks" / "tau-bench"

ALL_EVAL_RANKS = [BASE_RANK] + RANKS
DEFAULT_STAGES = ["base", "math_sft", "merged_dare", "dpo"]
USER_STRATEGIES = ["local_hf", "human"]


def tau_prompt_max_length(model_key: str) -> int:
    return max(3072, int(MODEL_SPECS[model_key]["prompt_max_length"]))


def load_results():
    if not RESULTS_JSON.exists():
        return {"benchmarks": {}}
    try:
        return json.loads(RESULTS_JSON.read_text())
    except Exception:
        return {"benchmarks": {}}


def save_results(payload: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))


def make_key(
    model: str,
    rank: int,
    stage: str,
    env_name: str,
    task_split: str,
    num_trials: int,
    agent_mode: str,
    user_strategy: str,
    user_model_key: str,
    user_rank: int,
    user_stage: str,
) -> str:
    return (
        f"model={model}|rank={rank}|stage={stage}|env={env_name}|split={task_split}|trials={num_trials}|"
        f"agent={agent_mode}|user={user_strategy}:{user_model_key}:{user_stage}:{user_rank}"
    )


def update_results_entry(key: str, outcome: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with RESULTS_LOCK.open("w") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload = load_results()
        payload.setdefault("benchmarks", {})[key] = outcome
        save_results(payload)
        write_summary(payload)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return payload


def write_summary(payload: dict):
    lines = [
        "# Tau-Bench Summary",
        "",
        "| Model | Rank | Stage | Env | Split | Trials | Agent | User | Score | Avg Reward | 95% CI | Success / Total | Status | Note |",
        "|-------|------|-------|-----|-------|--------|-------|------|-------|------------|---------|-----------------|--------|------|",
    ]
    for item_key, value in sorted(payload.get("benchmarks", {}).items()):
        parts = dict(part.split("=", 1) for part in item_key.split("|"))
        env_name = parts.get("env", parts.get("domain", "-"))
        task_split = parts.get("split", "-")
        trials = parts.get("trials", parts.get("repeats", "-"))
        agent_mode = parts.get("agent", "-")
        user_desc = parts.get("user", "-")
        score = value.get("score")
        avg_reward = value.get("average_reward")
        ci_low = value.get("ci95_low")
        ci_high = value.get("ci95_high")
        status = "OK"
        if value.get("metric") == "error":
            status = "ERR"
        elif value.get("metric") == "blocked":
            status = "BLOCKED"
        lines.append(
            f"| {parts.get('model', '-')} | {parts.get('rank', '-')} | {parts.get('stage', '-')} | {env_name} | {task_split} | "
            f"{trials} | {agent_mode} | {user_desc} | "
            f"{'-' if score is None else f'{score:.3f}'} | "
            f"{'-' if avg_reward is None else f'{avg_reward:.3f}'} | "
            f"{'-' if ci_low is None or ci_high is None else f'[{ci_low:.3f}, {ci_high:.3f}]'} | "
            f"{value.get('correct', 0)} / {value.get('total', 0)} | {status} | {value.get('note', '')} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


def repo_checkout_present() -> bool:
    return (TAU_BENCH_ROOT / "tau_bench").exists()


def api_key_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "GOOGLE_API_KEY": bool(os.environ.get("GOOGLE_API_KEY")),
        "MISTRAL_API_KEY": bool(os.environ.get("MISTRAL_API_KEY")),
    }


def detect_tau_bench():
    if not repo_checkout_present():
        return {"available": False, "reason": f"repo checkout missing at {TAU_BENCH_ROOT}"}

    repo_path = str(TAU_BENCH_ROOT)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    try:
        import tau_bench  # type: ignore

        return {"available": True, "path": getattr(tau_bench, "__file__", "local checkout")}
    except ModuleNotFoundError as exc:
        if exc.name != "litellm":
            return {"available": False, "reason": str(exc)}

    fake_litellm = types.ModuleType("litellm")

    def _completion(*args, **kwargs):
        raise RuntimeError("litellm is unavailable in this venv; use local_hf mode or install litellm")

    fake_litellm.completion = _completion
    fake_litellm.provider_list = []
    sys.modules["litellm"] = fake_litellm
    for module_name in list(sys.modules):
        if module_name == "tau_bench" or module_name.startswith("tau_bench."):
            sys.modules.pop(module_name, None)
    import tau_bench  # type: ignore

    return {
        "available": True,
        "path": getattr(tau_bench, "__file__", "local checkout"),
        "litellm_stubbed": True,
    }


def load_tau_interfaces():
    detect = detect_tau_bench()
    if not detect.get("available"):
        raise RuntimeError(detect.get("reason", "tau-bench unavailable"))
    from tau_bench.envs import get_env  # type: ignore
    from tau_bench.types import Action  # type: ignore

    return get_env, Action, detect


def render_chat_messages(tok, messages: list[dict]) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        lines = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            lines.append(f"{role}: {message.get('content', '')}")
        lines.append("Assistant:")
        return "\n".join(lines)


def generate_chat_text(model, tok, messages: list[dict], max_new_tokens: int, prompt_max_length: int):
    prompt = render_chat_messages(tok, messages)
    return generate_response(
        model=model,
        tok=tok,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        prompt_max_length=prompt_max_length,
    )


class LocalHFUserSimulator:
    def __init__(self, model, tok, prompt_max_length: int, max_new_tokens: int = 160):
        self.model = model
        self.tok = tok
        self.prompt_max_length = prompt_max_length
        self.max_new_tokens = max_new_tokens
        self.instruction = ""
        self.turns: list[dict] = []

    def build_system_prompt(self, instruction: str | None) -> str:
        instruction_display = f"\n\nInstruction: {instruction}\n" if instruction is not None else ""
        return (
            "You are a user interacting with an agent."
            f"{instruction_display}"
            "\nRules:"
            "\n- Generate exactly one user message per turn."
            "\n- Do not reveal the whole instruction immediately."
            "\n- Only provide information that is necessary for the current step."
            "\n- Do not hallucinate information not present in the instruction."
            "\n- If the instruction goal is satisfied, output ###STOP### by itself."
            "\n- Keep the conversation natural and stick to the persona in the instruction."
        )

    def reset(self, instruction: str | None = None) -> str:
        self.instruction = instruction or ""
        opening = self.instruction.strip()
        self.turns = [{"customer": opening}]
        return opening

    def step(self, content: str) -> str:
        self.turns.append({"agent": content})
        return self.generate_next_message()

    def build_prompt(self) -> str:
        lines = [self.build_system_prompt(instruction=self.instruction), "", "Conversation so far:"]
        for turn in self.turns:
            if "agent" in turn:
                lines.append(f"Agent: {turn['agent']}")
            if "customer" in turn:
                lines.append(f"Customer: {turn['customer']}")
        lines.append("Customer:")
        return "\n".join(lines)

    def generate_next_message(self) -> str:
        response = generate_response(
            model=self.model,
            tok=self.tok,
            prompt=self.build_prompt(),
            max_new_tokens=self.max_new_tokens,
            prompt_max_length=self.prompt_max_length,
        ).strip()
        self.turns[-1]["customer"] = response
        return response

    def get_total_cost(self) -> float:
        return 0.0


def available_targets(models: list[str], ranks: list[int], stages: list[str]):
    custom_adapter_path = getattr(args, "custom_adapter_path", None) if "args" in globals() else None
    custom_stage_label = getattr(args, "custom_stage_label", None) if "args" in globals() else None
    if custom_adapter_path:
        if len(models) != 1 or len(ranks) != 1 or len(stages) != 1:
            raise ValueError("custom adapter mode requires exactly one model, one rank, and one stage")
        stage_name = custom_stage_label or stages[0]
        return [(models[0], ranks[0], {stage_name: Path(custom_adapter_path)})]
    targets = []
    for model_key, rank, stage_map in eval_targets():
        if model_key not in models or rank not in ranks:
            continue
        filtered = {stage_name: stage_map.get(stage_name) for stage_name in stages if stage_name in stage_map}
        if filtered:
            targets.append((model_key, rank, filtered))
    return targets


def resolve_adapter_path(model_key: str, rank: int, stage_name: str):
    custom_adapter_path = getattr(args, "custom_adapter_path", None) if "args" in globals() else None
    custom_stage_label = getattr(args, "custom_stage_label", None) if "args" in globals() else None
    if custom_adapter_path:
        wanted_stage = custom_stage_label or (args.stages[0] if getattr(args, "stages", None) else stage_name)
        if model_key == args.models[0] and rank == args.ranks[0] and stage_name == wanted_stage:
            return Path(custom_adapter_path)
    for mk, rk, stage_map in eval_targets():
        if mk == model_key and rk == rank:
            return stage_map.get(stage_name)
    return None


def build_tool_spec(tools_info: list[dict]) -> str:
    simplified = []
    for tool in tools_info:
        fn = tool.get("function", {})
        params = fn.get("parameters", {})
        properties = params.get("properties", {})
        simplified.append(
            {
                "name": fn.get("name"),
                "description": str(fn.get("description", "")).split(".")[0],
                "parameters": {name: spec.get("type", "any") for name, spec in properties.items()},
                "required": params.get("required", []),
            }
        )
    return json.dumps(simplified, indent=2)


def filtered_tools_info(tools_info: list[dict]) -> list[dict]:
    return [tool for tool in tools_info if tool.get("function", {}).get("name") != "think"]


def extract_customer_facts(customer_text: str) -> dict:
    order_ids = sorted(set(re.findall(r"#W\d+", customer_text)))
    zip_codes = sorted(set(re.findall(r"\b\d{5}\b", customer_text)))
    email_match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", customer_text)
    name_match = re.search(
        r"(?:You are|Your name is)\s+([A-Z][a-z]+)(?:[_\s]+)([A-Z][a-z]+)",
        customer_text,
    )
    return {
        "customer_name": None if not name_match else f"{name_match.group(1)} {name_match.group(2)}",
        "first_name": None if not name_match else name_match.group(1),
        "last_name": None if not name_match else name_match.group(2),
        "zip_codes": zip_codes,
        "order_ids": order_ids,
        "email": None if not email_match else email_match.group(0),
    }


def build_agent_user_message(transcript: list[dict], observation: str) -> str:
    lines = ["Conversation transcript:"]
    for item in transcript:
        if item["kind"] == "customer":
            lines.append(f"Customer: {item['content']}")
        elif item["kind"] == "agent_message":
            lines.append(f"AgentMessage: {item['content']}")
        elif item["kind"] == "agent_tool":
            lines.append(
                f"AgentToolCall: {item['name']}({json.dumps(item['arguments'], sort_keys=True)})"
            )
        elif item["kind"] == "tool":
            lines.append(f"ToolResult[{item['tool_name']}]: {item['content']}")
    lines.append("")
    customer_text = "\n".join(item["content"] for item in transcript if item["kind"] == "customer")
    facts = extract_customer_facts(customer_text)
    known_facts = []
    if facts["customer_name"]:
        known_facts.append(f"customer_name={facts['customer_name']}")
    if facts["email"]:
        known_facts.append(f"email={facts['email']}")
    if facts["zip_codes"]:
        known_facts.append(f"zip_codes={facts['zip_codes']}")
    if facts["order_ids"]:
        known_facts.append(f"order_ids={facts['order_ids']}")
    if known_facts:
        lines.append("Known facts: " + "; ".join(known_facts))
        lines.append("")
    lines.append(f"New observation: {observation}")
    lines.append("Return the next action as a single JSON object.")
    return "\n".join(lines)


def preferred_action_text(text: str) -> str:
    for marker in ["Action:", '"name":', '{"name"']:
        idx = text.rfind(marker)
        if idx != -1:
            return text[idx:]
    return text


def extract_json_object(text: str):
    decoder = json.JSONDecoder()
    candidates = []
    preferred_text = preferred_action_text(text)
    for source_bias, source_text in [(2, preferred_text), (0, text)]:
        for idx, char in enumerate(source_text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(source_text[idx:])
                if isinstance(obj, dict):
                    score = source_bias
                    if any(key in obj for key in ("name", "action", "tool")):
                        score += 2
                    if any(key in obj for key in ("arguments", "kwargs", "content")):
                        score += 1
                    candidates.append((score, idx, obj))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return candidates[0][2]


def parse_agent_action(raw_text: str, allowed_tools: set[str]) -> tuple[str, dict, bool]:
    parsed = extract_json_object(raw_text)
    if not isinstance(parsed, dict):
        return "respond", {"content": raw_text.strip()}, False

    name = parsed.get("name") or parsed.get("action") or parsed.get("tool")
    arguments = parsed.get("arguments")
    if arguments is None:
        arguments = parsed.get("kwargs")

    # Some models emit nested action objects such as:
    # {"name": {"name": "...", "arguments": {...}}}
    # or {"action": {"tool": "...", "kwargs": {...}}}
    if isinstance(name, dict):
        nested = name
        name = nested.get("name") or nested.get("action") or nested.get("tool")
        if arguments is None:
            arguments = nested.get("arguments")
        if arguments is None:
            arguments = nested.get("kwargs")

    if name == "respond":
        if not isinstance(arguments, dict):
            content = parsed.get("content") or raw_text.strip()
            return "respond", {"content": str(content)}, True
        content = arguments.get("content") or parsed.get("content") or ""
        return "respond", {"content": str(content)}, True

    if isinstance(name, str) and name in allowed_tools and isinstance(arguments, dict):
        return str(name), arguments, True

    return "respond", {"content": raw_text.strip()}, False


def repair_action_json(
    model,
    tok,
    raw_text: str,
    allowed_tools: set[str],
    tools_info: list[dict],
    transcript: list[dict],
    observation: str,
    model_key: str,
):
    repair_messages = [
        {
            "role": "system",
            "content": (
                "Repair the invalid draft into one valid action JSON.\n"
                "Return exactly one JSON object and nothing else.\n"
                'Valid action forms are {"name":"respond","arguments":{"content":"..."}} or '
                '{"name":"<tool_name>","arguments":{...}}.\n'
                "No markdown. No explanation. No think tool.\n"
                "If enough tool evidence already exists, prefer the next real tool call over a summary response.\n"
                f"Allowed tools: {json.dumps(sorted(allowed_tools))}\n"
                f"Tool schemas: {build_tool_spec(tools_info)}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation summary:\n{build_agent_user_message(transcript, observation)}\n\n"
                f"Invalid draft action:\n{raw_text}\n\n"
                "Return the corrected action JSON now."
            ),
        },
    ]
    repaired = generate_chat_text(
        model=model,
        tok=tok,
        messages=repair_messages,
        max_new_tokens=120,
        prompt_max_length=tau_prompt_max_length(model_key),
    )
    return parse_agent_action(repaired, allowed_tools), repaired


def infer_forced_tool_action(transcript: list[dict], allowed_tools: set[str]) -> tuple[str, dict] | None:
    customer_text = "\n".join(item["content"] for item in transcript if item["kind"] == "customer")
    facts = extract_customer_facts(customer_text)
    used_tools = [item["name"] for item in transcript if item["kind"] == "agent_tool"]
    if not used_tools:
        if facts["email"] and "find_user_id_by_email" in allowed_tools:
            return "find_user_id_by_email", {"email": facts["email"]}
        if (
            facts["first_name"]
            and facts["last_name"]
            and facts["zip_codes"]
            and "find_user_id_by_name_zip" in allowed_tools
        ):
            return "find_user_id_by_name_zip", {
                "first_name": facts["first_name"],
                "last_name": facts["last_name"],
                "zip": facts["zip_codes"][0],
            }
    if "get_order_details" in allowed_tools and facts["order_ids"] and "get_order_details" not in used_tools:
        return "get_order_details", {"order_id": facts["order_ids"][0]}
    order_detail = None
    for item in reversed(transcript):
        if item["kind"] == "tool" and item.get("tool_name") == "get_order_details":
            parsed = item.get("parsed_observation")
            order_detail = parsed if isinstance(parsed, dict) else None
            break
    if order_detail and "get_product_details" in allowed_tools:
        customer_text_lower = customer_text.lower()
        seen_product_ids = {
            str(item["arguments"].get("product_id"))
            for item in transcript
            if item["kind"] == "agent_tool" and item["name"] == "get_product_details"
        }
        relevant_product_ids = []
        for order_item in order_detail.get("items", []):
            product_name = str(order_item.get("name", "")).lower()
            product_id = str(order_item.get("product_id", ""))
            if product_name and product_name in customer_text_lower and product_id:
                relevant_product_ids.append(product_id)
        for product_id in relevant_product_ids:
            if product_id not in seen_product_ids:
                return "get_product_details", {"product_id": product_id}
    return None


def sanitize_run_name(name: str) -> str:
    return (
        name.replace("|", "__")
        .replace("=", "-")
        .replace(":", "-")
        .replace("/", "_")
        .replace(".", "_")
    )


def normalize_action_kwargs(action_name: str, action_kwargs: dict) -> dict:
    normalized = dict(action_kwargs)
    order_id_tools = {
        "get_order_details",
        "exchange_delivered_order_items",
        "return_delivered_order_items",
        "modify_pending_order_address",
        "modify_pending_order_items",
        "modify_pending_order_payment",
        "cancel_pending_order",
    }
    if action_name in order_id_tools and isinstance(normalized.get("order_id"), str):
        if normalized["order_id"] and not normalized["order_id"].startswith("#"):
            normalized["order_id"] = f"#{normalized['order_id'].lstrip('#')}"
    return normalized


def latest_order_status(transcript: list[dict]) -> str | None:
    for item in reversed(transcript):
        if item["kind"] == "tool" and item.get("tool_name") == "get_order_details":
            parsed = item.get("parsed_observation")
            if isinstance(parsed, dict):
                return parsed.get("status")
            return None
    return None


def has_authenticated_user(transcript: list[dict]) -> bool:
    for item in transcript:
        if item["kind"] != "tool":
            continue
        if item.get("tool_name") not in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
            continue
        content = str(item.get("content", ""))
        if content and not content.startswith("Error:"):
            return True
    return False


def summarize_tool_observation(tool_name: str, observation) -> str:
    if isinstance(observation, str) and observation.startswith("Error:"):
        return observation
    try:
        parsed = json.loads(observation) if isinstance(observation, str) else observation
    except Exception:
        parsed = None
    if tool_name in {"find_user_id_by_email", "find_user_id_by_name_zip"}:
        return f"user_id={observation}"
    if tool_name == "get_user_details" and isinstance(parsed, dict):
        methods = sorted(parsed.get("payment_methods", {}).keys())
        orders = parsed.get("orders", [])
        return (
            f"user={parsed.get('name', {})} "
            f"email={parsed.get('email')} "
            f"payment_methods={methods} "
            f"orders={orders}"
        )
    if tool_name == "get_order_details" and isinstance(parsed, dict):
        items = [
            {
                "name": item.get("name"),
                "product_id": item.get("product_id"),
                "item_id": item.get("item_id"),
                "options": item.get("options"),
                "price": item.get("price"),
            }
            for item in parsed.get("items", [])
        ]
        payment_ids = [entry.get("payment_method_id") for entry in parsed.get("payment_history", [])]
        return (
            f"order_id={parsed.get('order_id')} "
            f"status={parsed.get('status')} "
            f"user_id={parsed.get('user_id')} "
            f"items={items} "
            f"payment_method_history={payment_ids}"
        )
    if tool_name == "get_product_details" and isinstance(parsed, dict):
        variants = []
        for variant_id, variant in parsed.get("variants", {}).items():
            if len(variants) >= 8:
                break
            variants.append(
                {
                    "item_id": variant_id,
                    "available": variant.get("available"),
                    "options": variant.get("options"),
                    "price": variant.get("price"),
                }
            )
        return f"product={parsed.get('name')} product_id={parsed.get('product_id')} variants={variants}"
    return str(observation)


def gold_prefix_metrics(env, steps: list[dict]) -> dict:
    predicted = [
        {"name": step["parsed_action"]["name"], "arguments": step["parsed_action"].get("arguments", {})}
        for step in steps
        if step["parsed_action"]["name"] != "respond"
    ]
    gold = [
        {"name": action.name, "arguments": normalize_action_kwargs(action.name, dict(action.kwargs))}
        for action in env.task.actions
        if action.name != "respond"
    ]
    tool_name_prefix = 0
    exact_prefix = 0
    for pred, target in zip(predicted, gold):
        if pred["name"] == target["name"]:
            tool_name_prefix += 1
        else:
            break
    for pred, target in zip(predicted, gold):
        pred_args = normalize_action_kwargs(pred["name"], dict(pred.get("arguments", {})))
        if pred["name"] == target["name"] and pred_args == target["arguments"]:
            exact_prefix += 1
        else:
            break
    denom = max(len(gold), 1)
    predicted_names = [item["name"] for item in predicted]
    gold_names = [item["name"] for item in gold]
    matched = 0
    for name in gold_names:
        if name in predicted_names:
            matched += 1
    terminal_tool_match = 1.0 if gold_names and gold_names[-1] in predicted_names else 0.0
    return {
        "gold_tool_name_prefix": tool_name_prefix,
        "gold_exact_prefix": exact_prefix,
        "gold_total_actions": len(gold),
        "gold_tool_name_prefix_frac": tool_name_prefix / denom,
        "gold_exact_prefix_frac": exact_prefix / denom,
        "gold_tool_name_recall_frac": matched / denom,
        "gold_terminal_tool_match": terminal_tool_match,
    }


def pass_k_from_results(episodes: list[dict], num_trials: int):
    grouped: dict[int, int] = {}
    for episode in episodes:
        grouped.setdefault(int(episode["task_id"]), 0)
        grouped[int(episode["task_id"])] += 1 if float(episode.get("reward", 0.0)) >= 1.0 else 0
    metrics = {}
    if not grouped:
        return metrics
    for k in range(1, num_trials + 1):
        total = 0.0
        for count in grouped.values():
            if count < k:
                continue
            total += math.comb(count, k) / max(math.comb(num_trials, k), 1)
        metrics[f"pass@{k}"] = total / len(grouped)
    return metrics


def preflight_for_entry(args, model_key: str, rank: int, stage_name: str):
    detect = detect_tau_bench()
    if not detect.get("available"):
        return False, detect.get("reason", "tau-bench unavailable")

    adapter_path = resolve_adapter_path(model_key, rank, stage_name)
    if rank != BASE_RANK and adapter_path is None:
        return False, f"adapter path missing for {model_key} rank {rank} stage {stage_name}"

    user_adapter_path = resolve_adapter_path(args.user_model_key, args.user_rank, args.user_stage)
    if args.user_rank != BASE_RANK and user_adapter_path is None:
        return False, (
            f"user simulator adapter path missing for {args.user_model_key} rank {args.user_rank} stage {args.user_stage}"
        )

    return True, f"tau-bench loaded from {detect.get('path')}"


def instantiate_env(get_env, args, task_index: int):
    env = get_env(
        args.env,
        user_strategy="human",
        user_model="dummy",
        task_split=args.task_split,
        task_index=task_index,
    )
    return env


def run_local_episode(
    env,
    Action,
    agent_model,
    agent_tok,
    user_model,
    user_tok,
    model_key: str,
    user_model_key: str,
    max_steps: int,
):
    env.user = LocalHFUserSimulator(
        model=user_model,
        tok=user_tok,
        prompt_max_length=tau_prompt_max_length(user_model_key),
    )
    reset_res = env.reset(task_index=env.task_index)
    observation = reset_res.observation
    transcript = [{"kind": "customer", "content": observation}]
    tools_info = filtered_tools_info(env.tools_info)
    allowed_tools = {tool["function"]["name"] for tool in tools_info}
    system_prompt = build_agent_system_prompt(env.__class__.__name__, env.wiki, env.rules, tools_info)
    steps = []
    tool_errors = 0
    invalid_actions = 0
    repaired_actions = 0
    exhausted = True
    reward = 0.0
    info = reset_res.info.model_dump()

    for _ in range(max_steps):
        user_message = build_agent_user_message(transcript, observation)
        raw_text = generate_chat_text(
            model=agent_model,
            tok=agent_tok,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_new_tokens=256,
            prompt_max_length=tau_prompt_max_length(model_key),
        )
        action_name, action_kwargs, parsed_cleanly = parse_agent_action(raw_text, allowed_tools)
        repaired_text = None
        if not parsed_cleanly:
            (action_name, action_kwargs, parsed_cleanly), repaired_text = repair_action_json(
                model=agent_model,
                tok=agent_tok,
                raw_text=raw_text,
                allowed_tools=allowed_tools,
                tools_info=tools_info,
                transcript=transcript,
                observation=observation,
                model_key=model_key,
            )
            repaired_actions += 1
        if not parsed_cleanly:
            invalid_actions += 1
        action_kwargs = normalize_action_kwargs(action_name, action_kwargs)
        forced = infer_forced_tool_action(transcript, allowed_tools)
        mutating_tools = {
            "cancel_pending_order",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "return_delivered_order_items",
            "exchange_delivered_order_items",
        }
        if action_name == "respond" and forced is not None:
            response_text = str(action_kwargs.get("content", "")).lower()
            if "?" not in response_text:
                action_name, action_kwargs = forced
        elif action_name in mutating_tools and forced is not None:
            action_name, action_kwargs = forced
        status = latest_order_status(transcript)
        if status == "delivered" and action_name == "modify_pending_order_items":
            action_name = "exchange_delivered_order_items"
        elif status == "pending" and action_name == "exchange_delivered_order_items":
            action_name = "modify_pending_order_items"
        auth_tools = {"find_user_id_by_email", "find_user_id_by_name_zip"}
        if not has_authenticated_user(transcript) and action_name not in auth_tools and forced is not None:
            forced_name, forced_kwargs = forced
            if forced_name in auth_tools:
                action_name, action_kwargs = forced_name, forced_kwargs
        action = Action(name=action_name, kwargs=action_kwargs)
        env_res = env.step(action)
        reward = float(env_res.reward)
        info = env_res.info.model_dump()
        if action_name == "respond":
            transcript.append({"kind": "agent_message", "content": str(action_kwargs.get("content", ""))})
            transcript.append({"kind": "customer", "content": env_res.observation})
        else:
            transcript.append({"kind": "agent_tool", "name": action_name, "arguments": action_kwargs})
            parsed_observation = None
            if isinstance(env_res.observation, str):
                try:
                    parsed_observation = json.loads(env_res.observation)
                except Exception:
                    parsed_observation = None
            transcript.append(
                {
                    "kind": "tool",
                    "tool_name": action_name,
                    "content": summarize_tool_observation(action_name, env_res.observation),
                    "raw_observation": env_res.observation,
                    "parsed_observation": parsed_observation,
                }
            )
        if str(env_res.observation).startswith("Error:"):
            tool_errors += 1
        steps.append(
            {
                "raw_model_text": raw_text,
                "repaired_model_text": repaired_text,
                "parsed_action": {"name": action_name, "arguments": action_kwargs},
                "parsed_cleanly": parsed_cleanly,
                "observation": env_res.observation,
                "done": env_res.done,
                "reward": reward,
            }
        )
        observation = env_res.observation
        if env_res.done:
            exhausted = False
            break

    prefix_metrics = gold_prefix_metrics(env, steps)
    return {
        "reward": reward,
        "info": info,
        "steps": steps,
        "num_steps": len(steps),
        "invalid_actions": invalid_actions,
        "repaired_actions": repaired_actions,
        "tool_errors": tool_errors,
        "exhausted_max_steps": exhausted,
        **prefix_metrics,
    }
def build_agent_system_prompt(env_name: str, wiki: str, rules: list[str], tools_info: list[dict]) -> str:
    policy_lines = "\n".join(f"- {rule}" for rule in rules[:8]) if rules else "- Follow the domain policy."
    return (
        f"You are a customer support agent operating in the {env_name} tau-bench environment.\n"
        "Follow the domain policy and tool constraints.\n"
        "Return exactly one JSON object and nothing else.\n"
        'Use {"name": "respond", "arguments": {"content": "..."} } to send a message to the customer.\n'
        'Use {"name": "<tool_name>", "arguments": {...}} to call one tool.\n'
        "Do not use any hidden scratchpad or think-style action.\n"
        "Take exactly one action per turn.\n"
        "If the customer already provided name and zip, prefer identity lookup instead of asking again.\n"
        "Do not repeat an identical tool call if you already have its result.\n"
        "Preserve the exact '#' prefix on order ids such as #W2378156.\n"
        "If a backend-changing tool would be needed, gather details first and ask for explicit confirmation before that change.\n"
        "If you need more information, ask a short follow-up question.\n"
        "Do not hallucinate ids when a tool can retrieve them.\n\n"
        "Operational rule:\n"
        "- If the customer already gave email, use find_user_id_by_email first.\n"
        "- Otherwise if the customer already gave name and zip, use find_user_id_by_name_zip first.\n"
        "- If the customer already gave an order id and you are authenticated, inspect the order before asking more questions.\n"
        "- Do not start with a polite acknowledgement when a lookup tool can be called immediately.\n\n"
        "Retail wiki:\n"
        f"{wiki[:4000]}\n\n"
        "Policy highlights:\n"
        f"{policy_lines}\n\n"
        "Available tools:\n"
        f"{build_tool_spec(tools_info)}"
    )


def run_local_hf_suite(args, model_key: str, rank: int, stage_name: str):
    get_env, Action, detect = load_tau_interfaces()
    started = time.time()
    agent_adapter_path = resolve_adapter_path(model_key, rank, stage_name)
    user_adapter_path = resolve_adapter_path(args.user_model_key, args.user_rank, args.user_stage)

    agent_model = user_model = None
    try:
        agent_model, agent_tok = load_model_and_tokenizer(
            model_key=model_key,
            adapter_path=agent_adapter_path,
            offload_name=f"tau_bench_agent_{model_key}_rank{rank}_{stage_name}",
        )
        if (
            model_key == args.user_model_key
            and rank == args.user_rank
            and stage_name == args.user_stage
            and agent_adapter_path == user_adapter_path
        ):
            user_model = agent_model
            user_tok = agent_tok
        else:
            user_model, user_tok = load_model_and_tokenizer(
                model_key=args.user_model_key,
                adapter_path=user_adapter_path,
                offload_name=(
                    f"tau_bench_user_{args.user_model_key}_rank{args.user_rank}_{args.user_stage}"
                ),
            )

        probe_env = instantiate_env(get_env, args, task_index=0)
        total_tasks = len(probe_env.tasks)
        if args.task_ids:
            task_ids = sorted(set(args.task_ids))
        else:
            task_ids = list(range(min(args.limit, total_tasks)))

        episodes = []
        for trial in range(args.num_trials):
            for task_id in task_ids:
                env = instantiate_env(get_env, args, task_index=task_id)
                episode = run_local_episode(
                    env=env,
                    Action=Action,
                    agent_model=agent_model,
                    agent_tok=agent_tok,
                    user_model=user_model,
                    user_tok=user_tok,
                    model_key=model_key,
                    user_model_key=args.user_model_key,
                    max_steps=args.max_steps,
                )
                episode["trial"] = trial
                episode["task_id"] = task_id
                episodes.append(episode)

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        run_name = sanitize_run_name(
            make_key(
                model=model_key,
                rank=rank,
                stage=stage_name,
                env_name=args.env,
                task_split=args.task_split,
                num_trials=args.num_trials,
                agent_mode="local_hf",
                user_strategy=args.user_strategy,
                user_model_key=args.user_model_key,
                user_rank=args.user_rank,
                user_stage=args.user_stage,
            )
        )
        run_path = RUNS_DIR / f"{run_name}.json"
        run_path.write_text(json.dumps({"episodes": episodes}, indent=2))

        successes = sum(1 for episode in episodes if float(episode.get("reward", 0.0)) >= 1.0)
        total = len(episodes)
        ci_low, ci_high = wilson_interval(successes, total)
        pass_metrics = pass_k_from_results(episodes, args.num_trials)
        avg_reward = sum(float(episode.get("reward", 0.0)) for episode in episodes) / max(total, 1)
        avg_invalid = sum(int(episode.get("invalid_actions", 0)) for episode in episodes) / max(total, 1)
        avg_repaired = sum(int(episode.get("repaired_actions", 0)) for episode in episodes) / max(total, 1)
        avg_tool_errors = sum(int(episode.get("tool_errors", 0)) for episode in episodes) / max(total, 1)
        avg_steps = sum(int(episode.get("num_steps", 0)) for episode in episodes) / max(total, 1)
        avg_name_prefix = sum(float(episode.get("gold_tool_name_prefix_frac", 0.0)) for episode in episodes) / max(total, 1)
        avg_exact_prefix = sum(float(episode.get("gold_exact_prefix_frac", 0.0)) for episode in episodes) / max(total, 1)
        avg_name_recall = sum(float(episode.get("gold_tool_name_recall_frac", 0.0)) for episode in episodes) / max(total, 1)
        avg_terminal_match = sum(float(episode.get("gold_terminal_tool_match", 0.0)) for episode in episodes) / max(total, 1)
        exhausted = sum(1 for episode in episodes if episode.get("exhausted_max_steps"))

        outcome = {
            "metric": "pass_rate",
            "score": pass_metrics.get("pass@1", avg_reward),
            "average_reward": avg_reward,
            "correct": successes,
            "total": total,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "pass_k": pass_metrics,
            "average_invalid_actions": avg_invalid,
            "average_repaired_actions": avg_repaired,
            "average_tool_errors": avg_tool_errors,
            "average_steps": avg_steps,
            "average_gold_tool_name_prefix_frac": avg_name_prefix,
            "average_gold_exact_prefix_frac": avg_exact_prefix,
            "average_gold_tool_name_recall_frac": avg_name_recall,
            "average_gold_terminal_tool_match": avg_terminal_match,
            "max_step_exhaustions": exhausted,
            "model": model_key,
            "rank": rank,
            "stage": stage_name,
            "env": args.env,
            "task_split": args.task_split,
            "num_trials": args.num_trials,
            "task_ids": task_ids,
            "tasks_per_trial": len(task_ids),
            "duration_seconds": time.time() - started,
            "run_file": str(run_path),
            "adapter_path": None if agent_adapter_path is None else str(agent_adapter_path),
            "user_simulator": {
                "strategy": args.user_strategy,
                "model_key": args.user_model_key,
                "rank": args.user_rank,
                "stage": args.user_stage,
                "adapter_path": None if user_adapter_path is None else str(user_adapter_path),
            },
            "tau_bench_path": detect.get("path"),
            "note": (
                "Executed with a local HF agent and a local HF user simulator. "
                "This is useful for offline ranking, but it is not an official API-backed leaderboard run."
            ),
        }
        return outcome
    finally:
        if user_model is not None and user_model is not agent_model:
            cleanup(model=user_model)
        cleanup(model=agent_model)


def parse_args():
    parser = argparse.ArgumentParser(description="Run tau-bench with local HF models or record preflight blockers.")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=["qwen_0.8b"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[0, 1, 2, 8, 128])
    parser.add_argument("--stages", nargs="+", default=DEFAULT_STAGES)
    parser.add_argument("--custom-adapter-path", default=None)
    parser.add_argument("--custom-stage-label", default=None)
    parser.add_argument("--env", choices=["retail", "airline"], default="retail")
    parser.add_argument("--task-split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--task-ids", nargs="+", type=int, default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--user-strategy", choices=USER_STRATEGIES, default="local_hf")
    parser.add_argument("--user-model-key", choices=MODEL_ORDER, default="qwen_0.8b")
    parser.add_argument("--user-rank", type=int, default=0)
    parser.add_argument("--user-stage", default="base")
    parser.add_argument("--record-blocked", action="store_true")
    return parser.parse_args()


def main():
    global args
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = load_results()
    targets = available_targets(args.models, args.ranks, args.stages)
    if not targets:
        raise RuntimeError("No tau-bench targets resolved from the requested models/ranks/stages")

    for model_key, rank, stage_map in targets:
        for stage_name in args.stages:
            if stage_name not in stage_map:
                continue
            entry_key = make_key(
                model=model_key,
                rank=rank,
                stage=stage_name,
                env_name=args.env,
                task_split=args.task_split,
                num_trials=args.num_trials,
                agent_mode="local_hf",
                user_strategy=args.user_strategy,
                user_model_key=args.user_model_key,
                user_rank=args.user_rank,
                user_stage=args.user_stage,
            )
            ready, note = preflight_for_entry(args, model_key, rank, stage_name)
            if not ready:
                if args.record_blocked:
                    update_results_entry(
                        entry_key,
                        {
                            "metric": "blocked",
                            "score": None,
                            "average_reward": None,
                            "correct": 0,
                            "total": 0,
                            "ci95_low": None,
                            "ci95_high": None,
                            "model": model_key,
                            "rank": rank,
                            "stage": stage_name,
                            "env": args.env,
                            "task_split": args.task_split,
                            "num_trials": args.num_trials,
                            "note": note,
                            "api_keys_present": api_key_status(),
                            "hf_token_present": bool(hf_token()),
                        },
                    )
                continue

            started = time.time()
            try:
                if args.user_strategy != "local_hf":
                    raise RuntimeError("Only local_hf user strategy is implemented in this runner")
                outcome = run_local_hf_suite(args, model_key, rank, stage_name)
            except Exception as exc:
                outcome = {
                    "metric": "error",
                    "score": None,
                    "average_reward": None,
                    "correct": 0,
                    "total": 0,
                    "ci95_low": None,
                    "ci95_high": None,
                    "model": model_key,
                    "rank": rank,
                    "stage": stage_name,
                    "env": args.env,
                    "task_split": args.task_split,
                    "num_trials": args.num_trials,
                    "duration_seconds": time.time() - started,
                    "note": note,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "api_keys_present": api_key_status(),
                    "hf_token_present": bool(hf_token()),
                }
            payload = update_results_entry(entry_key, outcome)

    print(RESULTS_JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
