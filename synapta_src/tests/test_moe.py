import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.lori_moe.config import LoRIMoEConfig
from src.lori_moe.model.lori_moe_model import LoRIMoEModel
from src.lori_moe.eval.run_benchmarks import evaluate_math
import torch
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "/home/learner/Desktop/mewtwo/adapters/lori_moe/qwen2.5_1.5b"
router_path = "/home/learner/Desktop/mewtwo/adapters/lori_moe/qwen2.5_1.5b/router/best/router.pt"

config = LoRIMoEConfig()
config.model.base_model = base_model_name

moe_model = LoRIMoEModel(config)
moe_model.build(load_experts=True, adapters_root=Path(adapter_dir))
router_state = torch.load(router_path, map_location="cuda")
moe_model.routers.load_state_dict(router_state["router_state_dict"])
moe_model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

result = evaluate_math(moe_model, tokenizer, "gsm8k", 30)
print(f"LoRI-MoE Full Routing GSM8K Score: {result.score}")
