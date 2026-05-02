import yaml

configs = {}
datasets = ["mixed_fincode", "mixed_mathlegal", "mixed_codephilo", "pure_math", "pure_code", "gen_wikitext", "gen_mmlu"]

for ds in datasets:
    filename = ds + ".jsonl" if "mixed" in ds or "wiki" in ds else ds + ".json"
    configs[f"exp_1_{ds}"] = {"method": "single_adapter", "k": 1, "dataset": filename}
    configs[f"exp_2_{ds}"] = {"method": "static_merge", "k": 2, "dataset": filename}
    configs[f"exp_3_{ds}"] = {"method": "unclamped_mix", "k": 2, "c": 1.0, "dataset": filename}
    configs[f"exp_4_adaptive_03_{ds}"] = {"method": "adaptive_clamp", "k": 2, "c": 0.3, "dataset": filename}
    configs[f"exp_5_adaptive_05_{ds}"] = {"method": "adaptive_clamp", "k": 2, "c": 0.5, "dataset": filename}
    configs[f"exp_6_adaptive_05_k3_{ds}"] = {"method": "adaptive_clamp", "k": 3, "c": 0.5, "dataset": filename}
    configs[f"exp_7_adaptive_07_{ds}"] = {"method": "adaptive_clamp", "k": 2, "c": 0.7, "dataset": filename}

with open("configs/uma_experiments.yaml", "w") as f:
    yaml.dump(configs, f)
