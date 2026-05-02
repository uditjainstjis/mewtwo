import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def reduced_svd_factors(a_weight: torch.Tensor, b_weight: torch.Tensor, retain_rank: int):
    a = a_weight.float().cpu()
    b = b_weight.float().cpu()
    rank = int(a.shape[0])
    keep = max(1, min(int(retain_rank), rank))

    qa, ra = torch.linalg.qr(a.T, mode="reduced")
    qb, rb = torch.linalg.qr(b, mode="reduced")
    small = rb @ ra.T
    u, s, vh = torch.linalg.svd(small, full_matrices=False)

    u_k = u[:, :keep]
    s_k = s[:keep]
    vh_k = vh[:keep, :]
    sqrt_s = torch.sqrt(s_k).diag()

    b_small = qb @ u_k @ sqrt_s
    a_small = sqrt_s @ vh_k @ qa.T

    out_dim = b.shape[0]
    in_dim = a.shape[1]
    b_new = torch.zeros((out_dim, rank), dtype=b.dtype)
    a_new = torch.zeros((rank, in_dim), dtype=a.dtype)
    b_new[:, :keep] = b_small.to(b.dtype)
    a_new[:keep, :] = a_small.to(a.dtype)
    return a_new.contiguous(), b_new.contiguous(), keep


def copy_support_files(src_dir: Path, out_dir: Path):
    for item in src_dir.iterdir():
        if item.name == "adapter_model.safetensors":
            continue
        target = out_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    parser = argparse.ArgumentParser(description="Create a truncated LoRA adapter variant with padded original rank.")
    parser.add_argument("--source", required=True, help="Path to source adapter directory")
    parser.add_argument("--output", required=True, help="Path to output adapter directory")
    parser.add_argument("--retain-rank", required=True, type=int, help="Number of singular directions to retain")
    parser.add_argument("--label", default=None, help="Optional label stored in metadata")
    args = parser.parse_args()

    src_dir = Path(args.source).resolve()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = load_file(str(src_dir / "adapter_model.safetensors"))
    output_tensors = {}
    summary = {
        "source": str(src_dir),
        "retain_rank": int(args.retain_rank),
        "label": args.label,
        "modules": {},
    }

    grouped = {}
    passthrough = {}
    for key, tensor in tensors.items():
        if key.endswith(".lora_A.weight"):
            stem = key[: -len(".lora_A.weight")]
            grouped.setdefault(stem, {})["A"] = tensor
        elif key.endswith(".lora_B.weight"):
            stem = key[: -len(".lora_B.weight")]
            grouped.setdefault(stem, {})["B"] = tensor
        else:
            passthrough[key] = tensor

    for stem, vals in grouped.items():
        if "A" not in vals or "B" not in vals:
            raise ValueError(f"Incomplete LoRA pair for module {stem}")
        a_new, b_new, keep = reduced_svd_factors(vals["A"], vals["B"], args.retain_rank)
        output_tensors[f"{stem}.lora_A.weight"] = a_new
        output_tensors[f"{stem}.lora_B.weight"] = b_new
        summary["modules"][stem] = {
            "original_rank": int(vals["A"].shape[0]),
            "retained_rank": int(keep),
        }

    output_tensors.update(passthrough)
    save_file(output_tensors, str(out_dir / "adapter_model.safetensors"))
    copy_support_files(src_dir, out_dir)

    (out_dir / "intervention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
