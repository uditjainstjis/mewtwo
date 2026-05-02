"""
LoRI-MoE Configuration
Central configuration for the entire LoRI-MoE pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Base model configuration."""
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    # Model architecture params (auto-detected, but defaults for Qwen2.5-3B)
    hidden_size: int = 2048
    num_layers: int = 36
    intermediate_size: int = 11008


@dataclass
class AdapterConfig:
    """LoRI adapter configuration."""
    rank: int = 32  # Higher than Synapta's 16 for more capacity
    alpha: float = 64.0  # LoRA scaling factor (alpha/rank = 2.0)
    shared_b_seed: int = 42  # Deterministic B initialization for reproducibility
    sparsity_level: float = 0.8  # 80% sparse A matrices (DARE-style)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    dropout: float = 0.05


@dataclass
class DomainConfig:
    """Domain-specific configuration."""
    name: str = ""
    train_dataset: str = ""
    train_subset: Optional[str] = None
    train_split: str = "train"
    eval_dataset: str = ""
    eval_subset: Optional[str] = None
    eval_split: str = "test"
    max_train_samples: int = 50000
    max_eval_samples: int = 1000


@dataclass
class RouterConfig:
    """Token-level router configuration."""
    bottleneck_dim: int = 64  # Router MLP bottleneck
    num_experts: int = 5  # Number of domain experts
    top_k: int = 2  # Top-K expert routing
    noise_std: float = 0.1  # Noisy gating for training
    load_balance_weight: float = 0.01  # Auxiliary loss weight
    entropy_collapse_threshold: float = 0.3  # Alert if below this


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Adapter training
    adapter_lr: float = 2e-4
    adapter_epochs: int = 3
    adapter_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # Router training
    router_lr: float = 1e-3
    router_epochs: int = 5
    router_batch_size: int = 16
    # General
    bf16: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = 500
    logging_steps: int = 50
    seed: int = 42


@dataclass
class PathConfig:
    """File paths."""
    project_root: Path = field(default_factory=lambda: Path("/home/learner/Desktop/mewtwo"))

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "adapters" / "lori_moe"

    @property
    def adapters_dir(self) -> Path:
        return self.checkpoints_dir / "adapters"

    @property
    def shared_b_path(self) -> Path:
        return self.checkpoints_dir / "shared_projection_B.pt"

    @property
    def router_dir(self) -> Path:
        return self.checkpoints_dir / "router"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "lori_moe"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / "lori_moe"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs" / "lori_moe"


# ─── Default Domain Configurations ───────────────────────────────────────────

DOMAINS = {
    "math": DomainConfig(
        name="math",
        train_dataset="meta-math/MetaMathQA",
        train_split="train",
        eval_dataset="lighteval/MATH",
        eval_split="test",
        max_train_samples=50000,
    ),
    "code": DomainConfig(
        name="code",
        train_dataset="sahil2801/CodeAlpaca-20k",
        train_split="train",
        eval_dataset="openai/openai_humaneval",
        eval_split="test",
        max_train_samples=20000,
    ),
    "science": DomainConfig(
        name="science",
        train_dataset="allenai/sciq",
        train_split="train",
        eval_dataset="allenai/ai2_arc",
        eval_subset="ARC-Challenge",
        eval_split="test",
        max_train_samples=30000,
    ),
    "legal": DomainConfig(
        name="legal",
        train_dataset="nguha/legalbench",
        train_subset="contract_nli_explicit_identification",
        train_split="test",  # legalbench uses test split
        eval_dataset="nguha/legalbench",
        eval_subset="contract_nli_explicit_identification",
        eval_split="test",
        max_train_samples=20000,
    ),
    "medical": DomainConfig(
        name="medical",
        train_dataset="bigbio/med_qa",
        train_subset="med_qa_en_4options_source",
        train_split="train",
        eval_dataset="bigbio/med_qa",
        eval_subset="med_qa_en_4options_source",
        eval_split="test",
        max_train_samples=30000,
    ),
}


@dataclass
class LoRIMoEConfig:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    domains: dict = field(default_factory=lambda: DOMAINS)

    def __post_init__(self):
        """Create directories."""
        self.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.paths.adapters_dir.mkdir(parents=True, exist_ok=True)
        self.paths.router_dir.mkdir(parents=True, exist_ok=True)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.results_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
