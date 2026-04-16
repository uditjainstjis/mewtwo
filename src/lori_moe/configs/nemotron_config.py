"""
Nemotron-specific LoRI-MoE configuration.

Current implementation note:
  The existing PEFT training path can safely target leaf-name modules such as
  `q_proj`, `k_proj`, `v_proj`, and `o_proj`.

  It cannot yet isolate only Nemotron's `shared_experts.up_proj/down_proj`
  because PEFT's current target_modules matching would also capture routed
  expert MLPs and other MLP-like leaves with the same names.

  Because of that, the default Nemotron config uses the conservative
  `attention_only` target set until custom module filtering is added.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from src.lori_moe.config import (
    AdapterConfig,
    LoRIMoEConfig,
    ModelConfig,
    PathConfig,
    RouterConfig,
    TrainingConfig,
)


@dataclass
class NemotronModelConfig(ModelConfig):
    base_model: str = "/home/learner/Desktop/mewtwo/models/nemotron"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    hidden_size: int = 2688
    num_layers: int = 52
    intermediate_size: int = 1856


@dataclass
class NemotronAdapterConfig(AdapterConfig):
    rank: int = 64
    alpha: float = 128.0
    shared_b_seed: int = 42
    sparsity_level: float = 0.8
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dropout: float = 0.05


@dataclass
class NemotronTrainingConfig(TrainingConfig):
    adapter_lr: float = 1e-4
    adapter_epochs: int = 2
    adapter_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    bf16: bool = True
    gradient_checkpointing: bool = True


@dataclass
class NemotronPathConfig(PathConfig):
    project_root: Path = field(default_factory=lambda: Path("/home/learner/Desktop/mewtwo"))

    @property
    def checkpoints_dir(self) -> Path:
        return self.project_root / "checkpoints" / "nemotron_lori"

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
        return self.project_root / "data" / "nemotron"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results" / "nemotron"

    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs" / "nemotron"


def get_nemotron_config() -> LoRIMoEConfig:
    """Factory for a conservative, currently-implementable Nemotron config."""
    return LoRIMoEConfig(
        model=NemotronModelConfig(),
        adapter=NemotronAdapterConfig(),
        router=RouterConfig(
            bottleneck_dim=128,
            num_experts=3,
            top_k=2,
            noise_std=0.1,
            load_balance_weight=0.01,
        ),
        training=NemotronTrainingConfig(),
        paths=NemotronPathConfig(),
    )
