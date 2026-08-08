"""
AFE — Model Architecture Configuration & Hierarchy Presets

Defines standard configurations for the AFE Foundation Model Family:
  • Nano (40M)   • Micro (250M)   • Small (2B)   • Base (7B)
  • Large (20B)  • XL (50B)       • XXL (150B)   • MoE-100B   • MoE-1T
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class ArchitectureConfig:
    name: str = "AFE-Base-7B"
    tier: str = "base"
    vocab_size: int = 64000
    d_model: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 8             # GQA ratio 4:1
    head_dim: int = 128
    intermediate_size: int = 14336     # ~3.5x for SwiGLU
    max_seq_len: int = 32768
    
    # Normalization & Activations
    norm_eps: float = 1e-6
    use_qk_norm: bool = True
    activation: str = "swiglu"        # swiglu, glu, silu
    
    # Hybrid Transformer-SSM
    use_ssm: bool = True
    ssm_attn_ratio: str = "1:7"        # 1 full-attention layer every 7 SSM layers
    ssm_d_state: int = 64              # Mamba state dimension
    ssm_d_conv: int = 4                # Conv1D kernel width
    ssm_expand: int = 2                # SSM projection expansion
    
    # Sparse Attention
    use_sliding_window: bool = True
    sliding_window_size: int = 4096
    
    # Mixture of Experts (MoE)
    use_moe: bool = False
    num_experts: int = 16
    top_k_experts: int = 2
    moe_intermediate_size: int = 3584  # Fine-grained expert width
    expert_capacity_factor: float = 1.25

    # Positional Encodings
    rope_theta: float = 500000.0       # Extended RoPE base
    rope_scaling_type: str = "yarn"    # yarn, dynamic, linear

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ArchitectureConfig:
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**filtered)


# ──────────────────────────────────────────────
# Standard Model Hierarchy Presets
# ──────────────────────────────────────────────
MODEL_PRESETS: Dict[str, ArchitectureConfig] = {
    "nano": ArchitectureConfig(
        name="AFE-Nano-40M", tier="nano", vocab_size=32000,
        d_model=384, num_layers=6, num_heads=6, num_kv_heads=2, head_dim=64,
        intermediate_size=1024, max_seq_len=2048, use_ssm=False, use_moe=False,
    ),
    "micro": ArchitectureConfig(
        name="AFE-Micro-250M", tier="micro", vocab_size=32000,
        d_model=768, num_layers=12, num_heads=12, num_kv_heads=4, head_dim=64,
        intermediate_size=2048, max_seq_len=4096, use_ssm=True, ssm_attn_ratio="1:3", use_moe=False,
    ),
    "small": ArchitectureConfig(
        name="AFE-Small-2B", tier="small", vocab_size=64000,
        d_model=2048, num_layers=24, num_heads=16, num_kv_heads=4, head_dim=128,
        intermediate_size=5632, max_seq_len=8192, use_ssm=True, ssm_attn_ratio="1:5", use_moe=False,
    ),
    "base": ArchitectureConfig(
        name="AFE-Base-7B", tier="base", vocab_size=64000,
        d_model=4096, num_layers=32, num_heads=32, num_kv_heads=8, head_dim=128,
        intermediate_size=14336, max_seq_len=32768, use_ssm=True, ssm_attn_ratio="1:7", use_moe=False,
    ),
    "large": ArchitectureConfig(
        name="AFE-Large-20B", tier="large", vocab_size=64000,
        d_model=5120, num_layers=48, num_heads=40, num_kv_heads=8, head_dim=128,
        intermediate_size=17920, max_seq_len=65536, use_ssm=True, ssm_attn_ratio="1:7", use_moe=False,
    ),
    "xl": ArchitectureConfig(
        name="AFE-XL-50B", tier="xl", vocab_size=64000,
        d_model=6144, num_layers=64, num_heads=48, num_kv_heads=8, head_dim=128,
        intermediate_size=21504, max_seq_len=131072, use_ssm=True, ssm_attn_ratio="1:7", use_moe=False,
    ),
    "xxl": ArchitectureConfig(
        name="AFE-XXL-150B", tier="xxl", vocab_size=100000,
        d_model=8192, num_layers=96, num_heads=64, num_kv_heads=8, head_dim=128,
        intermediate_size=28672, max_seq_len=131072, use_ssm=True, ssm_attn_ratio="1:7", use_moe=False,
    ),
    "moe_100b": ArchitectureConfig(
        name="AFE-MoE-100B", tier="moe_100b", vocab_size=64000,
        d_model=4096, num_layers=48, num_heads=32, num_kv_heads=8, head_dim=128,
        intermediate_size=14336, max_seq_len=65536, use_ssm=True, ssm_attn_ratio="1:7",
        use_moe=True, num_experts=16, top_k_experts=2, moe_intermediate_size=3584,
    ),
    "moe_1t": ArchitectureConfig(
        name="AFE-MoE-1T", tier="moe_1t", vocab_size=128000,
        d_model=8192, num_layers=64, num_heads=64, num_kv_heads=8, head_dim=128,
        intermediate_size=28672, max_seq_len=131072, use_ssm=True, ssm_attn_ratio="1:7",
        use_moe=True, num_experts=32, top_k_experts=4, moe_intermediate_size=4096,
    ),
}
