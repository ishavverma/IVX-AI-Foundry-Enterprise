"""
AFE — Neural Architecture Analytical Calculator

Computes parameter counts, active parameters per token, FLOPs, and VRAM
footprints across precisions (FP16, FP8, INT4) for training and inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Any

from engines.model.config import ArchitectureConfig


@dataclass
class ArchitectureMetrics:
    total_params: int
    active_params: int
    embedding_params: int
    attention_params: int
    ssm_params: int
    ffn_moe_params: int
    norm_params: int
    
    # Computational complexity
    gflops_forward_per_token: float
    tflops_backward_per_token: float
    
    # Memory Footprints (Megabytes)
    weights_fp16_mb: int
    weights_fp8_mb: int
    weights_int4_mb: int
    
    # Training VRAM estimates (Mixed Precision AdamW with gradient checkpointing)
    training_vram_fp16_mb: int
    training_vram_fp8_mb: int
    
    # Inference KV cache size per token (in KB)
    kv_cache_kb_per_token: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArchitectureCalculator:
    """Analytical engine for transformer and hybrid model resource budgeting."""

    @classmethod
    def compute_metrics(cls, cfg: ArchitectureConfig) -> ArchitectureMetrics:
        d = cfg.d_model
        L = cfg.num_layers
        V = cfg.vocab_size
        h_q = cfg.num_heads
        h_kv = cfg.num_kv_heads
        d_h = cfg.head_dim or (d // h_q)
        intermediate = cfg.intermediate_size

        # 1. Embeddings & LM Head
        emb_params = V * d
        lm_head_params = V * d  # standard untied head
        
        # 2. Attention Layer Params (GQA)
        # Q projection: d * (h_q * d_h), K proj: d * (h_kv * d_h), V proj: d * (h_kv * d_h), Out proj: (h_q * d_h) * d
        q_params = d * (h_q * d_h)
        k_params = d * (h_kv * d_h)
        v_params = d * (h_kv * d_h)
        out_params = (h_q * d_h) * d
        qk_norm_params = (2 * d_h * h_q) if cfg.use_qk_norm else 0
        attn_per_layer = q_params + k_params + v_params + out_params + qk_norm_params

        # 3. SSM Layer Params (Mamba-style)
        # in_proj: 2 * d * (expand * d), conv1d: expand * d * d_conv, x_proj: (expand * d) * (d_state + 2 * dt_rank), out_proj: (expand * d) * d
        ssm_expand_dim = cfg.ssm_expand * d
        ssm_dt_rank = math.ceil(d / 16)
        ssm_in_proj = d * (2 * ssm_expand_dim)
        ssm_conv = ssm_expand_dim * cfg.ssm_d_conv
        ssm_x_proj = ssm_expand_dim * (cfg.ssm_d_state * 2 + ssm_dt_rank)
        ssm_dt_proj = ssm_dt_rank * ssm_expand_dim
        ssm_out_proj = ssm_expand_dim * d
        ssm_per_layer = ssm_in_proj + ssm_conv + ssm_x_proj + ssm_dt_proj + ssm_out_proj

        # 4. Dense FFN (SwiGLU = 3 matrices: gate_proj, up_proj, down_proj)
        dense_ffn_per_layer = 3 * d * intermediate

        # 5. MoE FFN
        if cfg.use_moe:
            expert_ffn = 3 * d * cfg.moe_intermediate_size
            router_params = d * cfg.num_experts
            moe_ffn_per_layer = (cfg.num_experts * expert_ffn) + router_params
            active_ffn_per_layer = (cfg.top_k_experts * expert_ffn) + router_params
        else:
            moe_ffn_per_layer = dense_ffn_per_layer
            active_ffn_per_layer = dense_ffn_per_layer

        # 6. Normalization
        norm_per_layer = 2 * d # pre-attn norm + pre-ffn norm

        # 7. Aggregate Layers based on SSM ratio (e.g. 1:7 means 1 attn per 8 layers)
        if cfg.use_ssm and ":" in cfg.ssm_attn_ratio:
            parts = [int(p) for p in cfg.ssm_attn_ratio.split(":")]
            attn_weight = parts[0]
            ssm_weight = parts[1]
            period = attn_weight + ssm_weight
            num_attn_layers = round(L * (attn_weight / period))
            num_ssm_layers = L - num_attn_layers
        else:
            num_attn_layers = L
            num_ssm_layers = 0

        total_attn = num_attn_layers * attn_per_layer
        total_ssm = num_ssm_layers * ssm_per_layer
        total_ffn = L * moe_ffn_per_layer
        total_norm = L * norm_per_layer + d # + final norm

        total_params = emb_params + total_attn + total_ssm + total_ffn + total_norm + lm_head_params

        # Active parameters for forward pass
        active_ffn_total = L * active_ffn_per_layer
        active_params = emb_params + total_attn + total_ssm + active_ffn_total + total_norm + lm_head_params

        # FLOPs estimation: ~2 FLOPs per active parameter forward pass
        gflops_fwd = (2 * active_params) / 1e9
        tflops_bwd = (6 * active_params) / 1e12  # standard 6 * P for full forward+backward pass

        # Weight Storage (MB)
        weights_fp16_mb = int((total_params * 2) / (1024 ** 2))
        weights_fp8_mb = int((total_params * 1) / (1024 ** 2))
        weights_int4_mb = int((total_params * 0.5) / (1024 ** 2))

        # Training VRAM (AdamW states 8 bytes + grads 2 bytes + FP32 master weights 4 bytes + activations)
        # With ZeRO-3 / FSDP and gradient checkpointing:
        train_vram_fp16 = int(weights_fp16_mb * 3.8)
        train_vram_fp8 = int(weights_fp8_mb * 3.2)

        # KV cache size per token across all layers (bytes)
        # 2 (K and V) * 2 (fp16 bytes) * num_attn_layers * h_kv * d_h
        kv_cache_bytes_per_tok = 2 * 2 * num_attn_layers * h_kv * d_h
        kv_cache_kb_per_tok = round(kv_cache_bytes_per_tok / 1024, 2)

        return ArchitectureMetrics(
            total_params=total_params,
            active_params=active_params,
            embedding_params=emb_params,
            attention_params=total_attn,
            ssm_params=total_ssm,
            ffn_moe_params=total_ffn,
            norm_params=total_norm,
            gflops_forward_per_token=round(gflops_fwd, 3),
            tflops_backward_per_token=round(tflops_bwd, 3),
            weights_fp16_mb=weights_fp16_mb,
            weights_fp8_mb=weights_fp8_mb,
            weights_int4_mb=weights_int4_mb,
            training_vram_fp16_mb=train_vram_fp16,
            training_vram_fp8_mb=train_vram_fp8,
            kv_cache_kb_per_token=kv_cache_kb_per_tok,
        )
