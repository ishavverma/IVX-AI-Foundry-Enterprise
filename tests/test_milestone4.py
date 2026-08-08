"""
AFE — Milestone 4 Verification Script

Tests architecture configuration, analytical calculator (params, FLOPs, VRAM),
PyTorch hybrid module generator, and code exporter.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from engines.model.config import ArchitectureConfig, MODEL_PRESETS
from engines.model.calculator import ArchitectureCalculator
from engines.model.exporter import ModelExporter
from engines.model.builder import (
    RMSNorm,
    RotaryEmbedding,
    GroupedQueryAttention,
    SSMLayer,
    SwiGLUFeedForward,
    MoEFeedForward,
    HybridBlock,
    AFEHybridModel,
)
from core.database.engine import init_db, get_session
from core.database.models.model_def import ModelDefinition
from core.storage.paths import storage_paths
from core.storage.sdk import get_storage_backend


def run_tests():
    print("=======================================================")
    print("    AFE MILESTONE 4: MODEL FACTORY & COMPILER TEST     ")
    print("=======================================================")

    init_db()
    storage_paths.init_all()
    storage = get_storage_backend()

    # ── 1. Test Architecture Calculator across presets ──
    print("\n1. Testing Architecture Calculator on Standard Presets:")
    for tier in ["nano", "micro", "small", "base", "large", "xl", "moe_100b", "moe_1t"]:
        cfg = MODEL_PRESETS[tier]
        metrics = ArchitectureCalculator.compute_metrics(cfg)
        param_str = f"{metrics.total_params / 1e9:.2f}B" if metrics.total_params >= 1e9 else f"{metrics.total_params / 1e6:.1f}M"
        active_str = f"{metrics.active_params / 1e9:.2f}B" if metrics.active_params >= 1e9 else f"{metrics.active_params / 1e6:.1f}M"
        print(f"   [{tier.upper():<8}] {cfg.name:<18} | Params: {param_str:<8} | Active: {active_str:<8} | VRAM (FP16): {metrics.weights_fp16_mb / 1024:>5.1f} GB | FLOPs: {metrics.gflops_forward_per_token:>5.1f} GF")
        assert metrics.total_params > 0
        assert metrics.weights_fp16_mb > 0

    # ── 2. Test Model Exporter ──
    print("\n2. Testing Standalone PyTorch Code Generator:")
    base_cfg = MODEL_PRESETS["base"]
    py_code = ModelExporter.generate_pytorch_code(base_cfg)
    config_json = ModelExporter.generate_config_json(base_cfg)
    print(f"   Generated model.py ({len(py_code)} chars)")
    print(f"   Generated config.json ({len(config_json)} chars)")
    assert "class AFEModel" in py_code
    assert "GroupedQueryAttention" in py_code

    # ── 3. Test Hybrid Model Instantiation ──
    print("\n3. Testing Hybrid Model Architecture:")
    mini_cfg = {
        "d_model": 128,
        "num_layers": 4,
        "vocab_size": 1000,
        "num_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 32,
        "intermediate_size": 256,
        "max_seq_len": 512,
        "ssm_attn_ratio": "1:1",
        "use_ssm": True,
        "use_moe": True,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 128,
        "norm_eps": 1e-6,
    }
    model = AFEHybridModel(mini_cfg)
    print(f"   Hybrid Model instantiated with {len(model.blocks)} layers (SSM + GQA + MoE)")
    assert len(model.blocks) == 4

    # ── 4. Save to Database & Storage SDK ──
    model_id = "test-hybrid-7b"
    code_key = f"models/{model_id}/model.py"
    config_key = f"models/{model_id}/config.json"
    storage.write(code_key, py_code.encode("utf-8"))
    storage.write(config_key, config_json.encode("utf-8"))

    base_metrics = ArchitectureCalculator.compute_metrics(base_cfg)
    with get_session() as session:
        old = session.get(ModelDefinition, model_id)
        if old:
            session.delete(old)
            session.flush()

        m = ModelDefinition(
            id=model_id,
            name="AFE Hybrid 7B Base",
            slug="afe-hybrid-7b-base",
            architecture_type="hybrid",
            preset_tier="base",
            d_model=base_cfg.d_model,
            num_layers=base_cfg.num_layers,
            num_heads=base_cfg.num_heads,
            num_kv_heads=base_cfg.num_kv_heads,
            vocab_size=base_cfg.vocab_size,
            max_seq_len=base_cfg.max_seq_len,
            ssm_ratio=base_cfg.ssm_attn_ratio,
            use_moe=False,
            num_experts=16,
            top_k_experts=2,
            param_count=base_metrics.total_params,
            active_param_count=base_metrics.active_params,
            gflops_per_token=base_metrics.gflops_forward_per_token,
            vram_fp16_mb=base_metrics.weights_fp16_mb,
            vram_fp8_mb=base_metrics.weights_fp8_mb,
            vram_int4_mb=base_metrics.weights_int4_mb,
            storage_key=config_key,
            code_storage_key=code_key,
            config_json=config_json,
            status="compiled",
        )
        session.add(m)

    print(f"   Model '{model_id}' registered in DB and stored in Storage SDK.")

    print("\n=======================================================")
    print("         ALL MILESTONE 4 UNIT TESTS PASSED!          ")
    print("=======================================================\n")


if __name__ == "__main__":
    run_tests()
