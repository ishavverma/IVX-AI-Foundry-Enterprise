"""
AFE — Milestone 5 Verification Script

Tests distributed pretraining engine, LoRA/PEFT parameter reductions,
checkpoint saving, GPTQ quantization, and GGUF export.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add root directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.database.engine import init_db, get_session
from core.database.models.experiment import Experiment, Checkpoint
from core.database.models.model_def import ModelDefinition
from core.storage.paths import storage_paths
from core.storage.sdk import get_storage_backend
from engines.training.trainer import TrainingEngine, TrainingHyperparams
from engines.finetuning.peft import PEFTEngine, LoRAConfig
from engines.checkpoints.manager import CheckpointManager


def run_tests():
    print("=======================================================")
    print("  AFE MILESTONE 5: TRAINING, PEFT & CHECKPOINTS TEST   ")
    print("=======================================================")

    init_db()
    storage_paths.init_all()
    storage = get_storage_backend()

    # ── 1. Test LoRA Parameter Reduction Calculator ──
    print("\n1. Testing LoRA Parameter Reduction Calculator:")
    lora_cfg = LoRAConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "gate_proj"])
    lora_metrics = PEFTEngine.calculate_lora_params(d_model=4096, num_layers=32, lora_cfg=lora_cfg)
    print(f"   LoRA Trainable Params: {lora_metrics['trainable_adapter_params'] / 1e6:.2f}M")
    print(f"   Trainable Ratio: {lora_metrics['trainable_percentage']}% of full weights")
    print(f"   Memory Saved: {lora_metrics['memory_saved_percent']}%")
    assert lora_metrics["trainable_adapter_params"] > 0
    assert lora_metrics["trainable_percentage"] < 5.0

    # ── 2. Test Training Engine Loop ──
    print("\n2. Testing Pretraining Engine Loop:")
    exp_id = "test-pretrain-run-01"
    with get_session() as session:
        old = session.get(Experiment, exp_id)
        if old:
            session.delete(old)
            session.flush()

        exp = Experiment(
            id=exp_id,
            name="Test Pretraining Run 01",
            slug="test-pretrain-run-01",
            run_type="pretraining",
            status="queued",
            max_steps=200,
        )
        session.add(exp)

    hp = TrainingHyperparams(max_steps=200, learning_rate=1e-4, checkpoint_every_steps=100)
    train_res = TrainingEngine.run_training_loop("job-train-01", exp_id, hp.to_dict())
    print(f"   Training Result: final_step={train_res['final_step']}, best_loss={train_res['best_loss']:.4f}, tokens={train_res['total_tokens']}")
    assert train_res["final_step"] == 200
    assert train_res["best_loss"] < 3.0

    # ── 3. Test Fine-Tuning Engine Loop (LoRA) ──
    print("\n3. Testing Fine-Tuning Engine Loop (LoRA):")
    ft_exp_id = "test-lora-run-01"
    with get_session() as session:
        old_ft = session.get(Experiment, ft_exp_id)
        if old_ft:
            session.delete(old_ft)
            session.flush()

        ft_exp = Experiment(
            id=ft_exp_id,
            name="Test LoRA Instruction Run",
            slug="test-lora-instruction-run",
            run_type="lora",
            status="queued",
            max_steps=150,
        )
        session.add(ft_exp)

    ft_res = PEFTEngine.run_finetuning_loop("job-ft-01", ft_exp_id, {"run_type": "lora", "max_steps": 150})
    print(f"   Fine-tuning Result: run_type={ft_res['run_type']}, final_loss={ft_res['final_loss']:.4f}")
    assert ft_res["final_loss"] < 2.0

    # ── 4. Test Checkpoint Manager (Quantization & GGUF) ──
    print("\n4. Testing Checkpoint Manager (Quantization & GGUF):")
    with get_session() as session:
        ckpt = session.query(Checkpoint).filter(Checkpoint.experiment_id == exp_id).first()
        assert ckpt is not None
        target_ckpt_id = ckpt.id

    quant_res = CheckpointManager.quantize_weights(target_ckpt_id, "int4")
    print(f"   Quantized to INT4: orig={quant_res.original_size_bytes / (1024**2):.1f}MB -> quant={quant_res.quantized_size_bytes / (1024**2):.1f}MB ({quant_res.compression_ratio}x compression)")
    assert quant_res.compression_ratio > 3.0

    gguf_key = CheckpointManager.export_to_gguf(target_ckpt_id)
    print(f"   GGUF Export: created '{gguf_key}'")
    assert storage.exists(gguf_key)

    print("\n=======================================================")
    print("         ALL MILESTONE 5 UNIT TESTS PASSED!          ")
    print("=======================================================")


if __name__ == "__main__":
    run_tests()
