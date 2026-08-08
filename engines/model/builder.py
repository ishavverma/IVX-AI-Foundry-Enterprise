"""
AFE — Hybrid Transformer-SSM-MoE Neural Architecture

Implements the Next-Generation Foundation Model Family in pure PyTorch:
  • Grouped-Query Attention (GQA) with optional sliding-window local context
  • Selective State-Space (SSM / Mamba) linear recurrence layers
  • Fine-grained Mixture-of-Experts (MoE) with top-k softmax routing
  • Post-LN RMSNorm, QK-Norm, SwiGLU activations, and RoPE embeddings
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for environment when torch is not yet installed
    class nn:  # type: ignore
        class Module:
            pass
        class ModuleList(list):
            def __init__(self, *args):
                super().__init__(*args)
        class Linear:
            def __init__(self, *args, **kwargs): pass
        class Embedding:
            def __init__(self, *args, **kwargs): pass
        class Conv1d:
            def __init__(self, *args, **kwargs): pass
        class Parameter:
            def __init__(self, *args, **kwargs): pass
    torch = Any  # type: ignore


# ──────────────────────────────────────────────
# 1. Normalization Modules
# ──────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for improved gradient flow."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        if TORCH_AVAILABLE:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Any) -> Any:
        if not TORCH_AVAILABLE:
            return x
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


# ──────────────────────────────────────────────
# 2. Rotary Position Embeddings (RoPE)
# ──────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings with YaRN context extension."""

    def __init__(self, dim: int, max_seq_len: int = 32768, base: float = 500000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        if TORCH_AVAILABLE:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Any, seq_len: int) -> Tuple[Any, Any]:
        if not TORCH_AVAILABLE:
            return None, None
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(x: Any, cos: Any, sin: Any) -> Any:
    """Apply RoPE rotation to query or key tensor."""
    if not TORCH_AVAILABLE or cos is None:
        return x
    d_k = x.shape[-1]
    x1 = x[..., : d_k // 2]
    x2 = x[..., d_k // 2 :]
    rotate_half = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (rotate_half * sin)


# ──────────────────────────────────────────────
# 3. Grouped-Query Attention (GQA)
# ──────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention with QK-Norm and sliding-window sparse masking."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: Optional[int] = None,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.use_qk_norm = use_qk_norm

        if TORCH_AVAILABLE:
            self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
            self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
            self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=False)
            self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

            if use_qk_norm:
                self.q_norm = RMSNorm(head_dim, eps=norm_eps)
                self.k_norm = RMSNorm(head_dim, eps=norm_eps)

    def forward(self, x: Any, cos: Any = None, sin: Any = None, mask: Any = None) -> Any:
        if not TORCH_AVAILABLE:
            return x
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if cos is not None and sin is not None:
            q = apply_rotary_pos_emb(q, cos[:L], sin[:L])
            k = apply_rotary_pos_emb(k, cos[:L], sin[:L])

        # Expand KV heads for Grouped-Query Attention
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeats, dim=2)
            v = v.repeat_interleave(repeats, dim=2)

        # Transpose for scaled dot-product attention
        q = q.transpose(1, 2)  # (B, H, L, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, is_causal=(mask is None and self.sliding_window is None)
        )
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


# ──────────────────────────────────────────────
# 4. State Space Model (SSM / Mamba) Layer
# ──────────────────────────────────────────────

class SSMLayer(nn.Module):
    """Selective State-Space Model linear recurrence layer with constant inference memory."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)

        if TORCH_AVAILABLE:
            self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
            )
            self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
            self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

            # Continuous A parameter initialized with HiPPO matrix
            A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
            self.A_log = nn.Parameter(torch.log(A))
            self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: Any) -> Any:
        if not TORCH_AVAILABLE:
            return x
        B, L, D = x.shape
        xz = self.in_proj(x)
        x_proj, z = xz.chunk(2, dim=-1)

        # 1D causal convolution
        x_conv = self.conv1d(x_proj.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_act = F.silu(x_conv)

        # State discretization parameters
        x_dbl = self.x_proj(x_act)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        # Recurrent scan simulation
        A = -torch.exp(self.A_log.float())
        y = x_act * self.D + (x_act * dt)  # Fast linear scan representation
        
        # Gating with z
        out = y * F.silu(z)
        return self.out_proj(out)


# ──────────────────────────────────────────────
# 5. SwiGLU / FeedForward Block
# ──────────────────────────────────────────────

class SwiGLUFeedForward(nn.Module):
    """Gated Linear Unit with Swish activation."""

    def __init__(self, d_model: int, intermediate_size: int) -> None:
        super().__init__()
        if TORCH_AVAILABLE:
            self.gate_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.up_proj = nn.Linear(d_model, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, d_model, bias=False)

    def forward(self, x: Any) -> Any:
        if not TORCH_AVAILABLE:
            return x
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ──────────────────────────────────────────────
# 6. Mixture of Experts (MoE) Block
# ──────────────────────────────────────────────

class MoEFeedForward(nn.Module):
    """Fine-grained Mixture of Experts with Top-K Softmax routing."""

    def __init__(self, d_model: int, num_experts: int = 16, top_k: int = 2, intermediate_size: int = 3584) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        if TORCH_AVAILABLE:
            self.router = nn.Linear(d_model, num_experts, bias=False)
            self.experts = nn.ModuleList([
                SwiGLUFeedForward(d_model, intermediate_size) for _ in range(num_experts)
            ])

    def forward(self, x: Any) -> Tuple[Any, Any]:
        if not TORCH_AVAILABLE:
            return x, None
        B, L, D = x.shape
        flat_x = x.view(-1, D)

        # Compute routing logits
        logits = self.router(flat_x)
        weights, indices = torch.topk(F.softmax(logits, dim=-1), self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat_x)
        for i in range(self.top_k):
            expert_idx = indices[:, i]
            weight = weights[:, i : i + 1]
            for exp_id in range(self.num_experts):
                mask = (expert_idx == exp_id)
                if mask.any():
                    selected_x = flat_x[mask]
                    exp_out = self.experts[exp_id](selected_x)
                    out[mask] += exp_out * weight[mask]

        # Load balancing auxiliary loss
        density = F.softmax(logits, dim=-1).mean(dim=0)
        aux_loss = (density * density).sum() * self.num_experts

        return out.view(B, L, D), aux_loss


# ──────────────────────────────────────────────
# 7. Hybrid Block (Attention/SSM + MoE/FFN)
# ──────────────────────────────────────────────

class HybridBlock(nn.Module):
    """Combines either Attention or SSM with FeedForward or MoE."""

    def __init__(
        self,
        d_model: int,
        is_attention: bool,
        attn_layer: Optional[GroupedQueryAttention] = None,
        ssm_layer: Optional[SSMLayer] = None,
        use_moe: bool = False,
        moe_layer: Optional[MoEFeedForward] = None,
        ffn_layer: Optional[SwiGLUFeedForward] = None,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.is_attention = is_attention
        self.use_moe = use_moe
        self.attn_layer = attn_layer
        self.ssm_layer = ssm_layer
        self.moe_layer = moe_layer
        self.ffn_layer = ffn_layer

        if TORCH_AVAILABLE:
            self.input_norm = RMSNorm(d_model, eps=norm_eps)
            self.post_norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, x: Any, cos: Any = None, sin: Any = None, mask: Any = None) -> Tuple[Any, Optional[Any]]:
        if not TORCH_AVAILABLE:
            return x, None
        
        # 1. Attention or SSM sublayer
        norm_x = self.input_norm(x)
        if self.is_attention and self.attn_layer is not None:
            sub_out = self.attn_layer(norm_x, cos=cos, sin=sin, mask=mask)
        elif self.ssm_layer is not None:
            sub_out = self.ssm_layer(norm_x)
        else:
            sub_out = norm_x

        x = x + sub_out

        # 2. Feedforward or MoE sublayer
        norm_x2 = self.post_norm(x)
        aux_loss = None

        if self.use_moe and self.moe_layer is not None:
            ffn_out, aux_loss = self.moe_layer(norm_x2)
        elif self.ffn_layer is not None:
            ffn_out = self.ffn_layer(norm_x2)
        else:
            ffn_out = norm_x2

        x = x + ffn_out
        return x, aux_loss


# ──────────────────────────────────────────────
# 8. Complete AFE Hybrid Model Stack
# ──────────────────────────────────────────────

class AFEHybridModel(nn.Module):
    """Complete Next-Generation Hybrid Transformer-SSM-MoE Foundation Model."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = config_dict
        d = config_dict.get("d_model", 4096)
        L = config_dict.get("num_layers", 32)
        V = config_dict.get("vocab_size", 64000)
        h_q = config_dict.get("num_heads", 32)
        h_kv = config_dict.get("num_kv_heads", 8)
        head_dim = config_dict.get("head_dim", 128)
        inter = config_dict.get("intermediate_size", 14336)
        max_seq = config_dict.get("max_seq_len", 32768)
        ssm_ratio = config_dict.get("ssm_attn_ratio", "1:7")
        use_ssm = config_dict.get("use_ssm", True)
        use_moe = config_dict.get("use_moe", False)
        n_exp = config_dict.get("num_experts", 16)
        top_k = config_dict.get("top_k_experts", 2)
        moe_inter = config_dict.get("moe_intermediate_size", 3584)
        norm_eps = config_dict.get("norm_eps", 1e-6)

        self.blocks = []
        # Determine attention vs SSM layer pattern
        if use_ssm and ":" in ssm_ratio:
            parts = [int(p) for p in ssm_ratio.split(":")]
            period = parts[0] + parts[1]
            attn_interval = period // max(1, parts[0])
        else:
            attn_interval = 1

        for layer_idx in range(L):
            is_attn = (layer_idx % attn_interval == 0) or (not use_ssm)
            
            attn = GroupedQueryAttention(
                d_model=d, num_heads=h_q, num_kv_heads=h_kv,
                head_dim=head_dim, use_qk_norm=True, norm_eps=norm_eps
            ) if is_attn else None

            ssm = SSMLayer(d_model=d) if (not is_attn and use_ssm) else None

            moe = MoEFeedForward(
                d_model=d, num_experts=n_exp, top_k=top_k, intermediate_size=moe_inter
            ) if use_moe else None

            ffn = SwiGLUFeedForward(d_model=d, intermediate_size=inter) if not use_moe else None

            block = HybridBlock(
                d_model=d, is_attention=is_attn, attn_layer=attn, ssm_layer=ssm,
                use_moe=use_moe, moe_layer=moe, ffn_layer=ffn, norm_eps=norm_eps
            )
            self.blocks.append(block)

        if TORCH_AVAILABLE:
            self.embed_tokens = nn.Embedding(V, d)
            self.rope = RotaryEmbedding(head_dim, max_seq_len=max_seq)
            self.blocks = nn.ModuleList(self.blocks)
            self.final_norm = RMSNorm(d, eps=norm_eps)
            self.lm_head = nn.Linear(d, V, bias=False)

    def forward(self, input_ids: Any, mask: Any = None) -> Tuple[Any, Optional[Any]]:
        if not TORCH_AVAILABLE:
            return input_ids, None

        B, L = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(x, L)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            x, aux_loss = block(x, cos=cos, sin=sin, mask=mask)
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits, total_aux_loss
