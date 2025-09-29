import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from Model.config import Config

# cache definition
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]


class GPT(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.config = config

        # main transformer module which contains of embedding fucntion and multiple blocks
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, config.padded_vocab_size, bias=config.bias
        )
        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

    def _init_weights(
        self,
        module: nn.Module,
        n_layer: int,
    ):
        # smaller std will lead its weight closer to 0
        if isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * self.config.n_embd))
            )
        elif isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight, mean=0.0, std=math.sqrt(2.0 / (5.0 * self.config.n_embd))
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # GPT-Neox style
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (
                name == "proj.weight" and isinstance(module, CausalSelfAttention)
            ):
                nn.init.normal_(
                    p, mean=0.0, std=1.0 / math.sqrt(self.config.n_embd) / n_layer
                )

    # only for inference mode
    def reset_cache(
        self,
    ):
        self.kv_caches.clear()
        # TPU sepecific, XLA: Accelerated Linear Algebra
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            self.rope_cache = None
            self.mask_cache = None

    # rope cache and mask cache are all static
    def build_rope_cache(
        self,
        X: torch.Tensor,
    ):
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.float32,
            device=X.device,
            condense_ratio=self.config.condense_ratio,
        )

    def build_mask_cache(
        self,
        X: torch.Tensor,
    ):
        mat_ones = torch.ones(
            (self.config.block_size, self.config.block_size),
            device=X.device,
            dtype=torch.bool,
        )
        # for element-wise addition broadcasting batch dim and n_heads dim
        return torch.tril(mat_ones).unsqueeze(0).unsqueeze(0)

    def build_kv_caches(
        self,
        X: torch.Tensor,
        max_seq_len: int,
    ):
        B = X.shape[0]
        heads = self.config.n_query_groups  # the number of heads for k, v
        k_cache_shape = (B, max_seq_len, heads, self.config.head_size)
        v_cache_shape = (B, max_seq_len, heads, self.config.head_size)
        device = X.device
        return [
            (
                torch.zeros(k_cache_shape, device=device),
                torch.zeros(v_cache_shape, device=device),
            )
            for _ in range(self.config.n_layer)
        ]

    def forward(
        self,
        X: torch.Tensor,
        max_seq_len: Optional[int] = None,
        input_pos: Optional[torch.Tensor] = None,
    ):
        B, s = X.shape
        # only use kv_cache during inference
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_len is None:
            max_seq_len = block_size
        if use_kv_cache:
            assert (
                max_seq_len >= s
            ), f"Cannot forward sequence of length {s}, max seq length is only {max_seq_len}"
        assert (
            max_seq_len <= block_size
        ), f"Cannot attend to {max_seq_len}, block size is only {block_size}"
        assert (
            block_size >= s
        ), f"Cannot forward sequence of length {s}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(X)
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(X)

        cos, sin = self.rope_cache
        if use_kv_cache:
            cos = cos.index_select(dim=0, index=input_pos)
            sin = sin.index_select(dim=0, index=input_pos)
            mask = self.mask_cache.index_select(dim=2, index=input_pos)
            mask = mask[..., :max_seq_len]
        else:
            cos = cos[:s]
            sin = sin[:s]
            mask = None

        # main forward process
        # embedding
        X = self.transformer.wte(X)

        if not use_kv_cache:
            for block in self.transformer.h:
                X, *_ = block(X, (cos, sin), max_seq_len)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(X, max_seq_len)
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    X, (cos, sin), max_seq_len, mask, input_pos, self.kv_caches[i]
                )

        X = self.transformer.ln_f(X)
        return self.lm_head(X)


# Transformer block
class Block(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

    def forward(
        self,
        X: torch.Tensor,
        rope: RoPECache,
        max_seq_len: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        n_1 = self.norm_1(X)
        y, new_kv_cache = self.attn(n_1, rope, max_seq_len, mask, input_pos, kv_cache)

        # GPT-J or GPT-Neox style, attention module and FFN are parallelism
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(X)
            X = X + y + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            X = X + y
            X = X + self.mlp(self.norm_2(X))
        return y, new_kv_cache


# mask attention
class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        # as we are using GQA here for better GPU memory allocation
        # we inforce each query group to share one single key, value head from KV cache
        shape = (config.n_heads + 2 * config.n_query_groups) * config.head_size
        # qkv linear mapping
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    # Inference mode:
    # Pre-fill process will take prompt as input and generate first token, kv cache will be updated
    # Decoding process will take the last generated tokens as input to generate qkv, kv will then be updated with previous information
    def forward(
        self,
        X: torch.Tensor,
        rope: RoPECache,
        max_seq_len: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        B, s, d = X.shape
        qkv = self.attn(X)

        q_per_kv = self.config.n_heads // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(
            B, s, self.config.n_query_groups, total_qkv, self.config.head_size
        )

        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, s, -1, self.config.head_size)
        k = k.reshape(B, s, -1, self.config.head_size)
        v = v.reshape(B, s, -1, self.config.head_size)

        # apply rope embedding before calculating the attn scores, v will not apply with rope
        cos, sin = rope
        # partially rotate the tensor
        n_elem = int(self.config.rotary_percentage * self.config.head_size)
        cos, sin = cos.repeat(1, 2).unsqueeze(-2), sin.repeat(1, 2).unsqueeze(-2)
        q_roped = apply_rope(q[..., :n_elem], cos, sin, self.config.half)
        k_roped = apply_rope(k[..., :n_elem], cos, sin, self.config.half)
        q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        # inference mode
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # input_pos[-1] >= max_seq_len, we dump the frontest tokens
            if input_pos[-1] >= max_seq_len:
                input_pos = torch.tensor(max_seq_len - 1, device=input_pos.device)
                # shift 1 position to the left, sliding window
                # [T0, T1, T2, None, None, ...] -> [T1, T2, None, None, ..., T0]
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            # [T1, T2, None, None, ..., T0] -> [T1, T2, T3, None, ..., T0]
            # index_copy aims to operate on a specific position, replace empty or cache value with the newset k,v value
            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v  # update the k, v cache

        y = self.scale_dot_product_attention(q, k, v, mask)
        y = y.reshape(B, s, d)
        y = self.proj(y)
        return y, kv_cache

    def scale_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)

        # traspose seq_len and attention heads
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # in ususal settings, the n_heads for k and v is smaller than q
        if q.shape != k.shape:
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        # causal mask is essential, as it determines how the parallelism in next token prediction
        y = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            scale=scale,
            is_causal=mask is None,
        )
        return y.transpose(1, 2)  # n_heads is still maintained


class GPTNeoxMLP(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(
        self,
        X: torch.Tensor,
    ):
        X = self.fc(X)
        X = nn.functional.gelu(X)
        return self.proj(X)


class LLaMAMLP(nn.Module):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(
        self,
        X: torch.Tensor,
    ):
        X_fc_1 = self.fc_1(X)
        X_fc_2 = self.fc_2(X)
        # apply sigmoid to obtain its probabilistics
        X = nn.functional.silu(X_fc_1) * X_fc_2
        return self.proj(X)


# define rope cache
def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: float = 1.0,  # aims to build a mapping from longer context to its original length, 1.0 means no compression.
):
    # rotation angle: $theta_i = 10000^{\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** torch.arange(0, n_elem, 2, device=device) / n_elem)

    # create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device)

    # calculate the outer porduct of idx with theta, m * theat_i
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # bfloat16: 1-bit sign + 8-bits exponent + 7-bits mantissa, better for shifting
    # float16: 1-bit sign + 5-bits exponent + 10-bits mantissa
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(
    X: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half: bool = True,
):
    head_size = X.shape[-1]
    if half:
        # for efficiency, we simply reverse half of the input
        half_size = head_size // 2
        X1 = X[..., :half_size]
        X2 = X[..., half_size:]
        # rotated: [-X_half, ..., X1, ...]
        rotated = torch.cat((-X2, X1), dim=-1)
    else:
        # for rigorous definition implementation
        X_reshaped = X.view(*X.shape[:-1], -1, 2)  # B, s, n_h, head_size // 2, 2
        X1, X2 = X_reshaped.unbind(-1)
        # rotated: [-X2, X1, -X3, X4, ...]
        rotated = torch.stack((-X2, X1), dim=-1).view(*X.shape[:-1], -1)
    rope = X * cos + rotated * sin
    return rope.type_as(X)
