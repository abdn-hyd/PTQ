from dataclasses import dataclass
from typing import Any, Literal, Optional, Type

import torch
from typing_extensions import Self


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class Config:
    name: str = "lit-GPT"
    # block size defines how the text will be chunked
    block_size: int = 4096
    vocab_size: int = 50254
    padding_multiple: int = 512
    padded_vocab_size: Optional[int] = None
    n_layer: int = 16
    n_heads: int = 32
    n_embd: int = 4096
    rotary_percentage: float = 0.25
    parallel_residual: bool = True
    bias: bool = True
    n_query_groups: Optional[int] = None
    shared_attention_norm: bool = False
    _norm_class: Literal["LayerNorm"] = "LayerNorm"
    norm_eps: float = 1e-5
    _mlp_class: Literal["GPTNeoxMLP", "LLaMAMLP"] = "GPTNeoxMLP"
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    half: bool = False

    def __post_init__(self):
        # error checking
        assert self.n_embd % self.n_heads == 0
        # vocab size should be a power of 2 to be optimal on hardware. compute the closest value
        if self.padded_vocab_size is None:
            self.padded_vocab_size = find_multiple(
                self.vocab_size, self.padding_multiple
            )
        # compute the number of query groups
        if self.n_query_groups is not None:
            assert self.n_heads % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_heads
        # compute the intermediate size for MLP if not set
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("The config needs to set the `intermediate_size`")
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_heads

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        import Model.TinyLLaMA.GPT as GPT

        return getattr(GPT, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        return getattr(torch.nn, self._norm_class)


configs = []

tiny_LLaMA = [
    dict(
        name="tiny_LLaMA",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=1,
        n_heads=12,
        n_embd=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="LayerNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        n_query_groups=1,
    ),
]
configs.extend(tiny_LLaMA)


name_to_config = {config["name"]: config for config in configs}
