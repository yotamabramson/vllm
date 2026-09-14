# SPDX-License-Identifier: Apache-2.0
"""Per-layer side caches for full mode, allocated by vLLM's KV cache manager like
MiniMax-M3's indexer cache (vllm/models/minimax_m3/common/indexer.py):

  thin-K   one width-32 key per KV head per token  -> [NB, BS, Hkv * 32], model dtype
  score    accumulated stage-1 score per KV head   -> [NB, BS, Hkv], float32

Both are positional over the whole sequence (never freed while the request
lives). MLAAttentionSpec with num_kv_heads=1 budgets one vector per token
(no separate V), the same trick the MiniMax and DeepSeek indexer caches use.
Their page sizes (256 fp16 and 8 float32 bytes/token x 2 or 4) divide the
full-KV page exactly, which vLLM's page-size unification requires for MLA specs.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec


@dataclass
class CascadeSideMetadata(AttentionMetadata):
    slot_mapping: torch.Tensor   # positional slots, this cache's block size
    block_table: torch.Tensor
    seq_lens: torch.Tensor


class CascadeSideMetadataBuilder(AttentionMetadataBuilder[CascadeSideMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> CascadeSideMetadata:
        m = common_attn_metadata
        return CascadeSideMetadata(slot_mapping=m.slot_mapping, block_table=m.block_table_tensor,
                                   seq_lens=m.seq_lens)


class CascadeSideImpl:
    """Placeholder: side caches are read and written by the main layer's impl."""


class CascadeSideBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16, torch.float32]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CASCADE_SIDE_CACHE"

    @staticmethod
    def get_impl_cls():
        return CascadeSideImpl

    @staticmethod
    def get_builder_cls() -> type[CascadeSideMetadataBuilder]:
        return CascadeSideMetadataBuilder


class CascadeSideCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, head_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.layer_name = prefix
        self.head_size = head_size
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(block_size=vllm_config.cache_config.block_size, num_kv_heads=1,
                                head_size=self.head_size, dtype=self.dtype)

    def forward(self) -> None: ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CascadeSideBackend
