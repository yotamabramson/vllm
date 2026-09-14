# SPDX-License-Identifier: Apache-2.0
"""CascadeAttention: the Llama attention layer for cascaded attention.

Build step 3 (no-op): computes the thin (width-32) screening Q/K from the
pre-RoPE q/k and carries them into its own attention op, but the attention
itself is still stock FlashAttention over the full KV cache. Must match stock
vLLM token-for-token before any selection logic goes in.

Why a separate op instead of vllm::unified_attention_with_output: that op's
signature has no room for the thin Q/K, and attention must stay an opaque
custom op so torch.compile splits the graph around it (registered in
CompilationConfig._attention_ops).
"""

import functools

import torch

from vllm.cascade import THIN_WIDTH, projections_path
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

logger = init_logger(__name__)


@functools.lru_cache(maxsize=1)
def _load_projections(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    loaded = torch.load(path, map_location="cpu")
    return loaded[THIN_WIDTH]["w_q"], loaded[THIN_WIDTH]["w_k"]


class CascadeAttention(Attention):
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, **kwargs) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads=num_kv_heads, **kwargs)
        assert self.query_quant is None, "cascade: query quantization not supported"
        self.layer_idx = extract_layer_index(self.layer_name)
        # Filled from the projections file in process_weights_after_loading,
        # once the model is on its real device (not part of the checkpoint).
        self.register_buffer("w_q_thin", torch.empty(num_heads, head_size, THIN_WIDTH), persistent=False)
        self.register_buffer("w_k_thin", torch.empty(num_kv_heads, head_size, THIN_WIDTH), persistent=False)

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        super().process_weights_after_loading(act_dtype)
        w_q, w_k = _load_projections(projections_path())
        w_q, w_k = w_q[self.layer_idx], w_k[self.layer_idx]
        assert w_q.shape == self.w_q_thin.shape and w_k.shape == self.w_k_thin.shape, (
            f"cascade: projections {tuple(w_q.shape)}/{tuple(w_k.shape)} do not match layer "
            f"{self.layer_idx} shapes {tuple(self.w_q_thin.shape)}/{tuple(self.w_k_thin.shape)} "
            "(tensor parallelism is not supported yet)"
        )
        self.w_q_thin = w_q.to(self.w_q_thin.device, act_dtype)
        self.w_k_thin = w_k.to(self.w_k_thin.device, act_dtype)
        if self.layer_idx == 0:
            # fork_tests/step3_noop.py looks for this line to prove the layer is active
            logger.info("cascade: projections loaded (%s)", projections_path())

    def project_thin(self, q_pre: torch.Tensor, k_pre: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-RoPE q [T, Hq*D], k [T, Hkv*D] -> thin q [T, Hq, 32], thin k [T, Hkv, 32]
        (same per-head projection as the harness's einsum over q_pre/k_pre)."""
        q = q_pre.view(-1, self.num_heads, self.head_size)
        k = k_pre.view(-1, self.num_kv_heads, self.head_size)
        return (torch.einsum("thd,hde->the", q, self.w_q_thin),
                torch.einsum("thd,hde->the", k, self.w_k_thin))

    def forward(  # type: ignore[override]
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_thin: torch.Tensor,
        k_thin: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        hidden_size = self.num_heads * self.head_size_v
        output = torch.empty((num_tokens, hidden_size), dtype=query.dtype, device=query.device)
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size_v)
        if self.use_direct_call:
            dep = None
            if not self.attn_backend.forward_includes_kv_cache_update:
                dep = unified_kv_cache_update(key, value, self.layer_name)
            cascade_attention_with_output(query, key, value, q_thin, k_thin, output, self.layer_name,
                                          kv_cache_dummy_dep=dep)
        else:
            encoded = _encode_layer_name(self.layer_name)
            dep = None
            if not self.attn_backend.forward_includes_kv_cache_update:
                dep = torch.ops.vllm.unified_kv_cache_update(key, value, encoded)
            torch.ops.vllm.cascade_attention_with_output(query, key, value, q_thin, k_thin, output, encoded,
                                                         kv_cache_dummy_dep=dep)
        return output.view(-1, hidden_size)


@eager_break_during_capture
@maybe_transfer_kv_layer
def cascade_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_thin: torch.Tensor,
    k_thin: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    del kv_cache_dummy_dep, q_thin, k_thin  # step 3: thin Q/K carried but not used yet
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)
    self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=output)


def cascade_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_thin: torch.Tensor,
    k_thin: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="cascade_attention_with_output",
    op_func=cascade_attention_with_output,
    mutates_args=["output"],
    fake_impl=cascade_attention_with_output_fake,
)
