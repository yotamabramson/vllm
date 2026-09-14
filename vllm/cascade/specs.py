# SPDX-License-Identifier: Apache-2.0
"""KV cache spec + manager for the per-request working set (full mode).

The scheduler owns HOW MANY blocks a request holds; the worker owns WHAT is in
them (towards_70B_vllm/README.md, "Stage 1 design"):
  - prefill: blocks allocated positionally, exactly as full attention (dense
    prefill needs every prompt token's K/V on the GPU)
  - once the prompt is processed: every block past the working budget is freed
    (replaced by the null block); the worker has already copied the prompt's
    K/V to the CPU store chunk by chunk
  - decode: never allocates past the budget, so decode growth never preempts

The spec is its own uniform-type base on purpose: if it shared FullAttentionSpec's
base, vLLM could pack it into one block table with the thin-K side cache, and
freeing working blocks would free thin-K pages too.
"""

from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True, kw_only=True)
class CascadeWorkingSpec(FullAttentionSpec):
    working_slots: int

    @classmethod
    def merge(cls, specs: list["CascadeWorkingSpec"]) -> "CascadeWorkingSpec":
        assert all(isinstance(spec, CascadeWorkingSpec) for spec in specs)
        slots = {spec.working_slots for spec in specs}
        assert len(slots) == 1, f"cascade: layers disagree on working_slots {slots}"
        base = FullAttentionSpec.merge(specs)  # type: ignore[arg-type]
        return cls(
            block_size=base.block_size,
            num_kv_heads=base.num_kv_heads,
            head_size=base.head_size,
            head_size_v=base.head_size_v,
            dtype=base.dtype,
            kv_quant_mode=base.kv_quant_mode,
            page_size_padded=base.page_size_padded,
            num_head_slots=base.num_head_slots,
            state_content_bytes=base.state_content_bytes,
            tokens_per_state=base.tokens_per_state,
            sliding_window=base.sliding_window,
            attention_chunk_size=base.attention_chunk_size,
            non_causal=base.non_causal,
            working_slots=slots.pop(),
        )


class CascadeWorkingManager(FullAttentionManager):
    def __init__(self, kv_cache_spec: CascadeWorkingSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        assert not self.enable_caching, "cascade: prefix caching must be disabled (enable_prefix_caching=False)"
        self.keep_blocks = cdiv(kv_cache_spec.working_slots, self.block_size)
        self._decoding: set[str] = set()

    def _cap(self, request_id: str, num_tokens: int) -> int:
        if request_id in self._decoding:
            return min(num_tokens, self.keep_blocks * self.block_size)
        return num_tokens

    def remove_skipped_blocks(self, request_id: str, processed_computed_tokens: int,
                              num_prompt_tokens: int | None = None) -> None:
        if num_prompt_tokens is None or processed_computed_tokens < num_prompt_tokens:
            return
        self._decoding.add(request_id)
        num_blocks = len(self.req_to_blocks.get(request_id, ()))
        self._remove_blocks_in_range(request_id, self.keep_blocks, num_blocks)

    def get_num_blocks_to_allocate(self, request_id: str, num_tokens: int, *args, **kwargs) -> int:
        return super().get_num_blocks_to_allocate(request_id, self._cap(request_id, num_tokens), *args, **kwargs)

    def allocate_new_blocks(self, request_id: str, num_tokens: int, num_tokens_main_model: int):
        return super().allocate_new_blocks(request_id, self._cap(request_id, num_tokens), num_tokens_main_model)

    def free(self, request_id: str) -> None:
        self._decoding.discard(request_id)
        super().free(request_id)
