# SPDX-License-Identifier: Apache-2.0
"""Full-mode attention backend: FlashAttention for prefill, the working set for decode.

Per layer, per step (towards_70B_vllm/README.md, "Stage 1 design"; slot layout in
runtime.py):
  every token     thin-K written to its side cache (positional)
  prefill rows    K/V written to positional pages; copied to the CPU store; stock
                  FlashAttention over the positional pages
  decode rows     refresh rows: tokens since the last refresh flushed GPU->CPU, the
                  current token written to CPU
                  all rows: stage-1 score added into the accumulator (ops/score.py)
                  non-refresh rows: current K/V written to working slot a_end - 1
                  refresh rows: block-16 top-K per head (ops/select.py); floor and
                  selected blocks copied CPU->GPU into the working pages;
                  accumulator reset
                  all rows: attention over the working set (ops/working_attn.py)

Warm-up/dummy batches (no known request state) run as stock FlashAttention.
"""

import torch

from vllm.cascade import SEL_BLOCK
from vllm.cascade.ops.score import thin_score_decode
from vllm.cascade.ops.select import floor_start, select_blocks
from vllm.cascade.ops.working_attn import working_attn_decode
from vllm.cascade.runtime import CascadeRuntime, StepPlan, get_runtime
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)


class CascadeMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        self.runtime = get_runtime(num_layers=len(layer_names), num_kv_heads=kv_cache_spec.num_kv_heads,
                                   head_size=kv_cache_spec.head_size, block_size=kv_cache_spec.block_size,
                                   dtype=kv_cache_spec.dtype, max_len=vllm_config.model_config.max_model_len)

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec) -> AttentionCGSupport:
        return cls._cudagraph_support

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        md.cascade = self.runtime.plan_step(common_attn_metadata)
        return md


class CascadeImpl(FlashAttentionImpl):
    def cascade_forward(self, layer, query, key, value, q_thin, k_thin, kv_cache, attn_metadata, output):
        if attn_metadata is None:                      # profiling run
            return output.fill_(0)
        plan: StepPlan = attn_metadata.cascade
        T = attn_metadata.num_actual_tokens
        if plan.fallback:
            self.do_kv_cache_update(layer, key[:T], value[:T], kv_cache, attn_metadata.slot_mapping[:T])
            return self.forward(layer, query, key, value, kv_cache, attn_metadata, output=output)

        rt = plan.runtime
        l = layer.layer_idx
        assert kv_cache.shape[1:] == (rt.Hkv, rt.BS, 2 * rt.D), f"cascade: unexpected KV layout {tuple(kv_cache.shape)}"
        side = get_forward_context().attn_metadata
        thin_md = side[layer.thin_cache.layer_name]
        acc_md = side[layer.acc_cache.layer_name]
        thin_kv = layer.thin_cache.kv_cache
        acc_kv = layer.acc_cache.kv_cache

        # Thin-K for every scheduled token, positional. Its accumulator entry starts at
        # zero: side-cache blocks are recycled from finished requests.
        thin_kv.view(-1, thin_kv.shape[-1]).index_copy_(
            0, thin_md.slot_mapping[:T], k_thin[:T].reshape(T, -1).to(thin_kv.dtype))
        acc_kv.view(-1, acc_kv.shape[-1]).index_fill_(0, acc_md.slot_mapping[:T], 0.0)

        nd = plan.num_decodes
        if T > nd:
            self._prefill(layer, rt, plan, query, key, value, kv_cache, attn_metadata, output, T)
        if nd:
            self._decode(layer, rt, plan, l, query[:nd], key[:nd], value[:nd], q_thin[:nd], kv_cache,
                         attn_metadata, thin_md, acc_md, thin_kv, acc_kv, output[:nd])
        return output

    def _prefill(self, layer, rt: CascadeRuntime, plan: StepPlan, query, key, value, kv_cache, md, output, T):
        nd = plan.num_decodes
        self.do_kv_cache_update(layer, key[nd:T], value[nd:T], kv_cache, md.slot_mapping[nd:T])
        for r in plan.prefill_rows:
            q0, q1 = plan.query_start[r], plan.query_start[r + 1]
            start = plan.seq_lens[r] - (q1 - q0)
            rt.write_cpu(layer.layer_idx, plan.states[r], start, torch.cat([key[q0:q1], value[q0:q1]], dim=-1),
                         blocking=False)
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        flash_attn_varlen_func(
            q=query[nd:T], k=key_cache, v=value_cache, out=output[nd:T],
            cu_seqlens_q=md.query_start_loc[nd:] - nd, max_seqlen_q=md.max_query_len,
            seqused_k=md.seq_lens[nd:], max_seqlen_k=md.max_seq_len,
            softmax_scale=self.scale, causal=True, block_table=md.block_table[nd:],
            fa_version=self.vllm_flash_attn_version,
        )

    def _decode(self, layer, rt: CascadeRuntime, plan: StepPlan, l, query, key, value, q_thin, kv_cache, md,
                thin_md, acc_md, thin_kv, acc_kv, output):
        nd = plan.num_decodes
        bt = md.block_table[:nd]
        dev = query.device

        # Refresh rows, CPU side first: tokens since the last refresh, then the current token.
        for f in plan.flushes:
            slots = torch.arange(f.slot_start, f.slot_start + f.count, device=dev)
            kv = kv_cache[bt[f.row][slots // rt.BS].long(), :, slots % rt.BS]            # [count, Hkv, 2D]
            rt.write_cpu(l, plan.states[f.row], f.cpu_start, kv, blocking=True)
        for r in plan.refresh_rows:
            rt.write_cpu(l, plan.states[r], plan.seq_lens[r] - 1,
                         torch.cat([key[r], value[r]], dim=-1)[None], blocking=True)

        # Stage-1 score, every decode row, every step.
        thin_score_decode(q_thin, thin_kv, thin_md.block_table[:nd], acc_kv, acc_md.block_table[:nd],
                          thin_md.seq_lens[:nd], max_seq_len=plan.max_decode_seq_len)

        # Non-refresh rows: current K/V into its working slot (refresh rows are -1: skipped).
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        reshape_and_cache_flash(key, value, key_cache, value_cache, plan.decode_slots, self.kv_cache_dtype,
                                layer._k_scale, layer._v_scale)

        b_end = plan.b_end[l]
        for r in plan.refresh_rows:
            self._refresh(rt, plan, l, r, kv_cache, bt[r], acc_kv, acc_md.block_table[r], b_end)

        working_attn_decode(query, kv_cache, bt, plan.a_end, rt.S, b_end, output, self.scale,
                            max_slots=rt.working_slots)

    def _refresh(self, rt: CascadeRuntime, plan: StepPlan, l, r, kv_cache, bt_row, acc_kv, acc_bt_row, b_end):
        n = plan.seq_lens[r]
        state = plan.states[r]
        dev = kv_cache.device
        acc_pages = acc_bt_row[: (n + acc_kv.shape[1] - 1) // acc_kv.shape[1]].long()
        acc = acc_kv[acc_pages].reshape(-1, rt.Hkv)[:n]
        ids, counts, fs = select_blocks(acc, n, rt.caps[l])
        acc_kv[acc_pages] = 0

        pool = rt.cpu_layer(l, state)                                                  # [capacity, Hkv, 2D]
        floor = pool[fs:n].to(dev)
        slots = torch.arange(n - fs, device=dev)
        kv_cache[bt_row[slots // rt.BS].long(), :, slots % rt.BS] = floor

        k_max = ids.shape[1]
        if k_max:
            tok = (ids.clamp(min=0)[:, :, None] * SEL_BLOCK
                   + torch.arange(SEL_BLOCK, device=dev)).reshape(rt.Hkv, -1).cpu()      # [Hkv, M]
            heads = torch.arange(rt.Hkv)[:, None].expand_as(tok)
            selected = pool[tok, heads].to(dev)                                         # [Hkv, M, 2D]
            j = torch.arange(tok.shape[1], device=dev)
            valid = j[None, :] < (counts.to(dev) * SEL_BLOCK)[:, None]
            h_idx, j_idx = valid.nonzero(as_tuple=True)
            dst = rt.S + j_idx
            kv_cache[bt_row[dst // rt.BS].long(), h_idx, dst % rt.BS] = selected[h_idx, j_idx]

        ends = (rt.S + counts * SEL_BLOCK).to(torch.int32)
        state.b_end[l] = ends
        b_end[r] = ends.to(dev)


class CascadeBackend(FlashAttentionBackend):
    # The layer's op writes K/V itself: decode tokens go to working slots, not the
    # scheduler's positional slots (which point past the working budget).
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_impl_cls():
        return CascadeImpl

    @staticmethod
    def get_builder_cls():
        return CascadeMetadataBuilder
