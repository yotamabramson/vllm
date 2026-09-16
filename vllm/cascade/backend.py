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

import math
import time

import torch

import vllm.cascade as cascade
from vllm.cascade import SEL_BLOCK
from vllm.cascade import THIN_WIDTH as THIN
from vllm.cascade.ops.score import thin_score_decode
from vllm.cascade.ops.select import select_blocks
from vllm.cascade.ops.working_attn import working_attn_decode, working_attn_decode_reference
from vllm.logger import init_logger
from vllm.cascade.runtime import CascadeRuntime, StepPlan, get_runtime
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)
DEBUG_ATTN_CHECKS = 12


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
        # Padding slots are -1 (vLLM's cache-write kernel skips them); send them to the
        # null block's first slot, which nothing reads.
        thin_kv.view(-1, thin_kv.shape[-1]).index_copy_(
            0, thin_md.slot_mapping[:T].clamp(min=0), k_thin[:T].reshape(T, -1).to(thin_kv.dtype))
        acc_kv.view(-1, acc_kv.shape[-1]).index_fill_(0, acc_md.slot_mapping[:T].clamp(min=0), 0.0)

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
            self._refresh(rt, plan, l, r, kv_cache, bt[r], acc_kv, acc_md.block_table[r], b_end,
                          q_thin[r], thin_kv, thin_md.block_table[r])

        working_attn_decode(query, kv_cache, bt, plan.a_end, rt.S, b_end, output, self.scale,
                            max_slots=plan.max_slots[l])
        if cascade.debug() and l == 0 and rt.debug_attn_checks < DEBUG_ATTN_CHECKS:
            rt.debug_attn_checks += 1
            self._debug_check(rt, plan, query, key, value, kv_cache, bt, b_end, output)

    def _debug_check(self, rt: CascadeRuntime, plan: StepPlan, query, key, value, kv_cache, bt, b_end, output):
        """Layer 0, decode row 0: kernel vs plain-PyTorch attention over the same working slots
        (real vLLM cache layout and strides), floor slots vs the CPU store, new token's slot vs its K/V."""
        dev = query.device
        ref = torch.empty_like(output[:1])
        working_attn_decode_reference(query[:1], kv_cache, bt[:1], plan.a_end[:1], rt.S, b_end[:1], ref, self.scale)
        attn_diff = (ref.float() - output[:1].float()).abs().max().item()
        state = plan.states[0]
        lf = state.lf
        slots = torch.arange(lf, device=dev)
        floor_gpu = kv_cache[bt[0][slots // rt.BS].long(), :, slots % rt.BS].float()
        floor_cpu = rt.cpu_layer(0, state)[:, state.refresh_n - lf:state.refresh_n].to(dev).transpose(0, 1).float()
        floor_diff = (floor_gpu - floor_cpu).abs().max().item()
        new_diff = None
        if 0 not in plan.refresh_rows:
            s = int(plan.a_end[0]) - 1
            cur = kv_cache[bt[0][s // rt.BS].long(), :, s % rt.BS].float()
            new_diff = (cur - torch.cat([key[0], value[0]], dim=-1).float()).abs().max().item()
        logger.info("cascade debug L0 row0 n=%d refresh=%s a_end=%d lf=%d refresh_n=%d b_end=%s attn_vs_ref=%.3e "
                    "floor_vs_cpu=%.3e new_token_slot_vs_kv=%s kv_strides=%s", plan.seq_lens[0],
                    0 in plan.refresh_rows, int(plan.a_end[0]), lf, state.refresh_n, b_end[0].tolist(),
                    attn_diff, floor_diff, new_diff, kv_cache.stride())

    def _refresh(self, rt: CascadeRuntime, plan: StepPlan, l, r, kv_cache, bt_row, acc_kv, acc_bt_row, b_end,
                 q_thin_row, thin_kv, thin_bt_row):
        n = plan.seq_lens[r]
        state = plan.states[r]
        dev = kv_cache.device
        acc_pages = acc_bt_row[: (n + acc_kv.shape[1] - 1) // acc_kv.shape[1]].long()
        acc = acc_kv[acc_pages].reshape(-1, rt.Hkv)[:n]
        ids, counts, fs = select_blocks(acc, n, rt.caps[l])
        if cascade.debug() and r == 0 and l in (0, 15, 31) and state.prompt_len == n - 1:
            self._debug_select(rt, l, n, acc, ids, counts, q_thin_row, thin_kv, thin_bt_row)
        acc_kv[acc_pages] = 0

        t_start = time.perf_counter()
        pool = rt.cpu_layer(l, state)                                                  # [Hkv, capacity, 2D]
        lf = n - fs
        if state.prev_refresh_n == 0:
            # First refresh: the working pages still hold the prompt positionally, so the
            # floor has to come from the CPU store.
            floor = pool[:, fs:n].to(dev).transpose(0, 1).contiguous()                 # [lf, Hkv, 2D]
            slots = torch.arange(lf, device=dev)
            kv_cache[bt_row[slots // rt.BS].long(), :, slots % rt.BS] = floor
        else:
            # Floor and the tokens decoded since the last refresh are already on the GPU,
            # contiguous from slot (token - previous floor start): shift instead of re-fetching.
            shift = fs - (state.prev_refresh_n - state.prev_lf)
            if shift > 0:
                src = torch.arange(shift, shift + lf, device=dev)
                moved = kv_cache[bt_row[src // rt.BS].long(), :, src % rt.BS]
                dst = torch.arange(lf, device=dev)
                kv_cache[bt_row[dst // rt.BS].long(), :, dst % rt.BS] = moved
        t_floor = time.perf_counter()

        # Selected blocks: fetch only blocks that are not resident already, into the slots
        # whose blocks were dropped (measured overlap between refreshes: 85-99%).
        ids_cpu = ids.cpu()
        k_max = ids_cpu.shape[1]
        resident = state.resident.get(l)
        if resident is None or resident.shape[1] < k_max:
            grown = torch.full((rt.Hkv, k_max), -1, dtype=torch.int64)
            if resident is not None:
                grown[:, : resident.shape[1]] = resident
            resident = grown
        need_per_head, free_per_head = [], []
        for h in range(rt.Hkv):
            cnt = int(counts[h])
            new_h = ids_cpu[h, :cnt]
            old_h = resident[h]
            need = new_h[~torch.isin(new_h, old_h)] if cnt else new_h[:0]
            free = (~torch.isin(old_h, new_h)).nonzero(as_tuple=True)[0]
            free = free[free < cnt][: need.numel()]
            need = need[: free.numel()]
            old_h[free] = need
            need_per_head.append(need)
            free_per_head.append(free)
        state.resident[l] = resident
        fetched = int(sum(x.numel() for x in need_per_head))

        row_elems = SEL_BLOCK * 2 * rt.D
        t_gather = t_h2d = t_floor
        if fetched:
            blocks = pool.view(rt.Hkv, -1, row_elems)                                   # [Hkv, blocks, 8KB row]
            staging = rt.staging(fetched, row_elems)
            at = 0
            for h, need in enumerate(need_per_head):
                if need.numel():
                    torch.index_select(blocks[h], 0, need, out=staging[at:at + need.numel()])
                    at += need.numel()
            t_gather = time.perf_counter()
            data = staging.to(dev, non_blocking=False).view(-1, SEL_BLOCK, 2 * rt.D)
            t_h2d = time.perf_counter()
            slot0 = rt.S + torch.cat(free_per_head).to(dev) * SEL_BLOCK
            slots = (slot0[:, None] + torch.arange(SEL_BLOCK, device=dev)[None, :]).reshape(-1)
            head_idx = torch.cat([torch.full((f.numel(),), h, dtype=torch.int64)
                                  for h, f in enumerate(free_per_head)]).to(dev).repeat_interleave(SEL_BLOCK)
            kv_cache[bt_row[slots // rt.BS].long(), head_idx, slots % rt.BS] = data.reshape(-1, 2 * rt.D)

        ends = (rt.S + counts * SEL_BLOCK).to(torch.int32)
        state.b_end[l] = ends
        b_end[r] = ends.to(dev)
        if int(counts.max()) > 0:
            plan.max_slots[l] = max(plan.max_slots[l], int(ends.max()))
        if cascade.debug() and r == 0 and l in (0, 15, 31):
            self._debug_refresh(rt, state, l, n, ids, counts, fetched, time.perf_counter() - t_start,
                                t_floor - t_start, t_gather - t_floor, t_h2d - t_gather)

    def _debug_refresh(self, rt: CascadeRuntime, state, l, n, ids, counts, fetched, total_s, floor_s, gather_s,
                       h2d_s):
        """Per refresh: how much of the new selection was already selected last time (delta size),
        and where the refresh time goes (floor, CPU gather, host->GPU transfer)."""
        ids_cpu = ids.cpu()
        prev = state.prev_ids.get(l)
        overlap = None
        if prev is not None:
            kept = total = 0
            for h in range(rt.Hkv):
                new_h = set(ids_cpu[h][ids_cpu[h] >= 0].tolist())
                old_h = set(prev[h][prev[h] >= 0].tolist())
                kept += len(new_h & old_h)
                total += len(new_h)
            overlap = round(kept / max(1, total), 4)
        state.prev_ids[l] = ids_cpu
        selected = int(counts.sum())
        mb = fetched * SEL_BLOCK * rt.D * 2 * 2 / 2**20
        logger.info("cascade refresh L%d n=%d selected_blocks=%d fetched_blocks=%d (%.1fMB) total=%.0fms "
                    "floor=%.0fms gather=%.0fms h2d=%.0fms (%.2f GB/s) overlap_with_previous=%s",
                    l, n, selected, fetched, mb, total_s * 1e3, floor_s * 1e3, gather_s * 1e3, h2d_s * 1e3,
                    mb / 2**10 / max(h2d_s, 1e-9), overlap)


    def _debug_select(self, rt: CascadeRuntime, l, n, acc, ids, counts, q_thin_row, thin_kv, thin_bt_row):
        """First refresh after prefill (window = this step only): the accumulated score must equal the
        harness formula recomputed from the thin-K cache, and the selected blocks must match."""
        bs_t = thin_kv.shape[1]
        pages = thin_bt_row[: (n + bs_t - 1) // bs_t].long()
        k = thin_kv[pages].reshape(-1, rt.Hkv, THIN)[:n].float()                          # [n, Hkv, 32]
        q = q_thin_row.float().view(rt.Hkv, -1, THIN)
        w = torch.softmax(torch.einsum("hrd,nhd->hrn", q, k) / math.sqrt(THIN), dim=-1).max(dim=1).values.T
        rel = ((acc - w).abs() / w.abs().clamp_min(1e-12)).max().item()
        ref_ids, ref_counts, _ = select_blocks(w, n, rt.caps[l])
        overlap = []
        for h in range(rt.Hkv):
            got = set(ids[h][ids[h] >= 0].tolist())
            ref = set(ref_ids[h][ref_ids[h] >= 0].tolist())
            overlap.append(round(len(got & ref) / max(1, len(ref)), 3))
        logger.info("cascade debug select L%d n=%d acc_vs_formula_max_rel=%.3e counts=%s ref_counts=%s "
                    "overlap_per_head=%s acc_sum_per_head=%s", l, n, rel, counts.tolist(), ref_counts.tolist(),
                    overlap, [round(v, 3) for v in acc.sum(0).tolist()])

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
