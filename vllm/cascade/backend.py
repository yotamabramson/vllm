# SPDX-License-Identifier: Apache-2.0
"""Full-mode attention backend.

Every (request, KV group) is its own FlashAttention row: its own KV length and
block table, so no masks and no custom attention kernel (towards_70B_vllm/README.md,
"Stage 1 design"). Pages hold one group and interleave by group, so a group's block
table is bt[:, g::Hkv] (specs.py); rows are group-major, row = g * rows + i.

Per layer, per step:
  every token     thin-K written to its side cache (positional)
  prefill rows    K/V written positionally; copied to the CPU store; FlashAttention
                  over the positional pages
  decode rows     this step's K/V written to the group's next working slot; stage-1
                  score accumulated (ops/score.py); refresh rows additionally flush to
                  CPU, re-select (ops/select.py), shift their floor and fetch the blocks
                  they are missing; then FlashAttention over [0, length) of each row
  warm-up/dummy   treated exactly like prefill (positional), so vLLM's fake batches
                  run through real code without touching cascade state
"""

import math
import time

import torch

import vllm.cascade as cascade
from vllm.cascade import SEL_BLOCK
from vllm.cascade import THIN_WIDTH as THIN
from vllm.cascade.ops.layout import floor_shift, group_block_table, row_of, slot_pages
from vllm.cascade.ops.score import thin_score_decode
from vllm.cascade.ops.select import select_blocks
from vllm.cascade.runtime import CascadeRuntime, StepPlan, get_runtime
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func, reshape_and_cache_flash
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)
DEBUG_CHECKS = 8


class CascadeMetadataBuilder(FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1)
        self.runtime = get_runtime(num_layers=len(layer_names), num_kv_heads=kv_cache_spec.num_groups,
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
    # ---- helpers -------------------------------------------------------------
    @staticmethod
    def _to_rows(x: torch.Tensor, groups: int) -> torch.Tensor:
        """[tokens, groups * per_group, D] -> [groups * tokens, per_group, D] (group-major rows)."""
        t, _, d = x.shape[0], x.shape[1], x.shape[2]
        return x.view(t, groups, -1, d).permute(1, 0, 2, 3).reshape(groups * t, -1, d).contiguous()

    @staticmethod
    def _from_rows(x: torch.Tensor, groups: int, tokens: int) -> torch.Tensor:
        """Inverse of _to_rows."""
        return x.view(groups, tokens, -1, x.shape[-1]).permute(1, 0, 2, 3).reshape(tokens, -1, x.shape[-1])

    def _caches(self, kv_cache: torch.Tensor):
        """vLLM page [blocks, 1, block, 2 * D] -> FlashAttention's (K, V) views."""
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    # ---- entry point ---------------------------------------------------------
    def cascade_forward(self, layer, query, key, value, q_thin, k_thin, kv_cache, attn_metadata, output):
        if attn_metadata is None:                      # profiling run
            return output.fill_(0)
        plan: StepPlan = attn_metadata.cascade
        rt = plan.runtime
        T = attn_metadata.num_actual_tokens
        l = layer.layer_idx
        nd = 0 if plan.fallback else plan.num_decodes

        side = get_forward_context().attn_metadata
        thin_md = side[layer.thin_cache.layer_name]
        acc_md = side[layer.acc_cache.layer_name]
        thin_kv = layer.thin_cache.kv_cache
        acc_kv = layer.acc_cache.kv_cache
        # Thin-K for every scheduled token, positional. Its accumulator entry starts at
        # zero: side-cache blocks are recycled from finished requests. Padding slots are
        # -1 (vLLM's own cache-write kernel skips them); send those to the null block.
        thin_kv.view(-1, thin_kv.shape[-1]).index_copy_(
            0, thin_md.slot_mapping[:T].clamp(min=0), k_thin[:T].reshape(T, -1).to(thin_kv.dtype))
        acc_kv.view(-1, acc_kv.shape[-1]).index_fill_(0, acc_md.slot_mapping[:T].clamp(min=0), 0.0)

        if T > nd:
            self._positional(layer, rt, plan, query, key, value, kv_cache, attn_metadata, output, nd, T)
        if nd:
            self._decode(layer, rt, plan, l, query, key, value, q_thin, kv_cache, thin_md, acc_md,
                         thin_kv, acc_kv, output)
        return output

    # ---- prefill and warm-up: positional pages, per-group rows ---------------
    def _positional(self, layer, rt: CascadeRuntime, plan: StepPlan, query, key, value, kv_cache, md, output,
                    t0: int, t1: int):
        dev = query.device
        g = rt.Hkv
        tokens = t1 - t0
        r0 = plan.num_decodes if not plan.fallback else 0
        num_rows = md.seq_lens.shape[0] - r0

        qlens = md.query_start_loc[r0:] - md.query_start_loc[r0]
        seq_lens = md.seq_lens[r0:]
        bt_rows = group_block_table(md.block_table[r0:], g)

        # absolute position of every token in this slice, and the row it belongs to
        starts = (seq_lens - (qlens[1:] - qlens[:-1])).to(torch.int64)
        counts = (qlens[1:] - qlens[:-1]).to(torch.int64)
        req_of_token = torch.repeat_interleave(torch.arange(num_rows, device=dev), counts)
        within = torch.arange(tokens, device=dev) - qlens[:-1].to(torch.int64)[req_of_token]
        pos = starts[req_of_token] + within                                              # [tokens]

        groups = torch.arange(g, device=dev)[:, None]
        rows = groups * num_rows + req_of_token[None, :]                                 # [g, tokens]
        pages = bt_rows[rows.reshape(-1), (pos // rt.BS).repeat(g)].to(torch.int64)
        slots = pages * rt.BS + (pos % rt.BS).repeat(g)                                  # [g * tokens]

        key_cache, value_cache = self._caches(kv_cache)
        reshape_and_cache_flash(self._to_rows(key[t0:t1], g), self._to_rows(value[t0:t1], g),
                                key_cache, value_cache, slots, self.kv_cache_dtype,
                                layer._k_scale, layer._v_scale)
        if not plan.fallback:
            for r in plan.prefill_rows:
                q0, q1 = plan.query_start[r], plan.query_start[r + 1]
                start = plan.seq_lens[r] - (q1 - q0)
                rt.write_cpu(layer.layer_idx, plan.states[r], start,
                             torch.cat([key[q0:q1], value[q0:q1]], dim=-1), blocking=False)

        q_rows = self._to_rows(query[t0:t1], g)
        out_rows = torch.empty_like(q_rows)
        cu_q = torch.cat([qlens[:-1] + i * tokens for i in range(g)] + [torch.tensor([g * tokens], device=dev,
                                                                                    dtype=qlens.dtype)])
        flash_attn_varlen_func(
            q=q_rows, k=key_cache, v=value_cache, out=out_rows,
            cu_seqlens_q=cu_q.to(torch.int32), max_seqlen_q=md.max_query_len,
            seqused_k=seq_lens.repeat(g), max_seqlen_k=md.max_seq_len,
            softmax_scale=self.scale, causal=True, block_table=bt_rows,
            fa_version=self.vllm_flash_attn_version,
        )
        output[t0:t1] = self._from_rows(out_rows, g, tokens)

    # ---- decode --------------------------------------------------------------
    def _decode(self, layer, rt: CascadeRuntime, plan: StepPlan, l, query, key, value, q_thin, kv_cache,
                thin_md, acc_md, thin_kv, acc_kv, output):
        nd = plan.num_decodes
        g = rt.Hkv
        dev = query.device

        # Refresh rows, CPU side first: tokens since the last refresh, then this token.
        for f in plan.flushes:
            state = plan.states[f.row]
            slots = state.sel_len[l].to(dev)[:, None] + torch.arange(f.slot_start, f.slot_start + f.count,
                                                                     device=dev)[None, :]
            kv = self._read_slots(rt, plan, f.row, slots, kv_cache)                      # [count, Hkv, 2D]
            rt.write_cpu(l, state, f.cpu_start, kv, blocking=True)
        for r in plan.refresh_rows:
            rt.write_cpu(l, plan.states[r], plan.seq_lens[r] - 1,
                         torch.cat([key[r], value[r]], dim=-1)[None], blocking=True)

        # This step's K/V into each group's next working slot.
        slots = plan.write_slots[l]                                                      # [nd * Hkv]
        rows = torch.arange(nd * g, device=dev)
        pages = plan.bt_rows[rows, (slots.clamp(min=0) // rt.BS)].to(torch.int64)
        addr = torch.where(slots >= 0, pages * rt.BS + (slots % rt.BS), torch.full_like(pages, -1))
        key_cache, value_cache = self._caches(kv_cache)
        reshape_and_cache_flash(self._to_rows(key[:nd], g), self._to_rows(value[:nd], g),
                                key_cache, value_cache, addr, self.kv_cache_dtype,
                                layer._k_scale, layer._v_scale)

        # Stage-1 score, every decode row, every step.
        thin_score_decode(q_thin[:nd], thin_kv, thin_md.block_table[:nd], acc_kv, acc_md.block_table[:nd],
                          thin_md.seq_lens[:nd], max_seq_len=plan.max_decode_seq_len)

        for r in plan.refresh_rows:
            self._refresh(rt, plan, l, r, kv_cache, acc_kv, acc_md.block_table[r], q_thin[r], thin_kv,
                          thin_md.block_table[r])

        q_rows = self._to_rows(query[:nd], g)
        out_rows = torch.empty_like(q_rows)
        flash_attn_varlen_func(
            q=q_rows, k=key_cache, v=value_cache, out=out_rows,
            cu_seqlens_q=rt.decode_cu_seqlens(nd * g, dev), max_seqlen_q=1,
            seqused_k=plan.lens[l], max_seqlen_k=rt.working_slots,
            softmax_scale=self.scale, causal=False, block_table=plan.bt_rows,
            fa_version=self.vllm_flash_attn_version,
        )
        output[:nd] = self._from_rows(out_rows, g, nd)
        if cascade.debug() and l == 0 and rt.debug_checks < DEBUG_CHECKS:
            rt.debug_checks += 1
            logger.info("cascade debug L0 n=%s refresh=%s lens=%s write_slots=%s", plan.seq_lens[:2],
                        plan.refresh_rows, plan.lens[l][: min(8, nd * g)].tolist(),
                        plan.write_slots[l][: min(8, nd * g)].tolist())

    def _read_slots(self, rt: CascadeRuntime, plan: StepPlan, r: int, slots: torch.Tensor,
                    kv_cache: torch.Tensor) -> torch.Tensor:
        """slots: [Hkv, m] working slots per group -> [m, Hkv, 2D] from the working pages."""
        g, m = slots.shape
        rows = (torch.arange(g, device=slots.device) * plan.num_decodes + r)[:, None].expand(g, m)
        pages = plan.bt_rows[rows.reshape(-1), (slots // rt.BS).reshape(-1)].to(torch.int64)
        offs = (slots % rt.BS).reshape(-1)
        kv = kv_cache[pages, 0, offs]                                                     # [g * m, 2D]
        return kv.view(g, m, -1).transpose(0, 1)

    def _refresh(self, rt: CascadeRuntime, plan: StepPlan, l, r, kv_cache, acc_kv, acc_bt_row, q_thin_row,
                 thin_kv, thin_bt_row):
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
        pool = rt.cpu_layer(l, state)                                                    # [Hkv, capacity, 2D]
        lf = n - fs
        sel_old = state.sel_len[l].clone()
        sel_new = (counts * SEL_BLOCK).to(torch.int32)
        first = state.prev_refresh_n == 0

        # Floor: on the first refresh the working pages still hold the prompt positionally,
        # so it comes from the CPU store; afterwards it is already on the GPU (floor plus the
        # tokens decoded since), contiguous from slot sel_old, so it is shifted in place.
        prev_fs = state.prev_refresh_n - state.prev_lf
        for gi in range(rt.Hkv):
            src0, dst0, length = floor_shift(int(sel_old[gi]), int(sel_new[gi]), prev_fs, fs, lf)
            dst = dst0 + torch.arange(length, device=dev)
            if first:
                data = pool[gi, fs:n].to(dev)                                            # [lf, 2D]
            else:
                data = self._slot_view(rt, plan, r, gi, src0 + torch.arange(length, device=dev), kv_cache)
            self._slot_write(rt, plan, r, gi, dst, data, kv_cache)
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
        need_per_group, free_per_group = [], []
        for gi in range(rt.Hkv):
            cnt = int(counts[gi])
            new_g = ids_cpu[gi, :cnt]
            old_g = resident[gi]
            need = new_g[~torch.isin(new_g, old_g)] if cnt else new_g[:0]
            free = (~torch.isin(old_g, new_g)).nonzero(as_tuple=True)[0]
            free = free[free < cnt][: need.numel()]
            need = need[: free.numel()]
            old_g[free] = need
            need_per_group.append(need)
            free_per_group.append(free)
        state.resident[l] = resident
        fetched = int(sum(x.numel() for x in need_per_group))

        row_elems = SEL_BLOCK * 2 * rt.D
        t_gather = t_h2d = t_floor
        if fetched:
            blocks = pool.view(rt.Hkv, -1, row_elems)                                    # [Hkv, blocks, 8KB row]
            staging = rt.staging(fetched, row_elems)
            at = 0
            for gi, need in enumerate(need_per_group):
                if need.numel():
                    torch.index_select(blocks[gi], 0, need, out=staging[at:at + need.numel()])
                    at += need.numel()
            t_gather = time.perf_counter()
            data = staging.to(dev, non_blocking=False).view(-1, SEL_BLOCK, 2 * rt.D)
            t_h2d = time.perf_counter()
            at = 0
            for gi, free in enumerate(free_per_group):
                if free.numel():
                    dst = (free.to(dev)[:, None] * SEL_BLOCK
                           + torch.arange(SEL_BLOCK, device=dev)[None, :]).reshape(-1)
                    self._slot_write(rt, plan, r, gi, dst, data[at:at + free.numel()].reshape(-1, 2 * rt.D),
                                     kv_cache)
                    at += free.numel()

        state.sel_len[l] = sel_new
        plan.lens[l][torch.arange(rt.Hkv, device=dev) * plan.num_decodes + r] = (sel_new + lf).to(dev)
        if cascade.debug() and r == 0 and l in (0, 15, 31):
            self._debug_refresh(rt, state, l, n, ids, counts, fetched, time.perf_counter() - t_start,
                                t_floor - t_start, t_gather - t_floor, t_h2d - t_gather)

    # ---- slot helpers --------------------------------------------------------
    def _slot_addr(self, rt: CascadeRuntime, plan: StepPlan, r: int, gi: int, slots: torch.Tensor):
        rows = torch.full_like(slots, row_of(gi, r, plan.num_decodes))
        return slot_pages(plan.bt_rows, rows, slots, rt.BS)

    def _slot_view(self, rt, plan, r, gi, slots, kv_cache) -> torch.Tensor:
        pages, offs = self._slot_addr(rt, plan, r, gi, slots)
        return kv_cache[pages, 0, offs]

    def _slot_write(self, rt, plan, r, gi, slots, data, kv_cache) -> None:
        pages, offs = self._slot_addr(rt, plan, r, gi, slots)
        kv_cache[pages, 0, offs] = data

    # ---- debug ---------------------------------------------------------------
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
                    "overlap_per_group=%s", l, n, rel, counts.tolist(), ref_counts.tolist(), overlap)

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
        mb = fetched * SEL_BLOCK * rt.D * 2 * 2 / 2**20
        logger.info("cascade refresh L%d n=%d selected_blocks=%d fetched_blocks=%d (%.1fMB) total=%.0fms "
                    "floor=%.0fms gather=%.0fms h2d=%.0fms overlap_with_previous=%s",
                    l, n, int(counts.sum()), fetched, mb, total_s * 1e3, floor_s * 1e3, gather_s * 1e3,
                    h2d_s * 1e3, overlap)


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
