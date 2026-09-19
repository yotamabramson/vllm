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
from dataclasses import dataclass

import torch

import vllm.cascade as cascade
from vllm.cascade import SEL_BLOCK
from vllm.cascade import THIN_WIDTH as THIN
from vllm.cascade.ops.layout import (floor_shift, group_block_table, refresh_slots, row_of,
                                     shift_indices, slot_pages)
from vllm.cascade.ops.score import thin_score_decode
from vllm.cascade.async_io import PendingFetch
from vllm.cascade.ops.select import floor_start, select_blocks
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


@dataclass
class _Pick:
    """One (request, layer) refresh, carried across the batched phases below.

    `mode` says where its selection came from:
      sync  selected in this very step from this step's scores (G=0, and every first
            refresh whatever G is) -- it still has to fetch before it can attend
      swap  selected G steps ago; the blocks are already on the GPU, waiting in staging
      hold  neither: no usable early selection arrived, so the working set stays as it is
            and only the floor moves (safety valve, should not happen in a healthy run)
    """
    r: int
    n: int
    ids: torch.Tensor | None     # [Hkv, K] selected block ids, on the GPU (sync only)
    counts: torch.Tensor | None  # [Hkv] blocks kept per group (CPU)
    fs: int                      # first token of the recency floor
    state: object
    sel_old: torch.Tensor        # [Hkv] int32 selected tokens before this refresh
    sel_new: torch.Tensor        # [Hkv] int32 after it
    lf: int                      # floor length
    mode: str = "sync"
    pending: object = None
    ids_cpu: torch.Tensor | None = None


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
        with rt.timer.track("sidecache"):
            # Every layer's side caches share one slot mapping, so clamp it once per step.
            if plan.slots_src is not thin_md.slot_mapping:
                plan.slots_src = thin_md.slot_mapping
                plan.thin_slots = thin_md.slot_mapping[:T].clamp(min=0)
                plan.acc_slots = acc_md.slot_mapping[:T].clamp(min=0)
            thin_kv.view(-1, thin_kv.shape[-1]).index_copy_(
                0, plan.thin_slots, k_thin[:T].reshape(T, -1).to(thin_kv.dtype))
            acc_kv.view(-1, acc_kv.shape[-1]).index_fill_(0, plan.acc_slots, 0.0)

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
        with rt.timer.track("flush"):
            if plan.refresh_rows:
                self._flush_layer(rt, plan, l, kv_cache, key, value)

        # This step's K/V into each group's next working slot.
        with rt.timer.track("write"):
            key_cache, value_cache = self._caches(kv_cache)
            reshape_and_cache_flash(self._to_rows(key[:nd], g), self._to_rows(value[:nd], g),
                                    key_cache, value_cache, plan.write_addr[l], self.kv_cache_dtype,
                                    layer._k_scale, layer._v_scale)

        # Stage-1 score. Which rows it runs on is the aggregation setting: every row every
        # step for "mean", only the rows at a stride step (the refresh step included) for
        # "stride:N" and "last". The divisor is never applied -- every token of a group is
        # summed over the same steps, and top-K does not see a uniform scale.
        with rt.timer.track("score"):
            if len(plan.score_rows) == nd:
                sl = slice(0, nd)
                thin_score_decode(q_thin[sl], thin_kv, thin_md.block_table[sl], acc_kv,
                                  acc_md.block_table[sl], thin_md.seq_lens[sl],
                                  max_seq_len=plan.max_decode_seq_len)
            elif plan.score_rows:
                sel_rows = torch.tensor(plan.score_rows, device=dev)
                thin_score_decode(q_thin[sel_rows], thin_kv, thin_md.block_table[sel_rows], acc_kv,
                                  acc_md.block_table[sel_rows], thin_md.seq_lens[sel_rows],
                                  max_seq_len=plan.max_decode_seq_len)

        # Early selection (VLLM_CASCADE_G > 0): pick the next working set G steps before it
        # is swapped in, and let the fetch run on the worker thread meanwhile.
        with rt.timer.track("select"):
            if plan.select_rows:
                self._select_layer(rt, plan, l, acc_kv, acc_md)

        with rt.timer.track("refresh"):
            if plan.refresh_rows:
                self._refresh_layer(rt, plan, l, kv_cache, acc_kv, acc_md, q_thin, thin_kv, thin_md)

        with rt.timer.track("attn"):
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

    # ---- refresh: one batch of work per layer, never one per request ---------
    def _slots_head_major(self, rt: CascadeRuntime, plan: StepPlan, r: int, slots: torch.Tensor,
                          kv_cache: torch.Tensor) -> torch.Tensor:
        """slots: [Hkv, m] working slots per group -> [Hkv, m, 2D], the CPU store's own order."""
        g, m = slots.shape
        rows = (torch.arange(g, device=slots.device) * plan.num_decodes + r)[:, None].expand(g, m)
        addr = self._addr(rt, plan, rows.reshape(-1), slots.reshape(-1))
        return kv_cache.view(-1, 2 * rt.D).index_select(0, addr).view(g, m, -1)

    def _flush_layer(self, rt: CascadeRuntime, plan: StepPlan, l, kv_cache, key, value) -> None:
        """Tokens decoded since the last refresh, plus this step's own token, into the CPU
        store -- one gather and ONE device->host copy for every refreshing request in this
        layer. Per request it was a pipeline drain each: 256 of them per refresh step at N=8.

        With VLLM_CASCADE_ASYNC_FLUSH=1 the copy goes out on a side stream and the
        host-side scatter runs on the worker thread, so the step does not wait for either.
        Nothing reads these tokens for several windows (a token is only fetchable once it
        falls out of the ~1000-token recency floor), and the worker is FIFO, so a later
        fetch cannot overtake this write.
        """
        dev, g, D2 = kv_cache.device, rt.Hkv, 2 * rt.D
        parts, meta = [], []
        for f in plan.flushes:
            state = plan.states[f.row]
            slots = state.sel_len[l].to(dev)[:, None] + torch.arange(
                f.slot_start, f.slot_start + f.count, device=dev)[None, :]
            parts.append(self._slots_head_major(rt, plan, f.row, slots, kv_cache).reshape(-1, D2))
            meta.append((state, f.cpu_start, f.count))
        for r in plan.refresh_rows:
            parts.append(torch.cat([key[r], value[r]], dim=-1).reshape(-1, D2))           # [g, 2D]
            meta.append((plan.states[r], plan.seq_lens[r] - 1, 1))
        packed = parts[0] if len(parts) == 1 else torch.cat(parts)
        if rt.async_flush:
            targets = [(rt.cpu_layer(l, state), g, start, count) for state, start, count in meta]
            rt.io(dev).submit_flush(packed, targets)
            return
        host = rt.pinned_out(packed.shape[0], D2)
        host.copy_(packed)
        at = 0
        for state, start, count in meta:
            rt.cpu_layer(l, state)[:, start:start + count].copy_(host[at:at + g * count].view(g, count, D2))
            at += g * count

    # ---- early selection: choose now, swap in G steps later -------------------
    @staticmethod
    def _resident_row(state, l: int, groups: int, k_max: int) -> torch.Tensor:
        """This (request, layer)'s [Hkv, K] block-id-per-slot bookkeeping, grown to k_max."""
        resident = state.resident.get(l)
        if resident is None or resident.shape[1] < k_max:
            grown = torch.full((groups, k_max), -1, dtype=torch.int64)
            if resident is not None:
                grown[:, : resident.shape[1]] = resident
            resident = grown
        state.resident[l] = resident
        return resident

    def _select_layer(self, rt: CascadeRuntime, plan: StepPlan, l, acc_kv, acc_md) -> None:
        """Select this request's next working set from the scores it has NOW, and hand the
        fetch to the worker thread. The swap happens G steps later, in _refresh_layer.

        The selection uses the floor boundary that will be in force at swap time
        (`n_apply`), not the one in force now -- see ops/select.py -- so the set it picks is
        disjoint from the floor the swap will install.
        """
        acc_bs = acc_kv.shape[1]
        for r in plan.select_rows:
            state = plan.states[r]
            n = plan.seq_lens[r]
            n_apply = n + rt.gap
            pages = acc_md.block_table[r][: (n + acc_bs - 1) // acc_bs].long()
            acc = acc_kv[pages].reshape(-1, rt.Hkv)[:n]
            ids, counts, fs = select_blocks(acc, n, rt.caps[l], n_apply=n_apply)
            acc_kv[pages] = 0
            old = state.pending.pop(l, None)
            if old is not None:                      # never consumed (a skipped swap); give it back
                old.wait()
                if old.staged is not None:
                    rt.io(acc.device).dev_pool.release(old.staged, old.event)
            pending = PendingFetch(n_apply=n_apply, sel_new=(counts * SEL_BLOCK).to(torch.int32),
                                   fs=fs, lf=n_apply - fs, counts=counts)
            state.pending[l] = pending
            rt.io(acc.device).submit_fetch(pending, ids, rt.cpu_layer(l, state),
                                           self._resident_row(state, l, rt.Hkv, ids.shape[1]).clone(),
                                           SEL_BLOCK * 2 * rt.D, SEL_BLOCK)

    def _refresh_layer(self, rt: CascadeRuntime, plan: StepPlan, l, kv_cache, acc_kv, acc_md,
                       q_thin, thin_kv, thin_md) -> None:
        dev, g, D2 = kv_cache.device, rt.Hkv, 2 * rt.D
        kv_flat = kv_cache.view(-1, D2)
        acc_bs = acc_kv.shape[1]
        nd = plan.num_decodes

        # What each refreshing request swaps in: a selection made G steps ago if one is
        # ready, otherwise one made here and now.
        picks = []
        for r in plan.refresh_rows:
            n = plan.seq_lens[r]
            state = plan.states[r]
            pending = state.pending.pop(l, None)
            if pending is not None:
                pending.wait()                       # the worker has issued the copy by now
            if pending is not None and pending.failed is None and pending.n_apply == n:
                picks.append(_Pick(r=r, n=n, ids=None, counts=pending.counts, fs=pending.fs,
                                   state=state, sel_old=state.sel_len[l].clone(),
                                   sel_new=pending.sel_new, lf=pending.lf,
                                   mode="swap", pending=pending))
                continue
            if pending is not None:
                # Selected for a step this request never reached, or the fetch failed. The
                # slots it would write still hold live blocks, so drop it rather than apply
                # it to the wrong frame.
                if pending.staged is not None:
                    rt.io(dev).dev_pool.release(pending.staged, pending.event)
                logger.warning("cascade: discarding an early selection for layer %d (apply n=%s, now %s, "
                               "failed=%s)", l, pending.n_apply, n, pending.failed)
            if rt.gap and state.prev_refresh_n:
                # No usable selection, and with G>0 there is no score taken on this step to
                # make one from. Keep the working set and move only the floor: every
                # selected block lies before the previous floor start, hence before this
                # one, so the frame stays disjoint.
                sel_old = state.sel_len[l].clone()
                fs = floor_start(n)
                picks.append(_Pick(r=r, n=n, ids=None, counts=None, fs=fs, state=state,
                                   sel_old=sel_old, sel_new=sel_old.clone(), lf=n - fs, mode="hold"))
                continue
            pages = acc_md.block_table[r][: (n + acc_bs - 1) // acc_bs].long()
            acc = acc_kv[pages].reshape(-1, g)[:n]
            ids, counts, fs = select_blocks(acc, n, rt.caps[l])
            if cascade.debug() and r == 0 and l in (0, 15, 31) and state.prompt_len == n - 1:
                self._debug_select(rt, l, n, acc, ids, counts, q_thin[r], thin_kv, thin_md.block_table[r])
            acc_kv[pages] = 0
            picks.append(_Pick(r=r, n=n, ids=ids, counts=counts, fs=fs, state=state,
                               sel_old=state.sel_len[l].clone(),
                               sel_new=(counts * SEL_BLOCK).to(torch.int32), lf=n - fs))

        # ONE device->host sync for every request that selected here. Rows that selected
        # early already had their ids read back on the worker thread.
        sync_picks = [pk for pk in picks if pk.mode == "sync"]
        if sync_picks:
            flat = torch.cat([pk.ids.reshape(-1) for pk in sync_picks]).cpu()
            at = 0
            for pk in sync_picks:
                k = pk.ids.shape[1]
                pk.ids_cpu = flat[at:at + g * k].view(g, k)
                at += g * k

        # Floor: already on the GPU (the floor plus the tokens decoded since), contiguous from
        # sel_old, so it is shifted in place -- every request and group in one gather and one
        # scatter. On a request's FIRST refresh the working pages still hold the prompt
        # positionally, so that floor comes from the CPU store instead.
        src0, dst0, lens, rows, firsts = [], [], [], [], []
        for pk in picks:
            if pk.state.prev_refresh_n == 0:
                firsts.append(pk)
                continue
            prev_fs = pk.state.prev_refresh_n - pk.state.prev_lf
            for gi in range(g):
                a, b, length = floor_shift(int(pk.sel_old[gi]), int(pk.sel_new[gi]), prev_fs, pk.fs, pk.lf)
                src0.append(a)
                dst0.append(b)
                lens.append(length)
                rows.append(row_of(gi, pk.r, nd))
        if src0:
            src, dst = shift_indices(src0, dst0, lens, dev)
            row_idx = torch.tensor(rows, device=dev)[:, None].expand(-1, src.shape[1]).reshape(-1)
            dst_addr = self._addr(rt, plan, row_idx, dst.reshape(-1))
            src_addr = self._addr(rt, plan, row_idx, src.reshape(-1))
            kv_flat.index_copy_(0, dst_addr, kv_flat.index_select(0, src_addr))
        for pk in firsts:
            pool = rt.cpu_layer(l, pk.state)
            data = pool[:, pk.fs:pk.n].to(dev).reshape(-1, D2)
            row_idx = torch.repeat_interleave(
                torch.tensor([row_of(gi, pk.r, nd) for gi in range(g)], device=dev), pk.lf)
            slots = (pk.sel_new.to(dev)[:, None] + torch.arange(pk.lf, device=dev)[None, :]).reshape(-1)
            kv_flat.index_copy_(0, self._addr(rt, plan, row_idx, slots), data)

        # Selected blocks: fetch only what is not resident already (measured overlap between
        # refreshes: 85-99%), for every request at once -- one host gather, one host->device
        # copy left in flight, one scatter.
        fetches, total = [], 0
        for pk in sync_picks:
            k_max = pk.ids_cpu.shape[1]
            resident = self._resident_row(pk.state, l, g, k_max)
            for gi in range(g):
                need, free = refresh_slots(resident[gi], pk.ids_cpu[gi], int(pk.counts[gi]))
                if need.numel():
                    fetches.append((pk, gi, need, free))
                    total += need.numel()
        if total:
            row_elems = SEL_BLOCK * D2
            host, event = rt.staging(total, row_elems)
            at = 0
            dst_slots, dst_rows = [], []
            for pk, gi, need, free in fetches:
                blocks = rt.cpu_layer(l, pk.state).view(g, -1, row_elems)
                torch.index_select(blocks[gi], 0, need, out=host[at:at + need.numel()])
                at += need.numel()
                dst_slots.append((free[:, None] * SEL_BLOCK + torch.arange(SEL_BLOCK)[None, :]).reshape(-1))
                dst_rows.append(torch.full((free.numel() * SEL_BLOCK,), row_of(gi, pk.r, nd)))
            data = host.to(dev, non_blocking=True).view(-1, D2)
            event.record()
            addr = self._addr(rt, plan, torch.cat(dst_rows).to(dev), torch.cat(dst_slots).to(dev))
            kv_flat.index_copy_(0, addr, data)

        # Selections fetched G steps ago: the blocks are already in GPU staging, so this is
        # one stream wait (on a copy that has had G decode steps to finish) and one scatter.
        for pk in picks:
            if pk.mode != "swap":
                continue
            pending = pk.pending
            if pending.resident is not None:
                pk.state.resident[l] = pending.resident
            if pending.staged is None:                   # nothing changed: all blocks resident
                continue
            torch.cuda.current_stream(dev).wait_event(pending.event)
            rows = (pending.dst_groups * nd + pk.r).to(dev)      # the row may differ from the
            slots = pending.dst_slots.to(dev)                    # selection step's: rebuild it
            addr = self._addr(rt, plan, rows, slots)
            kv_flat.index_copy_(0, addr, pending.staged.view(-1, D2))
            pk.state.resident[l] = pending.resident      # commit the slot bookkeeping
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(dev))
            rt.io(dev).dev_pool.release(pending.staged, done)
            pending.staged = None

        # New frame lengths, all requests in one write.
        idx = torch.tensor([row_of(gi, pk.r, nd) for pk in picks for gi in range(g)], device=dev)
        plan.lens[l][idx] = torch.cat([(pk.sel_new + pk.lf) for pk in picks]).to(dev)
        for pk in picks:
            pk.state.sel_len[l] = pk.sel_new
            if cascade.debug() and pk.r == 0 and l in (0, 15, 31) and pk.ids is not None:
                self._debug_refresh(rt, pk.state, l, pk.n, pk.ids, pk.counts, total)


    # ---- slot helpers --------------------------------------------------------
    def _addr(self, rt: CascadeRuntime, plan: StepPlan, rows: torch.Tensor,
              slots: torch.Tensor) -> torch.Tensor:
        """Flat index of each (row, slot) into kv_cache.view(-1, 2D)."""
        pages, offs = slot_pages(plan.bt_rows, rows, slots, rt.BS)
        return pages * rt.BS + offs

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

    def _debug_refresh(self, rt: CascadeRuntime, state, l, n, ids, counts, fetched):
        """Per refresh: how much of the new selection was already selected last time -- the delta
        the fetch has to move. Where the time goes is VLLM_CASCADE_TIMING=1 now, since the phases
        are batched across requests and no longer have a per-request cost to attribute."""
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
        logger.info("cascade refresh L%d n=%d selected_blocks=%d fetched_blocks=%d (%.1fMB, this layer, "
                    "all requests) overlap_with_previous=%s", l, n, int(counts.sum()), fetched, mb, overlap)


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
