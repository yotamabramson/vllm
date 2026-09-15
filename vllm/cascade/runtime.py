# SPDX-License-Identifier: Apache-2.0
"""Worker-side state for full mode: per-request bookkeeping, the CPU KV store,
and the per-step plan the backend executes layer by layer.

Request identity: metadata carries no request ids, so a request is keyed by the
first physical block of its working-set block table. That block is never freed
while the request lives (the manager only frees blocks past the budget); a
prefill starting at token 0 resets the state, which also covers a recycled block
and a preempted-and-recomputed request. States not seen for 2 steps are dropped.

CPU store: a request starts with a small per-layer buffer (SMALL_CAP tokens,
unpinned) and moves to a full-length pinned slot as soon as it needs more.
vLLM's warm-up runs hundreds of 2-token dummy requests through the real path;
real long prompts cross SMALL_CAP in their first prefill chunk.

Working-page slot layout per request (same for every layer; contents per head):
    [0, lf)                  recency floor at the last refresh (lf = 1000..1015)
    [lf, lf + t)             tokens decoded since that refresh: token at position j
                             sits at slot lf + (j - refresh_n)
    [S, S + 16 * count_h)    head h's selected blocks, S = FLOOR_SLOTS + p
Refresh cadence (harness step_count): decode step t = n - prompt_len - 1 refreshes
when t % p == 0, so the first decode step after prefill refreshes.
"""

from dataclasses import dataclass, field

import torch

import vllm.cascade as cascade
from vllm.cascade.ops.select import floor_start
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_and_prefills

logger = init_logger(__name__)

SMALL_CAP = 64
DEBUG_STEPS = 60


@dataclass
class RequestState:
    b_end: torch.Tensor          # [L, Hkv] int32 CPU: end slot of each head's selection, per layer
    cpu_slot: int | None = None  # pinned slot, once the request outgrows SMALL_CAP
    small: list[torch.Tensor | None] = field(default_factory=list)   # per layer, until then
    prompt_len: int | None = None
    refresh_n: int = 0           # seq_len at the last refresh (0 = none yet)
    lf: int = 0                  # floor length at the last refresh
    last_seen: int = 0


@dataclass
class Flush:
    row: int
    slot_start: int              # working slot of the first token to flush
    count: int
    cpu_start: int               # its position in the sequence


@dataclass
class StepPlan:
    runtime: "CascadeRuntime"
    fallback: bool               # batch with no known request state: stock attention
    num_decodes: int = 0
    seq_lens: list[int] = field(default_factory=list)
    query_start: list[int] = field(default_factory=list)
    states: list[RequestState | None] = field(default_factory=list)
    prefill_rows: list[int] = field(default_factory=list)
    refresh_rows: list[int] = field(default_factory=list)
    flushes: list[Flush] = field(default_factory=list)
    a_end: torch.Tensor | None = None          # [nd] int32
    decode_slots: torch.Tensor | None = None   # [nd] int64, -1 = refresh row (no working write)
    b_end: torch.Tensor | None = None          # [L, nd, Hkv] int32
    max_slots: list[int] = field(default_factory=list)   # per layer: highest working slot any row attends
    max_decode_seq_len: int = 0


class CascadeRuntime:
    def __init__(self, num_layers: int, num_kv_heads: int, head_size: int, block_size: int,
                 dtype: torch.dtype, max_len: int) -> None:
        self.L, self.Hkv, self.D, self.BS = num_layers, num_kv_heads, head_size, block_size
        self.dtype = dtype
        self.max_len = max_len
        self.p = cascade.refresh_period()
        self.S = cascade.working_start()
        self.working_slots = cascade.working_slots()
        caps = cascade.capacities()
        assert len(caps) == num_layers and all(len(row) == num_kv_heads for row in caps), (
            f"cascade: capacities shape {len(caps)}x{len(caps[0])} != layers {num_layers} x kv heads {num_kv_heads}")
        self.caps = torch.tensor(caps, dtype=torch.long)        # [L, Hkv] CPU
        self.num_seqs = cascade.cpu_pool_seqs()
        self.pool: list[torch.Tensor | None] = [None] * num_layers
        self.free_slots = list(range(self.num_seqs))
        self.states: dict[int, RequestState] = {}
        self.step = 0
        self.debug_steps = 0
        self.debug_fallbacks = 0
        self.debug_attn_checks = 0

    # ---- CPU store ----
    def pool_layer(self, layer: int) -> torch.Tensor:
        """[num_seqs, max_len, Hkv, 2D] pinned, allocated on first use."""
        if self.pool[layer] is None:
            self.pool[layer] = torch.empty(self.num_seqs, self.max_len, self.Hkv, 2 * self.D,
                                           dtype=self.dtype, pin_memory=True)
        return self.pool[layer]

    def cpu_layer(self, layer: int, state: RequestState) -> torch.Tensor:
        """This request's CPU store for one layer: [capacity, Hkv, 2D]."""
        if state.cpu_slot is not None:
            return self.pool_layer(layer)[state.cpu_slot]
        if state.small[layer] is None:
            state.small[layer] = torch.empty(SMALL_CAP, self.Hkv, 2 * self.D, dtype=self.dtype)
        return state.small[layer]

    def ensure_capacity(self, state: RequestState, end: int) -> None:
        """Move the request to a pinned slot before it needs more than SMALL_CAP tokens."""
        if state.cpu_slot is not None or end <= SMALL_CAP:
            return
        if not self.free_slots:
            self._gc(max_age=0)
        if not self.free_slots:
            raise RuntimeError(f"cascade: CPU store holds {self.num_seqs} long sequences; "
                               "raise VLLM_CASCADE_CPU_SEQS")
        slot = self.free_slots.pop()
        for layer, small in enumerate(state.small):
            if small is not None:
                self.pool_layer(layer)[slot, :SMALL_CAP].copy_(small)
        state.small = [None] * self.L
        state.cpu_slot = slot

    def write_cpu(self, layer: int, state: RequestState, start: int, kv: torch.Tensor, blocking: bool) -> None:
        """kv: [m, Hkv, 2D] (GPU) -> CPU store positions [start, start + m)."""
        self.cpu_layer(layer, state)[start:start + kv.shape[0]].copy_(kv, non_blocking=not blocking)

    # ---- request states ----
    def _release(self, state: RequestState) -> None:
        if state.cpu_slot is not None:
            self.free_slots.append(state.cpu_slot)

    def _new_state(self, key: int) -> RequestState:
        old = self.states.pop(key, None)
        if old is not None:
            self._release(old)
        state = RequestState(b_end=torch.full((self.L, self.Hkv), self.S, dtype=torch.int32),
                             small=[None] * self.L)
        self.states[key] = state
        return state

    def _gc(self, max_age: int = 2) -> None:
        for key in [k for k, s in self.states.items() if s.last_seen < self.step - max_age]:
            self._release(self.states.pop(key))

    # ---- per-step plan (called once per step by the metadata builder) ----
    def plan_step(self, m: CommonAttentionMetadata) -> StepPlan:
        self.step += 1
        num_reqs = m.num_reqs
        qsl = m.query_start_loc_cpu[: num_reqs + 1].tolist()
        seq = m.seq_lens[:num_reqs].cpu().tolist()
        keys = m.block_table_tensor[:num_reqs, 0].cpu().tolist()
        nd, _, num_decode_tokens, _ = split_decodes_and_prefills(m, decode_threshold=1)
        assert num_decode_tokens == nd

        # Stock attention for batches we cannot track: vLLM's warm-up, autotune and
        # graph-capture batches use block 0 -- the reserved null block, never given to a
        # real request -- and a decode row we never saw prefill, or a continued prefill
        # chunk we never saw start, has no state to work with.
        for r in range(num_reqs):
            qlen = qsl[r + 1] - qsl[r]
            starts_here = r >= nd and seq[r] - qlen == 0
            if keys[r] == 0 or (qlen > 0 and not starts_here and keys[r] not in self.states):
                if cascade.debug() and self.debug_fallbacks < 3:
                    self.debug_fallbacks += 1
                    logger.info("cascade debug step=%d FALLBACK reqs=%d decodes=%d row=%d qlen=%d seq=%s "
                                "keys=%s known_keys=%s", self.step, num_reqs, nd, r, qlen, seq[:6], keys[:6],
                                list(self.states)[:6])
                return StepPlan(runtime=self, fallback=True)

        plan = StepPlan(runtime=self, fallback=False, num_decodes=nd, seq_lens=seq, query_start=qsl,
                        states=[None] * num_reqs)
        for r in range(num_reqs):
            if r >= nd and qsl[r + 1] > qsl[r] and seq[r] - (qsl[r + 1] - qsl[r]) == 0:
                self._new_state(keys[r])
            if keys[r] in self.states:
                self.states[keys[r]].last_seen = self.step

        for r in range(nd, num_reqs):
            if qsl[r + 1] == qsl[r]:
                continue
            state = self.states[keys[r]]
            self.ensure_capacity(state, seq[r])
            plan.states[r] = state
            plan.prefill_rows.append(r)

        a_end: list[int] = []
        slot_idx: list[int] = []
        for r in range(nd):
            state = self.states[keys[r]]
            plan.states[r] = state
            n = seq[r]
            self.ensure_capacity(state, n)
            if state.prompt_len is None:
                state.prompt_len = n - 1
            if (n - state.prompt_len - 1) % self.p == 0:
                if state.refresh_n:
                    count = n - 1 - state.refresh_n
                    if count > 0:
                        plan.flushes.append(Flush(r, state.lf, count, state.refresh_n))
                state.lf = n - floor_start(n)
                state.refresh_n = n
                plan.refresh_rows.append(r)
                a_end.append(state.lf)
                slot_idx.append(-1)
            else:
                end = state.lf + (n - state.refresh_n)
                assert end <= self.S, f"cascade: {end} slots since refresh exceed floor+p={self.S}"
                a_end.append(end)
                slot_idx.append(end - 1)
        self._gc()

        if nd:
            dev = m.seq_lens.device
            plan.max_decode_seq_len = max(seq[:nd])
            plan.a_end = torch.tensor(a_end, dtype=torch.int32).to(dev)
            idx = torch.tensor(slot_idx, dtype=torch.int64)
            safe = idx.clamp(min=0)
            rows = torch.arange(nd, dtype=torch.int64)
            pages = m.block_table_tensor[rows.to(dev), (safe // self.BS).to(dev)].to(torch.int64)
            slots = pages * self.BS + (safe % self.BS).to(dev)
            plan.decode_slots = torch.where((idx >= 0).to(dev), slots, torch.full_like(slots, -1))
            b_end_cpu = torch.stack([plan.states[r].b_end for r in range(nd)], dim=1)      # [L, nd, Hkv]
            plan.b_end = b_end_cpu.to(dev)
            # Sizes the attention kernel's per-chunk buffers to this batch, not to the
            # worst-case budget (1024 warm-up requests x full budget ran out of memory).
            # b_end == S means "nothing selected" and adds no slots.
            max_a = max(a_end)
            selected_end = torch.where(b_end_cpu > self.S, b_end_cpu, 0).amax(dim=(1, 2)).tolist()
            plan.max_slots = [max(max_a, int(v)) for v in selected_end]
        if cascade.debug() and self.debug_steps < DEBUG_STEPS:
            self.debug_steps += 1
            logger.info("cascade debug step=%d reqs=%d decodes=%d prefills=%s refresh=%s flushes=%s seq=%s "
                        "qlens=%s keys=%s a_end=%s slot_idx=%s states=%d", self.step, num_reqs, nd,
                        plan.prefill_rows, plan.refresh_rows,
                        [(f.row, f.slot_start, f.count, f.cpu_start) for f in plan.flushes], seq[:6],
                        [qsl[r + 1] - qsl[r] for r in range(min(num_reqs, 6))], keys[:6], a_end[:6],
                        slot_idx[:6], len(self.states))
        return plan


_RUNTIME: CascadeRuntime | None = None


def get_runtime(**kwargs) -> CascadeRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = CascadeRuntime(**kwargs)
    return _RUNTIME
