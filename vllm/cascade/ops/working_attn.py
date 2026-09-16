# SPDX-License-Identifier: Apache-2.0
"""Decode attention over a request's working set (towards_70B_vllm/README.md,
"Per-request working pages").

Each KV head h of request r attends to two slot ranges of that request's
working pages (same physical pages for every head, different contents):

    [0, a_end[r])               recency floor + tokens decoded since refresh
                                (identical for every head)
    [b_start, b_end[r, h])      head h's selected 16-token blocks

Slots in between are unwritten and get no weight. Attention is order-invariant
over keys (K is stored post-RoPE), so slot order does not matter.

GPU parallelism: every (request, page chunk, KV head) is its own Triton program.
Each writes its chunk's softmax-weighted value average and log-sum-exp; torch
then merges chunks with softmax weights over those log-sum-exps (same split as
MiniMax-M3's chunked decode kernel). Measured on an RTX 4070 SUPER, llama3.1-8b
geometry, margin-0.3 working set: looping all pages inside one program per
request/head cost 16.6 ms/layer; element-wise products inside per-page programs
were no better; tl.dot products brought it to 0.27 ms/layer. Chunks are 64
slots because full-precision tl.dot over a whole 128-slot page exceeds the
per-program GPU shared-memory limit.

Imports nothing from vLLM (torch + Triton only).

Layouts:
    q            [R, Hq, D]
    kv_cache     [NB, Hkv, BS, 2 * D]   K in [..., :D], V in [..., D:] (FlashAttention layout)
    block_table  [R, MB] int32          working pages of each request
    a_end        [R] int32
    b_end        [R, Hkv] int32         b_end == b_start means no selected blocks
    output       [R, Hq, D]             written in place
"""

import torch
import triton
import triton.language as tl

SUB_CHUNK = 128


@triton.jit
def _working_attn_chunk_kernel(
    Q, KV, BT, A_END, B_END, LSE, OUT,
    stride_q_r, stride_q_h, stride_q_d,
    stride_kv_b, stride_kv_h, stride_kv_t, stride_kv_d,
    stride_bt_r,
    stride_be_r, stride_be_h,
    stride_l_r, stride_l_c, stride_l_h,
    stride_o_r, stride_o_c, stride_o_h, stride_o_d,
    b_start, scale, n_pages,
    D: tl.constexpr, N_REP: tl.constexpr, BS: tl.constexpr, SUB: tl.constexpr, NSUB: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    c = pid0 % NSUB
    rp = pid0 // NSUB
    pg = rp % n_pages
    r = rp // n_pages
    a_end = tl.load(A_END + r).to(tl.int64)
    b_end = tl.load(B_END + r * stride_be_r + h * stride_be_h).to(tl.int64)

    reps = tl.arange(0, N_REP).to(tl.int64)
    dims = tl.arange(0, D).to(tl.int64)
    offs = c * SUB + tl.arange(0, SUB).to(tl.int64)
    q_heads = h * N_REP + reps
    slot = pg * BS + offs
    valid = (slot < a_end) | ((slot >= b_start) & (slot < b_end))

    page = tl.load(BT + r * stride_bt_r + pg).to(tl.int64)
    # Q/K/V stay in the cache dtype for the dots (tensor cores accumulate in fp32, as the
    # harness's own fp16 matmuls do); converting them to fp32 first doubled shared memory
    # and capped chunks at 64 slots.
    q = tl.load(Q + r * stride_q_r + q_heads[:, None] * stride_q_h + dims[None, :] * stride_q_d)
    base = KV + page * stride_kv_b + h * stride_kv_h
    # K as [D, SUB] and V as [SUB, D] so both products are tl.dot (fused matmul).
    k = tl.load(base + offs[None, :] * stride_kv_t + dims[:, None] * stride_kv_d,
                mask=valid[None, :], other=0.0)                                       # [D, SUB]
    v = tl.load(base + offs[:, None] * stride_kv_t + (D + dims[None, :]) * stride_kv_d,
                mask=valid[:, None], other=0.0)                                       # [SUB, D]

    s = tl.dot(q, k) * scale                                                          # [N_REP, SUB] fp32
    s = tl.where(valid[None, :], s, float("-inf"))
    m = tl.max(s, axis=1)
    safe_m = tl.where(m > float("-inf"), m, 0.0)
    p = tl.where(valid[None, :], tl.exp(s - safe_m[:, None]), 0.0)
    wsum = tl.sum(p, axis=1)
    lse = tl.where(wsum > 0, safe_m + tl.log(wsum), float("-inf"))
    avg = tl.dot(p.to(v.dtype), v) / tl.maximum(wsum, 1e-30)[:, None]                 # [N_REP, D]

    chunk = pg * NSUB + c
    tl.store(LSE + r * stride_l_r + chunk * stride_l_c + q_heads * stride_l_h, lse)
    tl.store(OUT + r * stride_o_r + chunk * stride_o_c + q_heads[:, None] * stride_o_h + dims[None, :] * stride_o_d, avg)


def working_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    a_end: torch.Tensor,
    b_start: int,
    b_end: torch.Tensor,
    output: torch.Tensor,
    scale: float,
    max_slots: int | None = None,
) -> None:
    """max_slots (CPU int, >= every a_end and b_end) avoids a device sync when given."""
    R, Hq, D = q.shape
    _, Hkv, BS, D2 = kv_cache.shape
    assert D2 == 2 * D and Hq % Hkv == 0 and b_end.shape == (R, Hkv)
    assert block_table.stride(1) == 1 and output.shape == q.shape
    if R == 0:
        return
    if max_slots is None:
        max_slots = max(int(a_end.max()), int(b_end.max()))
    n_pages = (max_slots + BS - 1) // BS
    sub = min(SUB_CHUNK, BS)
    assert BS % sub == 0
    nsub = BS // sub
    lse = torch.empty(R, n_pages * nsub, Hq, dtype=torch.float32, device=q.device)
    avg = torch.empty(R, n_pages * nsub, Hq, D, dtype=torch.float32, device=q.device)
    _working_attn_chunk_kernel[(R * n_pages * nsub, Hkv)](
        q, kv_cache, block_table, a_end, b_end, lse, avg,
        q.stride(0), q.stride(1), q.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        block_table.stride(0),
        b_end.stride(0), b_end.stride(1),
        lse.stride(0), lse.stride(1), lse.stride(2),
        avg.stride(0), avg.stride(1), avg.stride(2), avg.stride(3),
        b_start, scale, n_pages,
        D=D, N_REP=Hq // Hkv, BS=BS, SUB=sub, NSUB=nsub,
    )
    weights = torch.softmax(lse, dim=1)                                               # over chunks
    output.copy_(torch.einsum("rch,rchd->rhd", weights, avg))


def working_attn_decode_reference(q, kv_cache, block_table, a_end, b_start, b_end, output, scale) -> None:
    """Plain softmax attention over the same slots, one request/head at a time (for tests)."""
    R, Hq, D = q.shape
    _, Hkv, BS, _ = kv_cache.shape
    n_rep = Hq // Hkv
    for r in range(R):
        for h in range(Hkv):
            end = max(int(a_end[r]), int(b_end[r, h]))
            pages = block_table[r, : (end + BS - 1) // BS].long()
            kv = kv_cache[pages, h].reshape(-1, 2 * D).float()                         # [pages*BS, 2D]
            slot = torch.arange(kv.shape[0], device=kv.device)
            valid = (slot < int(a_end[r])) | ((slot >= b_start) & (slot < int(b_end[r, h])))
            k, v = kv[valid, :D], kv[valid, D:]
            qg = q[r, h * n_rep:(h + 1) * n_rep].float()
            w = torch.softmax(qg @ k.T * scale, dim=-1)
            output[r, h * n_rep:(h + 1) * n_rep] = (w @ v).to(output.dtype)
