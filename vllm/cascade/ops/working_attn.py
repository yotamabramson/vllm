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

Softmax is accumulated page by page relative to a running max (the same
pattern as MiniMax-M3's sparse decode kernel), with guards so a page with no
valid slot contributes nothing.

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


@triton.jit
def _working_attn_decode_kernel(
    Q, KV, BT, A_END, B_END, O,
    stride_q_r, stride_q_h, stride_q_d,
    stride_kv_b, stride_kv_h, stride_kv_t, stride_kv_d,
    stride_bt_r,
    stride_be_r, stride_be_h,
    stride_o_r, stride_o_h, stride_o_d,
    b_start, scale,
    D: tl.constexpr, N_REP: tl.constexpr, BS: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    a_end = tl.load(A_END + r).to(tl.int64)
    b_end = tl.load(B_END + r * stride_be_r + h * stride_be_h).to(tl.int64)
    end = tl.maximum(a_end, b_end)
    num_pages = (end + BS - 1) // BS

    reps = tl.arange(0, N_REP).to(tl.int64)
    dims = tl.arange(0, D).to(tl.int64)
    offs = tl.arange(0, BS).to(tl.int64)
    q_heads = h * N_REP + reps
    q = tl.load(Q + r * stride_q_r + q_heads[:, None] * stride_q_h + dims[None, :] * stride_q_d).to(tl.float32)

    m = tl.full((N_REP,), float("-inf"), tl.float32)     # running max logit
    wsum = tl.zeros((N_REP,), tl.float32)                 # sum exp(logit - m)
    acc = tl.zeros((N_REP, D), tl.float32)                # sum exp(logit - m) * v
    for pg in range(0, num_pages):
        page = tl.load(BT + r * stride_bt_r + pg).to(tl.int64)
        slot = pg * BS + offs
        valid = (slot < a_end) | ((slot >= b_start) & (slot < b_end))
        base = KV + page * stride_kv_b + h * stride_kv_h + offs[:, None] * stride_kv_t
        k = tl.load(base + dims[None, :] * stride_kv_d, mask=valid[:, None], other=0.0).to(tl.float32)
        v = tl.load(base + (D + dims[None, :]) * stride_kv_d, mask=valid[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale                    # [N_REP, BS]
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        ratio = tl.where(m > float("-inf"), tl.exp(m - m_new), 0.0)
        p = tl.where(valid[None, :], tl.exp(s - m_new[:, None]), 0.0)                # [N_REP, BS]
        acc = acc * ratio[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
        wsum = wsum * ratio + tl.sum(p, axis=1)
        m = m_new

    out = acc / tl.maximum(wsum, 1e-30)[:, None]
    tl.store(O + r * stride_o_r + q_heads[:, None] * stride_o_h + dims[None, :] * stride_o_d, out)


def working_attn_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    a_end: torch.Tensor,
    b_start: int,
    b_end: torch.Tensor,
    output: torch.Tensor,
    scale: float,
) -> None:
    R, Hq, D = q.shape
    _, Hkv, BS, D2 = kv_cache.shape
    assert D2 == 2 * D and Hq % Hkv == 0 and b_end.shape == (R, Hkv)
    assert block_table.stride(1) == 1 and output.shape == q.shape
    if R == 0:
        return
    _working_attn_decode_kernel[(R, Hkv)](
        q, kv_cache, block_table, a_end, b_end, output,
        q.stride(0), q.stride(1), q.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), kv_cache.stride(3),
        block_table.stride(0),
        b_end.stride(0), b_end.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        b_start, scale,
        D=D, N_REP=Hq // Hkv, BS=BS,
    )


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
