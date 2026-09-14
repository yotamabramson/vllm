# SPDX-License-Identifier: Apache-2.0
"""Stage-1 screening score, accumulated every decode step.

Exactly the harness's per-step signal (cascaded-attention repo,
post_paper/scripts/_periodic_refresh.py, periodic_attention_step_batched):

    raw   = einsum("hrd,nhd->hrn", q_thin grouped [Hkv, n_rep, 32], k_thin [n, Hkv, 32]) / sqrt(32)
    w     = softmax(raw, dim=-1).max(dim=1)           # [Hkv, n]: softmax over ALL tokens, max over the group's query heads

and adds w into a per-token, per-KV-head accumulator. The harness keeps the last
p vectors and averages them at refresh; a running sum reset at every refresh is
the same set of steps, and dividing by the window length does not change top-K,
so only the sum is kept.

Deliberately imports nothing from vLLM (torch + Triton only), so it can be
tested against the reference below on any CUDA machine.

Layouts (decode only: one query token per request):
    q_thin          [R, Hq, W]            float16/bfloat16/float32
    k_cache         [NB, BS_K, Hkv * W]   thin-K side cache, paged
    k_block_table   [R, MB]  int32
    acc_cache       [NB2, BS_A, Hkv]      float32 accumulator, paged
    acc_block_table [R, MB2] int32
    seq_lens        [R]      int32        tokens per request, current token included
                                          (its thin-K must already be written)
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _thin_score_decode_kernel(
    Q, K_CACHE, K_BT, ACC, ACC_BT, SEQ_LENS,
    stride_q_r, stride_q_h,
    stride_k_b, stride_k_t,
    stride_kbt_r,
    stride_a_b, stride_a_t,
    stride_abt_r,
    scale,
    W: tl.constexpr, N_REP: tl.constexpr, BS_K: tl.constexpr, BS_A: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    n = tl.load(SEQ_LENS + r).to(tl.int64)
    num_blocks = (n + BS_K - 1) // BS_K

    reps = tl.arange(0, N_REP).to(tl.int64)
    dims = tl.arange(0, W).to(tl.int64)
    offs = tl.arange(0, BS_K).to(tl.int64)

    # The group's query heads: h * n_rep + rep -> [N_REP, W]
    q = tl.load(Q + r * stride_q_r + (h * N_REP + reps)[:, None] * stride_q_h + dims[None, :]).to(tl.float32)

    # Pass 1: log-sum-exp of the scores over all n tokens, per query head,
    # combined block by block so no global max is needed up front.
    lse = tl.full((N_REP,), float("-inf"), tl.float32)
    for b in range(0, num_blocks):
        phys = tl.load(K_BT + r * stride_kbt_r + b).to(tl.int64)
        pos = b * BS_K + offs
        tok = pos < n
        k = tl.load(K_CACHE + phys * stride_k_b + offs[:, None] * stride_k_t + (h * W + dims)[None, :],
                    mask=tok[:, None], other=0.0).to(tl.float32)              # [BS_K, W]
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale            # [N_REP, BS_K]
        s = tl.where(tok[None, :], s, float("-inf"))
        m = tl.max(s, axis=1)                                                # finite: a block always has >= 1 token
        blse = m + tl.log(tl.sum(tl.exp(s - m[:, None]), axis=1))
        lse = tl.maximum(lse, blse) + tl.log(1.0 + tl.exp(-tl.abs(lse - blse)))

    # Pass 2: softmax probability, max over the group's query heads, add into the accumulator.
    for b in range(0, num_blocks):
        phys = tl.load(K_BT + r * stride_kbt_r + b).to(tl.int64)
        pos = b * BS_K + offs
        tok = pos < n
        k = tl.load(K_CACHE + phys * stride_k_b + offs[:, None] * stride_k_t + (h * W + dims)[None, :],
                    mask=tok[:, None], other=0.0).to(tl.float32)
        s = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale
        s = tl.where(tok[None, :], s, float("-inf"))
        p = tl.max(tl.exp(s - lse[:, None]), axis=0)                         # [BS_K]
        a_phys = tl.load(ACC_BT + r * stride_abt_r + pos // BS_A, mask=tok, other=0).to(tl.int64)
        a_ptr = ACC + a_phys * stride_a_b + (pos % BS_A) * stride_a_t + h
        cur = tl.load(a_ptr, mask=tok, other=0.0)
        tl.store(a_ptr, cur + p, mask=tok)


def thin_score_decode(
    q_thin: torch.Tensor,
    k_cache: torch.Tensor,
    k_block_table: torch.Tensor,
    acc_cache: torch.Tensor,
    acc_block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    """Add this step's stage-1 score into acc_cache, in place. Layouts: module docstring."""
    R, Hq, W = q_thin.shape
    _, BS_K, HW = k_cache.shape
    Hkv = HW // W
    assert HW == Hkv * W and Hq % Hkv == 0
    assert acc_cache.dtype == torch.float32 and acc_cache.shape[2] == Hkv
    q = q_thin.contiguous()
    assert q.stride(2) == 1 and k_cache.stride(2) == 1 and acc_cache.stride(2) == 1
    assert k_block_table.stride(1) == 1 and acc_block_table.stride(1) == 1
    if R == 0:
        return
    _thin_score_decode_kernel[(R, Hkv)](
        q, k_cache, k_block_table, acc_cache, acc_block_table, seq_lens,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1),
        k_block_table.stride(0),
        acc_cache.stride(0), acc_cache.stride(1),
        acc_block_table.stride(0),
        1.0 / math.sqrt(W),
        W=W, N_REP=Hq // Hkv, BS_K=BS_K, BS_A=acc_cache.shape[1],
    )


def thin_score_decode_reference(q_thin, k_cache, k_block_table, acc_cache, acc_block_table, seq_lens) -> None:
    """Same result with the harness's own PyTorch formula, one request at a time (slow; for tests)."""
    R, Hq, W = q_thin.shape
    _, BS_K, HW = k_cache.shape
    Hkv = HW // W
    BS_A = acc_cache.shape[1]
    for r in range(R):
        n = int(seq_lens[r])
        blocks = k_block_table[r, : (n + BS_K - 1) // BS_K].long()
        k = k_cache[blocks].reshape(-1, Hkv, W)[:n].float()                  # [n, Hkv, W]
        q = q_thin[r].float().view(Hkv, Hq // Hkv, W)
        raw = torch.einsum("hrd,nhd->hrn", q, k) / math.sqrt(W)
        w = torch.softmax(raw, dim=-1).max(dim=1).values                     # [Hkv, n]
        pos = torch.arange(n, device=k_cache.device)
        a_blocks = acc_block_table[r].long()[pos // BS_A]
        acc_cache[a_blocks, pos % BS_A] += w.T
