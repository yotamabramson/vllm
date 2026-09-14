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

GPU parallelism: every (request, 128-token chunk, KV head) is its own Triton
program. A first pass writes each chunk's log-sum-exp, torch combines them per
request/head/query head, and a second pass adds the probabilities into the
accumulator. (A first version looped over the whole context inside one program
per request/head: correct, but ~1000x too slow -- a Triton program is a single
GPU thread.)

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

SUB_CHUNK = 128


@triton.jit
def _chunk_scores(Q, K_CACHE, K_BT, SEQ_LENS, pid0, h,
                  stride_q_r, stride_q_h, stride_k_b, stride_k_t, stride_kbt_r,
                  scale, n_blocks, W: tl.constexpr, N_REP: tl.constexpr, BS_K: tl.constexpr,
                  SUB: tl.constexpr, NSUB: tl.constexpr):
    c = pid0 % NSUB
    rb = pid0 // NSUB
    b = rb % n_blocks
    r = rb // n_blocks
    n = tl.load(SEQ_LENS + r).to(tl.int64)
    reps = tl.arange(0, N_REP).to(tl.int64)
    dims = tl.arange(0, W).to(tl.int64)
    offs = c * SUB + tl.arange(0, SUB).to(tl.int64)
    pos = b * BS_K + offs
    tok = pos < n
    phys = tl.load(K_BT + r * stride_kbt_r + b).to(tl.int64)
    q = tl.load(Q + r * stride_q_r + (h * N_REP + reps)[:, None] * stride_q_h + dims[None, :]).to(tl.float32)
    k = tl.load(K_CACHE + phys * stride_k_b + offs[None, :] * stride_k_t + (h * W + dims)[:, None],
                mask=tok[None, :], other=0.0).to(tl.float32)                       # [W, SUB]
    s = tl.dot(q, k, input_precision="ieee") * scale                              # [N_REP, SUB]
    return r, b, c, pos, tok, s


@triton.jit
def _score_lse_kernel(
    Q, K_CACHE, K_BT, SEQ_LENS, LSE,
    stride_q_r, stride_q_h, stride_k_b, stride_k_t, stride_kbt_r,
    stride_l_r, stride_l_c, stride_l_h,
    scale, n_blocks,
    W: tl.constexpr, N_REP: tl.constexpr, BS_K: tl.constexpr, SUB: tl.constexpr, NSUB: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    r, b, c, pos, tok, s = _chunk_scores(Q, K_CACHE, K_BT, SEQ_LENS, pid0, h,
                                         stride_q_r, stride_q_h, stride_k_b, stride_k_t, stride_kbt_r,
                                         scale, n_blocks, W, N_REP, BS_K, SUB, NSUB)
    s = tl.where(tok[None, :], s, float("-inf"))
    m = tl.max(s, axis=1)
    safe_m = tl.where(m > float("-inf"), m, 0.0)
    total = tl.sum(tl.where(tok[None, :], tl.exp(s - safe_m[:, None]), 0.0), axis=1)
    lse = tl.where(total > 0, safe_m + tl.log(total), float("-inf"))
    reps = tl.arange(0, N_REP).to(tl.int64)
    tl.store(LSE + r * stride_l_r + (b * NSUB + c) * stride_l_c + h * stride_l_h + reps, lse)


@triton.jit
def _score_acc_kernel(
    Q, K_CACHE, K_BT, SEQ_LENS, LSE, ACC, ACC_BT,
    stride_q_r, stride_q_h, stride_k_b, stride_k_t, stride_kbt_r,
    stride_l_r, stride_l_h,
    stride_a_b, stride_a_t, stride_abt_r,
    scale, n_blocks,
    W: tl.constexpr, N_REP: tl.constexpr, BS_K: tl.constexpr, SUB: tl.constexpr, NSUB: tl.constexpr,
    BS_A: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    r, b, c, pos, tok, s = _chunk_scores(Q, K_CACHE, K_BT, SEQ_LENS, pid0, h,
                                         stride_q_r, stride_q_h, stride_k_b, stride_k_t, stride_kbt_r,
                                         scale, n_blocks, W, N_REP, BS_K, SUB, NSUB)
    reps = tl.arange(0, N_REP).to(tl.int64)
    lse = tl.load(LSE + r * stride_l_r + h * stride_l_h + reps)                   # [N_REP]
    p = tl.max(tl.where(tok[None, :], tl.exp(s - lse[:, None]), 0.0), axis=0)    # [SUB]
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
    max_seq_len: int | None = None,
) -> None:
    """Add this step's stage-1 score into acc_cache, in place. Layouts: module docstring.
    max_seq_len (CPU int) avoids a device sync when given."""
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
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max())
    n_blocks = (max_seq_len + BS_K - 1) // BS_K
    sub = min(SUB_CHUNK, BS_K)
    assert BS_K % sub == 0
    nsub = BS_K // sub
    n_rep = Hq // Hkv
    scale = 1.0 / math.sqrt(W)
    grid = (R * n_blocks * nsub, Hkv)
    common = dict(W=W, N_REP=n_rep, BS_K=BS_K, SUB=sub, NSUB=nsub)

    chunk_lse = torch.empty(R, n_blocks * nsub, Hkv, n_rep, dtype=torch.float32, device=q.device)
    _score_lse_kernel[grid](
        q, k_cache, k_block_table, seq_lens, chunk_lse,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_block_table.stride(0),
        chunk_lse.stride(0), chunk_lse.stride(1), chunk_lse.stride(2),
        scale, n_blocks, **common,
    )
    lse = torch.logsumexp(chunk_lse, dim=1).contiguous()                          # [R, Hkv, n_rep]
    _score_acc_kernel[grid](
        q, k_cache, k_block_table, seq_lens, lse, acc_cache, acc_block_table,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_block_table.stride(0),
        lse.stride(0), lse.stride(1),
        acc_cache.stride(0), acc_cache.stride(1), acc_block_table.stride(0),
        scale, n_blocks, BS_A=acc_cache.shape[1], **common,
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
