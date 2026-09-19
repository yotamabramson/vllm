# SPDX-License-Identifier: Apache-2.0
"""Refresh-time selection: which 16-token blocks each KV head keeps resident.

Harness behavior (cascaded-attention repo, _periodic_refresh.py):
  - score per block = mean of the accumulated per-token score over its 16 tokens
    (_topk_block_ids); the accumulator is a running sum, and scaling does not
    change top-K
  - blocks kept per KV head = max(1, capacity_tokens // 16), capacity =
    mass-calibrated capacity x margin
  - recency floor of the last 1000 tokens always resident

One adaptation, needed because each head's resident set is laid out in fixed
slots (towards_70B_vllm/README.md, "Per-request working pages"): top-K and floor
must be disjoint by construction, as the batched harness already does by masking
the floor out of top-K. The floor therefore starts on a 16-token boundary
(floor_start = (n - 1000) // 16 * 16, so 1000..1015 tokens) and only blocks
entirely before it are candidates. Relative to the block-16 harness (which
unions top-K with the floor), a head can only gain tokens, never lose one.

Imports nothing from vLLM (torch only).
"""

import torch

FLOOR_MIN = 1000
SEL_BLOCK = 16


def floor_start(n: int, floor_min: int = FLOOR_MIN, block: int = SEL_BLOCK) -> int:
    """First token of the recency floor for a request with n tokens."""
    return max(0, (n - floor_min) // block * block)


def blocks_per_head(caps_tokens: torch.Tensor, n: int, floor_min: int = FLOOR_MIN,
                    block: int = SEL_BLOCK) -> torch.Tensor:
    """CPU long tensor [Hkv]: blocks each head keeps, clamped to the candidates available."""
    n_cand = floor_start(n, floor_min, block) // block
    if n_cand == 0:
        return torch.zeros_like(caps_tokens)
    return (caps_tokens // block).clamp(min=1, max=n_cand)


def select_blocks(acc: torch.Tensor, n: int, caps_tokens: torch.Tensor, floor_min: int = FLOOR_MIN,
                  block: int = SEL_BLOCK, n_apply: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """acc: [n, Hkv] float32 accumulated score (GPU). caps_tokens: [Hkv] long capacity in tokens (CPU).

    Returns (block_ids [Hkv, K_max] long on acc's device, ascending per head, -1 past the head's count;
             counts [Hkv] long on CPU; floor_start). No GPU->CPU sync: counts come from CPU inputs.

    n_apply (default n): the sequence length the selection will be SWAPPED IN at, which is
    later than n when the selection runs early (VLLM_CASCADE_G > 0). The floor boundary
    has to be the one in force at SWAP time: the frame is one contiguous range per group
    with no mask, so the selected blocks and the floor must not overlap, and it is the
    swap-time floor they will sit next to.

    The floor slides forward as the sequence grows, so candidates are blocks before
    floor_start(n_apply) -- a LARGER set than floor_start(n) would give, not a smaller
    one. The blocks that gap adds, [floor_start(n), floor_start(n_apply)), are tokens the
    floor still covers at selection time but will have dropped by the swap: exactly the
    ones that have to be selected if they are to stay resident. They are fetched from the
    CPU store like any other selected block.

    `acc` must cover those candidates, which holds because the gap is far below floor_min
    (G < p <= a few hundred, floor_min = 1000), so floor_start(n_apply) < n.
    """
    fs = floor_start(n if n_apply is None else n_apply, floor_min, block)
    assert fs <= n, (f"cascade: selecting at n={n} for apply at n={n_apply} needs scores up to "
                     f"token {fs}, which do not exist yet -- the gap must stay well below floor_min")
    n_cand_avail = fs // block
    counts = (torch.zeros_like(caps_tokens) if n_cand_avail == 0
              else (caps_tokens // block).clamp(min=1, max=n_cand_avail))
    Hkv = acc.shape[1]
    k_max = int(counts.max()) if counts.numel() else 0
    if k_max == 0:
        return torch.full((Hkv, 0), -1, dtype=torch.long, device=acc.device), counts, fs
    n_cand = fs // block
    block_scores = acc[: n_cand * block].reshape(n_cand, block, Hkv).mean(dim=1)   # [n_cand, Hkv]
    top = block_scores.topk(k_max, dim=0).indices.T                                  # [Hkv, k_max], best first
    valid = torch.arange(k_max, device=acc.device)[None, :] < counts.to(acc.device)[:, None]
    sentinel = torch.iinfo(torch.long).max
    ids = torch.where(valid, top, sentinel).sort(dim=1).values                      # keep the best counts[h], ascending
    ids[ids == sentinel] = -1
    return ids, counts, fs
