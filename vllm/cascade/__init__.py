# SPDX-License-Identifier: Apache-2.0
"""Cascaded attention (periodic-refresh KV screening) for Llama.

Design: cascaded-attention repo, towards_70B_vllm/README.md, "Stage 1 design".
Behavior mirrors that repo's harness (_periodic_refresh.py,
periodic_attention_step_batched with fast bookkeeping); the harness is the
ground truth this package is checked against.

Enabled per process by environment variables (read in every vLLM process, so
the scheduler and the workers agree):
  VLLM_CASCADE=1                       turn it on (Llama only)
  VLLM_CASCADE_MODE=noop|full          noop (default): thin Q/K computed, attention
                                       stock -- must match stock vLLM exactly.
                                       full: CPU KV store + GPU working set.
  VLLM_CASCADE_PROJECTIONS=<path|hf:>  trained screening projections, the harness's
                                       trained_projections_*.pt format:
                                       {32: {"w_q": [L, Hq, D, 32], "w_k": [L, Hkv, D, 32]}}
  VLLM_CASCADE_CAPACITIES=<path|hf:>   (full) mass-calibrated capacities JSON [L][Hkv]
  VLLM_CASCADE_MARGIN=0.3              (full) capacity margin
  VLLM_CASCADE_CALIB_CTX=64000         (full) context the capacities were calibrated at
  VLLM_CASCADE_P=64                    (full) refresh period
  VLLM_CASCADE_CPU_SEQS=4              (full) sequences the pinned CPU store holds
Paths may be hf:<owner>/<repo>/<file> (HF dataset repo, uses HF_TOKEN).
Prefix caching must be disabled in full mode.
"""

import functools
import json
import math
import os

THIN_WIDTH = 32
SEL_BLOCK = 16
# The recency floor holds 1000..1015 tokens (ops/select.py floor_start), so 1016 slots.
FLOOR_SLOTS = 1016


def is_enabled() -> bool:
    return os.environ.get("VLLM_CASCADE", "0") == "1"


def mode() -> str:
    value = os.environ.get("VLLM_CASCADE_MODE", "noop")
    assert value in ("noop", "full"), f"VLLM_CASCADE_MODE must be noop or full, got {value!r}"
    return value


def is_full() -> bool:
    return is_enabled() and mode() == "full"


def resolve_path(value: str) -> str:
    """Local path as-is; hf:<owner>/<repo>/<file> downloaded from that HF dataset repo."""
    if not value.startswith("hf:"):
        return value
    owner, repo, filename = value[len("hf:"):].split("/", 2)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(f"{owner}/{repo}", filename, repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))


def projections_path() -> str:
    value = os.environ.get("VLLM_CASCADE_PROJECTIONS")
    if not value:
        raise ValueError("VLLM_CASCADE=1 requires VLLM_CASCADE_PROJECTIONS=<path.pt or hf:owner/repo/file>")
    return resolve_path(value)


def refresh_period() -> int:
    return int(os.environ.get("VLLM_CASCADE_P", "64"))


def working_start() -> int:
    """First working slot of the selected blocks (after the floor and the tokens since refresh)."""
    return FLOOR_SLOTS + refresh_period()


@functools.lru_cache(maxsize=1)
def capacities() -> tuple[tuple[int, ...], ...]:
    """Per-layer, per-KV-head capacity in tokens: harness margin_capacities(base, margin, ctx)."""
    value = os.environ.get("VLLM_CASCADE_CAPACITIES")
    if not value:
        raise ValueError("VLLM_CASCADE_MODE=full requires VLLM_CASCADE_CAPACITIES=<path.json or hf:...>")
    with open(resolve_path(value)) as f:
        base = json.load(f)
    margin = float(os.environ.get("VLLM_CASCADE_MARGIN", "0.3"))
    ctx = int(os.environ.get("VLLM_CASCADE_CALIB_CTX", "64000"))
    return tuple(tuple(min(ctx, math.ceil(k * margin)) for k in row) for row in base)


def working_slots() -> int:
    """Slots per request per layer in the working pages: floor + tokens since refresh + largest selection."""
    max_selected = max(max(1, c // SEL_BLOCK) * SEL_BLOCK for row in capacities() for c in row)
    return working_start() + max_selected


def cpu_pool_seqs() -> int:
    return int(os.environ.get("VLLM_CASCADE_CPU_SEQS", "4"))
