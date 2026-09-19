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
  VLLM_CASCADE_AGG=mean|last|stride:N  (full) which steps the stage-1 score runs on
  VLLM_CASCADE_G=0                     (full) decode steps between selecting a working set
                                       and swapping it in, so the fetch has that long to
                                       run off the critical path (see gap() below)
  VLLM_CASCADE_ASYNC_FLUSH=0           (full) flush to the CPU store off the critical path
  VLLM_CASCADE_CPU_SEQS=4              (full) sequences the pinned CPU store holds
Paths may be hf:<owner>/<repo>/<file> (HF dataset repo, uses HF_TOKEN).

Timing-only stand-ins, for a box with no HF credentials (they change what is
selected, never the shapes or the work done, so speed is real and accuracy is
meaningless -- never use them for a correctness or accuracy run):
  VLLM_CASCADE_PROJECTIONS=random              deterministic random projections
  VLLM_CASCADE_CAPACITIES=synthetic[:L,Hkv]    capacities whose post-margin sizes
                                               reproduce the real margin-0.3 spread
                                               (7,792 / 15,329 / 17,568 tokens per
                                               group, towards_70B_vllm/README.md)
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
    if value == "random":
        return value
    return resolve_path(value)


def aggregation() -> tuple[str, int]:
    """VLLM_CASCADE_AGG: which decode steps the stage-1 score runs on.

      mean     (default) every step, averaged over the window -- the harness's behavior
      stride:N every Nth step of the window, including the refresh step
      last     the refresh step only -- 1 step in 64 at p=64

    Measured with new_degisn/exp_agg_variants.py. On llama3.1-8b at ctx=64000, margin 0.3,
    p=64 -- the configuration the fork actually serves -- `last` costs 0.19 points of
    coverage against 56.19 for mean, and stride:8 costs 0.01; on qwen2.5-7b at ctx=8000 the
    same comparison cost 0.26 and 0.02. `last` is therefore the intended setting.
    The divisor never matters: every token of a group is divided by the same count, and
    top-K is invariant to that. Normalization WITHIN a step still does matter (the score
    maxes over query heads that each have their own softmax denominator), so the kernel
    keeps both of its passes -- it just runs on fewer steps.

    Returns (name, stride) where stride is the step period, p for "last".
    """
    value = os.environ.get("VLLM_CASCADE_AGG", "mean")
    if value == "mean":
        return "mean", 1
    if value == "last":
        return "last", refresh_period()
    if value.startswith("stride:"):
        n = int(value.split(":", 1)[1])
        assert n >= 1 and refresh_period() % n == 0, (
            f"VLLM_CASCADE_AGG=stride:{n} must divide p={refresh_period()}")
        return "stride", n
    raise ValueError(f"VLLM_CASCADE_AGG must be mean, last or stride:N, got {value!r}")


def refresh_period() -> int:
    return int(os.environ.get("VLLM_CASCADE_P", "64"))


def gap() -> int:
    """VLLM_CASCADE_G: decode steps between selecting a working set and swapping it in.

    G=0 (default) is the behavior this package had before the option existed: the refresh
    step scores, selects, fetches from the CPU store and attends, all in one forward pass,
    so the step waits for the whole round trip (~15.4 ms/step amortised, measured).

    G>0 splits that in two. The selection runs at q = p - G steps into the window, off
    this step's own scores; the fetch is issued asynchronously (async_io.py) and the swap
    happens G steps later, at the refresh step, which only waits on an event that has
    almost certainly already completed. The transfer gets G decode steps (~40 ms each) to
    finish instead of zero.

    What it costs is staleness: the scores that choose the resident set are G steps older
    than they would have been. Nothing else changes -- same selection rule, same capacity,
    same floor. Pick G just large enough to cover the fetch (the measured refresh is ~1 s
    per event at N=8 spread over 32 layers, so a handful of steps), not p/2: staleness is
    paid one-for-one with transfer time, and there is no benefit to a bigger gap than the
    transfer needs.

    Only the gap matters, not where the selection sits in the window: the scores in use at
    any moment are between G and G + p - 1 steps old, so (p, q) and (p, G) describe the
    same grid. The first refresh after prefill has no earlier step to select at and stays
    synchronous whatever G is.

    Untested on a GPU: no accuracy or speed number exists for G>0 yet.
    """
    value = int(os.environ.get("VLLM_CASCADE_G", "0"))
    p = refresh_period()
    assert 0 <= value < p, f"VLLM_CASCADE_G={value} must be in [0, p={p})"
    return value


def async_flush() -> bool:
    """VLLM_CASCADE_ASYNC_FLUSH=1: send the flush (GPU -> CPU store) down a side stream and
    scatter it on the worker thread, instead of blocking the step on it.

    Independent of the gap, and free of any quality effect: the flush only writes tokens
    the CPU store will need much later (a token is only fetchable once it falls out of the
    ~1000-token recency floor). Measured cost of the blocking version: 6.8 ms of wall time
    per step against 0.1 ms of GPU time. Off by default only because it has not run on a
    GPU yet."""
    return os.environ.get("VLLM_CASCADE_ASYNC_FLUSH", "0") == "1"


def uses_async() -> bool:
    return is_full() and (gap() > 0 or async_flush())


def working_start() -> int:
    """First working slot of the selected blocks (after the floor and the tokens since refresh)."""
    return FLOOR_SLOTS + refresh_period()


@functools.lru_cache(maxsize=1)
def capacities() -> tuple[tuple[int, ...], ...]:
    """Per-layer, per-KV-head capacity in tokens: harness margin_capacities(base, margin, ctx)."""
    value = os.environ.get("VLLM_CASCADE_CAPACITIES")
    if not value:
        raise ValueError("VLLM_CASCADE_MODE=full requires VLLM_CASCADE_CAPACITIES=<path.json or hf:...>")
    margin = float(os.environ.get("VLLM_CASCADE_MARGIN", "0.3"))
    if value.split(":")[0] == "synthetic":
        base = _synthetic_base(value, margin)
    else:
        with open(resolve_path(value)) as f:
            base = json.load(f)
    ctx = int(os.environ.get("VLLM_CASCADE_CALIB_CTX", "64000"))
    return tuple(tuple(min(ctx, math.ceil(k * margin)) for k in row) for row in base)


def _synthetic_base(value: str, margin: float) -> list[list[float]]:
    """Capacities for a timing run with no capacities file: the same shape and the same
    spread of working-set sizes as the real margin-0.3 calibration, so every tensor and
    every kernel launch is the size it would really be. Deterministic, no randomness.

    Real margin-0.3 sizes: 7,792 min / 15,329 mean / 17,568 max tokens per KV group.
    u^q over a uniform u has mean 1/(q+1), so q fixes the mean between min and max.
    """
    lo, mean, hi = 7792, 15329, 17568
    dims = value.split(":", 1)[1] if ":" in value else "32,8"
    layers, heads = (int(x) for x in dims.split(","))
    n = layers * heads
    q = (hi - lo) / (mean - lo) - 1.0
    sizes = [lo + (hi - lo) * (i / (n - 1)) ** q for i in range(n)]
    return [[sizes[l * heads + h] / margin for h in range(heads)] for l in range(layers)]


def working_slots() -> int:
    """Slots per request per layer in the working pages: floor + tokens since refresh + largest selection."""
    max_selected = max(max(1, c // SEL_BLOCK) * SEL_BLOCK for row in capacities() for c in row)
    return working_start() + max_selected


def cpu_pool_seqs() -> int:
    return int(os.environ.get("VLLM_CASCADE_CPU_SEQS", "4"))


def timing() -> bool:
    """VLLM_CASCADE_TIMING=1: log where each decode step's time goes, once per refresh period."""
    return os.environ.get("VLLM_CASCADE_TIMING", "0") == "1"


def signature() -> str:
    """Everything about the cascade configuration that changes the compiled graph or the
    shapes the attention op works on.

    vLLM's compile-cache key is built from its OWN environment variables and config, so it
    cannot see ours: a stock engine and a cascade engine sharing a cache directory produce
    the same key, and whichever compiles first wins. The other then silently executes the
    wrong graph -- a cascade engine running stock's graph attends over a KV cache truncated
    to the working budget and emits fluent nonsense, with no error anywhere
    (towards_70B_vllm/README.md, "The compile-cache collision"). This string goes into that
    key so the two can never collide.
    """
    if not is_enabled():
        return "cascade:off"
    parts = [mode(), f"p={refresh_period()}", f"agg={os.environ.get('VLLM_CASCADE_AGG', 'mean')}",
             f"g={os.environ.get('VLLM_CASCADE_G', '0')}"]
    if mode() == "full":
        parts += [f"margin={os.environ.get('VLLM_CASCADE_MARGIN', '0.3')}",
                  f"ctx={os.environ.get('VLLM_CASCADE_CALIB_CTX', '64000')}",
                  f"proj={os.environ.get('VLLM_CASCADE_PROJECTIONS', '')}",
                  f"caps={os.environ.get('VLLM_CASCADE_CAPACITIES', '')}"]
    return "cascade:" + "|".join(parts)


def debug() -> bool:
    """VLLM_CASCADE_DEBUG=1: log per-step plans and layer-0 consistency checks (slow)."""
    return os.environ.get("VLLM_CASCADE_DEBUG", "0") == "1"
