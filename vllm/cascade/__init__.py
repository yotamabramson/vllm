# SPDX-License-Identifier: Apache-2.0
"""Cascaded attention (periodic-refresh KV screening) for Llama.

Design: cascaded-attention repo, towards_70B_vllm/README.md, "Stage 1 design".
Behavior mirrors that repo's harness (_periodic_refresh.py,
periodic_attention_step_batched with fast bookkeeping); the harness is the
ground truth this package is checked against.

Enabled per process by environment variables:
  VLLM_CASCADE=1                      turn it on (Llama only)
  VLLM_CASCADE_PROJECTIONS=<path.pt>  trained screening projections, the
                                      harness's trained_projections_*.pt
                                      format: {32: {"w_q": [L, Hq, D, 32],
                                                    "w_k": [L, Hkv, D, 32]}}
                                      or hf:<dataset repo>/<file> to download
                                      it (uses HF_TOKEN)
"""

import os

THIN_WIDTH = 32


def is_enabled() -> bool:
    return os.environ.get("VLLM_CASCADE", "0") == "1"


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
