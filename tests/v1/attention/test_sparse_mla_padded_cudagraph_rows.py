# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded FULL-CUDA-graph spec-decode rows and the sparse-MLA top-k contract.

Regression for flashinfer-ai/flashinfer#5015 (same defect class as
vllm-project/vllm#51593, fixed by #51538 commits 6-7).

Scenario: MTP k=1 (``next_n == 2``), three live requests (6 token rows) replayed
through the 8-row FULL CUDA graph. vLLM materialises a fourth, padded request
with ``seq_len == 0``. Its two rows are CUDA-graph padding. The contract that
keeps the padded rows harmless is:

1. the DSA indexer's per-token context length for a padded request must be
   clamped to 0 (``0 - next_n + j + 1`` is negative for ``j < next_n - 1``);
2. ``persistent_topk`` must not wedge (or read out of range) if it is ever
   handed a negative length anyway;
3. with (1) in place the top-k emits an all ``-1`` row, and the SM120
   request->physical index conversion keeps it all ``-1`` so no physical cache
   index is published for a padding row.

Removing (1) at vLLM 487ecf187 makes ``test_uniform_decode_kernel...`` fail
with a ``-1`` in row 6 and, on a build without (2), makes
``test_persistent_topk_survives_padded_row_negative_length`` time out on a
wedged kernel (the #5015 signature: GPU pinned, host idle).

Runs without any model; needs one CUDA GPU.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.mla import indexer as indexer_mod
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)

NEXT_N = 2  # MTP k=1
NUM_REAL_REQS = 3
CAPTURE_TOKENS = 8  # vLLM default capture list 1,2,4,8 -> 6 real rows pad to 8
NUM_REQS_PADDED = CAPTURE_TOKENS // NEXT_N  # 4
CTX = 300  # short contexts, like the reporter (kv usage < 1%)
TOPK = 2048
BLOCK_SIZE = 64
POISON = 0x7FFFFFF0


def _seq_lens(device: torch.device) -> torch.Tensor:
    # Real requests have CTX tokens; the padded request has seq_len 0
    # (gpu_model_runner: ``self.seq_lens[num_reqs:].fill_(0)``).
    s = torch.full((NUM_REQS_PADDED,), CTX, dtype=torch.int32, device=device)
    s[NUM_REAL_REQS:] = 0
    return s


def _expected_per_token_lens() -> list[int]:
    out = []
    for r in range(NUM_REQS_PADDED):
        L = CTX if r < NUM_REAL_REQS else 0
        for j in range(NEXT_N):
            out.append(max(L - NEXT_N + j + 1, 0))
    return out


def _run_uniform_decode_kernel(
    seq_lens,
    decode_seq_lens,
    block_table,
    expanded_bt,
    decode_lens,
    num_tokens,
    max_decode_len,
):
    """Invoke the indexer's uniform-decode Triton kernel on either API
    (``PrepareUniformDecodeKernel`` wrapper on main, raw ``@triton.jit`` before)."""
    wrapped = getattr(indexer_mod, "_PREPARE_UNIFORM_DECODE_KERNEL", None)
    if wrapped is not None:
        wrapped(
            seq_lens,
            decode_seq_lens,
            block_table,
            expanded_bt,
            decode_lens,
            num_tokens,
            max_decode_len,
        )
        return
    indexer_mod._prepare_uniform_decode_kernel[(num_tokens,)](
        seq_lens,
        decode_seq_lens,
        block_table,
        block_table.stride(0),
        expanded_bt,
        expanded_bt.stride(0),
        decode_lens,
        max_decode_len,
        BLOCK_SIZE=1024,
    )


def test_uniform_decode_kernel_clamps_padded_request_context_len():
    """Contract (1), uniform/flatten path used outside SM100 with next_n <= 2."""
    device = torch.device("cuda")
    seq_lens = _seq_lens(device)
    block_table = torch.arange(
        1, 1 + NUM_REQS_PADDED * 8, dtype=torch.int32, device=device
    ).view(NUM_REQS_PADDED, 8)
    block_table[NUM_REAL_REQS:] = NULL_BLOCK_ID
    # Poison the destination buffers so a skipped store cannot pass by luck.
    decode_seq_lens = torch.full(
        (CAPTURE_TOKENS,), POISON, dtype=torch.int32, device=device
    )
    expanded_bt = torch.full(
        (CAPTURE_TOKENS, 8), POISON, dtype=torch.int32, device=device
    )
    decode_lens = torch.full(
        (CAPTURE_TOKENS,), POISON, dtype=torch.int32, device=device
    )

    _run_uniform_decode_kernel(
        seq_lens,
        decode_seq_lens,
        block_table,
        expanded_bt,
        decode_lens,
        CAPTURE_TOKENS,
        NEXT_N,
    )
    torch.cuda.synchronize()
    got = decode_seq_lens.tolist()
    print(f"\nper-token context lens (uniform kernel): {got}")
    assert got == _expected_per_token_lens(), (
        "padded request rows must have a non-negative (zero) context length; "
        f"got {got} — a negative value is read as ~4e9 by persistent_topk"
    )
    assert decode_lens.tolist() == [1] * CAPTURE_TOKENS
    assert (expanded_bt[NUM_REAL_REQS * NEXT_N :] == NULL_BLOCK_ID).all()


def test_native_next_n_seq_lens_clamped_for_padded_request():
    """Contract (1), native (B, next_n) path (SM90/SM100 style spec decode)."""
    device = torch.device("cuda")
    b = object.__new__(indexer_mod.DeepseekV32IndexerMetadataBuilder)
    b.decode_seq_lens_buffer = torch.full(
        (64,), POISON, dtype=torch.int32, device=device
    )
    b.offsets_buffer = torch.arange(NEXT_N, dtype=torch.int32, device=device)
    b.decode_lens_buffer = torch.zeros(64, dtype=torch.int32, device=device)
    b.expanded_block_table_buffer = torch.zeros(
        (64, 8), dtype=torch.int32, device=device
    )
    b.arange_buffer = torch.arange(64, dtype=torch.int32, device=device)
    b.supports_varlen = False
    b.vllm_config = SimpleNamespace(speculative_config=None)  # newer builders read it

    seq_lens = _seq_lens(device)
    block_table = torch.zeros((NUM_REQS_PADDED, 8), dtype=torch.int32, device=device)
    decode_lens = torch.full(
        (NUM_REQS_PADDED,), NEXT_N, dtype=torch.int32, device=device
    )
    qsl = torch.arange(0, CAPTURE_TOKENS + 1, NEXT_N, dtype=torch.int32, device=device)

    out_seq_lens, _, _, _, requires_padding = b._prepare_decode_tensors(
        seq_lens,
        block_table,
        decode_lens,
        decode_lens.cpu(),
        qsl,
        NUM_REQS_PADDED,
        CAPTURE_TOKENS,
        True,  # use_native
        NEXT_N,
        NEXT_N,  # max_decode_len
    )
    torch.cuda.synchronize()
    assert not requires_padding
    got = out_seq_lens.reshape(-1).tolist()
    print(f"\nper-token context lens (native path): {got}")
    assert got == _expected_per_token_lens(), got


_TOPK_CHILD = textwrap.dedent(
    """
    import sys, torch
    import vllm  # noqa: F401  (registers the torch.ops._C library)
    from vllm import _custom_ops  # noqa: F401
    rows, stride, topk = 8, 32768, 2048
    torch.manual_seed(5015)
    logits = torch.rand(rows, stride, device="cuda", dtype=torch.float32)
    # rows 0..5 real (context 300 -> per-token 299/300), rows 6..7 padded
    lengths = torch.tensor([299, 300, 299, 300, 299, 300, -1, 0],
                           dtype=torch.int32, device="cuda")
    out = torch.full((rows, topk), 0x7FFFFFF0, dtype=torch.int32, device="cuda")
    ws = torch.zeros(1024 * 1024, dtype=torch.uint8, device="cuda")
    # exactly sparse_attn_indexer's call: max_seq_len = logits.shape[1]
    torch.ops._C.persistent_topk(logits, lengths, out, ws, topk, logits.shape[1])
    torch.cuda.synchronize()
    pad = out[6:]
    real = out[:6]
    ok_pad = bool((pad == -1).all())
    ok_real = bool((real[real != -1] >= 0).all()) and all(
        int(((real[r] >= 0) & (real[r] < int(lengths[r]))).sum()) == int(lengths[r])
        for r in range(6)
    )
    print("TOPK_RESULT pad_all_minus1=%s real_ok=%s" % (ok_pad, ok_real))
    sys.exit(0 if ok_pad and ok_real else 4)
    """
)


def test_persistent_topk_survives_padded_row_negative_length():
    """Contract (2). A padded row with length -1 (what an unclamped indexer
    emits for ``0 - 2 + 0 + 1``) must neither wedge the kernel nor publish
    indices. ``max_seq_len == 32768 == RADIX_THRESHOLD`` is the reporter's
    ``--max-model-len``; it makes the non-leader CTAs take the host-side early
    exit, so an unclamped leader that lands on the radix path waits forever."""
    if not hasattr(torch.ops._C, "persistent_topk"):
        pytest.skip("vllm._C without persistent_topk")
    cmd = ["timeout", "-s", "KILL", "60", sys.executable, "-c", _TOPK_CHILD]
    p = subprocess.run(cmd, capture_output=True, text=True)
    print(p.stdout[-500:], p.stderr[-800:])
    assert p.returncode not in (-9, 137), (
        "persistent_topk wedged on a padded row with negative length "
        "(flashinfer-ai/flashinfer#5015 signature: GPU pinned, host idle)"
    )
    assert p.returncode == 0, f"rc={p.returncode}: {p.stdout[-300:]} {p.stderr[-800:]}"


def _conversion_inputs(device: torch.device, pad_logical: torch.Tensor):
    """Metadata exactly as FlashInferMLASparseSM120Impl.forward_mqa sees it
    for the 8-row replay: req_id_per_token, block_table, topk_indices_buffer."""
    req_id_per_token = torch.repeat_interleave(
        torch.arange(NUM_REQS_PADDED, dtype=torch.int32, device=device), NEXT_N
    )
    max_blocks = 8
    block_table = torch.arange(
        1, 1 + NUM_REQS_PADDED * max_blocks, dtype=torch.int32, device=device
    ).view(NUM_REQS_PADDED, max_blocks)
    block_table[NUM_REAL_REQS:] = (
        NULL_BLOCK_ID  # gpu_model_runner pads with the null block
    )
    topk = torch.full((CAPTURE_TOKENS, TOPK), -1, dtype=torch.int32, device=device)
    for r in range(NUM_REAL_REQS * NEXT_N):
        topk[r, :CTX] = torch.arange(CTX, dtype=torch.int32, device=device)
    topk[NUM_REAL_REQS * NEXT_N :] = pad_logical
    return req_id_per_token, block_table, topk


def _dump_rows(tag, req_id_per_token, topk, phys, lens):
    print(
        f"\n[{tag}] q.shape[0] (CUDA-graph extent) = {CAPTURE_TOKENS}, "
        f"actual tokens = {NUM_REAL_REQS * NEXT_N}"
    )
    for r in range(CAPTURE_TOKENS):
        print(
            f"  row {r}: req={int(req_id_per_token[r])} "
            f"valid={r < NUM_REAL_REQS * NEXT_N} "
            f"ctx_len={lens[r]} logical[:3]={topk[r, :3].tolist()} "
            f"n_logical_valid={int((topk[r] >= 0).sum())} "
            f"phys[:3]={phys[r, :3].tolist()} n_phys_valid={int((phys[r] >= 0).sum())}"
        )


def test_sm120_padded_rows_publish_no_physical_index():
    """Contract (3): with a clamped indexer (all -1 padded rows) the physical
    conversion at the padded CUDA-graph extent publishes no physical index for
    rows 6..7, and rows 0..5 are bitwise identical to the exact 6-row batch."""
    device = torch.device("cuda")
    pad = torch.full((NEXT_N, TOPK), -1, dtype=torch.int32, device=device)
    req_id_per_token, block_table, topk = _conversion_inputs(device, pad)
    phys8 = triton_convert_req_index_to_global_index(
        req_id_per_token, block_table, topk, BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=TOPK
    )
    phys6 = triton_convert_req_index_to_global_index(
        req_id_per_token[:6],
        block_table,
        topk[:6],
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_TOPK_TOKENS=TOPK,
    )
    torch.cuda.synchronize()
    _dump_rows(
        "clamped indexer", req_id_per_token, topk, phys8, _expected_per_token_lens()
    )
    assert torch.equal(phys8[:6], phys6), "valid rows changed by the padded extent"
    assert (phys8[6:] == -1).all(), "padding rows published physical cache indices"
    real = phys8[:6]
    assert ((real >= BLOCK_SIZE) | (real == -1)).all(), (
        "real rows must not resolve into the null block"
    )


def test_sm120_stale_padded_rows_would_publish_physical_indices():
    """Negative control documenting why contract (1) is load-bearing: if the
    top-k left stale in-range logical indices in the padded rows (e.g. the
    out-of-range garbage an unclamped radix pass emits when max_model_len >
    32768, or a previous batch's row), the conversion happily publishes
    physical slots for them — into the null block 0 here."""
    device = torch.device("cuda")
    stale = torch.randint(0, CTX, (NEXT_N, TOPK), dtype=torch.int32, device=device)
    req_id_per_token, block_table, topk = _conversion_inputs(device, stale)
    phys8 = triton_convert_req_index_to_global_index(
        req_id_per_token, block_table, topk, BLOCK_SIZE=BLOCK_SIZE, NUM_TOPK_TOKENS=TOPK
    )
    torch.cuda.synchronize()
    _dump_rows(
        "stale padded rows", req_id_per_token, topk, phys8, _expected_per_token_lens()
    )
    published = phys8[6:]
    assert (published >= 0).all(), "stale logical indices were unexpectedly masked"
    assert (published < BLOCK_SIZE).all(), "expected resolution into null block 0"
