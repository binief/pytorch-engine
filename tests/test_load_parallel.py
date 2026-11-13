"""Parallel row-split weight-load tests (PYGGUF_LOAD_WORKERS).

Guarantees:
  * row-split repack is bitwise-identical to the serial path (GGML blocks never span rows);
  * from_gguf honors the auto threshold and the workers env override;
  * _load_workers parses/clamps PYGGUF_LOAD_WORKERS correctly.
"""

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

import gguf as gq

import pygguf_engine.quant_kernels as qk
from pygguf_engine import GGUFFile
from pygguf_engine.quant_kernels import (
    GGUFQuantLinear,
    _load_workers,
    _repack_parallel,
    repack_gguf_to_w4a16,
    repack_gguf_to_w8a16,
    SUB_5BIT_W4_QTYPES,
)


def _make_raw_quant_tensor(
    rng: np.random.Generator,
    shape: Tuple[int, ...],
    qtype: gq.GGMLQuantizationType,
) -> np.ndarray:
    """Valid raw GGUF quantized tensor bytes for the formats exercised here."""
    if qtype in (gq.GGMLQuantizationType.Q4_0, gq.GGMLQuantizationType.Q8_0):
        w_f32 = (rng.standard_normal(shape, dtype=np.float32) * 0.1).astype(np.float32)
        return gq.quantize(w_f32, qtype)

    cols = shape[-1]
    rows = int(np.prod(shape[:-1]))
    if qtype == gq.GGMLQuantizationType.Q4_K:
        n_blocks = rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 144), dtype=np.uint8)
        raw[:, 0:2] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        raw[:, 2:4] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(rows, (cols // 256) * 144)

    if qtype == gq.GGMLQuantizationType.Q5_K:
        n_blocks = rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 176), dtype=np.uint8)
        raw[:, 0:2] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        raw[:, 2:4] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(rows, (cols // 256) * 176)

    if qtype == gq.GGMLQuantizationType.Q6_K:
        n_blocks = rows * (cols // 256)
        raw = rng.integers(0, 256, size=(n_blocks, 210), dtype=np.uint8)
        raw[:, 208:210] = (rng.uniform(0.0001, 0.0005, size=(n_blocks, 1)).astype(np.float16)).view(np.uint8)
        return raw.reshape(rows, (cols // 256) * 210)

    raise ValueError(f"Unsupported test qtype: {qtype}")


def _serial_reference(raw_flat: np.ndarray, qtype: int, rows: int, cols: int):
    use_w4 = qtype in SUB_5BIT_W4_QTYPES and cols % 32 == 0 and (
        qtype != int(gq.GGMLQuantizationType.Q4_K) or cols % 256 == 0
    )
    if use_w4:
        return repack_gguf_to_w4a16(raw_flat, qtype, rows, cols, torch.device("cpu"))
    return repack_gguf_to_w8a16(raw_flat, qtype, rows, cols, torch.device("cpu"))


@pytest.mark.parametrize(
    "qtype",
    [
        gq.GGMLQuantizationType.Q4_0,
        gq.GGMLQuantizationType.Q8_0,
        gq.GGMLQuantizationType.Q4_K,
        gq.GGMLQuantizationType.Q5_K,
        gq.GGMLQuantizationType.Q6_K,
    ],
)
def test_row_split_repack_bitwise_matches_serial(qtype: gq.GGMLQuantizationType) -> None:
    rng = np.random.default_rng(123)
    rows, cols = 20, 512  # deliberately not divisible by the worker count below
    raw = _make_raw_quant_tensor(rng, (rows, cols), qtype)
    raw_flat = raw.reshape(-1)

    ref_qw, ref_sc, ref_zr = _serial_reference(raw_flat, int(qtype), rows, cols)
    got_qw, got_sc, got_zr = _repack_parallel(raw_flat, int(qtype), rows, cols, torch.device("cpu"), workers=3)

    assert torch.equal(ref_qw, got_qw), f"qweight mismatch on {qtype.name}"
    assert torch.equal(ref_sc, got_sc), f"scales mismatch on {qtype.name}"
    if (ref_zr is None) != (got_zr is None):
        pytest.fail(f"zeros presence mismatch on {qtype.name}")
    if ref_zr is not None:
        assert torch.equal(ref_zr, got_zr), f"zeros mismatch on {qtype.name}"


def _write_single_tensor_gguf(path: Path, qb: np.ndarray, qtype: gq.GGMLQuantizationType) -> None:
    writer = gq.GGUFWriter(str(path), "qwen3")
    writer.add_tensor("token_embd.weight", qb, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.mark.parametrize(
    "qtype",
    [gq.GGMLQuantizationType.Q4_K, gq.GGMLQuantizationType.Q6_K],
)
def test_from_gguf_row_split_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qtype: gq.GGMLQuantizationType) -> None:
    rng = np.random.default_rng(7)
    rows, cols = 24, 512
    qb = _make_raw_quant_tensor(rng, (rows, cols), qtype)
    path = tmp_path / "split_candidate.gguf"
    _write_single_tensor_gguf(path, qb, qtype)

    with GGUFFile(path) as gf:
        # Serial path (threshold effectively unreachable)
        serial = GGUFQuantLinear.from_gguf(gf, "token_embd.weight", device=torch.device("cpu"), workers=1)
    with GGUFFile(path) as gf:
        # Force row-split with 3 workers on a tiny tensor
        monkeypatch.setattr(qk, "_ROW_SPLIT_MIN_BYTES", 1)
        monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "3")
        split = GGUFQuantLinear.from_gguf(gf, "token_embd.weight", device=torch.device("cpu"))

    assert torch.equal(serial.qweight, split.qweight)
    assert torch.equal(serial.scales, split.scales)
    if serial.zeros is None:
        assert split.zeros is None
    else:
        assert torch.equal(serial.zeros, split.zeros)


def test_load_workers_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYGGUF_LOAD_WORKERS", raising=False)
    assert _load_workers() == max(1, min(4, os.cpu_count() or 1))
    monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "1")
    assert _load_workers() == 1
    monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "3")
    assert _load_workers() == 3
    monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "999")
    assert _load_workers() == 16
    monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "0")
    assert _load_workers() == 1
    monkeypatch.setenv("PYGGUF_LOAD_WORKERS", "not-a-number")
    assert _load_workers() == max(1, min(4, os.cpu_count() or 1))
