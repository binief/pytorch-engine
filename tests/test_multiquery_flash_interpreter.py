"""Numerical regression: run with TRITON_INTERPRET=1 python -m tests.test_multiquery_flash_interpreter."""
import math
import torch
import triton
from pygguf_engine.kv_cache import (
    _int4_gqa_flash_decode_split_kernel, _int8_gqa_flash_decode_split_kernel,
    _flash_decode_combine_kernel,
)


def main():
    torch.manual_seed(123)
    heads, kvheads, dim, cap, splits = 4, 2, 32, 24, 4
    for quant in (4, 8):
        for length in (1, 2, 3, 8):
            for ring, end, window in ((False, 12, 0), (True, 42, 9)):
                start = max(0, end - (window + length - 1)) if ring else 0
                total = end - start
                q = torch.randn(heads, length, dim, dtype=torch.float16)
                if quant == 8:
                    k = torch.randint(-8, 9, (kvheads, cap, dim), dtype=torch.int8)
                    v = torch.randint(-8, 9, (kvheads, cap, dim), dtype=torch.int8)
                    ks = torch.rand(kvheads, cap, dtype=torch.float16) * 0.1 + 0.01
                    vs = torch.rand(kvheads, cap, dtype=torch.float16) * 0.1 + 0.01
                    kd = k.float() * ks.float().unsqueeze(-1)
                    vd = v.float() * vs.float().unsqueeze(-1)
                else:
                    k = torch.randint(0, 256, (kvheads, cap, dim // 2), dtype=torch.uint8)
                    v = torch.randint(0, 256, (kvheads, cap, dim // 2), dtype=torch.uint8)
                    ks = torch.rand(kvheads, cap, 2, dtype=torch.float16) * 0.1 + 0.01
                    vs = torch.rand(kvheads, cap, 2, dtype=torch.float16) * 0.1 + 0.01
                    kd = torch.cat(((k & 15).float().sub(8) * ks[..., :1].float(),
                                    (k >> 4).float().sub(8) * ks[..., 1:].float()), -1)
                    vd = torch.cat(((v & 15).float().sub(8) * vs[..., :1].float(),
                                    (v >> 4).float().sub(8) * vs[..., 1:].float()), -1)
                part = torch.empty(heads * length, splits, dim, dtype=torch.float32)
                lse = torch.empty(heads * length, splits, dtype=torch.float32)
                out = torch.empty(heads * length, dim, dtype=torch.float16)
                common = dict(WIN_RING=ring, Q_LEN=length, WINDOW_SIZE=window,
                              GQA_GROUP=heads // kvheads, NUM_SPLITS=splits, BLOCK_S=4)
                args = (q, k, ks, v, vs, part, lse, total, cap, start, 1 / math.sqrt(dim))
                if quant == 4:
                    _int4_gqa_flash_decode_split_kernel[(heads * length, splits)](*args, HALF_D=dim // 2, **common)
                else:
                    _int8_gqa_flash_decode_split_kernel[(heads * length, splits)](*args, HEAD_DIM=dim, **common)
                _flash_decode_combine_kernel[(heads * length,)](part, lse, out, NUM_SPLITS=splits, HEAD_DIM=dim)
                ref = torch.empty(heads, length, dim)
                for h in range(heads):
                    for j in range(length):
                        qend = end - length + j + 1
                        qstart = max(start, qend - window) if ring else 0
                        indices = torch.arange(qstart, qend) % cap
                        hkv = h // (heads // kvheads)
                        scores = kd[hkv, indices] @ q[h, j].float() / math.sqrt(dim)
                        ref[h, j] = torch.softmax(scores, 0) @ vd[hkv, indices]
                err = (out.float().view_as(ref) - ref).abs().max().item()
                assert err < 0.001, (quant, length, ring, err)
    print('Multi-query INT4/INT8 flash attention: causal and wrapped-window parity passed')


if __name__ == '__main__':
    main()
