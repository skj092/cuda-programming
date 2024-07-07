import torch
import time
import math
import re
from types import SimpleNamespace as ns
from pathlib import Path
from torch.utils.cpp_extension import load_inline


def blk_kernel2d(f, blocks, threads, *args):
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            for j0 in range(threads.y):
                for j1 in range(threads.x):
                    # print(f"blockidx: {i0, i1}, threadidx: {j0, j1}")
                    f(ns(x=i1, y=i0), ns(x=j1, y=j0), threads, *args)


def matmul_bk(blockidx, threadidx, blockdim, m, n, out, h, w, k):
    r = blockidx.y*blockdim.y + threadidx.y
    c = blockidx.x*blockdim.x + threadidx.x

    if (r >= h or c >= w):
        return
    o = 0.
    for i in range(k):
        o += m[r*k+i] * n[i*w+c]
    out[r*w+c] = o


def mm_py():
    h, k, k2, w = 100, 100, 100, 100
    A = torch.ones((h, k))
    B = torch.ones((k2, w))
    A = torch.tensor([i for i in range(1, h*k+1)]).reshape(h, k)
    B = torch.tensor([i for i in range(1, k2*w+1)]).reshape(k2, w)
    tik = time.time()
    C = matmul_2d(A, B)
    tok = time.time()
    print(f"time take in manual multiplication: {tok-tik}")

    tik = time.time()
    D = torch.mm(A, B)
    tok = time.time()
    print(f"time take in torch multiplication: {tok-tik}")

    print(torch.isclose(C, D).all())


def matmul_2d(m, n):
    h, k = m.shape
    k2, w = n.shape
    assert k == k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = ns(x=16, y=16)
    blocks = ns(x=math.ceil(w/tpb.x), y=math.ceil(h/tpb.y))
    blk_kernel2d(matmul_bk, blocks, tpb,
                 m.flatten(), n.flatten(), output.flatten(), h, w, k)
    return output


# ============CUDA================
# =================================

def compile_extension():
    cuda_source = Path("multiply_kernel_2d.cu").read_text()
    cpp_source = "torch::Tensor matmul(torch::Tensor m, torch::Tensor n);"

    # Load the CUDA kernel as a PyTorch extension
    module = load_inline(
        name="multiply_in_cuda_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["matmul"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return module


def matmul_2d_cuda():
    h, k, k2, w = 100, 100, 100, 100
    m = torch.ones((h, k)).cuda()
    n = torch.ones((k2, w)).cuda()
    assert k == k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = ns(x=16, y=16)
    blocks = ns(x=math.ceil(w/tpb.x), y=math.ceil(h/tpb.y))
    module = compile_extension()
    tik = time.time()
    module.matmul(m, n)
    tok = time.time()
    print(f"time taken : {tok-tik:.4f}")
    return output


def main():
    mm_py()
    matmul_2d_cuda()


if __name__ == "__main__":
    main()

