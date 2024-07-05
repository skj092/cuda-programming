import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline
import time
import math

# Python code


def blk_kernal(f, blocks, threads, *args):
    for i in range(blocks):
        for j in range(threads):
            f(i, j, threads, *args)


def adder(blockidx, threadidx, blockdim, x, y, res, n):
    i = blockidx*blockdim + threadidx
    if i < n:
        res[i] = x[i] + y[i]


def add(x, y):
    h, w = x.shape
    res = torch.empty(h*w)
    x = x.view(-1)
    y = y.view(-1)
    n = h * w
    threads = 256
    blocks = int(math.ceil(h*w/threads))
    blk_kernal(adder, blocks, threads, x, y, res, n)
    return res.view(h, w)


def compile_extension():
    cuda_source = Path("add_kernel.cu").read_text()
    cpp_source = "torch::Tensor add_in_cuda(torch::Tensor tensor1, torch::Tensor tensor2);"

    # Load the CUDA kernel as a PyTorch extension
    add_in_cuda_extension = load_inline(
        name="add_in_cuda_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["add_in_cuda"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return add_in_cuda_extension


def main():
    ext = compile_extension()

    h, w = 5*4**3, 5*4**3
    a = torch.rand((h, w))
    b = torch.rand((h, w))
    a = a.to(device='cuda')
    b = b.to(device='cuda')
    tik = time.time()
    res = ext.add_in_cuda(a, b).cpu()
    tok = time.time()
    print(f"Time taken by cuda code: {tok-tik: .4f}")
    h, w = res.shape
    print(h, w, h*w)

    # CPU code
    tik = time.time()
    res = add(a, b)
    tok = time.time()
    print(f"Time taken by cpu code: {tok-tik: .4f}")


if __name__ == "__main__":
    main()
