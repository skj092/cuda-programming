import torch
from collections import namedtuple

# os.environ['CUDA_LAUNCH_BLOCKING']='1'
torch.manual_seed(42)

dim3 = namedtuple('dim3', ['x', 'y', 'z'], defaults=(1, 1))


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b


def blk_kernel2d(f, blocks, threads, *args):
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            for j0 in range(threads.y):
                for j1 in range(threads.x):
                    f(dim3(i1, i0), dim3(j1, j0), threads, *args)


def matmul_bk(blockIdx, threadIdx, blockDim, m, n, out, h, w, k):
    r = blockIdx.y*blockDim.y + threadIdx.y
    c = blockIdx.x*blockDim.x + threadIdx.x

    if (r >= h or c >= w):
        return
    o = 0.
    for i in range(k):
        o += m[r*k+i] * n[i*w+c]
    out[r*w+c] = o


def matmul_2d(m, n):
    h, k = m.shape
    k2, w = n.shape
    assert k == k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = dim3(16, 16)
    blocks = dim3(cdiv(w, tpb.x), cdiv(h, tpb.y))
    blk_kernel2d(matmul_bk, blocks, tpb,
                 m.flatten(), n.flatten(), output.flatten(), h, w, k)
    return output


m1 = torch.rand(5120, 256)
m1s = m1[:4]
m2 = torch.rand(256, 5120)
m2s = m2[:, :4]
print(torch.isclose(matmul_2d(m1s, m2s), m1s@m2s).all())
