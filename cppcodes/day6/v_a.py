import torch
from collections import namedtuple

dim3 = namedtuple('dim3', ['x', 'y', 'z'], defaults=(1, 1))


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b


N = 8
a = torch.arange(N).float()
b = torch.arange(N, N + N).float()


def blk_kernel2d_shar(f, blocks, threads, sh_sz, *args, **kwargs):
    for i1 in range(blocks.x):
        shared = torch.zeros(sh_sz)
        f(dim3(i1), threads, shared, *args, **kwargs)


def matmul_tiled_bk(blockIdx, blockDim, shared, m, n, out, N, tw):
    ms, ns = shared[:tw], shared[tw:]

    # update ms and ns
    for tc in range(N/tw):
        idx = blockIdx.x
        ms[tc] =
        ns[tc] =


def matmul_2d(m, n, tw=4):
    k = len(m)
    k2 = len(n)
    assert k == k2, "Size mismatch!"
    output = torch.zeros(k, dtype=m.dtype)
    tpb = dim3(tw)
    blocks = dim3(cdiv(k, tpb.x))
    blk_kernel2d_shar(matmul_tiled_bk, blocks, tpb, tw*2,
                      m.flatten(), n.flatten(), output.flatten(), N, tw=tw)
    return output


print(torch.isclose(matmul_2d(a, b), a+b).all())
