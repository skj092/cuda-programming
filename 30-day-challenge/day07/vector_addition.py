import torch
from collections import namedtuple

dim3 = namedtuple('dim3', ['x', 'y', 'z'], defaults=(1, 1))


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b


m1 = torch.rand(5120, 256)
m1s = m1[:4]
m2 = torch.rand(256, 5120)
m2s = m2[:, :4]
m1s = torch.arange(16).float()
m2s = torch.arange(16).float()


def blk_kernel2d_shar(f, blocks, threads, sh_sz, *args, **kwargs):
    for i1 in range(blocks.x):
        shared = torch.zeros(sh_sz)
        f(dim3(i1), threads, shared, *args, **kwargs)


def matadd_tiled_bk(blockIdx, blockDim, shared, m, n, out, k, tw):
    shar_sz = tw
    ms, ns = shared[:shar_sz], shared[shar_sz:]

    # Load data into shared memory
    for tc in range(blockDim.x):
        c = blockIdx.x * blockDim.x + tc
        if c < k:
            ms[tc] = m[c]
            ns[tc] = n[c]
        else:
            ms[tc] = 0.0
            ns[tc] = 0.0

    # Perform addition and store result
    for tc in range(blockDim.x):
        c = blockIdx.x * blockDim.x + tc
        if c < k:
            out[c] = ms[tc] + ns[tc]


def matadd_2d(m, n, tw=4):
    k = len(m)
    k2 = len(n)
    assert k == k2, "Size mismatch!"
    output = torch.zeros(k, dtype=m.dtype)
    tpb = dim3(tw, tw)
    blocks = dim3(cdiv(k, tpb.x))
    blk_kernel2d_shar(matadd_tiled_bk, blocks, tpb, tw*2,
                      m, n, output, k, tw=tw)
    return output


print(m1s)
out = matadd_2d(m1s, m2s)
print(out)

