import torch
import numpy as np
from collections import namedtuple

dim3 = namedtuple('dim3', ['x', 'y', 'z'], defaults=(1, 1))


def cdiv(a, b):
    "Int ceiling division of `a` over `b`"
    return (a+b-1)//b


np.set_printoptions(precision=2, linewidth=140)
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)

m1 = torch.rand(5120, 256)
m1s = m1[:4]
m2 = torch.rand(256, 5120)
m2s = m2[:, :4]
m1s = torch.arange(16).reshape(4, 4).float()
m2s = torch.arange(16).reshape(4, 4).float()


def blk_kernel2d_shar(f, blocks, threads, sh_sz, *args, **kwargs):
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            shared = torch.zeros(sh_sz)
            f(dim3(i1, i0), threads, shared, *args, **kwargs)


def matadd_tiled_bk(blockIdx, blockDim, shared, m, n, out, h, w, k, tw):
    shar_sz = tw * tw
    ms, ns = shared[:shar_sz], shared[shar_sz:]

    # Load data into shared memory
    for tr in range(blockDim.y):
        for tc in range(blockDim.x):
            r = blockIdx.y * blockDim.y + tr
            c = blockIdx.x * blockDim.x + tc
            if r < h and c < k:
                ms[tr * tw + tc] = m[r * k + c]
                ns[tr * tw + tc] = n[r * k + c]
            else:
                ms[tr * tw + tc] = 0.0
                ns[tr * tw + tc] = 0.0

    # Perform addition and store result
    for tr in range(blockDim.y):
        for tc in range(blockDim.x):
            r = blockIdx.y * blockDim.y + tr
            c = blockIdx.x * blockDim.x + tc
            if r < h and c < w:
                out[r * w + c] = ms[tr * tw + tc] + ns[tr * tw + tc]


def matadd_2d(m, n, tw=2):
    h, k = m.shape
    k2, w = n.shape
    assert k == k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = dim3(tw, tw)
    blocks = dim3(cdiv(w, tpb.x), cdiv(h, tpb.y))
    blk_kernel2d_shar(matadd_tiled_bk, blocks, tpb, tw*tw*2,
                      m.flatten(), n.flatten(), output.flatten(),
                      h, w, k, tw=tw)
    return output


out = matadd_2d(m1s, m2s)
print(torch.isclose(matadd_2d(m1s, m2s, tw=2), m1s+m2s).all())
