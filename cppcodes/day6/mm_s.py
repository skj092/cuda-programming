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


def blk_kernel2d_shar(f, blocks, threads, sh_sz, *args, **kwargs):
    for i0 in range(blocks.y):
        for i1 in range(blocks.x):
            shared = torch.zeros(sh_sz)
            f(dim3(i1, i0), threads, shared, *args, **kwargs)


def matmul_tiled_bk(blockIdx, blockDim, shared, m, n, out, h, w, k, tw):
    shar_sz = tw*tw
    ms, ns = shared[:shar_sz], shared[shar_sz:]

    for ph in range(cdiv(k, tw)):
        idx = ph*tw
        # fill shared
        for tr in range(blockDim.y):
            for tc in range(blockDim.x):
                r, c = blockIdx.y*blockDim.y + tr, blockIdx.x*blockDim.x + tc
                ms[tr*tw+tc] = m[tc+idx + r*k] if r < h and idx+tc < k else 0.
                ns[tr*tw+tc] = n[(tr+idx)*w + c] if c < w and idx + \
                    tr < k else 0.

        # do dotprods from shared
        for tr in range(blockDim.y):
            for tc in range(blockDim.x):
                r, c = blockIdx.y*blockDim.y + tr, blockIdx.x*blockDim.x + tc
                for i in range(tw):
                    if r*w+c < len(out):
                        out[r*w+c] += ms[tr*tw+i] * ns[tw*i+tc]


def matmul_2d(m, n, tw=16):
    h, k = m.shape
    k2, w = n.shape
    assert k == k2, "Size mismatch!"
    output = torch.zeros(h, w, dtype=m.dtype)
    tpb = dim3(tw, tw)
    blocks = dim3(cdiv(w, tpb.x), cdiv(h, tpb.y))
    blk_kernel2d_shar(matmul_tiled_bk, blocks, tpb, tw*tw*2,
                      m.flatten(), n.flatten(), output.flatten(),
                      h, w, k, tw=tw)
    return output


print(torch.isclose(matmul_2d(m1s, m2s, tw=16), m1s@m2s).all())

