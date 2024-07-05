from pathlib import Path
import torch
from torchvision.io import read_image, write_png
from torch.utils.cpp_extension import load_inline
from urllib.request import urlretrieve
import time
import math


def compile_extension():
    # cuda_source = Path("grayscale_kernel.cu").read_text()
    cuda_source = Path("grayscale_kernel_1d.cu").read_text()
    cpp_source = "torch::Tensor rgb_to_grayscale(torch::Tensor image);"

    # Load the CUDA kernel as a PyTorch extension
    rgb_to_grayscale_extension = load_inline(
        name="rgb_to_grayscale_extension",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        functions=["rgb_to_grayscale"],
        with_cuda=True,
        extra_cuda_cflags=["-O2"],
        # build_directory='./cuda_build',
    )
    return rgb_to_grayscale_extension


# python code to benchmark the cpu and gpu
def blk_kernel(f, blocks, threads, *args):
    for i in range(blocks):
        for j in range(threads): f(i, j, threads, *args)

def rgb2grey_bk(blockidx, threadidx, blockdim, x, out, n):
    i = blockidx*blockdim + threadidx
    if i<n: out[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]

def rgb2grey_pybk(x):
    c,h,w = x.shape
    n = h*w
    x = x.flatten()
    res = torch.empty(n, dtype=x.dtype, device=x.device)
    threads = 256
    blocks = int(math.ceil(h*w/threads))
    blk_kernel(rgb2grey_bk, blocks, threads, x, res, n)
    return res.view(h,w)

def main():
    """
    Use torch cpp inline extension function to compile the kernel in grayscale_kernel.cu.
    Read input image, convert it to grayscale via custom cuda kernel and write it out as png.
    """
    ext = compile_extension()

    # Download image
    url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/1600px-Cute_dog.jpg?20140729055059'
    path_img = Path('puppy.jpg')
    if not path_img.exists(): urlretrieve(url, path_img)

    x = read_image("puppy.jpg").permute(1, 2, 0).cuda()
    print("mean:", x.float().mean())
    print("Input image:", x.shape, x.dtype)

    assert x.dtype == torch.uint8


    tik = time.time()
    y = ext.rgb_to_grayscale(x)
    tok = time.time()
    print("Time taken for GPU:", tok - tik)

    print("Output image:", y.shape, y.dtype)
    print("mean", y.float().mean())
    # write_png(y.permute(2, 0, 1).cpu(), "output.png")

    # CPU
    x = read_image("puppy.jpg").permute(1, 2, 0).cpu()
    print("Input image:", x.shape, x.dtype)
    tik = time.time()
    y = rgb2grey_pybk(x)
    tok = time.time()
    print("Time taken for CPU:", tok - tik)


if __name__ == "__main__":
    main()

