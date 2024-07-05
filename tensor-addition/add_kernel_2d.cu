#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__
void add_tensors_cuda_kernel(float *a, float *b, float *out, int w, int h) {
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int r = blockIdx.y*blockDim.y + threadIdx.y;
    if (c<w && r<h) {
      int i = r*w + c;
      out[i] = a[i] + b[i];
  }
}

// helper function for ceiling unsigned integer division
inline unsigned int cdiv(unsigned int a, unsigned int b) {
  return (a + b - 1) / b;
}

torch::Tensor add_in_cuda(torch::Tensor tensor1, torch::Tensor tensor2) {
    CHECK_INPUT(tensor1);
    CHECK_INPUT(tensor2);
    AT_ASSERTM(tensor1.sizes() == tensor2.sizes(), "tensors must have the same size");
    int h = tensor1.size(0);
    int w = tensor1.size(1);

    // auto result = torch::empty_like(tensor1);
    auto result = torch::zeros({h,w}, tensor1.options());
    printf("h*w: %d*%d\n", h, w);

    dim3 tpb(16,16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));

    add_tensors_cuda_kernel<<<blocks, tpb>>>(
      tensor1.data_ptr<float>(),
      tensor2.data_ptr<float>(),
      result.data_ptr<float>(),
      w,
      h
    );

    return result;
  }
