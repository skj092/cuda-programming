#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__
void add_tensors_cuda_kernel(float *a, float *b, float *out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
      out[index] = a[index] + b[index];
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

    auto n = tensor1.numel();
    auto result = torch::empty_like(tensor1);

    const int threads = 256;
    const int blocks = cdiv(n, (unsigned int)threads);

    add_tensors_cuda_kernel<<<blocks, threads>>>(
      tensor1.data_ptr<float>(),
      tensor2.data_ptr<float>(),
      result.data_ptr<float>(),
      n
    );

    return result;
  }
