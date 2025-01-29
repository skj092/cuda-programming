#include <stdio.h>
#include <cuda_runtime.h> // cuda header

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


// Kernel function (run on the GPU)
__global__ void mykernel(int *d_array, int N) {
    int idx = blockIdx.x * blockIdx.x + threadIdx.x;
    if (idx < N){
        d_array[idx] = idx ;
    }
}

int main() {
    // Host (CPU) variables
    int N = 100; // size of the array
    int h_array1[N]; // Host array
    int *d_array; // Device array

    // 1. Allocate memory on the device
    CHECK_CUDA(cudaMalloc((void**)&d_array, N* sizeof(int)));

    // 2. Launch the kernel
    int blockSize = 256; /// Threads per block
    int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks
    mykernel<<<gridSize, blockSize>>>(d_array, N);

    // 3. check the kernel launch error
    CHECK_CUDA(cudaGetLastError());


    // 4 Copy results back to the host
    CHECK_CUDA(cudaMemcpy(h_array, d_array, N * sizeof(int), cudaMemcpyDeviceToHost));

    // 5. Verify the results
    for (int i = 0; i < N; i++){
        printf("h_array[%d] = %d\n", i, out_array[i]);
    }

    // 6. Free device memory
    CHECK_CUDA(cudaFree(d_array));

    return 0;
}
