#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Transfer data from host to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Kernel invocation with N threads
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    VecAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C);

    // Transfer results from device to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verification
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(1);
        }
    }
    printf("Test PASSED\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
