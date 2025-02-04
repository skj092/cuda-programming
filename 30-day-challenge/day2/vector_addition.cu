// vector addition
#include <stdio.h>
#include <cuda_runtime.h>


// Kernel function (run on the GPU)
__global__ void mykernel(float *array1, float *array2, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        out[idx] = array1[idx] + array2[idx] ;
    }
}

int main() {
    int N = 1024;
    float *h_a, *h_b, *h_c; // host array
    float *d_a, *d_b, *d_c; // device array

    // Allocate host memory
    h_a = (float *)malloc(N * sizeof(float));
    h_b = (float *)malloc(N * sizeof(float));
    h_c = (float *)malloc(N * sizeof(float));

    // Initialize host array
    for (int i= 0; i < N; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256; /// Threads per block
    int gridSize = (N + blockSize - 1) / blockSize; // Number of blocks

    mykernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    // Copy data to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the results
    for (int i = 0; i< N; i++){
        printf("h_c[%d] = %.1f\n", i, h_c[i]);
    }
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}

