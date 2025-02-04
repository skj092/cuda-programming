#include <stdio.h>
#include <cuda_runtime.h>

__global__
void mykernel_neive(int *a, int *b, int *c, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        c[idx] = a[idx] + b[idx];
    }
}

__global__
void mykernel(int *a, int *b, int *c, int N, int TW){

    // loading from shared memory
    extern __shared__ int shared_mem[];
    int *as = shared_mem;
    int *bs = shared_mem + blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Update tiled vectors
    if (idx < N){
        as[threadIdx.x] = a[idx];
        bs[threadIdx.x] = b[idx];
    }
    __syncthreads();

    // perform addition
    if (idx < N){
        c[idx] = as[threadIdx.x] + bs[threadIdx.x];
    }
    __syncthreads();

}


int main(){
    int N = 16;
    int TW = 4;
    int *h_a, *h_b, *h_c; // host array
    int *d_a, *d_b, *d_c; // device array

    // acclocate hosty memory
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));

    // Initialize host array
    for (int i = 0; i < N; i ++){
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * sizeof(int), cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 tpd(TW);
    dim3 gridDim((N + TW - 1) / TW);
    // shared size
    int size_t = TW * 2 * sizeof(int);
    mykernel<<<gridDim, tpd, size_t>>>(d_a, d_b, d_c, N, TW);
    // mykernel_neive<<<gridDim, tpd>>>(d_a, d_b, d_c, N);

    // copy data from device to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // verify the result
    for (int i = 0; i < N; i ++){
        printf("h_c[%d] = %d\n", i, h_c[i]);
    }

    // release
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

