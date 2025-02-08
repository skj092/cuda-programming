#include <stdio.h>
#include <cuda_runtime.h>

__global__
void mykernel(int *a, int *b, int *c, int N, int TW){

    // loading from shared memory
    extern __shared__ int shared_mem[];
    int *as = shared_mem;
    int *bs = &as[TW * TW ];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = row * N + col;

    // Update tiled vectors
    int t_idx = threadIdx.y * TW + threadIdx.x;
    if (row < N and col < N){
        as[t_idx] = a[row * N + col];
        bs[t_idx] = b[row * N + col];
    }
    __syncthreads();

    // perform addition
    if (row < N and col < N){
        c[row * N + col] = as[t_idx] + bs[t_idx];
    }
    __syncthreads();

}


int main(){
    int N = 4;
    int TW = 2;
    int *h_a, *h_b, *h_c; // host array
    int *d_a, *d_b, *d_c; // device array

    // acclocate hosty memory
    h_a = (int *)malloc(N * N * sizeof(int));
    h_b = (int *)malloc(N * N * sizeof(int));
    h_c = (int *)malloc(N * N * sizeof(int));

    // Initialize host array
    for (int i = 0; i < N; i ++){
        for (int j = 0; j < N; j++){
            h_a[i * N + j] = i * N + j;
            h_b[i * N + j] = i * N + j;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * N * sizeof(int));
    cudaMalloc(&d_b, N * N * sizeof(int));
    cudaMalloc(&d_c, N * N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, h_a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N * N * sizeof(int), cudaMemcpyHostToDevice);


    // Launch kernel
    dim3 tpd(TW, TW);
    dim3 gridDim((N + TW - 1) / TW, (N + TW -1) / TW);
    // shared size
    int size_t = TW * TW * 2 * sizeof(int);
    mykernel<<<gridDim, tpd, size_t>>>(d_a, d_b, d_c, N, TW);

    // copy data from device to host
    cudaMemcpy(h_c, d_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // verify the resultz
    for (int i = 0; i < N; i ++){
        for (int j = 0; j < N; j++){
            printf("h_c[%d][%d] = %d\n", i, j, h_c[i * N + j]);

        }
    }

    // release
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

