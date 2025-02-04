#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

/*
matrix multiplication
*/
__global__
void mykernel(int *a, int *b, int *o, int n, int m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= m) return;
    int c = 0;
    for (int i = 0; i < m; i ++) c += a[row * n + i] * b[n * i + col];
    o[row * 3 + col] = c;
}

inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}



int main(){
    int n = 10;
    int m = 10;

    int *h_a, *h_b, *h_o;
    int *d_a, *d_b, *d_o;

    srand(time(NULL));


    // allocate host memory
    h_a = (int *)malloc(n * m * sizeof(int));
    h_b = (int *)malloc(n * m * sizeof(int));
    h_o = (int *)malloc(n * m * sizeof(int));

    // initialize the metrices
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            h_a[i * m + j] = i * m + j;
            h_b[i * m + j] = i * m + j;
        }
    }

    // allocate device memory
    cudaMalloc(&d_a, n * m * sizeof(int));
    cudaMalloc(&d_b, n * m * sizeof(int));
    cudaMalloc(&d_o, n * m * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_a, h_a, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * m * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 threadsperBlock(4, 4);
    dim3 numBlocks(cdiv(n, threadsperBlock.x), cdiv(m, threadsperBlock.y));
    mykernel<<<numBlocks, threadsperBlock>>>(d_a, d_b, d_o, n, m);


    // Copy data from device to host
    cudaMemcpy(h_o, d_o, n * m * sizeof(int), cudaMemcpyDeviceToHost);


    // print and verify
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("h_a[%d][%d] = %d\n",i, j, h_o[i * m + j]);
        }
    }

    return 0;

}
