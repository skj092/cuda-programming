#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

/*
2. Element-wise Matrix Operations
- Challenge: Perform element-wise multiplication of two 1024x1024 matrices
- Learning goals: 2D thread indexing, handling larger datasets
- Extra: Add support for different matrix sizes
*/
__global__
void mykernel(int *array1, int *array2, int *out, int n, int m){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < m){
        int idx = row * m + col;
        out[idx] = array1[idx] * array2[idx];
    }
}

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


inline unsigned int cdiv(unsigned int a, unsigned int b){
    return (a + b - 1) / b;
}


int main(){
    int n = 64;
    int m = 32;
    int *h_matrix1, *h_matrix2, *h_out_matrix; // host array
    int *d_matrix1, *d_matrix2, *d_out_matrix; // device array
    // (n x m)

    // Seed random number generator
    srand(time(NULL));

    // Allocate host memory
    h_matrix1 = (int *)malloc(n * m * sizeof(int));
    h_matrix2 = (int *)malloc(n * m * sizeof(int));
    h_out_matrix = (int *)malloc(n * m * sizeof(int));

    // fill the matrix with random numbers (initialize)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < m; j++){
            h_matrix1[i * m + j] = rand() % 10; // generate number between 0 and 9;
            h_matrix2[i * m + j] = rand() % 10;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_matrix1, n * m * sizeof(int));
    cudaMalloc(&d_matrix2, n * m * sizeof(int));
    cudaMalloc(&d_out_matrix, n * m * sizeof(int));

    // copy data to device
    cudaMemcpy(d_matrix1, h_matrix1, n * m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix2, h_matrix2, n * m * sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dim3 threadsperBlock(32, 32);
    dim3 numBlocks(cdiv(n, threadsperBlock.x), cdiv(m, threadsperBlock.y)); // (32, 16)
    mykernel<<<numBlocks, threadsperBlock>>>(d_matrix1, d_matrix2, d_out_matrix, n, m);
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();

    // copy data to host
    cudaMemcpy(h_out_matrix, d_out_matrix, n * m * sizeof(int), cudaMemcpyDeviceToHost);

    // verify
   for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("h_out_matrix[%d][%d] = %d\n", i, j, h_out_matrix[i * n + j]);
        }
    }

    // free memory
    cudaFree(d_matrix1);
    cudaFree(d_matrix2);
    cudaFree(d_out_matrix);
    free(h_matrix1);
    free(h_matrix2);
    free(h_out_matrix);
    return 0;

}

