/*
matrix addition
2. Write a kernel that has each thread producing one output matrix col. Fill in the execution configuration parameters for the design.

*/
#include <stdio.h>
#include <cuda_runtime.h>

__global__
void mykernel(int *a, int *b, int *o, int n){

    // get the absolute co-ordinate of thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        for (int i = 0; i < n ; i ++){
            o[idx + n * i] = a[idx + n * i] + b[idx + n * i];
        }
    }

}
/*
idx = 0 + 3 * (0, 1, 2) = (0, 3, 6) - col1
idx = 1 + 3 * (0, 1, 2) = (1, 4, 7) - col2
idx = 2 + 3 * (0, 1, 2) = (2, 5, 8) - col3
*/

int main(){
    int n = 3;
    int *h_a, *h_b, *h_o;
    int *d_a, *d_b, *d_o;

    // allocate host memory
    h_a = (int *)malloc(n * n * sizeof(int));
    h_b = (int *)malloc(n * n * sizeof(int));
    h_o = (int *)malloc(n * n * sizeof(int));

    // initialize the variable
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            h_a[i * n + j] = i * n + j;
            h_b[i * n + j] = i * n + j;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_a, n * n * sizeof(int));
    cudaMalloc(&d_b, n * n * sizeof(int));
    cudaMalloc(&d_o, n * n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // define blocksize and grid size
    int blockDim = 3;
    int gridDim = (n + blockDim - 1) /  blockDim;

    // launch the kernel
    mykernel<<<gridDim, blockDim>>>(d_a, d_b, d_o, n);

    // synchronize
    cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(h_o, d_o, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    // verify the result
    for (int i = 0; i < n; i ++ ){
        for (int j = 0; j < n; j ++){
            printf("h_o[%d][%d] = %d\n", i, j, h_o[i * n + j]);
        }
    }

    // release the memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_o);
    free(h_a);
    free(h_b);
    free(h_o);
}
