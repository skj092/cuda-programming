#include <stdio.h>
#include <cuda_runtime.h>
/*
Write a CUDA program that:
1. Takes an array of integers as input
2. Creates a square function that runs on the GPU (device)
3. Uses a CUDA kernel to square each element of the input array
4. Returns the squared values in an output array
5. The program should:
   - Handle an array of size 10
   - Allocate memory on both host (CPU) and device (GPU)
   - Use appropriate CUDA memory transfers
   - Print the squared values
   - Clean up allocated memory properly

Requirements:
- Use 256 threads per block
- Include proper CUDA error handling
- The square function should be a device function
- Fill the input array with values from 1 to 10
*/

__device__ float square(int x){
    return x * x;
}
__global__ void mykernel(int *array1, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N){
        array1[idx] = square(array1[idx]);
    }
}


int main(){
    const int N = 10;
    int *h_a; // host array
    int *d_a; // device array

    // Allocate host memory
    h_a = (int *)malloc(N * sizeof(int));

    // initialize host array
    for (int i = 0; i < N; i++){
        h_a[i] = i+1;
    }

    // allocate device memory
    cudaMalloc(&d_a, N * sizeof(int));

    // copy data to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256; // thread per block
    int gridSize = (N + blockSize - 1) / blockSize;
    mykernel<<<gridSize, blockSize>>>(d_a, N);
    cudaDeviceSynchronize();

    // copy data from device to host
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // verify
    for (int i = 0; i< N; i++){
        printf("h[%d] = %d\n", i, h_a[i]);
    }

    // free memory
    cudaFree(d_a);
    free(h_a);
    return 0;


}
