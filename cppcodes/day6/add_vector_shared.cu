#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h> // For clock() and CLOCKS_PER_SEC

void printDeviceProperties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("\n=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("Device %d: %s\n", i, prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Number of SMs: %d\n", prop.multiProcessorCount);
        printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
            prop.maxThreadsDim[0],
            prop.maxThreadsDim[1],
            prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n",
            prop.maxGridSize[0],
            prop.maxGridSize[1],
            prop.maxGridSize[2]);
        // printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Total global memory: %zu bytes\n", prop.totalGlobalMem);
        printf("Warp size: %d\n", prop.warpSize);
        printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("Memory clock rate: %d kHz\n", prop.memoryClockRate);
        printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("Clock rate %d KHz \n", prop.clockRate/1000);
        printf("================================\n\n");
    }
}

__global__
void mykernel(int *a, int *b, int *c, int N, int tw){
    extern __shared__ int sharedMem[];
    int* ms = sharedMem;
    int* ns = &sharedMem[tw];

    // copy into shared memory
    for (int tc = 0; tc < N/tw; tc++){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N){
            ms[tc] = a[idx];
            ns[tc] = b[idx];
        }
    }

    // add the tiles
    for (int tc = 0; tc < N/tw; tc++){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < N){
            c[idx] = ms[tc] + ns[tc];
        }
    }
}

void cpu_kernel(int *a, int *b, int *c, int N){
    for (int i = 0; i < N; i ++){
        c[i] = a[i] + b[i];
    }
}

int main(){
    printDeviceProperties();
    int N = 16;
    int *h_a, *h_b, *h_c, *h_c_cpu; // host variable
    int *d_a, *d_b, *d_c; // device variable

    // allocate memory to host
    h_a = (int *)malloc(N * sizeof(int));
    h_b = (int *)malloc(N * sizeof(int));
    h_c = (int *)malloc(N * sizeof(int));
    h_c_cpu = (int *)malloc(N * sizeof(int)); // For CPU result

    // Initialize host variables
    for (int i = 0; i < N; i ++){
        h_a[i] = i;
        h_b[i] = i;
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    float gpu_alloc_time = 0.0f, h2d_time = 0.0f, kernel_time = 0.0f, d2h_time = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing GPU allocation
    cudaEventRecord(start);
    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_alloc_time, start, stop);

    // Start timing host to device transfer
    cudaEventRecord(start);
    // Copy data into device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2d_time, start, stop);

    // Start timing kernel execution
    cudaEventRecord(start);
    // launch kernel
    int tw = 4;
    dim3 blockdim(4);
    dim3 gridDim(4);
    int sharedSize = 2 * blockdim.x * sizeof(int); // for both array
    mykernel<<<gridDim, blockdim, sharedSize>>>(d_a, d_b, d_c, N, tw);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);

    // Start timing device to host transfer
    cudaEventRecord(start);
    // Copy data from device to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2h_time, start, stop);

    // Print timing information
    printf("GPU allocation time: %.6f ms\n", gpu_alloc_time);
    printf("Host to device transfer time: %.6f ms\n", h2d_time);
    printf("Kernel execution time: %.6f ms\n", kernel_time);
    printf("Device to host transfer time: %.6f ms\n", d2h_time);

    // CPU timing
    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    cpu_start = clock(); // Start CPU timing
    cpu_kernel(h_a, h_b, h_c_cpu, N); // Execute CPU kernel
    cpu_end = clock(); // End CPU timing

    cpu_time_used = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000; // Convert to milliseconds
    printf("CPU execution time: %.6f ms\n", cpu_time_used);

    // Verify GPU and CPU results
    int match = 1; // Assume results match initially
    for (int i = 0; i < N; i++) {
        if (h_c[i] != h_c_cpu[i]) {
            match = 0; // Set to 0 if any mismatch is found
            break;
        }
    }

    if (match) {
        printf("Verification: GPU and CPU results match!\n");
    } else {
        printf("Verification: GPU and CPU results do NOT match!\n");
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cpu);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
