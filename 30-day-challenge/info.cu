#include <stdio.h>
#include <cuda_runtime.h>

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


int main() {
    printDeviceProperties();
}
