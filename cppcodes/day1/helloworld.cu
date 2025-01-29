#include <stdio.h>

__global__ void helloFromGPU() {
    printf("hello world from GPU!\n");
}

int main() {
    helloFromGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

