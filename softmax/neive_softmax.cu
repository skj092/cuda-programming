#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

__device__
float max_val(float a, float b){
    return (a > b) ? a : b;
}

__global__
void mykernel(float *m, float *n, int N){
    int index = threadIdx.x; /// (0, 1, 2)

    float ma = 0;
    for (int i = 0; i < N; i++){
        ma = max_val(m[index*N + i], ma);
    }

    // find denominator values
    float s = 0;
    for (int i = 0; i < N; i++){
        s += exp(m[index*N + i] - ma);
    }

    // devide each element by denominator
    for (int i = 0; i < N; i++){
        n[index * N + i] = exp(m[index * N + i] - ma) / s;
    }
}

float* softmax(float *m, float *n, int N){
    for (int k = 0; k < N; k++){

        float ma = 0;
        for (int i = 0; i < N; i++){
            ma = max(m[k*N + i], ma);
        }

        // find denominator values
        float s = 0;
        for (int i = 0; i < N; i++){
            s += exp(m[k*N + i] - ma);
        }

        // devide each element by denominator
        for (int i = 0; i < N; i++){
            n[k * N + i] = exp(m[k * N + i] - ma) / s;
        }
    }
    return n;
}

int main(){
    int N = 4095;
    float *h_m, *h_n; // host variable
    float *d_m, *d_n; // device variable

    // Memory allocation in host
    int size = N * N * sizeof(float);
    h_m = (float *)malloc(size);
    h_n = (float *)malloc(size);

    // initialize the matrix
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            h_m[i * N + j] = i * N + j;
        }
    }
    // Timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;
    
    // Allocate device memory
    cudaEventRecord(start);
    cudaMalloc(&d_m, size);
    cudaMalloc(&d_n, size);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);


    // copy data from host to device
    cudaEventRecord(start);
    cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    // launch kernel
    int blocksize = N;
    cudaEventRecord(start);
    mykernel<<<1, blocksize>>>(d_m, d_n, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> kernel execution time: %f ms\n", ms);

    // copy data from device to host
    cudaMemcpy(h_n, d_n, size, cudaMemcpyDeviceToHost);
    // verify
    // cpu benchmark
    clock_t t;
    t = clock();
    h_n  = softmax(h_m, h_n, N);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf(">> CPU execution time %f ms\n", time_taken);

    cudaFree(d_m);
    cudaFree(d_n);
    free(h_m);
    free(h_n);


    return 0;


}
