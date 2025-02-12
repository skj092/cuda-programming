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

    float m_prev = -INFINITY;
    float d_prev = 0.0;

    for (int i = 0; i < N; i++){
        float x_i = m[index * N + i];
        float m_curr = max_val(x_i, m_prev);
        float d_curr = d_prev * exp(m_prev - m_curr) + exp(x_i - m_curr);

        m_prev = m_curr;
        d_prev = d_curr;
    }

    // devide each element by denominator
    for (int i = 0; i < N; i++){
        n[index * N + i] = exp(m[index * N + i] - m_prev) / d_prev;
    }
}

float max(float a, float b){
    return (a > b) ? a : b;
}

float* softmax(float *m, float *n, int N){
    for (int k = 0; k < N; k++){

        float m_prev = -INFINITY;
        float d_prev = 0;

        for (int i = 0; i < N; i++){

            float x_i = m[k * N + i];
            float m_curr = fmax(x_i, m_prev);
            float d_curr = d_prev * exp(m_prev - m_curr) + exp(x_i - m_curr);
            m_prev = m_curr;
            d_prev = d_curr;
        }
        // devide each element by denominator
        for (int i = 0; i < N; i++){
            n[k * N + i] = exp(m[k * N + i] - m_prev) / d_prev;
        }
    }
    return n;
}

int main(){
    int N = 3;
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

    // Allocate device memory
    cudaMalloc(&d_m, size);
    cudaMalloc(&d_n, size);

    // copy data from host to device
    cudaMemcpy(d_m, h_m, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, size, cudaMemcpyHostToDevice);

    // launch kernel
    int blocksize = N;
    mykernel<<<1, blocksize>>>(d_m, d_n, N);
    cudaDeviceSynchronize();

    // copy data from device to host
    cudaMemcpy(h_n, d_n, size, cudaMemcpyDeviceToHost);
    // verify
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%f \n", h_n[i * N + j]);
        }
    }

    // cpu benchmark
    clock_t t;
    t = clock();
    h_n  = softmax(h_m, h_n, N);
    t = clock() - t;
    double time_taken = ((double)t)/CLOCKS_PER_SEC;
    printf("fun() took %f seconds to execute \n", time_taken);
    return 0;
}

