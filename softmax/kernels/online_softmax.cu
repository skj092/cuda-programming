#include <cuda_runtime.h>
#include <iostream>

__global__ void online_softmax(float *m, float *n, int N) {
    int index = threadIdx.x; /// (0, 1, 2)

    float m_prev = -INFINITY;
    float d_prev = 0.0;

    for (int i = 0; i < N; i++){
        float x_i = m[index * N + i];
        float m_curr = fmax(x_i, m_prev);
        float d_curr = d_prev * exp(m_prev - m_curr) + exp(x_i - m_curr);

        m_prev = m_curr;
        d_prev = d_curr;
}


    // devide each element by denominator
    for (int i = 0; i < N; i++){
        n[index * N + i] = exp(m[index * N + i] - m_prev) / d_prev;
    }
}

void run_online_softmax(float *d_input, float *d_output, int N) {
    online_softmax<<<(N + 255) / 256, 256>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
}
