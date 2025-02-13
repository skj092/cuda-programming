#include <math.h>
#include <stdlib.h>
#include <time.h>

float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
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