// write a softmax function for a matrix in c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


float max(float a, float b){
    return (a > b) ? a : b;
}

float* softmax(float *m, float *n, int N){
    for (int v = 0; v < N; v++){
        float *a;
        a = (float *)malloc(N * sizeof(float));
        for (int i = 0; i < N; i++){
            a[i] = m[v  * 3 + i];
        }
        // find the maximu
        float m = 0;
        for (int i = 0; i < N; i++){
            m = max(a[i], m);
        }
        // find denominator values
        float s = 0;
        for (int i = 0; i < N; i++){
            s += exp(a[i] - m);
        }

        for (int i = 0; i < N; i++){
            n[v * 3 + i] = exp(a[i] - m) / s;
        }
    }
    return n;
}





int main(){
    float *h_a, *h_o;
    int N = 3;

    h_a = (float *)malloc(N * N* sizeof(float));
    h_o = (float *)malloc(N * N* sizeof(float));

    for(int i = 0; i < N; i++){
        for (int j = 0; j< N; j++){
            h_a[i * N + j] = i * N +j;
        }
    }

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%f ", h_a[i * N + j]);
        }
    }
    printf("\n");
    h_o  = softmax(h_a, h_o, N);
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            printf("%f ", h_o[i * N + j]);
        }
    }
    return 0;

}

