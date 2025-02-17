#include <cuda_runtime.h>

__device__
float max_val(float a, float b){
    return (a > b) ? a : b;
}