#include <cuda_runtime.h>

// __device__
// float max_val(float a, float b){
//     return (a > b) ? a : b;
// }


#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
    
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}