# include <stdio.h>
# include <cuda_runtime.h>

# define TW 3

int cdiv(int n, int tw){
    return (n + tw - 1) / tw
}

__global__
void mykernel(float * h_m, float * h_n, float * h_o, int N){
    __shared__ float ms[TW][TW]
    __shared__ float ns[TW][TW]

    int row = blockIdx.y * blockDim.y + threadIdx.y
    int col = blockIdx.x * blockDim.x + threadIdx.x

    int ty = threadIdx.y
    int tx = threadIdx.x

    // fill ms and ns
    float sum = 0
    if (row >= N | | col >= N) return
    for (int t=0
         t < (N + TW - 1)/TW
         t++){
        ms[threadIdx.y][threadIdx.x] = h_m[row * N + t * TW + tx]
        ns[threadIdx.y][threadIdx.x] = h_n[(TW * t + ty) * N + col]

        __syncthreads()

        for (int k=0
             k < TW
             k++){
            sum += ms[ty][k] * ns[k][tx]
        }
        __syncthreads()
    }

    h_o[row * N + col] = sum

    // compute dot product
}

void verify_multiplication(float * A, float * B, float * C, int N) {
    printf("\nExpected values:\n")
    for (int i=0
         i < N
         i++) {
        for (int j=0
             j < N
             j++) {
            float expected = 0
            for (int k=0
                 k < N
                 k++) {
                expected += A[i * N + k] * B[k * N + j]
            }
            printf("Position [%d][%d]: Expected %.2f, Got %.2f\n",
                   i, j, expected, C[i * N + j])
        }
    }
}


int main(){
    int N = 6
    float * h_m, *h_n, *h_o
    // host variable
    float * d_m, *d_n, *d_o
    // device variable

    // Allocate memory on host
    int size_ = N * N * sizeof(float)
    h_m = (float * )malloc(size_)
    h_n = (float * )malloc(size_)
    h_o = (float * )malloc(size_)

    // Initialize the host variables
    for (int i=0
         i < N
         i++){
        for (int j=0
             j < N
             j++){
            h_m[i * N + j] = j + 1
            h_n[i * N + j] = j + 1

        }
    }

    // Allocate memory on device
    cudaMalloc(& d_m, size_)
    cudaMalloc(& d_n, size_)
    cudaMalloc(& d_o, size_)

    // Copy memory from host to device
    cudaMemcpy(d_m, h_m, size_, cudaMemcpyHostToDevice)
    cudaMemcpy(d_n, h_n, size_, cudaMemcpyHostToDevice)
    cudaMemcpy(d_o, h_o, size_, cudaMemcpyHostToDevice)

    // Grid and Block size configuration
    dim3 block(TW, TW)
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y)

    // launch the kernel
    mykernel << <grid, block >> >(d_m, d_n, d_o, N)

    // Copy data from device to host
    cudaMemcpy(h_o, d_o, size_, cudaMemcpyDeviceToHost)

    // Verify the data
    // for (int i=0
            i < N
            i + +){
        // for (int j=0
                j < N
                j++){
            // printf("h_o[%d][%d] = %f\n", i, j, h_o[i * N + j])
            // }
        //}
    cudaMemcpy(h_o, d_o, size_, cudaMemcpyDeviceToHost)
    // After cudaMemcpy(h_o, d_o, size_, cudaMemcpyDeviceToHost)
    verify_multiplication(h_m, h_n, h_o, N)

    // release the memory
    cudaFree(d_m)
    cudaFree(d_n)
    cudaFree(d_o)

    // relesae host memory
    free(h_m)
    free(h_n)
    free(h_o)

    return 0

}
