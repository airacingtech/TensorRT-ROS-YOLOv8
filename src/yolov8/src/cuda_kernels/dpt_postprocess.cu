#include <cuda_runtime.h>

// TODO: Think about mememory coalescing
__global__ void postprocessKernel(float* depth, int totalElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements){
        float d = depth[idx];
        d = fminf(fmaxf(d, 0.0f), 120.0f);
        depth[idx] = 120.0f - d;
    }
}

// Wrapper functionto call CUDA kernel from C++ code
extern "C" void launchPostprocessKernel(float* depth, int totalElements, cudaStream_t stream) {
    int threadsPerBlock = 256;
    int blocks = (totalElements + threadsPerBlock -1) / threadsPerBlock;
    postprocessKernel<<<blocks, threadsPerBlock, 0, stream>>>(depth, totalElements);
    cudaStreamSynchronize(stream);
}