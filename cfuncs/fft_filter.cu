#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "fft_filter.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CHECK_CUFFT(call) do { \
    cufftResult r = (call); \
    if (r != CUFFT_SUCCESS) { \
        fprintf(stderr, "CUFFT error %s:%d: %d\n", \
            __FILE__, __LINE__, r); \
        return -1; \
    } \
} while(0)

// Kernel: zero out frequencies outside cutoff
__global__ void applyFilter(cufftDoubleComplex *data, int N, int cutoff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int freq = (i <= N/2) ? i : (i - N);
        if (abs(freq) > cutoff) {
            data[i].x = 0.0;
            data[i].y = 0.0;
        }
    }
}

// Initialize
int fft_filter_init(FFTFilterCtx *ctx, int N, int cutoff) {
    ctx->N = N;
    ctx->cutoff = cutoff;

    // Allocate device buffer
    CHECK_CUDA(cudaMalloc(&ctx->d_data, N * sizeof(cufftDoubleComplex)));

    // Create FFT plan
    cufftHandle *plan = new cufftHandle;
    CHECK_CUFFT(cufftPlan1d(plan, N, CUFFT_Z2Z, 1));
    ctx->plan = plan;

    return 0;
}

// Run one FFT-filter-IFFT
int fft_filter_run(FFTFilterCtx *ctx, double _Complex *arr) {
    int N = ctx->N;
    int cutoff = ctx->cutoff;
    cufftHandle *plan = (cufftHandle*)ctx->plan;
    cufftDoubleComplex *d_data = (cufftDoubleComplex*)ctx->d_data;

    // Copy in
    CHECK_CUDA(cudaMemcpy(d_data, arr, N * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice));

    // Forward FFT
    CHECK_CUFFT(cufftExecZ2Z(*plan, d_data, d_data, CUFFT_FORWARD));

    // Filter
    int block = 256;
    int grid = (N + block - 1) / block;
    applyFilter<<<grid, block>>>(d_data, N, cutoff);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Inverse FFT
    CHECK_CUFFT(cufftExecZ2Z(*plan, d_data, d_data, CUFFT_INVERSE));

    // Copy back
    CHECK_CUDA(cudaMemcpy(arr, d_data, N * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost));

    // Normalize
    for (int i = 0; i < N; i++) {
        arr[i] /= N;
    }

    return 0;
}

// Cleanup
void fft_filter_destroy(FFTFilterCtx *ctx) {
    if (ctx->plan) {
        cufftHandle *plan = (cufftHandle*)ctx->plan;
        cufftDestroy(*plan);
        delete plan;
        ctx->plan = NULL;
    }
    if (ctx->d_data) {
        cudaFree(ctx->d_data);
        ctx->d_data = NULL;
    }
}
