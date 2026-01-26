#include "diode_cavity.h"
#ifdef DIODE_CAVITY_H
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

__host__ __device__
inline cuDoubleComplex cmul_real(double a, cuDoubleComplex z) {
    return make_cuDoubleComplex(a * z.x, a * z.y);
}
__host__ __device__
inline cuDoubleComplex cneg(cuDoubleComplex z) {
    return make_cuDoubleComplex(-z.x, -z.y);
}

__global__ void diode_cavity_round_trip_kernel(DiodeCavityCtx *data, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* position on the left and right beams*/
    int index_1 = ((data->diode_pos_1[i] + offset) % data->N) * data->N_x + threadIdx.x;
    int index_2 = ((data->diode_pos_2[i] + offset) % data->N) * data->N_x + threadIdx.x;

    /* access arrays with restrict pointers */
    double *__restrict__ pN0 = data->diode_N0;
    double N0 = pN0[i];
    cuDoubleComplex *__restrict__ pP1 = data->diode_P_dir_1;
    cuDoubleComplex *__restrict__ pP2 = data->diode_P_dir_2;
    cuDoubleComplex P1 = pP1[i];
    cuDoubleComplex P2 = pP2[i];
    cuDoubleComplex *__restrict__ pE1 = data->amplitude;
    cuDoubleComplex E1 = pE1[index_1];
    cuDoubleComplex *__restrict__ pE2 = data->amplitude;
    cuDoubleComplex E2 = pE2[index_2];

    /* ------ the P update equation */
    cuDoubleComplex drive = cuCmul(data->I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E1));
    P1 = cuCsub(cmul_real(data->alpha, P1), drive);
    drive = cuCmul(data->I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E2));
    P2 = cuCsub(cmul_real(data->alpha, P2), drive);

    /* ------ the N update equation */
    //averageP = 0.5 * (gainP[iN] + gainP[i]);
    double exchange = (cuCadd(cuCmul(cuConj(E1), P1), cuCmul(cuConj(E2), P2))).y;
    if (data->diode_type[blockIdx.x] == 1 /* gain */) {
        N0 = N0 + data->dt * ((- N0) / data->tGain + data->C_gain * exchange + data->Pa);
    } else if (data->diode_type[blockIdx.x] == 2 /* absorber */) {
        N0 = N0 + data->dt * ((data->N0b - N0) / data->tLoss + data->C_loss * exchange);
    }

    /* ------ the E update equation */
    pE1[index_1] = cuCadd(pE1[index_1], cmul_real(data->dt * data->coupling_out_gain , cuCmul(data->I1, P1)));
    pE2[index_2] = cuCadd(pE2[index_2], cmul_real(data->dt * data->coupling_out_gain , cuCmul(data->I1, P2)));

    /* store back updated values */
    pN0[i] = N0;
    pP1[i] = P1;
    pP2[i] = P2;
}

// Initialize
int cuAllocZero(void **devPtr, size_t size) {
    CHECK_CUDA(cudaMalloc(devPtr, size));
    CHECK_CUDA(cudaMemset(*devPtr, 0, size));
    return 0;
}
int cuAllocValueDouble(double **devPtr, size_t n, double value) {
    CHECK_CUDA(cudaMalloc(devPtr, sizeof(double) * n));
    // Wrap raw pointer and fill
    thrust::device_ptr<double> dev_ptr(*devPtr);
    thrust::fill(dev_ptr, dev_ptr + n, value);
    return 0;
}

int diode_cavity_init(DiodeCavityCtx *ctx_host) {

    DiodeCavityCtx ctx_local; 
    memcpy(&ctx_local, ctx_host, sizeof(DiodeCavityCtx));
    if (ctx_host->d_ctx == NULL) {
        /* first time calling for initialization */
        /* allocate cuda arrays */
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_type, sizeof(int) * ctx_host->diode_length)); /* type: 1 gain, 2 absorber*/
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_1, sizeof(int) * ctx_host->diode_length)); /* position index for left to right beam*/
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_2, sizeof(int) * ctx_host->diode_length)); /* position index for right to left beam*/
        cuAllocValueDouble(&ctx_local.diode_N0, ctx_host->diode_length * ctx_host->N_x, 1.0); /* inversion density */
        cuAllocZero((void **)&ctx_local.diode_P_dir_1, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for left to right beam*/
        cuAllocZero((void **)&ctx_local.diode_P_dir_2, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for right to left beam*/ 
        cuAllocZero((void **)&ctx_local.amplitude, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x); /* beam amplitude values */
        /* copy host to cuda arrays */
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_type, ctx_host->diode_type, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_1, ctx_host->diode_pos_1, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_2, ctx_host->diode_pos_2, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(ctx_local.diode_N0, ctx_host->diode_N0, sizeof(double) * ctx_host->diode_length * ctx_host->N_x, cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(ctx_local.diode_P_dir_1, ctx_host->diode_P_dir_1, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x, cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(ctx_local.diode_P_dir_2, ctx_host->diode_P_dir_2, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x, cudaMemcpyHostToDevice));
        // CHECK_CUDA(cudaMemcpy(ctx_local.amplitude, ctx_host->amplitude, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x, cudaMemcpyHostToDevice));

        /* copy ctx_local to device */
        DiodeCavityCtx *ctx_cuda = NULL;
        CHECK_CUDA(cudaMalloc(&ctx_cuda, sizeof(DiodeCavityCtx)));
        CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_local, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));
        ctx_host->d_ctx = ctx_cuda;
    } else {
        DiodeCavityCtx *ctx_cuda = ctx_host->d_ctx;
        DiodeCavityCtx ctx_help;
        /* collect the existing pointers from the device*/
        CHECK_CUDA(cudaMemcpy(&ctx_help, ctx_cuda, sizeof(DiodeCavityCtx), cudaMemcpyDeviceToHost));
        /* copy the pointers from the device */
        ctx_local.diode_type = ctx_help.diode_type;
        ctx_local.diode_pos_1 = ctx_help.diode_pos_1;
        ctx_local.diode_pos_2 = ctx_help.diode_pos_2;
        ctx_local.diode_N0 = ctx_help.diode_N0;
        ctx_local.diode_P_dir_1 = ctx_help.diode_P_dir_1;
        ctx_local.diode_P_dir_2 = ctx_help.diode_P_dir_2;
        ctx_local.amplitude = ctx_help.amplitude;
        ctx_local.d_ctx = NULL;
        /* copy ctx_local to device so that new parameters are used but old buffers are preserved */
        CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_local, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));
    }

    return 0;
}

// Run one Cavity tick
int diode_cavity_run(DiodeCavityCtx *ctx) {

    /* perform n_rounds of cavity round trips */
    for (int i_round = 0; i_round < ctx->n_rounds; i_round++) {
        /* perform one round trip */
        for (int offset = 0; offset < ctx->N_x; offset++) {
            // Call the CUDA kernel to process a single step in ther round trip
            int threads = ctx->N_x;
            int blocks = ctx->diode_length;
            diode_cavity_round_trip_kernel<<<blocks, threads>>>(ctx->d_ctx, offset);
            CHECK_CUDA(cudaDeviceSynchronize());
        }

    }

    return 0;
}

// Cleanup
void diode_cavity_destroy(DiodeCavityCtx *ctx) {
    cudaFree(ctx->diode_type);
    cudaFree(ctx->diode_pos_1);
    cudaFree(ctx->diode_pos_2);
    cudaFree(ctx->diode_N0);
    cudaFree(ctx->diode_P_dir_1);
    cudaFree(ctx->diode_P_dir_2);
    cudaFree(ctx->amplitude);
    cudaFree(ctx->d_ctx);

}
#endif
