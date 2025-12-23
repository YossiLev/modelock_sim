#include "diode_cavity.h"
#ifdef DIODE_CAVITY_H
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>

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

    int index_1 = ((data->diode_pos_1[i] + offset) % data->N) * data->N_x + threadIdx.x;
    int index_2 = ((data->diode_pos_2[i] + offset) % data->N) * data->N_x + threadIdx.x;

    double *N0 = &(data->diode_N0[i]);
    cuDoubleComplex *P1 = &(data->diode_P_dir_1[i]);
    cuDoubleComplex *P2 = &(data->diode_P_dir_2[i]);
    cuDoubleComplex *E1 = &(data->amplitude[index_1]);
    cuDoubleComplex *E2 = &(data->amplitude[index_2]);

    /* ------ the P update equation */
    cuDoubleComplex drive = cneg(cuCmul(data->I1, cmul_real(data->kappa * *N0, *E1)));
    *P1 = cuCadd(cmul_real(data->alpha, *P1), cmul_real(data->one_minus_alpha_div_a, drive));
    drive = cneg(cuCmul(data->I1, cmul_real(data->kappa * *N0, *E2)));
    *P2 = cuCadd(cmul_real(data->alpha, *P2), cmul_real(data->one_minus_alpha_div_a, drive));

    /* ------ the N update equation */
    //averageP = 0.5 * (gainP[iN] + gainP[i]);
    double exchange = (cuCadd(cuCmul(cuConj(*E1), *P1), cuCmul(cuConj(*E2), *P2))).y;
    if (data->diode_type[blockIdx.x] == 1 /* gain */) {
        *N0 = *N0 + data->dt * ((- *N0) / data->tGain + data->C_gain * exchange + data->Pa);
    } else if (data->diode_type[blockIdx.x] == 2 /* absorber */) {
        *N0 = *N0 + data->dt * ((data->N0b - *N0) / data->tLoss + data->C_loss * exchange);
    }

    /* ------ the E update equation */
    *E1 = cuCadd(*E1, cmul_real(data->dt * data->coupling_out_gain , cuCmul(data->I1, *P1)));
    *E2 = cuCadd(*E2, cmul_real(data->dt * data->coupling_out_gain , cuCmul(data->I1, *P2)));
}

// Initialize
int diode_cavity_init(DiodeCavityCtx *ctx) {

    DiodeCavityCtx ctx_local;
    memcpy(&ctx_local, ctx, sizeof(DiodeCavityCtx));
    if (ctx->d_ctx == NULL) {
        /* allocate cuda arrays */
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_type, sizeof(int) * ctx->diode_length));
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_1, sizeof(int) * ctx->diode_length));
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_2, sizeof(int) * ctx->diode_length));
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_N0, sizeof(double) * ctx->diode_length * ctx->N_x));
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_P_dir_1, sizeof(cuDoubleComplex) * ctx->diode_length * ctx->N_x));
        CHECK_CUDA(cudaMalloc(&ctx_local.diode_P_dir_2, sizeof(cuDoubleComplex) * ctx->diode_length * ctx->N_x));
        CHECK_CUDA(cudaMalloc(&ctx_local.amplitude, sizeof(cuDoubleComplex) * ctx->N * ctx->N_x));
        /* fill cuda arrays */
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_type, ctx->diode_type, sizeof(int) * ctx->diode_length, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_1, ctx->diode_pos_1, sizeof(int) * ctx->diode_length, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_2, ctx->diode_pos_2, sizeof(int) * ctx->diode_length, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_N0, ctx->diode_N0, sizeof(double) * ctx->diode_length * ctx->N_x, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_P_dir_1, ctx->diode_P_dir_1, sizeof(cuDoubleComplex) * ctx->diode_length * ctx->N_x, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.diode_P_dir_2, ctx->diode_P_dir_2, sizeof(cuDoubleComplex) * ctx->diode_length * ctx->N_x, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx_local.amplitude, ctx->amplitude, sizeof(cuDoubleComplex) * ctx->N * ctx->N_x, cudaMemcpyHostToDevice));

        /* copy ctx_local to device */
        DiodeCavityCtx *ctx_cuda = NULL;
        CHECK_CUDA(cudaMalloc(&ctx_cuda, sizeof(DiodeCavityCtx)));
        CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_local, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));
        ctx->d_ctx = ctx_cuda;
    } else {
        DiodeCavityCtx *ctx_cuda = ctx->d_ctx;
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
        /* copy ctx_local to device so that new parameters are used but old buffers are preserved */
        CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_local, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));
    }

    return 0;
}

// Run one Cavity tick
int diode_cavity_run(DiodeCavityCtx *ctx) {

    for (int i_round = 0; i_round < ctx->n_rounds; i_round++) {
        for (int offset = 0; offset < ctx->N_x; offset++) {
            // Call the CUDA kernel to process the cavity round trip
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
