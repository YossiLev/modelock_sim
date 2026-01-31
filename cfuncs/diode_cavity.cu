#ifdef USE_CUDA_CODE

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
    return make_cuDoubleComplex(a * z.x, a * z.y);w
}
__host__ __device__
inline cuDoubleComplex cneg(cuDoubleComplex z) {
    return make_cuDoubleComplex(-z.x, -z.y);
}

__global__ void diode_cavity_extract_kernel(DiodeCavityCtx *data, double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int io = (lround(i * factor) + data->target_slice_start) * data->N_x + data->N_x / 2;
    int io1 = lround(i * factor) + data->target_slice_start;
    data->ext_beam_in[i] = data->amplitude[io];
    data->ext_beam_out[i] = data->amplitude_out[io];
    data->ext_gain_N[i] = data->gain_N[io1];
    data->ext_gain_polarization[i] = data->gain_polarization[io1];
    data->ext_loss_N[i] = data->loss_N[io1];
    data->ext_loss_polarization[i] = data->loss_polarization[io1];
}

__global__ void diode_cavity_round_trip_kernel(DiodeCavityCtx *data, int offset, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /* position on the left and right beams*/
    int index_1 = ((data->diode_pos_1[i] + offset) % data->N) * data->N_x + threadIdx.x;
    int index_2 = ((data->diode_pos_2[i] + offset) % data->N) * data->N_x + threadIdx.x;

    cuDoubleComplex *__restrict__ pE1 = data->amplitude;
    cuDoubleComplex E1 = pE1[index_1];
    cuDoubleComplex *__restrict__ pE2 = data->amplitude;
    cuDoubleComplex E2 = pE2[index_2];
    if (data->diode_type[blockIdx.x] == 3 /* output coupler and initializer */) {
        if (mode & 1 /* first round trip */) {
            /* initialize the beam according to the selected type */
            if (data->beam_init_type == 0 /* pulse */) {
                double pulseWidth = data->beam_init_parameter;
                double x = offset;
                double pulseVal = exp(-((x - data->N / 2) * (x - data->N / 2) / (2 * pulseWidth * pulseWidth)));
                E1 = make_cuDoubleComplex(pulseVal, 0.0);
                //E2 = make_cuDoubleComplex(pulseVal, 0.0);
            } else if (data->beam_init_type == 1 /* noise */) {
                double noise_amplitude = data->beam_init_parameter;
                double real_part = noise_amplitude * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
                double imag_part = noise_amplitude * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
                E1 = make_cuDoubleComplex(real_part, imag_part);
                //E2 = make_cuDoubleComplex(real_part, imag_part);
            } else if (data->beam_init_type == 2 /* cw */) {
                double cw_amplitude = data->beam_init_parameter;
                E1 = make_cuDoubleComplex(cw_amplitude, 0.0);
                //E2 = make_cuDoubleComplex(cw_amplitude, 0.0);
            } else if (data->beam_init_type == 3 /* flat */) {
                double flat_amplitude = data->beam_init_parameter;
                E1 = make_cuDoubleComplex(flat_amplitude, 0.0);
                //E2 = make_cuDoubleComplex(flat_amplitude, 0.0);
            }
            pE1[index_1] = E1;
            //pE2[index_2] = E2;
        }
        if (mode & 2 /* last round trip */) {
            /* copy to outside of the cavity */
            data->amplitude_out[threadIdx.x] = E1;
        }
        return;
    }
    /* access arrays with restrict pointers */
    double *__restrict__ pN0 = data->diode_N0;
    double N0 = pN0[i];
    cuDoubleComplex *__restrict__ pP1 = data->diode_P_dir_1;
    cuDoubleComplex *__restrict__ pP2 = data->diode_P_dir_2;
    cuDoubleComplex P1 = pP1[i];
    cuDoubleComplex P2 = pP2[i];
    cuDoubleComplex I1 = make_cuDoubleComplex(0.0, 1.0);

    /* ------ the P update equation */
    cuDoubleComplex drive = cuCmul(I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E1));
    P1 = cuCsub(cmul_real(data->alpha, P1), drive);
    drive = cuCmul(I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E2));
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
    pE1[index_1] = cuCadd(pE1[index_1], cmul_real(data->dt * data->coupling_out_gain , cuCmul(I1, P1)));
    pE2[index_2] = cuCadd(pE2[index_2], cmul_real(data->dt * data->coupling_out_gain , cuCmul(I1, P2)));

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

int diode_cavity_build(DiodeCavityCtx *ctx_host) {

    DiodeCavityCtx ctx_local;
    memcpy(&ctx_local, ctx_host, sizeof(DiodeCavityCtx));
    /* first time calling for initialization */
    /* allocate cuda arrays which are common to C and cuda */
    CHECK_CUDA(cudaMalloc(&ctx_local.diode_type, sizeof(int) * ctx_host->diode_length)); /* type: 1 gain, 2 absorber*/
    CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_1, sizeof(int) * ctx_host->diode_length)); /* position index for left to right beam*/
    CHECK_CUDA(cudaMalloc(&ctx_local.diode_pos_2, sizeof(int) * ctx_host->diode_length)); /* position index for right to left beam*/
    /* copy host to cuda arrays */
    CHECK_CUDA(cudaMemcpy(ctx_local.diode_type, ctx_host->diode_type, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_1, ctx_host->diode_pos_1, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ctx_local.diode_pos_2, ctx_host->diode_pos_2, sizeof(int) * ctx_host->diode_length, cudaMemcpyHostToDevice));

    /* allocate and initialize cuda arrays thart are internal to cuda */
    cuAllocValueDouble(&ctx_local.diode_N0, ctx_host->diode_length * ctx_host->N_x, 1.0); /* inversion density */
    cuAllocZero((void **)&ctx_local.diode_P_dir_1, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for left to right beam*/
    cuAllocZero((void **)&ctx_local.diode_P_dir_2, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for right to left beam*/ 
    cuAllocZero((void **)&ctx_local.amplitude, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x); /* beam amplitude values */
    cuAllocZero((void **)&ctx_local.amplitude_out, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x); /* beam amplitude as it comes out of the cavity */
    cuAllocZero((void **)&ctx_local.gain_N, sizeof(double) * ctx_host->N); /* gain carrier density internal */
    cuAllocZero((void **)&ctx_local.gain_polarization, sizeof(cuDoubleComplex) * ctx_host->N); /* gain polarization */
    cuAllocZero((void **)&ctx_local.loss_N, sizeof(double) * ctx_host->N); /* loss carrier density */
    cuAllocZero((void **)&ctx_local.loss_polarization, sizeof(cuDoubleComplex) * ctx_host->N); /* loss polarization */

    /* allocate cuda arrays for returning data from cuda to C */
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_beam_in, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_beam_out, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_gain_N, sizeof(double) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_gain_polarization, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_loss_N, sizeof(double) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_loss_polarization, sizeof(cuDoubleComplex) * ctx_host->ext_len));

    /* copy ctx_local to device */
    DiodeCavityCtx *ctx_cuda = NULL;
    CHECK_CUDA(cudaMalloc(&ctx_cuda, sizeof(DiodeCavityCtx)));
    CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_local, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));

    ctx_host->d_ctx = ctx_cuda;

    return 0;
}

int diode_cavity_prepare(DiodeCavityCtx *ctx_host) {
    DiodeCavityCtx ctx_help;
    DiodeCavityCtx *ctx_cuda = ctx_host->d_ctx;

    /* collect the existing pointers from the device*/
    CHECK_CUDA(cudaMemcpy(&ctx_help, ctx_cuda, sizeof(DiodeCavityCtx), cudaMemcpyDeviceToHost));
    /* copy the partial parameters from the host */
    ctx_help.diode_type = ctx_help.diode_type;
    ctx_help.diode_pos_1 = ctx_help.diode_pos_1;
    ctx_help.diode_pos_2 = ctx_help.diode_pos_2;
    ctx_help.diode_N0 = ctx_help.diode_N0;
    ctx_help.diode_P_dir_1 = ctx_help.diode_P_dir_1;
    ctx_help.diode_P_dir_2 = ctx_help.diode_P_dir_2;

    ctx_help.n_rounds = ctx_host->n_rounds;
    ctx_help.target_slice_length = ctx_host->target_slice_length;
    ctx_help.target_slice_start = ctx_host->target_slice_start;
    ctx_help.target_slice_end = ctx_host->target_slice_end;

    ctx_help.dt = ctx_host->dt;

    ctx_help.beam_init_type = ctx_host->beam_init_type;
    ctx_help.beam_init_parameter = ctx_host->beam_init_parameter;
    
    ctx_help.tGain = ctx_host->tGain;
    ctx_help.tLoss = ctx_host->tLoss;
    ctx_help.C_gain = ctx_host->C_gain;
    ctx_help.C_loss = ctx_host->C_loss;
    ctx_help.N0b = ctx_host->N0b;
    ctx_help.Pa = ctx_host->Pa;
    ctx_help.kappa = ctx_host->kappa;
    ctx_help.alpha = ctx_host->alpha;
    ctx_help.one_minus_alpha_div_a = ctx_host->one_minus_alpha_div_a;
    ctx_help.coupling_out_gain = ctx_host->coupling_out_gain;
    ctx_help.ext_len = ctx_host->ext_len;
#ifdef USE_CUDA_CODE
    ctx_help.ext_beam_in = ctx_host->ext_beam_in;
    ctx_help.ext_beam_out = ctx_host->ext_beam_out;
    ctx_help.ext_gain_N = ctx_host->ext_gain_N;
    ctx_help.ext_gain_polarization = ctx_host->ext_gain_polarization;
    ctx_help.ext_loss_N = ctx_host->ext_loss_N;
    ctx_help.ext_loss_polarization = ctx_host->ext_loss_polarization;
#endif
    CHECK_CUDA(cudaMemcpy(ctx_help.left_linear_cavity, ctx_host->left_linear_cavity, sizeof(ctx_host->left_linear_cavity), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(ctx_help.right_linear_cavity, ctx_host->right_linear_cavity, sizeof(ctx_host->right_linear_cavity), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_help, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));

    return 0;
}

int diode_cavity_extract(DiodeCavityCtx *ctx_host) {
    DiodeCavityCtx ctx_local; 
    DiodeCavityCtx *ctx_cuda = ctx_host->d_ctx;

    int threads = ctx_host->ext_len / 32;
    double factor = (double)(ctx_host->target_slice_end - ctx_host->target_slice_start) / (double)(ctx_host->ext_len);

    diode_cavity_extract_kernel<<<32, threads>>>(ctx_host->d_ctx, factor);

    /* collect the existing pointers from the device*/
    CHECK_CUDA(cudaMemcpy(&ctx_local, ctx_cuda, sizeof(DiodeCavityCtx), cudaMemcpyDeviceToHost));

    /* copy ctx_local to device so that new parameters are used but old buffers are preserved */
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_beam_in, ctx_local.ext_beam_in, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_beam_out, ctx_local.ext_beam_out, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_gain_N, ctx_local.ext_gain_N, sizeof(double) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_gain_polarization, ctx_local.ext_gain_polarization, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_loss_N, ctx_local.ext_loss_N, sizeof(double) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_loss_polarization, ctx_local.ext_loss_polarization, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));

    return 0;
}

// Run one Cavity tick
int diode_cavity_run(DiodeCavityCtx *ctx_host) {

    DiodeCavityCtx *ctx = ctx_host->d_ctx;

    /* perform n_rounds of cavity round trips */
    int mode = 1; // flagging type of round trip (flags: bit 0 - first round trip, bit 1 - last round trip)
    for (int i_round = 0; i_round < ctx->n_rounds; i_round++) {
        /* perform one round trip */
        if (i_round == ctx->n_rounds - 1) {
            mode |= 2; // flagging last round trips
        }
        for (int offset = 0; offset < ctx->N_x; offset++) {
            // Call the CUDA kernel to process a single step in ther round trip
            int threads = ctx->N_x;
            int blocks = ctx->diode_length;
            diode_cavity_round_trip_kernel<<<blocks, threads>>>(ctx->d_ctx, offset, mode);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        mode = 0; 
    }

    return 0;
}

// Cleanup
void diode_cavity_destroy(DiodeCavityCtx *ctx_host) {

    cudaFree(ctx_host->diode_type);
    cudaFree(ctx_host->diode_pos_1);
    cudaFree(ctx_host->diode_pos_2);
    cudaFree(ctx_host->diode_N0);
    cudaFree(ctx_host->diode_P_dir_1);
    cudaFree(ctx_host->diode_P_dir_2);
    cudaFree(ctx_host->amplitude);
    cudaFree(ctx_host->d_ctx);

}
#endif

#endif