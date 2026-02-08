#define USE_CUDA_CODE 1

#include "diode_cavity.h"
#ifdef DIODE_CAVITY_H
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <curand_kernel.h>


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

__global__ void diode_cavity_extract_kernel(DiodeCavityCtx *data, double factor) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int io = (lround(i * factor) + data->target_slice_start) * data->N_x + data->N_x / 2;
    int io1 = lround(i * factor) + data->target_slice_start;
    // if (i < 10) {
    //     printf("i = %d io = %d io1 = %d\n", i, io, io1);
    // }
    data->ext_beam_in[i] = data->amplitude[io];
    data->ext_beam_out[i] = data->amplitude_out[io];
    data->ext_gain_N[i] = data->gain_N[io1];
    data->ext_gain_polarization_dir1[i] = data->gain_polarization_dir1[io1];
    data->ext_gain_polarization_dir2[i] = data->gain_polarization_dir2[io1];
    data->ext_loss_N[i] = data->loss_N[io1];
    data->ext_loss_polarization_dir1[i] = data->loss_polarization_dir1[io1];
    data->ext_loss_polarization_dir2[i] = data->loss_polarization_dir2[io1];
}

__global__ void init_rng(curandStatePhilox4_32_10_t *rng, unsigned long seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, i, 0, &rng[i]);
}

__global__ void diode_cavity_round_trip_kernel(DiodeCavityCtx *data, int offset, int mode) {
    int i = blockIdx.x; // 0:130
    int j = threadIdx.x; // 0:31

    /* position on the left and right beams*/
    int index_1 = ((data->diode_pos_1[i] + offset) % data->N) * data->N_x + j;
    int index_2 = ((data->diode_pos_2[i] + offset) % data->N) * data->N_x + j;

    // if (offset < 1 && i < 2 && j < 3) {
    //     printf("bx = %d tx=%d bd = %d --- i1 = %d i2 = %d\n", blockIdx.x, threadIdx.x, blockDim.x, index_1, index_2);
    // }

    // if (blockIdx.x == 0 && threadIdx.x == 0 && offset == 0) {
    //     for (int ii = 0; ii < data->diode_length * blockDim.x; ii++) {
    //         printf("ii = %d, diode = %f\n", ii, data->diode_N0[ii]);
    //     }

    // }
    cuDoubleComplex *__restrict__ pE1 = data->amplitude;
    cuDoubleComplex E1 = pE1[index_1];
    cuDoubleComplex *__restrict__ pE2 = data->amplitude;
    cuDoubleComplex E2 = pE2[index_2];
    if (data->diode_type[blockIdx.x] == 3 /* output coupler and initializer */) {
        if ((mode & 1) != 0 /* first round trip */) {

            /* initialize the beam according to the selected type */
            if (data->beam_init_type == 0 /* pulse */) {
                double pulseWidth = data->beam_init_parameter;
                double x = (offset + data->N / 2) % data->N;
                double pulseVal = 4.0E08 * exp(-((x - data->N / 2) * (x - data->N / 2) / (2 * pulseWidth * pulseWidth)));

                E1 = make_cuDoubleComplex(pulseVal, 0.0);
                //E2 = make_cuDoubleComplex(pulseVal, 0.0);
            } else if (data->beam_init_type == 1 /* noise */) {
                //double noise_amplitude = data->beam_init_parameter;
                double real_part = 1.0;//noise_amplitude * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
                double imag_part = 1.0;//noise_amplitude * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
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
            // pE1[index_1] = E1;
            //pE2[index_2] = E2;
        }
        if ((mode & 2) != 0 /* last round trip */) {
            /* copy to outside of the cavity */        
            data->amplitude_out[index_1] = cmul_real(data->oc_out_val, E1);
        }
        pE1[index_1] = cmul_real(data->oc_val_sqrt, E1);

        return;
    }
    /* access arrays with restrict pointers */
    int idi = blockIdx.x * blockDim.x + threadIdx.x;
    double *__restrict__ pN0 = data->diode_N0;
    double N0 = pN0[idi];
    cuDoubleComplex *__restrict__ pP1 = data->diode_P_dir_1;
    cuDoubleComplex *__restrict__ pP2 = data->diode_P_dir_2;
    cuDoubleComplex P1 = pP1[idi];
    cuDoubleComplex P2 = pP2[idi];
    cuDoubleComplex I1 = make_cuDoubleComplex(0.0, 1.0);
    // if (offset < 1 && i + j < 10) {
    //     printf("idi %d b%d t%d \n", idi, blockIdx.x, threadIdx.x);
    // }
    auto local_rng = ((curandStatePhilox4_32_10_t *)(data->rng))[idi];
    // printf("ok\n");
    double2 z_rng = curand_normal2_double(&local_rng);
    if (offset < 1 && i + j < 2) {
        printf("idi %d rnd = %f + i%f\n", idi, z_rng.x, z_rng.y);
    }
    cuDoubleComplex noise;
    noise.x = z_rng.x * 1.0E-06;   // real
    noise.y = z_rng.y * 1.0E-06;   // imag

    /* ------ the P update equation */
    cuDoubleComplex drive = cuCmul(I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E1));
    P1 = cuCsub(cmul_real(data->alpha, P1), drive);
    drive = cuCmul(I1, cmul_real(data->one_minus_alpha_div_a * data->kappa * N0, E2));
    P2 = cuCsub(cmul_real(data->alpha, P2), drive);

    /* ------ the N update equation */
    //averageP = 0.5 * (gainP[iN] + gainP[i]);
    double exchange = 1.0E-27 * (cuCadd(cuCmul(cuConj(E1), P1), cuCmul(cuConj(E2), P2))).y;
    if (data->diode_type[blockIdx.x] == 1 /* gain */) {
        N0 = N0 + data->dt * ((- N0) / data->tGain + data->C_gain * exchange + data->Pa);
    } else if (data->diode_type[blockIdx.x] == 2 /* absorber */) {
        N0 = N0 + data->dt * ((data->N0b - N0) / data->tLoss + data->C_loss * exchange);
    }

    /* ------ the E update equation */
    if (offset < 1 && i + j < 2) {
        printf("idi %d rnd = %f + i%f\n", idi, noise.x, noise.y);
    }
    pE1[index_1] = cuCadd(cuCadd(pE1[index_1], noise), cmul_real(data->dt * 1.0E-25 * data->coupling_out_gain , cuCmul(I1, P1)));
    pE2[index_2] = cuCadd(cuCadd(pE2[index_2], noise), cmul_real(data->dt * 1.0E-25 * data->coupling_out_gain , cuCmul(I1, P2)));

    //pE1[index_1] = cuCadd(pE1[index_1], cmul_real(data->dt * 1.0E-25 * data->coupling_out_gain , cuCmul(I1, P1)));
    //pE2[index_2] = cuCadd(pE2[index_2], cmul_real(data->dt * 1.0E-25 * data->coupling_out_gain , cuCmul(I1, P2)));

    /* store back updated values */
    pN0[idi] = N0;
    pP1[idi] = P1;
    pP2[idi] = P2;
    if (j == threadIdx.x / 2) {
        if (blockIdx.x == 5) {
            int idb = (data->diode_pos_1[i] + offset) % data->N;
            // if (offset < 20) {
            //     printf("idi = %d, idb = %d, lgain = %d\n", idi, idb, data->gain_length);
            // }
            data->gain_N[idb] = N0;
            data->gain_polarization_dir1[idb] = P1;
            data->gain_polarization_dir2[idb] = P2;
        }
        if (blockIdx.x == data->gain_length + data->loss_length/ 2) {
            int idb = (data->diode_pos_1[i] + offset) % data->N;
            data->loss_N[idb] = N0;
            data->loss_polarization_dir1[idb] = P1;
            data->loss_polarization_dir2[idb] = P2;
        }
    }
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
int cuMemsetValueDouble(double **devPtr, size_t s, size_t n, double value) {
    // Wrap raw pointer and fill
    thrust::device_ptr<double> dev_ptr(*devPtr);
    thrust::fill(dev_ptr + s, dev_ptr + s + n, value);
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
    cuAllocZero((void **)&ctx_local.diode_N0, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* inversion density zero init*/
    printf("fill %f into %d values\n", ctx_host->Pa * ctx_host->tGain,ctx_host->gain_length * ctx_host->N_x);
    cuMemsetValueDouble(&ctx_local.diode_N0, 0, ctx_host->gain_length * ctx_host->N_x, ctx_host->Pa * ctx_host->tGain); /* inversion start density for gain */
    cuMemsetValueDouble(&ctx_local.diode_N0, ctx_host->gain_length * ctx_host->N_x, ctx_host->loss_length * ctx_host->N_x, ctx_host->N0b); /* inversion start density for loss */
    cuAllocZero((void **)&ctx_local.diode_P_dir_1, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for left to right beam*/
    cuAllocZero((void **)&ctx_local.diode_P_dir_2, sizeof(cuDoubleComplex) * ctx_host->diode_length * ctx_host->N_x); /* polarization for right to left beam*/ 
    cudaMalloc((void **)&ctx_local.rng, ctx_host->diode_length * ctx_host->N_x * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<ctx_host->diode_length, ctx_host->N_x>>>((curandStatePhilox4_32_10_t *)ctx_local.rng, 345);

    cuAllocZero((void **)&ctx_local.amplitude, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x); /* beam amplitude values */
    cuAllocZero((void **)&ctx_local.amplitude_out, sizeof(cuDoubleComplex) * ctx_host->N * ctx_host->N_x); /* beam amplitude as it comes out of the cavity */
    cuAllocZero((void **)&ctx_local.gain_N, sizeof(double) * ctx_host->N); /* gain carrier density internal */
    cuAllocZero((void **)&ctx_local.gain_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->N); /* gain polarization */
    cuAllocZero((void **)&ctx_local.gain_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->N); /* gain polarization */
    cuAllocZero((void **)&ctx_local.loss_N, sizeof(double) * ctx_host->N); /* loss carrier density */
    cuAllocZero((void **)&ctx_local.loss_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->N); /* loss polarization */
    cuAllocZero((void **)&ctx_local.loss_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->N); /* loss polarization */

    /* allocate cuda arrays for returning data from cuda to C */
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_beam_in, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_beam_out, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_gain_N, sizeof(double) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_gain_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_gain_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_loss_N, sizeof(double) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_loss_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->ext_len));
    CHECK_CUDA(cudaMalloc(&ctx_local.ext_loss_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->ext_len));

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

    ctx_help.n_rounds = ctx_host->n_rounds;
    ctx_help.target_slice_length = ctx_host->target_slice_length;
    ctx_help.target_slice_start = ctx_host->target_slice_start;
    ctx_help.target_slice_end = ctx_host->target_slice_end;
    ctx_help.start_round = ctx_host->start_round;

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
    ctx_help.oc_val_sqrt = ctx_host->oc_val_sqrt;
    ctx_help.oc_out_val = ctx_host->oc_out_val;
    ctx_help.ext_len = ctx_host->ext_len;

    memcpy(ctx_help.left_linear_cavity, ctx_host->left_linear_cavity, sizeof(ctx_host->left_linear_cavity));
    memcpy(ctx_help.right_linear_cavity, ctx_host->right_linear_cavity, sizeof(ctx_host->right_linear_cavity));

    CHECK_CUDA(cudaMemcpy(ctx_cuda, &ctx_help, sizeof(DiodeCavityCtx), cudaMemcpyHostToDevice));

    return 0;
}

// Run one Cavity tick
int diode_cavity_run(DiodeCavityCtx *ctx_host) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);   // device 0
    printf("multiProcessorCount %d\nmaxThreadsPerMultiProcessor %d\nmaxThreadsPerBlock %d\nwarpSize %d\nsharedMemPerBlock %ld\nregsPerBlock %d\n",
         prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.maxThreadsPerBlock ,prop.warpSize ,prop.sharedMemPerBlock ,prop.regsPerBlock);


    DiodeCavityCtx *ctx_dev = ctx_host->d_ctx;
    int threads = ctx_host->N_x;
    int blocks = ctx_host->diode_length;
    printf("threads %d blocks %d\n", threads, blocks);

    /* perform n_rounds of cavity round trips */
    int mode = 0;
    if (ctx_host->start_round == 0) {
        mode = 1; // flagging type of round trip (flags: bit 0 - first round trip, bit 1 - last round trip)
    }
    for (int i_round = 0; i_round < ctx_host->n_rounds; i_round++) {
        /* perform one round trip */
        if (i_round == ctx_host->n_rounds - 1) {
            mode |= 2; // flagging last round trips
        }
        printf("Cavity round %d\n", i_round + 1);        
        for (int offset = 0; offset < ctx_host->N; offset++) {
            // Call the CUDA kernel to process a single step in ther round trip
            diode_cavity_round_trip_kernel<<<blocks, threads>>>(ctx_dev, offset, mode);
            CHECK_CUDA(cudaDeviceSynchronize());
        }
        mode = 0; 
    }

    return 0;
}


int diode_cavity_extract(DiodeCavityCtx *ctx_host) {
    printf("CUDA extract Context %p %p\n", ctx_host, ctx_host->d_ctx);

    DiodeCavityCtx ctx_local; 
    DiodeCavityCtx *ctx_cuda = ctx_host->d_ctx;

    int threads = ctx_host->ext_len / 32;
    double factor = (double)(ctx_host->target_slice_end - ctx_host->target_slice_start) / (double)(ctx_host->ext_len);

    printf("CUDA extract start\n");

    diode_cavity_extract_kernel<<<32, threads>>>(ctx_cuda, factor);
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("CUDA extract done\n");

    /* collect the existing pointers from the device*/
    CHECK_CUDA(cudaMemcpy(&ctx_local, ctx_cuda, sizeof(DiodeCavityCtx), cudaMemcpyDeviceToHost));

    /* copy ctx_local to device so that new parameters are used but old buffers are preserved */
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_beam_in, ctx_local.ext_beam_in, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_beam_out, ctx_local.ext_beam_out, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_gain_N, ctx_local.ext_gain_N, sizeof(double) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_gain_polarization_dir1, ctx_local.ext_gain_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_gain_polarization_dir2, ctx_local.ext_gain_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_loss_N, ctx_local.ext_loss_N, sizeof(double) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_loss_polarization_dir1, ctx_local.ext_loss_polarization_dir1, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(ctx_host->ext_loss_polarization_dir2, ctx_local.ext_loss_polarization_dir2, sizeof(cuDoubleComplex) * ctx_host->ext_len, cudaMemcpyDeviceToHost));

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
    cudaFree(ctx_host->rng);
    cudaFree(ctx_host->d_ctx);

}
#endif
