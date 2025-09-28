
#ifndef FFT_FILTER_H
#define FFT_FILTER_H
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int N;
    int cutoff;
    void *d_data;     // device buffer
    void *plan;       // cufftHandle (opaque here)
} FFTFilterCtx;

// Initialize context
int fft_filter_init(FFTFilterCtx *ctx, int N, int cutoff);

// Run one FFT-filter-IFFT on given array
int fft_filter_run(FFTFilterCtx *ctx, double _Complex *arr);

// Cleanup
void fft_filter_destroy(FFTFilterCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
