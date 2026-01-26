

#ifndef DIODE_CAVITY_H
#define DIODE_CAVITY_H
#include <complex.h>
#include <cuComplex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _DiodeParams {
    int n_cavity_bits; // log base 2 of size of cavity
    int n_x_bits; // log base 2 of size of transverse dimension
    int n_rounds; // number of round trips per call
    int target_slice_lenghth;
    int target_slice_start;
    int target_slice_end;

    int N; // number of spatial cells in cavity (2^n_cavity_bits)
    int N_x; // number of transverse cells (2^n_x_bits)
    int diode_length; // number of diode total components

    double dt;
    double tGain;
    double tLoss;
    double C_gain;
    double C_loss;
    double N0b;
    double Pa;
    double kappa;
    double alpha;
    double one_minus_alpha_div_a;
    double coupling_out_gain;

    double left_linear_cavity[4]; // ABCD matrix elements for left linear cavity section
    double right_linear_cavity[4]; // ABCD matrix elements for right linear cavity section

} DiodeParams;

typedef struct _DiodeCavityCtx {
    int n_cavity_bits; // log base 2 of size of cavity
    int n_x_bits; // log base 2 of size of transverse dimension
    int n_rounds; // number of round trips per call
    int target_slice_lenghth;
    int target_slice_start;
    int target_slice_end;

    int N; // number of spatial cells in cavity (2^n_cavity_bits)
    int N_x; // number of transverse cells (2^n_x_bits)

    double dt;
    double tGain;
    double tLoss;
    double C_gain;
    double C_loss;
    double N0b;
    double Pa;
    double kappa;
    double alpha;
    double one_minus_alpha_div_a;
    double coupling_out_gain;
    cuDoubleComplex I1;

    int diode_length; // number of diode total components
    int *diode_type; // type of diode component (1=gain, 2=absorber)
    int *diode_pos_1; // position index of each diode component at the left to right beam direction
    int *diode_pos_2; // position index of each diode component at the right to left beam direction

    double *diode_N0; // equilibrium inversion for each diode component (unified for both directions, size diode_length * N_x)
    cuDoubleComplex *diode_P_dir_1; // polarization for each diode component, left to right direction (size diode_length * N_x)
    cuDoubleComplex *diode_P_dir_2; // polarization for each diode component, right to left direction (size diode_length * N_x)

    cuDoubleComplex *amplitude; // buffer for field amplitude between diode components (size N * N_x)

    double left_linear_cavity[4]; // ABCD matrix elements for left linear cavity section
    double right_linear_cavity[4]; // ABCD matrix elements for right linear cavity section

    struct _DiodeCavityCtx *d_ctx; // device context pointer
} DiodeCavityCtx;

// Initialize context
int diode_cavity_build(DiodeCavityCtx *ctx_host);

int diode_cavity_prepare(DiodeCavityCtx *ctx_host);

int diode_cavity_run(DiodeCavityCtx *ctx_host);

int diode_cavity_extract(DiodeCavityCtx *ctx_host);

// Cleanup
void diode_cavity_destroy(DiodeCavityCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif 
