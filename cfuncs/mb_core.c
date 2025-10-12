/*
 mb_core.c
 Maxwell-Bloch style core for a point-like Gain + Saturable Absorber element.
 C99 (uses <complex.h>) single-file implementation suitable as a kernel.

 Author: ChatGPT
 Date: 2025-10-12

 API summary:
   typedef struct TwoLevelMedium TwoLevelMedium;
   void tlm_init(TwoLevelMedium *m, double dt, double Gamma, double omega0,
                 double kappa, double C, double T1, double N_eq, double pump,
                 double coupling_out, double initial_N, double complex initial_P);
   void tlm_drive_emit(TwoLevelMedium *m, double complex E, double complex *deltaE_out);
   void tlm_update_inversion(TwoLevelMedium *m, double complex E);
   void process_block(TwoLevelMedium *gain, TwoLevelMedium *abs,
                      const double complex *x1, const double complex *x2,
                      double complex *y1, double complex *y2, size_t L);

 Notes:
 - Work in complex-envelope representation (baseband).
 - dt should resolve the fastest dynamics (Gamma, Rabi).
 - The model uses a simple Euler update for N; use smaller dt or better integrator if needed.
 - This code intentionally keeps the per-sample loop simple and sequential.
   For speed: compile with -O3 and consider replacing loop body with a platform-optimized routine.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stddef.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    /* parameters */
    double dt;            /* time step */
    double Gamma;         /* polarization decay (real) */
    double omega0;        /* carrier offset (rad/s); 0 for envelope at center */
    double kappa;         /* coupling into polarization */
    double C;             /* converts Im(conj(E)*P) into inversion rate */
    double T1;            /* inversion recovery time */
    double N_eq;          /* equilibrium inversion (pump equilibrium) */
    double pump;          /* external pump term (units of N per second) */
    double coupling_out;  /* how strongly P radiates to fields */

    /* state */
    double complex alpha; /* recurrence coefficient = exp(-(Gamma + i*omega0) * dt) */
    double complex P;     /* polarization (complex) */
    double N;             /* inversion (real) */
} TwoLevelMedium;


/* Initialize a TwoLevelMedium (call before use) */
static inline void tlm_init(TwoLevelMedium *m,
                            double dt, double Gamma, double omega0,
                            double kappa, double C, double T1, double N_eq,
                            double pump, double coupling_out,
                            double initial_N, double complex initial_P)
{
    m->dt = dt;
    m->Gamma = Gamma;
    m->omega0 = omega0;
    m->kappa = kappa;
    m->C = C;
    m->T1 = T1;
    m->N_eq = N_eq;
    m->pump = pump;
    m->coupling_out = coupling_out;

    /* compute complex alpha */
    double complex z = -(Gamma + I * omega0) * dt;
    m->alpha = cexp(z);

    m->P = initial_P;
    m->N = initial_N;
}

/* Drive polarization with total field E and return emitted deltaE (P-dependent) */
static inline void tlm_drive_emit(TwoLevelMedium *m, double complex E, double complex *deltaE_out)
{
    /* P <- alpha*P + (1 - alpha) * (kappa * E * N) */
    double complex one_minus_alpha = 1.0 + 0.0*I - m->alpha; /* (1 - alpha) */
    /* compute drive = kappa * E * N */
    double complex drive = m->kappa * E * (m->N + 0.0); /* ensure complex multiply */
    m->P = m->alpha * m->P + one_minus_alpha * drive;

    /* emitted field contribution */
    *deltaE_out = m->coupling_out * m->P;
}

/* Update inversion using coherent exchange term: N <- N + dt * ( (N_eq - N)/T1 - C * Im(conj(E)*P) + pump ) */
static inline void tlm_update_inversion(TwoLevelMedium *m, double complex E)
{
    /* exchange = Im(conj(E) * P) */
    double complex conjE = conj(E);
    double complex tmp = conjE * m->P;
    double exchange = carg(tmp) == 0.0 ? creal(tmp) * 0.0 : 0.0; /* placeholder to avoid unused variable warnings (overwritten) */

    /* compute Im(conjE * P) correctly: imag(tmp) */
    exchange = cimag(tmp);

    double dN = (m->N_eq - m->N) / m->T1 - m->C * exchange + m->pump;
    m->N += m->dt * dN;
}

/* Process a block of samples (length L). Arrays must be allocated by caller.
   x1, x2: input complex envelopes (forward/backward)
   y1, y2: outputs
   gain, abs: pointers to TwoLevelMedium for gain and absorber (can share params)
*/
void process_block(TwoLevelMedium *gain, TwoLevelMedium *absr,
                   const double complex *x1, const double complex *x2,
                   double complex *y1, double complex *y2, size_t L)
{
    size_t i;
    for (i = 0; i < L; ++i) {
        double complex xi1 = x1[i];
        double complex xi2 = x2[i];
        double complex E = xi1 + xi2; /* coherent total field */

        /* drive gain polarization and absorber polarization */
        double complex delta_gain = 0.0 + 0.0*I;
        double complex delta_abs  = 0.0 + 0.0*I;
        tlm_drive_emit(gain, E, &delta_gain);
        tlm_drive_emit(absr, E, &delta_abs);

        /* emitted contributions are added to both directions (thin-medium approx).
           Geometry-dependent splitting can be added here if needed. */
        double complex out1 = xi1 + delta_gain + delta_abs;
        double complex out2 = xi2 + delta_gain + delta_abs;

        /* update inversions after emission (coherent exchange) */
        tlm_update_inversion(gain, E);
        tlm_update_inversion(absr, E);

        /* optional scalar multiplicative gain/loss step:
           many models do not require this because deltaE already depends on N via P.
           If you want a small-signal scalar, apply it here. */
        /* Example (disabled by default): */
        /* double sg_gain = 1.0 + 0.0 * (gain->N / gain->N_eq); */
        /* double sg_abs  = 1.0 - 0.0 * (absr->N / absr->N_eq); */
        /* out1 *= sg_gain * sg_abs; out2 *= sg_gain * sg_abs; */

        y1[i] = out1;
        y2[i] = out2;
    }
}

/* ---------------- Example main() showing usage ---------------- */
#ifdef MB_CORE_TEST
int main(void)
{
    /* Example parameters — adjust to your units */
    double dt = 1e-12; /* 1 ps */
    double Gamma_gain = 2.0 * M_PI * 1e9;  /* 1 GHz */
    double Gamma_abs  = 2.0 * M_PI * 5e9;  /* 5 GHz */

    TwoLevelMedium gain;
    TwoLevelMedium absr;

    /* init: dt, Gamma, omega0, kappa, C, T1, N_eq, pump, coupling_out, initial_N, initial_P */
    tlm_init(&gain, dt, Gamma_gain, 0.0, 1.0, 1.0, 1e-9, 1.0, 1e6, 0.01, 1.0, 0.0 + 0.0*I);
    tlm_init(&absr, dt, Gamma_abs, 0.0, 1.0, 1.0, 5e-12, 0.0, 0.0, 1.0, 0.0 + 0.0*I);

    size_t L = 2000;
    double complex *x1 = (double complex*) malloc(sizeof(double complex) * L);
    double complex *x2 = (double complex*) malloc(sizeof(double complex) * L);
    double complex *y1 = (double complex*) malloc(sizeof(double complex) * L);
    double complex *y2 = (double complex*) malloc(sizeof(double complex) * L);

    if (!x1 || !x2 || !y1 || !y2) {
        fprintf(stderr, "alloc fail\n"); return 1;
    }

    /* small test tones */
    size_t i;
    for (i = 0; i < L; ++i) {
        double t = (double)i * dt;
        x1[i] = 1e-3 * cexp(I * 2.0 * M_PI * 1e6 * t);
        x2[i] = 1e-3 * cexp(-I * 2.0 * M_PI * 1e6 * t);
    }

    process_block(&gain, &absr, x1, x2, y1, y2, L);

    /* print first few outputs */
    for (i = 0; i < 8 && i < L; ++i) {
        printf("y1[%zu] = %.6e + %.6e i,  y2[%zu] = %.6e + %.6e i\n",
               i, creal(y1[i]), cimag(y1[i]), i, creal(y2[i]), cimag(y2[i]));
    }

    free(x1); free(x2); free(y1); free(y2);
    return 0;
}
#endif
