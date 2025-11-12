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
/* Utility: squared magnitude */
static inline double cabs_square(complex double z) { double a = creal(z), b = cimag(z); return a*a + b*b; }

/* You should already have gaussian_rand01() defined (Box-Muller). */

/*
  Revised Maxwell-Bloch round-trip function with:
   - separate forward/backward polarizations for gain & absorber
   - exponential integrator for P, trapezoidal(P) for field & N updates
   - phase-aware splitting of stimulated emission between the two counterpropagating samples
   - pump-dependent dephasing (optional via pump_eta)
   - per-sample spontaneous noise
   - output coupler and mirror handling (reflection coefficient)
 
  NOTE: Units must be consistent. Comments inside the code indicate units/expectations.
*/
void mb_diode_round_trip(
    /* state arrays (length N) */
    double *gainN,                /* N per spatial cell for gain (shared inversion) */
    complex double *gainPf,       /* forward polarization per cell (complex) */
    complex double *gainPb,       /* backward polarization per cell (complex) */
    double *lossN,                /* inversion for absorber (shared) */
    complex double *lossPf,       /* absorber forward polarization */
    complex double *lossPb,       /* absorber backward polarization */

    /* diagnostics / outputs (length N) */
    double *gain_value,           /* optional diagnostics: local gain change fraction */
    double *loss_value,

    /* field arrays */
    complex double *pulse_amplitude,       /* full-round complex samples array (length N) */
    complex double *pulse_amplitude_after, /* output coupler extracted amplitude */

    /* simulation control */
    int n_rounds,
    int N,
    int loss_shift,        /* index separation so absorber sees pair (i, i+loss_shift) */
    int oc_shift,          /* output coupler position offset (component moving convention) */
    int gain_distance,     /* distance between absorber and gain indices in your mapping */

    /* physical & numeric parameters */
    double dt,             /* time step (s) */
    double gainWidth_THz,  /* gain linewidth in THz (converted to rad/s inside) */
    double Pa,             /* pump rate (gain) (units: carriers per sec per cell or normalized) */
    double Ta_ps,          /* gain T1 (ps) */
    double Ga0,            /* intrinsic polarization decay for gain (rad/s) or use gainWidth_THz */
    double Pb_pump,        /* pump for absorber (if any) */
    double Tb_ps,          /* absorber T1 (ps) */
    double Gb0,            /* intrinsic polarization decay for absorber (rad/s) */
    double N0a,            /* equilibrium inversion for gain (setpoint) */
    double oc_val,         /* amplitude retention of OC (R) 0..1 (amplitude^2 = power) */
    double pump_eta,       /* factor converting pump rate -> extra dephasing (s per pump unit) */
    complex double r_left, /* left mirror complex reflection coefficient (-1 for perfect) */
    complex double r_right /* right mirror reflection coeff */
) {
    /* ---- Tunable simulation constants (adjust to your unit normalization) ---- */
    const double kappa = 3.0e4;           /* drive strength in P ODE (units depend on normalization) */
    const double C_exch_gain = 1.0;       /* converts Im(E*P) -> population change (tune to match units) */
    const double C_exch_loss = 1.0;
    const double coupling_out_gain = 8e-16; /* converts P -> delta E (scale to your normalization) */
    const double coupling_out_loss = 6e-16;
    const double noise_prefactor = 1e-5;  /* spontaneous noise amplitude prefactor (tune) */
    const double eps = 1e-18;             /* small regularizer for intensity division (tune) */

    /* ---- derived / unit conversions ---- */
    double tGain = Ta_ps * 1e-12;
    double tLoss = Tb_ps * 1e-12;
    double Gamma_from_THz = gainWidth_THz * 2.0 * M_PI * 1e12; /* rad/s (if you supply this) */

    /* Use provided Ga0/Gb0 if nonzero; otherwise use Gamma_from_THz as default for gain */
    double Gamma_gain0 = (Ga0 > 0.0) ? Ga0 : Gamma_from_THz;
    double Gamma_loss0 = (Gb0 > 0.0) ? Gb0 : Gamma_gain0; /* default same as gain if not provided */

    double oc_val_sqrt = sqrt(fmax(0.0, oc_val));         /* amplitude retention */
    double oc_out_val = sqrt(fmax(0.0, 1.0 - oc_val));    /* transmitted amplitude */

    int bugs = 0;

    /* main round-trip loop */
    for (int i_round = 0; i_round < n_rounds; ++i_round) {
        for (int ii = 0; ii < N; ++ii) {
            int i = ii % N;
            int iN = (i + 1) % N; /* history-next index */

            /* ----------------- ABSORBER (loss) interaction ----------------- */
            int idx_loss_a = i;
            int idx_loss_b = (i + loss_shift) % N;

            complex double E_a = pulse_amplitude[idx_loss_a];
            complex double E_b = pulse_amplitude[idx_loss_b];
            complex double E_loss_drive = E_a + E_b; /* total local field driving absorber */

            /* pump dependent dephasing (additive) */
            double Gamma_loss = Gamma_loss0 + pump_eta * fmax(0.0, lossN[i]); /* pump_eta * lossN approx */
            complex double alpha_loss = cexp( - (Gamma_loss + I * 0.0) * dt );
            complex double one_m_alpha_loss = 1.0 + 0.0*I - alpha_loss;

            /* P updates for forward/backward polarizations (exponential integrator) */
            complex double P_f_old = lossPf[i];
            complex double P_b_old = lossPb[i];
            /* drive uses instantaneous N (old) and local direction field */
            complex double drive_f_loss = kappa * E_a * (lossN[i] + 0.0);
            complex double drive_b_loss = kappa * E_b * (lossN[i] + 0.0);

            complex double P_f_next = alpha_loss * P_f_old + one_m_alpha_loss * drive_f_loss;
            complex double P_b_next = alpha_loss * P_b_old + one_m_alpha_loss * drive_b_loss;

            /* midpoint polarizations for field & N updates */
            complex double P_f_avg = 0.5 * (P_f_old + P_f_next);
            complex double P_b_avg = 0.5 * (P_b_old + P_b_next);

            /* stimulated emission contributions (complex) from absorber per direction */
            complex double delta_loss_total_f = coupling_out_loss * P_f_avg;
            complex double delta_loss_total_b = coupling_out_loss * P_b_avg;

            /* per-mode exchange rates (power per second in simulation units) */
            double S_a = C_exch_loss * cimag(conj(E_a) * P_f_avg);
            double S_b = C_exch_loss * cimag(conj(E_b) * P_b_avg);

            /* amplitude increments aligned with existing phases to give requested power change */
            double I_a = cabs_square(E_a);
            double I_b = cabs_square(E_b);

            if (I_a + I_b > eps) {
                double dW_a = S_a * dt;
                double dW_b = S_b * dt;
                /* ΔE_j = (dW_j / I_j) * E_j  -> if I_j small, fallback below */
                if (I_a > eps) pulse_amplitude[idx_loss_a] += (dW_a / (I_a + eps)) * E_a;
                else pulse_amplitude[idx_loss_a] += 0.5 * delta_loss_total_f; /* fallback seeding */

                if (I_b > eps) pulse_amplitude[idx_loss_b] += (dW_b / (I_b + eps)) * E_b;
                else pulse_amplitude[idx_loss_b] += 0.5 * delta_loss_total_b;
            } else {
                /* both beams effectively zero: seed equally using P_avg */
                pulse_amplitude[idx_loss_a] += 0.5 * (delta_loss_total_f + delta_loss_total_b);
                pulse_amplitude[idx_loss_b] += 0.5 * (delta_loss_total_f + delta_loss_total_b);
            }

            /* spontaneous-like noise injection for absorber (independent) */
            {
                double sigma_sp = noise_prefactor * sqrt(fmax(0.0, lossN[i])) * sqrt(dt);
                complex double n1 = sigma_sp * (gaussian_rand01() + I * gaussian_rand01());
                complex double n2 = sigma_sp * (gaussian_rand01() + I * gaussian_rand01());
                pulse_amplitude[idx_loss_a] += n1;
                pulse_amplitude[idx_loss_b] += n2;
            }

            /* update inversion using SAME midpoint polarizations and the local total driving field */
            double exch_loss = cimag(conj(E_loss_drive) * (P_f_avg + P_b_avg));
            lossN[iN] = lossN[i] + dt * ( ( /* relaxation towards eq */ N0a - lossN[i]) / tLoss - C_exch_loss * exch_loss + Pb_pump );

            /* store next polarizations into history */
            lossPf[iN] = P_f_next;
            lossPb[iN] = P_b_next;

            /* diagnostics */
            {
                double oldI = I_a + I_b;
                double newI = cabs_square(pulse_amplitude[idx_loss_a]) + cabs_square(pulse_amplitude[idx_loss_b]);
                loss_value[i] = newI / (1e-6 + oldI);
            }

            /* ----------------- GAIN interaction (two polarizations) ----------------- */
            int idx_gain_a = (i + gain_distance) % N;
            int idx_gain_b = (i + loss_shift - gain_distance) % N;

            complex double Eg_a = pulse_amplitude[idx_gain_a];
            complex double Eg_b = pulse_amplitude[idx_gain_b];
            complex double E_gain_drive = Eg_a + Eg_b;

            /* pump dependent dephasing for gain */
            double Gamma_gain = Gamma_gain0 + pump_eta * fmax(0.0, gainN[i]);
            complex double alpha_gain = cexp( - (Gamma_gain + I * 0.0) * dt );
            complex double one_m_alpha_gain = 1.0 + 0.0*I - alpha_gain;

            
            complex double Gp_f_old = gainPf[i];
            complex double Gp_b_old = gainPb[i];
            
            complex double drive_f_gain = kappa * Eg_a * (gainN[i] + 0.0);
            complex double drive_b_gain = kappa * Eg_b * (gainN[i] + 0.0);

            complex double Gp_f_next = alpha_gain * Gp_f_old + one_m_alpha_gain * drive_f_gain;
            complex double Gp_b_next = alpha_gain * Gp_b_old + one_m_alpha_gain * drive_b_gain;

            
            complex double Gp_f_avg = 0.5 * (Gp_f_old + Gp_f_next);
            complex double Gp_b_avg = 0.5 * (Gp_b_old + Gp_b_next);

            
            complex double delta_gain_f = coupling_out_gain * Gp_f_avg;
            complex double delta_gain_b = coupling_out_gain * Gp_b_avg;

            /* per-mode stimulated exchange (phase-aware) */
            double S_g_a = C_exch_gain * cimag(conj(Eg_a) * Gp_f_avg);
            double S_g_b = C_exch_gain * cimag(conj(Eg_b) * Gp_b_avg);

            double Ig_a = cabs_square(Eg_a);
            double Ig_b = cabs_square(Eg_b);

            if (Ig_a + Ig_b > eps) {
                double dW_ga = S_g_a * dt;
                double dW_gb = S_g_b * dt;
                if (Ig_a > eps) pulse_amplitude[idx_gain_a] += (dW_ga / (Ig_a + eps)) * Eg_a;
                else pulse_amplitude[idx_gain_a] += 0.5 * delta_gain_f;
                if (Ig_b > eps) pulse_amplitude[idx_gain_b] += (dW_gb / (Ig_b + eps)) * Eg_b;
                else pulse_amplitude[idx_gain_b] += 0.5 * delta_gain_b;
            } else {
                /* seed equally */
                pulse_amplitude[idx_gain_a] += 0.5 * (delta_gain_f + delta_gain_b);
                pulse_amplitude[idx_gain_b] += 0.5 * (delta_gain_f + delta_gain_b);
            }

            /* spontaneous-like noise at gain */
            {
                double sigma_sp = noise_prefactor * sqrt(fmax(0.0, gainN[i])) * sqrt(dt);
                complex double n1 = sigma_sp * (gaussian_rand01() + I * gaussian_rand01());
                complex double n2 = sigma_sp * (gaussian_rand01() + I * gaussian_rand01());
                pulse_amplitude[idx_gain_a] += n1;
                pulse_amplitude[idx_gain_b] += n2;
            }

            /* inversion update for gain (using SAME midpoint polars) */
            double exch_gain = cimag(conj(E_gain_drive) * (Gp_f_avg + Gp_b_avg));
            gainN[iN] = gainN[i] + dt * ( (N0a - gainN[i]) / tGain - C_exch_gain * exch_gain + Pa );

            /* clamp negative inversion */
            if (gainN[iN] < 0.0) {
                gainN[iN] = 0.0;
                if (++bugs > 10) return; /* stop if runaway */
            }

            /* store polarizations */
            gainPf[iN] = Gp_f_next;
            gainPb[iN] = Gp_b_next;

            /* diagnostics */
            {
                double oldI = Ig_a + Ig_b;
                double newI = cabs_square(pulse_amplitude[idx_gain_a]) + cabs_square(pulse_amplitude[idx_gain_b]);
                gain_value[i] = newI / (1e-6 + oldI);
            }

            /* ---------- Output coupler (partial transmission) ----------
               apply OC at oc_loc similar to before: transmit then keep reflected amplitude
               order: compute transmitted out field, then scale local amplitude by reflection amplitude
            */
            int oc_loc = (oc_shift + i) % N;
            pulse_amplitude_after[oc_loc] = pulse_amplitude[oc_loc] * oc_out_val;
            pulse_amplitude[oc_loc] *= oc_val_sqrt;

            /* ---------- Mirror reflections (treat mirrors as moving components too) ----------
               apply local mirror reflections at indices (example: left=0, right=N/2) -- user may adjust
               Here we assume mirror positions are fixed shifts — you can set them globally outside.
               For safety we apply both reflections at their indices when encountered (idempotent if not at this step).
            */
            /* Example mirror positions (customize as needed) */
            int mirror_left_idx = (0 + i) % N;              /* if you want dynamic movement change base */
            int mirror_right_idx = (N/2 + i) % N;

            /* apply reflection */
            pulse_amplitude[mirror_left_idx] *= r_left;
            pulse_amplitude[mirror_right_idx] *= r_right;

        } /* end spatial loop */
    } /* end rounds loop */
}
