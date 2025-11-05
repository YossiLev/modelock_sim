#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "fft_filter.h"

// for linux compilation:
// if using cuda:
// nvcc -c -Xcompiler -fPIC ./cfuncs/fft_filter.cu -o ./cfuncs/fft_filter.o
// gcc -c -fPIC -DUSE_FFT_FILTER_CUDA ./cfuncs/diode_actions.c -o ./cfuncs/diode_actions.o 
// nvcc -shared -o ./cfuncs/libs/libdiode.so ./cfuncs/diode_actions.o ./cfuncs/fft_filter.o -lcufft -lcudart
// or without cuda:
// gcc -shared -o ./cfuncs/libs/libdiode.so -fPIC ./cfuncs/diode_actions.c
//

// for windows compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dll -Wl ./cfuncs/diode_actions.c
//

// for macos compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dylib -fPIC ./cfuncs/diode_actions.c

double abs_square(double _Complex z) {
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

void diode_gain(double *pulse, double *gain, double *gain_value, double *pulse_after, 
    int N, double dt, double Pa, double Ta, double Ga, double gain_factor) {
   
    int i;

    double xh1 = Ga * 4468377122.5 * gain_factor * 16.5;
    double xh2 = Ga * 4468377122.5 * gain_factor * 0.32 * exp(0.000000000041*14E+10);

    double rand_factor = 0.00000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    // gain medium calculations
    for (i = 0; i < N; i++) {
        int iN = (i + 1) % N; // wrap around for circular array behavior
        double gGain = xh1 - xh2 * exp(-0.000000000041 * gain[i]);
        gain_value[i] = 1 + gGain;
        //gGain *= pulse[i];
        pulse_after[i] = pulse[i] * gain_value[i] + rand_factor * gain[i] * (double)rand();
        gain[iN] = gain[i] + dt * (-gGain * pulse[i] + Pa - gain[i] / (Ta * 1E-12));
        // if (i == 1000) {
        //     printf("int gGain = %f gain_value[i] = %f\n", gGain, gain_value[i]);
        //     printf("int gain[i] = %f gain[iN] = %f pulse[i] = %f after = %f rf = %e\n", gain[i], gain[iN], pulse[i], pulse_after[i], rand_factor * gain[i]);
        // }

    }
}

void cmp_diode_gain(double _Complex *pulse, double *gain, double *gain_value, double _Complex *pulse_after, 
    int N, double dt, double Pa, double Ta, double Ga, double gain_factor) {
   
    int i;

    double xh1 = Ga * 4468377122.5 * gain_factor * 16.5;
    double xh2 = Ga * 4468377122.5 * gain_factor * 0.32 * exp(0.000000000041*14E+10);

    double rand_factor = 0.00000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    // gain medium calculations
    for (i = 0; i < N; i++) {
        int iN = (i + 1) % N; // wrap around for circular array behavior
        double gGain = xh1 - xh2 * exp(-0.000000000041 * gain[i]);
        gain_value[i] = 1 + gGain;
        gain[iN] = gain[i] + dt * (- gGain * abs_square(pulse[i]) + Pa - gain[i] / (Ta * 1E-12));
    
        pulse_after[i] = pulse[i] * sqrt(gain_value[i]) + rand_factor * gain[i] * (double)rand();
        // if (i == 1000) {
        //     printf("cmp gGain = %f gain_value[i] = %f\n", gGain, gain_value[i]);
        //     printf("cmp gain[i] = %f gain[iN] = %f pulse[i] = %f after = %f rf = %e\n", gain[i], gain[iN], abs_square(pulse[i]), abs_square(pulse_after[i]), rand_factor * gain[i]);
        // }

    }
}

void diode_loss(double *loss, double *loss_value, double *pulse_after,
                int N, double dt, double Pb, double Tb, double Gb, double N0b) {

    // absorber calculations
    int i;
    double gAbs;

    for (i = 0; i < N; i++) {
        int iN = (i + 1) % N; // wrap around for circular array behavior
        gAbs = Gb * 0.02 * (loss[i] - N0b);
        loss_value[i] = 1 + gAbs;
        gAbs *= pulse_after[i];
        loss[iN] = loss[i] + dt * (- gAbs + Pb - loss[i] / (Tb * 1E-12));
        // if (i == 1000) {
        //     printf("int loss[i] = %f loss[iN] = %f pulse[i] = %f \n", loss[i], loss[iN], pulse_after[i]);
        // }
        pulse_after[i] += gAbs;// + 0.25 * dt * loss[i] / (Tb * 1E-12);
    }
}

void cmp_diode_loss(double *loss, double *loss_value, double _Complex *pulse_after,
                int N, double dt, double Pb, double Tb, double Gb, double N0b) {

    // absorber calculations
    int i;
    double gAbs;

    for (i = 0; i < N; i++) {
        int iN = (i + 1) % N; // wrap around for circular array behavior
        gAbs = Gb * 0.02 * (loss[i] - N0b);
        loss_value[i] = 1 + gAbs;
        loss[iN] = loss[i] + dt * (- gAbs * abs_square(pulse_after[i]) + Pb - loss[i] / (Tb * 1E-12));
        // if (i == 1000) {
        //     printf("cmp loss[i] = %f loss[iN] = %f pulse[i] = %f\n", loss[i], loss[iN], abs_square(pulse_after[i]));
        // }
        pulse_after[i] *= sqrt(loss_value[i]);
    }
}

void diode_round_trip(double *gain, double *loss, double *gain_value, double *loss_value,
                   double *pulse_intensity, double *pulse_intensity_after,
                   int n_rounds, int N, int loss_shift, int oc_shift, int gain_distance,
                   double dt, double gainWidth, double Pa, double Ta, double Ga, double Pb, double Tb, double Gb, double N0b, double oc_val) {
    int i, m_shift = 0;
    double gAbs;

    double xh1 = Ga * 4468377122.5 * 0.46 * 16.5;
    double xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * exp(0.000000000041*14E+10);
    double rand_factor = 0.000000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    double oc_out_val = 1.0 - oc_val;

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (int ii = m_shift; ii < N + m_shift; ii++) {
            if (ii % 100 == 0) {
                printf("Round %d step %d\r", i_round, ii);
                fflush(stdout);
            }   
            int i = ii % N;
            int iN = (i + 1) % N;
            // twin segment that meets us on the absorber
            int i_match_loss = (i + loss_shift) % N;
            // total intensity in the absorber
            double intensity_loss = pulse_intensity[i] + pulse_intensity[i_match_loss];

            // ---------- loss calculation
            // loss[i] is the number of charge carrier in the absorber at step i. the allows us to calculate the gain(loss) at the absorber (gAbs)
            gAbs = Gb * 0.02 * (loss[i] - N0b);
            // loss_value[i] is the factor on the intensity of a beam segment passsing through the absorber at step i.
            loss_value[i] = 1 + gAbs;
            // make the change to the charge carriers in the absorber
            loss[iN] = loss[i] + dt * (- gAbs * intensity_loss + Pb - loss[i] / (Tb * 1E-12));
            // update the pulse intensity of the two beams after the absorber
            pulse_intensity[i] *= loss_value[i];
            pulse_intensity[i_match_loss] *= loss_value[i];

            // get the indices of the two beam segments at the gain medium for the gain calculation
            int i_gain = (i + gain_distance) % N;
            int i_match_gain = (i + loss_shift - gain_distance) % N;
            // total intensity in the gain medium
            double intensity_gain = pulse_intensity[i_gain] + pulse_intensity[i_match_gain];

            // ---------- gain calculation
            // gain[i] is the number of charge carrier in the gain medium at step i. the allows us to calculate the gain at the gain medium (gGain)
            double gGain = xh1 - xh2 * exp(-0.000000000041 * gain[i]);
            if (gGain < 0) {
                printf("Negative gain detected at index %d: %f %f\n", i, gGain, gain[i]);
            }
            // gain_value[i] is the factor on the intensity of a beam segment passsing through the gain medium at step i.
            gain_value[i] = 1 + gGain;
            // make the change to the charge carriers in the gain medium
            gain[iN] = gain[i] + dt * (-gGain * intensity_gain + Pa - gain[i] / (Ta * 1E-12));
            if (gain[iN] < 0) {
                printf("Negative gain carrier detected at index %d: %f %f %f\n", i, gain[iN], gain[i], intensity_gain);
            }
            // update the pulse intensity of the two beams after the gain medium
            pulse_intensity[i_gain] *= gain_value[i];
            pulse_intensity[i_match_gain] *= gain_value[i];
            // add random noise (at the point of the gain medium. this can be moved to other places)
            pulse_intensity[i_gain] += rand_factor * gain[i] * (double)rand();
            pulse_intensity[i_match_gain] += rand_factor * gain[i] * (double)rand();

            // ---------- output coupler calculation
            int oc_loc = (oc_shift + i) % N;
            // store the intensity of the output beam at it goes outside the cavity
            pulse_intensity_after[oc_loc] = pulse_intensity[oc_loc] * oc_out_val;

            pulse_intensity[oc_loc] *= oc_val;
            //pulse_intensity[i] += rand_factor * gain[i] * (double)rand();

        }

    }
}

// complex version of the above function for electric field amplitude
void cmp_diode_round_trip(double *gain, double *loss, double *gain_value, double *loss_value,
                   double _Complex *pulse_amplitude, double _Complex *pulse_amplitude_after,
                   int n_rounds, int N, int loss_shift, int oc_shift, int gain_distance,
                   double dt, double gainWidth, double Pa, double Ta, double Ga, double Pb, double Tb, double Gb, double N0b, double oc_val) {
    int i, m_shift = 0;
    double gAbs;

    double xh1 = Ga * 4468377122.5 * 0.46 * 16.5;
    double xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * exp(0.000000000041*14E+10);
    double rand_factor = 0.000000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    double oc_val_sqrt = sqrt(oc_val);
    double oc_out_val = sqrt(1.0 - oc_val);

#ifdef USE_FFT_FILTER_CUDA
    FFTFilterCtx ctx;

    if (fft_filter_init(&ctx, N, gainWidth) != 0) {
        fprintf(stderr, "Init failed\n");
        return;
    }
#endif

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (int ii = m_shift; ii < N + m_shift; ii++) {
            int i = ii % N;
            int iN = (i + 1) % N;
            // twin segment that meets us on the absorber
            int i_match_loss = (i + loss_shift) % N;
            // total intensity in the absorber
            double intensity_loss = abs_square(pulse_amplitude[i]) + abs_square(pulse_amplitude[i_match_loss]);

            // ---------- loss calculation
            // loss[i] is the number of charge carrier in the absorber at step i. the allows us to calculate the gain(loss) at the absorber (gAbs)
            gAbs = Gb * 0.02 * (loss[i] - N0b);
            // loss_value[i] is the factor on the intensity of a beam segment passsing through the absorber at step i.
            loss_value[i] = 1 + gAbs;
            // make the change to the charge carriers in the absorber
            loss[iN] = loss[i] + dt * (- gAbs * intensity_loss + Pb - loss[i] / (Tb * 1E-12));
            // update the pulse intensity of the two beams after the absorber
            pulse_amplitude[i] *= sqrt(loss_value[i]);
            pulse_amplitude[i_match_loss] *= sqrt(loss_value[i]);

            // get the indices of the two beam segments at the gain medium for the gain calculation
            int i_gain = (i + gain_distance) % N;
            int i_match_gain = (i + loss_shift - gain_distance) % N;
            // total intensity in the gain medium
            double intensity_gain = abs_square(pulse_amplitude[i_gain]) + abs_square(pulse_amplitude[i_match_gain]);

            // ---------- gain calculation
            // gain[i] is the number of charge carrier in the gain medium at step i. the allows us to calculate the gain at the gain medium (gGain)
            double gGain = xh1 - xh2 * exp(-0.000000000041 * gain[i]);
            if (gGain < 0) {
                printf("Negative gain detected at index %d: %f %f\n", i, gGain, gain[i]);
            }
            // gain_value[i] is the factor on the intensity of a beam segment passsing through the gain medium at step i.
            gain_value[i] = 1 + gGain;
            // make the change to the charge carriers in the gain medium
            gain[iN] = gain[i] + dt * (-gGain * intensity_gain + Pa - gain[i] / (Ta * 1E-12));
            if (gain[iN] < 0) {
                printf("Negative gain carrier detected at index %d: %f %f %f\n", i, gain[iN], gain[i], intensity_gain);
            }
            // update the pulse amplitude of the two beams after the gain medium
            pulse_amplitude[i_gain] *= sqrt(gain_value[i]);
            pulse_amplitude[i_match_gain] *= sqrt(gain_value[i]);
            // add random noise (at the point of the gain medium. this can be moved to other places)
            pulse_amplitude[i_gain] += sqrt(rand_factor * gain[i]) * (double)rand();
            pulse_amplitude[i_match_gain] += sqrt(rand_factor * gain[i]) * (double)rand();

            // ---------- output coupler calculation
            int oc_loc = (oc_shift + i) % N;
            // store the intensity of the output beam at it goes outside the cavity
            pulse_amplitude_after[oc_loc] = pulse_amplitude[oc_loc] * oc_out_val;

            pulse_amplitude[oc_loc] *= oc_val_sqrt;

        }
#ifdef USE_FFT_FILTER_CUDA
        fft_filter_run(&ctx, pulse_amplitude);
#endif

    }

#ifdef USE_FFT_FILTER_CUDA
    fft_filter_destroy(&ctx);
#endif
}

static inline double cabs_square(double _Complex z) {
    return creal(z) * creal(z) + cimag(z) * cimag(z);
}

/* safe modulo for possibly negative operands */
static inline int imod(int x, int n) {
    int r = x % n;
    return (r < 0) ? r + n : r;
}

/* Box-Muller gaussian N(0,1) generator */
static double gaussian_rand01(void) {
    static int has_spare = 0;
    static double spare;
    if (has_spare) {
        has_spare = 0;
        return spare;
    }
    double u, v, s;
    do {
        u = rand() / (double)RAND_MAX;
        v = rand() / (double)RAND_MAX;
    } while (u <= 1e-300);
    double mag = sqrt(-2.0 * log(u));
    spare = mag * sin(2.0 * 3.14159 * v);
    has_spare = 1;
    return mag * cos(2.0 * 3.14159 * v);
}

// maxwell bloch method function for electric field amplitude
void mb_diode_round_trip(double *gainN, double _Complex *gainP, double *lossN, double _Complex *lossP, 
                   double *gain_value, double *loss_value,
                   double _Complex *pulse_amplitude, double _Complex *pulse_amplitude_after,
                   int n_rounds, int N, int loss_shift, int oc_shift, int gain_distance,
                   double dt, double gainWidth, double Pa, double Ta, double Ga, double N0a, double Pb, double Tb, double Gb, double N0b, double oc_val) {
                    
    int m_shift = 0;
    double gAbs;

    double rand_factor = 0.0000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    double oc_val_sqrt = sqrt(oc_val); // output coupler retention amplitude factor
    double oc_out_val = sqrt(1.0 - oc_val); // output coupler output amplitude factor
    double omega0 = 0.0; // transition frequency, set to zero for simplicity
    double kappa = 3.0E04; // coupling constant, adjust as needed
    double C_loss = - 1.0E-05; // inversion to polarization coupling, adjust as needed
    double C_gain = - 5.0E-06; // inversion to polarization coupling, adjust as needed
    double coupling_out_gain = 8E-16; // coupling from polarization to field, adjust as needed
    double coupling_out_loss = 6E-16; // coupling from polarization to field, adjust as needed
    double Gamma =  gainWidth * 2.0 * 3.14159 * 1E12; // gain width is given in THz, convert to rad/s

    double _Complex z = -(Gamma + I * omega0) * dt;
    double _Complex alpha = cexp(z);
    double _Complex one_minus_alpha = 1.0 + 0.0*I - alpha; /* (1 - alpha) */

    /* noise prefactor: tune to your units */
    const double noise_prefactor = 1e-5;

    int idx_gain_a, idx_gain_b, idx_loss_a, idx_loss_b;
    double _Complex amplitude_gain, amplitude_loss;
    double _Complex drive;
    double _Complex delta_gain, delta_loss;
    double exchange, I_tot;
    double tGain = Ta * 1E-12, tLoss = Tb * 1E-12;
    double old_intensity;
    int bugs = 0;

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (int ii = m_shift; ii < N + m_shift; ii++) {
            int i = ii % N;
            int iN = (i + 1) % N;

            /* -------------------- ABSORBER (loss) interaction -------------------- */
            // two segments that meet at the absorber
            idx_loss_a = i;
            idx_loss_b = (i + loss_shift) % N;

            // amplitude sum in the absorber
            amplitude_loss = pulse_amplitude[idx_loss_a] + pulse_amplitude[idx_loss_b];

            drive = kappa * amplitude_loss * (double _Complex)lossN[i];
            lossP[iN] = alpha * lossP[i] + one_minus_alpha * drive;

            delta_loss = coupling_out_loss * lossP[iN];
            
            exchange = cimag(conj(amplitude_loss) * lossP[i]);
            lossN[iN] = lossN[i] + dt * ((N0b - lossN[i]) / tLoss - C_loss * exchange);

            // light amplitude change due to absorber
            I_tot = cabs_square(amplitude_loss);
            if(I_tot > 1e-30) {
                old_intensity = cabs_square(pulse_amplitude[idx_loss_a]) + cabs_square(pulse_amplitude[idx_loss_b]);
                pulse_amplitude[idx_loss_a] += delta_loss * cabs_square(pulse_amplitude[idx_loss_a]) / I_tot;
                pulse_amplitude[idx_loss_b] += delta_loss * cabs_square(pulse_amplitude[idx_loss_b]) / I_tot;
                loss_value[i] = (cabs_square(pulse_amplitude[idx_loss_a]) + cabs_square(pulse_amplitude[idx_loss_b])) / (0.000001 + old_intensity);
            } else {
                loss_value[i] = 0.0;
            }

            /* -------------------- Gain interaction -------------------- */
            // two beam segments at the gain medium for the gain calculation
            idx_gain_a = (i + gain_distance) % N;
            idx_gain_b = (i + loss_shift - gain_distance) % N;
            
            // amplitude sum in the gain medium
            amplitude_gain = pulse_amplitude[idx_gain_a] + pulse_amplitude[idx_gain_b];
            
            drive = kappa * amplitude_gain * (double _Complex)gainN[i]; /* ensure complex multiply */
            gainP[iN] = alpha * gainP[i] + one_minus_alpha * drive;

            delta_gain = coupling_out_gain * gainP[iN];
            exchange = cimag(conj(amplitude_gain) * gainP[i]);
            gainN[iN] = gainN[i] + dt * ((N0a - gainN[i]) / tGain - C_gain * exchange + Pa);
            if (gainN[iN] < 0) {
                printf("-\nNegative gain carrier detected at index %d: %f %f %f\n", i, gainN[iN], gainN[i], cabs(amplitude_gain));
                printf("Negative gain carrier Data1: %f %f %f %e %f %f %f\n", C_gain, exchange, Pa, dt, (N0a - gainN[i]) / tGain, N0a, tGain);
                printf("Negative gain carrier Data2: amp(%f + i%f) pol(%f + i%f)\n", creal(amplitude_gain), cimag(amplitude_gain), creal(gainP[i]), cimag(gainP[i]));
                bugs += 1;
                if (bugs > 5) {
                    return;
                }
                
            }
            // light amplitude change due to gain medium
            I_tot = cabs(amplitude_gain);

            if(I_tot > 1e-30) {
                old_intensity = cabs_square(pulse_amplitude[idx_gain_a]) + cabs_square(pulse_amplitude[idx_gain_b]);
                pulse_amplitude[idx_gain_a] += delta_gain * cabs(pulse_amplitude[idx_gain_a]) / I_tot;
                pulse_amplitude[idx_gain_b] += delta_gain * cabs(pulse_amplitude[idx_gain_b]) / I_tot;
                gain_value[i] = (cabs_square(pulse_amplitude[idx_gain_a]) + cabs_square(pulse_amplitude[idx_gain_b])) / (0.000001 + old_intensity);
            } else {
                gain_value[i] = 0.0;
            }

            // /* inject small complex Gaussian noise at gain interaction points (spontaneous-like) */
            double sigma = noise_prefactor * sqrt(fmax(0.0, gainN[i])) * sqrt(dt);
            double g_re = gaussian_rand01();
            double g_im = gaussian_rand01();
            double _Complex noise = sigma * (g_re + I * g_im);
            pulse_amplitude[idx_gain_a] += noise;
            pulse_amplitude[idx_gain_b] += noise;

            // ---------- output coupler calculation
            int oc_loc = (oc_shift + i) % N;
            // store the intensity of the output beam at it goes outside the cavity
            pulse_amplitude_after[oc_loc] = pulse_amplitude[oc_loc] * oc_out_val;

            pulse_amplitude[oc_loc] *= oc_val_sqrt;
        }

    }


}

/*
    cufftDoubleComplex *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, N * sizeof(cufftDoubleComplex)));
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1));

    double complex *target = malloc(N * sizeof(double complex));
    for (int it = 0; it < ITER; it++) {
        createSomeData(target);

        CHECK_CUDA(cudaMemcpy(d_data, target, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));
        CHECK_CUDA(cudaMemcpy(target, d_data, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
        
        // multiply by a bandwidth filter in frequency domain
        limitBandwidth(target);

        CHECK_CUDA(cudaMemcpy(d_data, target, N * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));
        CHECK_CUDA(cudaMemcpy(target, d_data, N * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    }

*/