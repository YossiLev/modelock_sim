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
                   double dt, double Pa, double Ta, double Ga, double Pb, double Tb, double Gb, double N0b, double oc_val) {
    int i, m_shift = 0;
    double gAbs;

    double xh1 = Ga * 4468377122.5 * 0.46 * 16.5;
    double xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * exp(0.000000000041*14E+10);
    double rand_factor = 0.000000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    double oc_out_val = 1.0 - oc_val;

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (int ii = m_shift; ii < N + m_shift; ii++) {
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
                   double dt, double Pa, double Ta, double Ga, double Pb, double Tb, double Gb, double N0b, double oc_val) {
    int i, m_shift = 0;
    double gAbs;

    double xh1 = Ga * 4468377122.5 * 0.46 * 16.5;
    double xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * exp(0.000000000041*14E+10);
    double rand_factor = 0.000000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;
    double oc_val_sqrt = sqrt(oc_val);
    double oc_out_val = sqrt(1.0 - oc_val);

#ifdef USE_FFT_FILTER_CUDA
    FFTFilterCtx ctx;

    if (fft_filter_init(&ctx, N, N / 4) != 0) {
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