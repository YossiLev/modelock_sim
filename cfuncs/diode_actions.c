#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// for linux compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.so -fPIC ./cfuncs/diode_actions.c
//
// for windows compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dll -Wl ./cfuncs/diode_actions.c
//
// for macos compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dylib -fPIC ./cfuncs/diode_actions.c

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
        gGain *= pulse[i];
        pulse_after[i] = pulse[i] + gGain + rand_factor * gain[i] * (double)rand();
        gain[iN] = gain[i] + dt * (-gGain + Pa - gain[i] / (Ta * 1E-12));
    }
}

    // iN = i + 1 if i < N - 1 else 0
    // #gGain = self.Ga * self.gain_factor * (self.diode_gain[i] - self.N0a) * self.diode_pulse[i]
    // #gGain = self.Ga * 4468377122.5 * self.gain_factor * (16.5-0.32*np.exp(-0.000000000041*(self.diode_gain[i]-14E+10)))
    // gGain = xh1 - xh2 * np.exp(-0.000000000041 * self.diode_gain[i])
    // #print(f"i={i}, gGain={gGain}, gGaint={gGaint}")
    // self.diode_gain_value[i] = 1 + gGain
    // gGain *= self.diode_pulse[i]
    // self.diode_pulse_after[i] = self.diode_pulse[i] + gGain
    // self.diode_gain[iN] = self.diode_gain[i] + self.dt * (- gGain + self.Pa - self.diode_gain[i] / (self.Ta * 1E-12))

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
        pulse_after[i] += gAbs;// + 0.25 * dt * loss[i] / (Tb * 1E-12);
    }
}

    // for i in range(N):
    //     iN = i + 1 if i < N - 1 else 0
    //     gAbs = self.Gb * self.loss_factor * (self.diode_loss[i] - self.N0b)
    //     self.diode_loss_value[i] = 1 + gAbs
    //     gAbs *= self.diode_pulse_after[i]
    //     self.diode_loss[iN] = self.diode_loss[i] + self.dt * (- gAbs + self.Pb - self.diode_loss[i] / (self.Tb * 1E-12))
    //     self.diode_pulse_after[i] += gAbs

void diode_round_trip(double *gain, double *loss, double *gain_value, double *loss_value,
                   double *pulse_intensity, double *pulse_intensity_after,
                   int n_rounds, int N, int loss_shift, int oc_shift, int gain_distance,
                   double dt, double Pa, double Ta, double Ga, double Pb, double Tb, double Gb, double N0b, double oc_val) {
    int i;
    double gAbs;

    double xh1 = Ga * 4468377122.5 * 0.46 * 16.5;
    double xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * exp(0.000000000041*14E+10);
    double rand_factor = 0.000000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (int ii = 300; ii < N + 300; ii++) {
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
            //loss_value[i] = 0.5;
            // make the change to the charge carriers in the absorber
            loss[iN] = loss[i] + dt * (- gAbs * intensity_loss + Pb - loss[i] / (Tb * 1E-12));
            // update the pulse intensity of the two beams after the absorber
            pulse_intensity[i] *= loss_value[i];
            pulse_intensity[i_match_loss] *= loss_value[i];

            //;// + 0.25 * dt * loss[i] / (Tb * 1E-12);

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
            //gain_value[i] = 2;
            //pulse[i] = pulse[i] + gGain;// + rand_factor * gain[i] * (double)rand();
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
            pulse_intensity[oc_loc] *= oc_val;
            pulse_intensity[i] += rand_factor * gain[i] * (double)rand();

            // store the intensity of the output beam at it goes outside the cavity
            pulse_intensity_after[oc_loc] = pulse_intensity[oc_loc];
            // if (i % 10 == 0 || i < 20 || i > N - 20) {
            //     printf("%5d %5d %5d %5d %5d %f %f %f\n", i, i_match_loss, i_gain, i_match_gain, oc_loc, pulse_intensity[4095], pulse_intensity[0], pulse_intensity[2048]);
            // }
        }
        // for (i = 0; i < 10; i++) {
        //     printf("%.1f ", pulse_intensity[i]);
        // }
        // printf("\n");
        // for (i = N / 2 - 10; i < N / 2; i++) {
        //     printf("%.1f ", pulse_intensity[i]);
        // }
        // printf("\n");
        // print pulse_intensity after 10 in one line
        // for (int i = 0; i < N; i++) {
        //     printf("%.0f ", pulse_intensity_after[i]);
        //     if ((i + 1) % 10 == 0) {
        //         printf("\n");
        //     }
        // }
        // printf("\n");
    }

}
