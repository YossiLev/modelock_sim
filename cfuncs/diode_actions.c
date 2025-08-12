#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

// for linux compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.so -fPIC ./cfuncs/diode_actions.c
//
// for windows compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dll -Wl
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
    double rand_factor = 0.00000000005 * dt / (Ta * 1E-12)  / (double)RAND_MAX;

    for (int i_round = 0; i_round < n_rounds; i_round++) {
        for (i = 0; i < N; i++) {
            int iN = (i + 1) % N;
            int i_match_loss = (i + loss_shift) % N;
            double intensity_loss = pulse_intensity[i] + pulse_intensity[i_match_loss];

            // loss calculation
            gAbs = Gb * 0.02 * (loss[i] - N0b);
            loss_value[i] = 1 + gAbs;
            loss[iN] = loss[i] + dt * (- gAbs * intensity_loss + Pb - loss[i] / (Tb * 1E-12));
            pulse_intensity[i] *= loss_value[i];
            pulse_intensity[i_match_loss] *= loss_value[i];

            ;// + 0.25 * dt * loss[i] / (Tb * 1E-12);

            // gain calculation
            int i_gain = (i + gain_distance) % N;
            int i_match_gain = (i + loss_shift - gain_distance) % N;
            double intensity_gain = pulse_intensity[i_gain] + pulse_intensity[i_match_gain];

            double gGain = xh1 - xh2 * exp(-0.000000000041 * gain[i]);
            gain_value[i] = 1 + gGain;
            //pulse[i] = pulse[i] + gGain;// + rand_factor * gain[i] * (double)rand();
            gain[iN] = gain[i] + dt * (-gGain * intensity_gain + Pa - gain[i] / (Ta * 1E-12));
            pulse_intensity[i_gain] *= gain_value[i];
            pulse_intensity[i_match_gain] *= gain_value[i];

            pulse_intensity[(oc_shift + i) % N] *= oc_val;

            //pulse_intensity[i] += rand_factor * gain[i] * (double)rand();

            pulse_intensity_after[i] = pulse_intensity[i];
        }
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
