#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// for linux compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.so -fPIC ./cfuncs/diode_actions.c
//
// for windows compilation:
// gcc -shared -o ./cfuncs/libs/libdiode.dll -Wl
//
// ffor macos compilation:
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


