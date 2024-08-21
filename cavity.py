import numpy as np

class CavityDate():
    def __init__(self):
        self.type = 1

        self.SNR = 0e-3  # signal-to-noise ratio
        self.lambda_ = 780e-9  # wavelength in meters

        self.n = 2 ** np.ceil(np.log2(2000)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
        self.bw = self.n  # simulated bandwidth
        self.n = self.n + 1  # to make the space between frequencies 1
        self.w = np.linspace(-self.bw/2, self.bw/2, n)  # frequency is in units of reprate, time is in units of round-trip time
        self.dw = self.bw / (self.n - 1)
        self.t = np.linspace(-1/(2*self.dw), 1/(2*self.dw), self.n)
        self.dt = 1 / (self.n * self.dw)


        self.n2 = 3e-20  # n2 of sapphire in m^2/W
        self.L = 3e-3  # crystal length in meters
        self.kerr_par = 4 * self.L * self.n2
        self.N = 5  # number of NL lenses in the crystal
        self.Ikl = self.kerr_par / self.N / 50
        self.Is = 2.6 * n ** 2 * 500  # saturation power
        self.Wp = 30e-6  # waist parameter in meters
        self.mirror_loss = 0.95  # loss of the OC
        self.spec_G_par = 200  # Gaussian linewidth parameter
        self.delta = 0.001  # how far we go into the stability gap
        self.deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
        self.disp_par = 0*1e-3 * 2 * np.pi / self.spec_G_par  # net dispersion
        self.epsilon = 0.2  # small number to add to the linear gain

        self.cbuf = 0
        self.nbuf = 1
        
        global Ew, Et, It, phaseShift, ph2pi, R, waist, q, g0, W
