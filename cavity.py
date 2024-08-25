import numpy as np

class CavityData():
    pass

class CavityDataKerr(CavityData):
    def __init__(self):
        self.type = 1

        self.SNR = 0e-3  # signal-to-noise ratio
        self.lambda_ = 780e-9  # wavelength in meters

        self.n = 2 ** np.ceil(np.log2(2000)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
        self.bw = self.n  # simulated bandwidth
        self.n = self.n + 1  # to make the space between frequencies 1
        self.w = np.linspace(-self.bw/2, self.bw/2, self.n)  # frequency is in units of reprate, time is in units of round-trip time
        self.expW = np.exp(-1j * 2 * np.pi * self.w)

        self.dw = self.bw / (self.n - 1)
        self.t = np.linspace(-1/(2*self.dw), 1/(2*self.dw), self.n)
        self.dt = 1 / (self.n * self.dw)


        self.n2 = 3e-20  # n2 of sapphire in m^2/W
        self.L = 3e-3  # crystal length in meters
        self.kerr_par = 4 * self.L * self.n2
        self.N = 5  # number of NL lenses in the crystal
        self.Ikl = self.kerr_par / self.N / 50
        self.Is = 2.6 * self.n ** 2 * 500  # saturation power
        self.Wp = 30e-6  # waist parameter in meters
        self.mirror_loss = 0.95  # loss of the OC
        self.spec_G_par = 200  # Gaussian linewidth parameter
        self.delta = 0.001  # how far we go into the stability gap
        self.deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
        self.disp_par = 0*1e-3 * 2 * np.pi / self.spec_G_par  # net dispersion
        self.epsilon = 0.2  # small number to add to the linear gain
        self.D = np.exp(-1j * self.disp_par * self.w**2)  # exp(-i phi(w)) dispersion

        self.cbuf = 0
        self.nbuf = 1
        
        global Ew, Et, It, phaseShift, ph2pi, R, waist, q, g0, W
        # Initializations
        self.Ew = np.zeros((2, self.n), dtype=complex)  # field in frequency
        self.Et = np.zeros((2, self.n), dtype=complex)  # field in time
        self.It = np.zeros((2, self.n))  # instantaneous intensity in time
        self.phaseShift = np.zeros(self.n)
        self.ph2pi = np.ones(self.n) * 2 * np.pi
        self.R = np.zeros((2, self.n))  # instantaneous waist size
        self.waist = np.zeros((2, self.n))  # instantaneous waist size
        self.q = np.zeros((2, self.n), dtype=complex)  # instantaneous waist size

        # Starting terms
        self.Ew[self.cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(self.n) + 2j * np.random.rand(self.n) - 1j) / 2.0  # initialize field to noise
        self.waist[self.cbuf, :] = np.ones(self.n) * 3.2e-5  # initial waist size probably
        self.R[self.cbuf, :] = -np.ones(self.n) * 3.0e-2  # initial waist size probably
        self.q[self.cbuf, :] = 1.0 / (1 / self.R[self.cbuf, :] - 1j * 
                                      (self.lambda_ / (np.pi * self.waist[self.cbuf, :]**2)))

        self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain

        ##
        self.W = 1 / (1 + (self.w / self.spec_G_par)**2)  # s(w) spectral gain function
        self.Et[self.cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.Ew[self.cbuf, :])))

    def get_state(self):
        return np.abs(self.Et[self.cbuf, :])**2, np.abs(self.Ew[self.cbuf, :]), self.waist[self.cbuf, :], np.angle(self.Ew[self.cbuf, :])

class Beam():
    pass

class SimComponent1():
    def __init__(self):
        self.lineColor = (0, 0, 0)
        self.backColor = (255, 255, 255)
    
    def light(self, beam):
        return beam

    def render(self):
        pass
"""
[
{
    "type": "Propogator",
    "parameters": {
        "distance": 10,
        "index": 1.2
    }
},
{
    "type": "Lens",
    "parameters": {
        "focal": 80,
        "radius": 10,
        "index": 1.2
    }
},
{
    "type": "Mirror",
    "parameters": {
        "radius": 20,
    }
},
{
    "type": "KerrMedium",
    "parameters": {
        "radius": 20,
    }
},
{
    "type": "Aperture, GainMedium, Absorver, Diode, Prism",
    "parameters": {
    }
}
]
"""