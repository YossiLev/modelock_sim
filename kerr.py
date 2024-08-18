import numpy as np
import matplotlib.pyplot as plt
# _sCP5Az_F.e9n5m  matlab

def NLloss(w, Wp):
    l = np.zeros_like(w)
    for i in range(len(w)):
        if w[i] <= Wp:
            l[i] = 1
        else:
            loss_fun = lambda ww: 1 / (1 + ((ww - Wp) ** 2 / (2 * Wp ** 2)))
            l[i] = loss_fun(w[i])
    return l

def MLSpatial_gain(delta, Etx, Ptx, q1x, W1x, Ikl, L, deltaPlane):
    deltaPoint = delta - deltaPlane
    Etp = Etx.copy()
    Ptp = Ptx.copy()
    q1p = q1x.copy()
    W1p = W1x.copy()
    lambda_ = 780e-9
    RM = 150e-3
    FM = 75e-3
    L1 = 0.5
    L2 = 0.9
    V = 1 / (2 / RM - 1 / L2)
    N = 5  # number of NL lenses
    n0 = 1#1.76  # linear refractive index of Ti:S
    LCO = n0 * L  # OPL of the crystal

    def Mcur(RM):
        return np.array([[1, 0], [-2 / RM, 1]])

    def distance(d):
        return np.array([[1, d], [0, 1]])

    def lens(fL):
        return np.array([[1, 0], [-1 / fL, 1]])

    def lensL(fL, fl):
        return np.array([[1, 0], [-1 / fL - 1j / fl, 1]])
    
    def ABCD(MX, qx):
        return (MX[0, 0] * qx + MX[0, 1]) / (MX[1, 0] * qx + MX[1, 1])

    def WaistOfQ(qx):
        return (-np.imag(1 / qx) * np.pi / lambda_) ** (-1 / 2)

    lens_aperture = 56e-6
    f = ((2 * np.pi * lens_aperture ** 2) / lambda_)

    def phiKerr(Ptxx, Wxx):
        a = (Ikl * Ptxx) / (lambda_ * Wxx ** 2)
        v = np.exp(1j * a)
        return v

    qt = np.zeros(len(Etp), dtype=complex)

    MRight = distance(RM / 2 + deltaPlane - 1e-10 - L / 2) @ Mcur(RM) @ distance(L1) @ distance(L1) @ Mcur(RM) @ distance(RM / 2 + deltaPlane - 1e-10 - L / 2)
    MLeft = distance(V + deltaPoint - L / 2) @ lens(FM) @ distance(L2) @ distance(L2) @ lens(FM) @ distance(V + deltaPoint - L / 2)

    def stepKerr(e, w, Ikl, dist, q):
        p = np.abs(e) ** 2
        Feff = (w ** 4) / (Ikl * p)
        M = distance(dist) @ lensL(Feff, f)
        qt = ABCD(M, q)
        wt = WaistOfQ(qt)
        et = phiKerr(p, w) * e
        pt = np.abs(et) ** 2

        return qt, wt, et, pt
    
    for i in range(len(Etp)):

        q2, W2, Et2, Pt2 = stepKerr(Etp[i], W1p[i], Ikl, LCO / N, q1p[i])

        q3, W3, Et3, Pt3 = stepKerr(Et2, W2, Ikl, LCO / N, q2)

        q4, W4, Et4, Pt4 = stepKerr(Et3, W3, Ikl, LCO / N, q3)

        q5, W5, Et5, Pt5 = stepKerr(Et4, W4, Ikl, LCO / N, q4)

        Feff51 = (W5 ** 4) / (Ikl * Pt5)

        M = distance(LCO / (2 * N)) @ \
            MRight @ \
            distance(LCO / (2 * N)) @ \
            lensL(Feff51, f)
        
        ##q5 = (M[0, 0] * q5 + M[0, 1]) / (M[1, 0] * q5 + M[1, 1])
        q5 = ABCD(M, q5)
        ##W5 = (-np.imag(1 / q5) * np.pi / lambda_) ** (-0.5)
        W5 = WaistOfQ(q5)

        Et5 = phiKerr(np.abs(Et5) ** 2, W5) * Et5
        Pt5 = np.abs(Et5) ** 2

        q4, W4, Et4, Pt4 = stepKerr(Et5, W5, Ikl, LCO / N, q5)

        q3, W3, Et3, Pt3 = stepKerr(Et4, W4, Ikl, LCO / N, q4)

        q2, W2, Et2, Pt2 = stepKerr(Et3, W3, Ikl, LCO / N, q3)

        q1p[i], W1p[i], Etp[i], Ptp[i] = stepKerr(Et2, W2, Ikl, LCO / N, q2)

        Feff12 = (W1p[i] ** 4) / (Ikl * Ptp[i])

        M = distance(LCO / (2 * N)) @ \
            MLeft @ \
            distance(LCO / (2 * N)) @ \
            lensL(Feff12, f)

        ##qt[i] = (M[0, 0] * q1[i] + M[0, 1]) / (M[1, 0] * q1[i] + M[1, 1])
        qt[i] = ABCD(M, q1p[i])
        W1p[i] = WaistOfQ(qt[i])
        Etp[i] = phiKerr(np.abs(Etp[i]) ** 2, W1p[i]) * Etp[i]
        Ptp[i] = np.abs(Etp[i]) ** 2

    return qt, W1p, Etp

def SatGain(Ew, w, g0, Is, Wp):
    Imean = np.mean(np.abs(Ew)**2)  # mean roundtrip intensity
    Wmin = np.min(w)
    if Wmin < Wp:
        Iss = lambda waistMin: Is * waistMin**2 / Wp**2
        g = g0 / (1 +  Imean / Iss(Wmin))
    else:
        g = g0 / (1 + Imean / Is)
    return g

def kerrInit():
    global n, bw, w, dw, t, dt, cbuf, nbuf
    global n2 ,L, kerr_par, N, Ikl, Is, Wp
    global mirror_loss, spec_G_par, SNR, lambda_, delta, deltaPlane, disp_par, epsilon, num_rounds
    global Ew, Et, It, phaseShift, tata, ph2pi, Imean, R, waist, q, F, g0, W

    # fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 6))

    seed = int(np.random.rand() * (2**32 - 1))
    #seed = 693039070
    np.random.seed(seed)
    print('Seed - ', seed)

    # Simulation initialization
    n = 2 ** np.ceil(np.log2(2000)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
    bw = n  # simulated bandwidth
    n = n + 1  # to make the space between frequencies 1
    w = np.linspace(-bw/2, bw/2, n)  # frequency is in units of reprate, time is in units of round-trip time

    dw = bw / (n - 1)
    t = np.linspace(-1/(2*dw), 1/(2*dw), n)
    dt = 1 / (n * dw)

    # Kerr lens parameters
    n2 = 3e-20  # n2 of sapphire in m^2/W
    L = 3e-3  # crystal length in meters
    kerr_par = 4 * L * n2
    N = 5  # number of NL lenses in the crystal
    Ikl = kerr_par / N / 50 ## / np.pi ## 50
    Is = 2.6 * n**2 * 500  # saturation power
    Wp = 30e-6  # waist parameter in meters

    # Units
    mirror_loss = 0.95  # loss of the OC
    spec_G_par = 200  # Gaussian linewidth parameter
    SNR = 0e-3  # signal-to-noise ratio
    lambda_ = 780e-9  # wavelength in meters
    delta = 0.001  # how far we go into the stability gap

    deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
    disp_par = 0*1e-3 * 2 * np.pi / spec_G_par  # net dispersion
    epsilon = 0.2  # small number to add to the linear gain

    # Simulation parameters
    num_rounds = 2  # number of simulated round-trips

    cbuf = 0
    nbuf = 1
    # Function definitions
    ##A = lambda w: np.random.rand(n)
    ##theta = (-1 + 2 * np.random.rand(n))*np.pi
    ##noise = SNR * A(w) * (np.cos(theta) + 1j * np.sin(theta))

    # Initializations
    Ew = np.zeros((num_rounds, n), dtype=complex)  # field in frequency
    Et = np.zeros((num_rounds, n), dtype=complex)  # field in time
    It = np.zeros((num_rounds, n))  # instantaneous intensity in time
    phaseShift = np.zeros(n)
    tata = 0.0
    ph2pi = np.ones(n) * 2 * np.pi
    Imean = np.zeros(num_rounds)
    R = np.zeros((num_rounds, n))  # instantaneous waist size
    waist = np.zeros((num_rounds, n))  # instantaneous waist size
    q = np.zeros((num_rounds, n), dtype=complex)  # instantaneous waist size
    F = np.zeros((num_rounds, n), dtype=complex)  # kerr lens focus

    # Starting terms
    Ew[cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(n) + 2j * np.random.rand(n) - 1j) / 2.0  # initialize field to noise
    waist[cbuf, :] = np.ones(n) * 3.2e-5  # initial waist size probably
    R[cbuf, :] = -np.ones(n) * 3.0e-2  # initial waist size probably
    q[cbuf, :] = 1.0 / (1 / R[cbuf, :] - 1j * (lambda_ / (np.pi * waist[cbuf, :]**2)))

    g0 = 1 / mirror_loss + epsilon  # linear gain

    ##
    W = 1 / (1 + (w / spec_G_par)**2)  # s(w) spectral gain function
    Et[cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Ew[cbuf, :])))
    return np.abs(Et[cbuf, :])**2, np.abs(Ew[cbuf, :]), waist[cbuf, :], np.angle(Ew[cbuf, :])

def kerrStep(m):
    global n, bw, w, dw, t, dt, cbuf, nbuf
    global n2 ,L, kerr_par, N, Ikl, Is, Wp
    global mirror_loss, spec_G_par, SNR, lambda_, delta, deltaPlane, disp_par, epsilon, num_rounds
    global Ew, Et, It, phaseShift, tata, ph2pi, Imean, R, waist, q, F, g0, W

    phiKerr = lambda Itxx, Wxx: np.exp((1j * Ikl * Itxx) / (lambda_ * Wxx**2)) # non-linear instantenous phase accumulated due to Kerr effect
    expW = np.exp(-1j * 2 * np.pi * w)

    # Initialize fields based on past round-trip
    Et[cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Ew[cbuf, :])))
    It[cbuf, :] = np.abs(Et[cbuf, :])**2

    # Nonlinear effects calculated in time
    q[nbuf, :], waist[nbuf, :], Et[nbuf, :] = MLSpatial_gain(delta, Et[cbuf, :], It[cbuf, :], q[cbuf, :], waist[cbuf, :], Ikl, L, deltaPlane)
    waist[nbuf, :] = (-(1 / q[nbuf, :]).imag * np.pi / lambda_)**(-0.5) 
    #Et[m, :] = phiKerr(It[m - 1, :], waist[m - 1, :]) * NLloss(waist[m - 1, :], Wp) * Et[m - 1, :]
    sd = NLloss(waist[cbuf, :], Wp)
    Et[nbuf, :] = phiKerr(It[cbuf, :], waist[nbuf, :]) * sd * Et[cbuf, :]

    Ew[nbuf, :] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Et[nbuf, :])))

    ##Imean[m] = np.mean(np.abs(Ew[m, :])**2)  # mean roundtrip intensity

    g = SatGain(Ew[cbuf, :], waist[cbuf, :], g0, Is, Wp)
    ## W = 1 / (1 + (w / spec_G_par)**2)  # s(w) spectral gain function
    D = np.exp(-1j * disp_par * w**2)  # exp(-i phi(w)) dispersion
    G = g * W * D  # Overall gain
    T = 0.5 * (1 + mirror_loss * G * expW) ##np.exp(-1j * 2 * np.pi * w))

    Et[nbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Ew[nbuf, :])))

    Ew[nbuf, :] = T * Ew[nbuf, :]

    Pt = np.abs(Et[nbuf, :])**2
    am = np.argmax(Pt)
    phaseShift = np.angle(Ew[nbuf, :])
    if Pt[am] > 14 * np.mean(Pt):
        for ii in range(len(phaseShift)):
            phaseShift[ii] += (am - 1024) / (326.0 + tata) * (ii - 1024)
        phaseShift = np.mod(phaseShift, ph2pi)

    cbuf = nbuf
    nbuf = 1 - nbuf

    return Pt, np.abs(Ew[cbuf, :]), waist[cbuf, :], phaseShift
    # # if m < 20:
    # #     print(Ew[m][1000])
    # # Creating noise
    # ##theta = (-1 + 2 * np.random.rand(n)*np.pi)
    # ##noise = SNR * A(w) * (np.cos(theta) + 1j * np.sin(theta))
    # ##Ew[m, :] = Ew[m, :] + np.sqrt(np.mean(np.abs(Ew[m, :])**2)) * noise

    # def plotGraph(i, x, y, t, s=False, color='royalblue'):
    #     axes[i].cla()
    #     axes[i].plot(x, y, color=color)
    #     if s:
    #         axes[i].set_title(t + " {:.4e} -  {:.4e}".format(np.min(y), np.max(y)))
    #     else:
    #         axes[i].set_title(t + " {:.8f} -  {:.8f}".format(np.min(y), np.max(y)))

    # if m % 10 <= 9:
    #     Pt = np.abs(Et[m, :])**2
    #     color = 'royalblue'
    #     am = np.argmax(Pt)
    #     phaseShift = np.angle(Ew[m, :])
    #     if Pt[am] > 14 * np.mean(Pt):
    #         color = 'firebrick'
    #         for ii in range(len(phaseShift)):
    #             phaseShift[ii] += (am - 1024) / (326.0 + tata) * (ii - 1024)
    #         phaseShift = np.mod(phaseShift, ph2pi)
    #         print(am)
    #     plotGraph(0, t, np.abs(Et[m, :])**2, 'Power', s=True, color=color)
    #     plotGraph(1, w, np.abs(Ew[m, :]), 'Spectrum', True)
    #     plotGraph(2, w, phaseShift, 'Phase')
    #     plotGraph(3, t, waist[m, :], 'Waist')
    #     #plotGraph(4, t, g, 'Gain')
    #     plotGraph(4, t, sd, 'Loss')

    #     fig.tight_layout()

    #     #figg = plt.figure() 
    #     fig.canvas.manager.set_window_title('{:d} / {:d}'.format(m + 1, num_rounds)) 
    #     plt.pause(0.1)

    # if m % 10 <= 9:
    #     print('- ', m, seed)

    #     # Plotting
    #     plt.cla()
    #     plt.plot(t, np.abs(Et[m, :])**2)
    #     #plt.plot(t, np.abs(Ew[m, :]/100.0)**2)
    #     plt.title('Power')
    #     #plt.plot(t, np.abs(W))
    #     #plt.title('W')

    #     #plt.draw()

    #     plt.pause(0.1)

