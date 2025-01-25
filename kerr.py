import numpy as np

def NLloss(waist, Wp): #pure function
    loss = np.ones_like(waist)
    for i in range(len(waist)):
        if waist[i] > Wp:
            loss[i] = 1 / (1 + ((waist[i] - Wp) ** 2 / (2 * Wp ** 2)))
            
    return loss

def SatGain(Ew, w, g0, Is, Wp):
    Imean = np.mean(np.abs(Ew)**2)  # mean roundtrip intensity
    Wmin = np.min(w)
    if Wmin < Wp:
        factor = (Wmin / Wp) **2
        g = g0 / (1 +  Imean / (Is * factor))
    else:
        g = g0 / (1 + Imean / Is)
    return g

def MLSpatial_gain(sim):
    Ikl = sim.Ikl
    deltaPoint =  - sim.deltaPlane
    qp = sim.q[sim.cbuf, :].copy()
    Wp = sim.waist[sim.cbuf, :].copy()
    Ep = sim.Et[sim.cbuf, :].copy()
    lambda_ = sim.lambda_
    N = round(sim.nLenses)  # number of NL lenses
    n0 = 1#1.76  # linear refractive index of Ti:S
    LCO = n0 * sim.L  # OPL of the crystal
    if sim.matlab:
        Nmid = N
        fullStep = LCO / (N + 1)
        edgeFactor = 1
    else:
        Nmid = N
        fullStep = LCO / N
        edgeFactor = 0.5
    steps = [1] * (Nmid - 1) + [edgeFactor]
    lens_aperture = 56e-6
    # if sim.step > 8000:
    #     lens_aperture = 56e-6 * 2
    fAper = ((2 * np.pi * lens_aperture ** 2) / sim.lambda_)

    def Mcur(rm):
        return np.array([[1, 0], [-2 / rm, 1]])

    def distance(d):
        return np.array([[1, d], [0, 1]])

    def lens(fL):
        return np.array([[1, 0], [-1 / fL, 1]])

    # def lensL(fL, fl):
    #     return np.array([[1, 0], [-1 / fL - 1j / fl, 1]])
    
    # def ABCD(MX, qx):
    #     return (MX[0, 0] * qx + MX[0, 1]) / (MX[1, 0] * qx + MX[1, 1])

    def ABCDVec(d, tf, qx):
        qnew = ((1 + d * tf) * qx + d) / (tf * qx + 1)
        return qnew

    def ABCDVecM(M, qx):
        qnew = (M[0][0] * qx + M[0][1]) / (M[1][0] * qx + M[1][1])
        return qnew

    def WaistOfQ(qx):
        return (-np.imag(1 / qx) * np.pi / lambda_) ** (-1 / 2)

    def phiKerr(Ptxx, Wxx):
        a = (Ikl * Ptxx) / (lambda_ * Wxx ** 2)
        v = np.exp(1j * a)
        return v

    MRight = distance(fullStep * edgeFactor) @ distance(sim.RMD + sim.deltaPlane - 1e-10 - sim.L / 2) @ Mcur(sim.RM) @ \
            distance(sim.L1) @ distance(sim.L1) @ Mcur(sim.RM) @ distance(sim.RMD + sim.deltaPlane - 1e-10 - sim.L / 2)
    MLeft = distance(fullStep * edgeFactor) @ distance(sim.FMD + deltaPoint - sim.L / 2) @ lens(sim.FM) @ distance(sim.L2) @ \
            distance(sim.L2) @ lens(sim.FM) @ distance(sim.FMD + deltaPoint - sim.L / 2)
    
    # print("MRight", MRight)
    # print("MLeft", MLeft)

    def thinKerr(q, w, e, Ikl, dist, M = None):
        p = np.abs(e) ** 2
        Feff = (w ** 4) / (Ikl * p)
        tf = -1 / Feff - 1j / fAper
        qt = ABCDVec(dist, tf, q)
        if M is not None:
            qt = ABCDVecM(M, qt)
        wt = WaistOfQ(qt)
        et = phiKerr(p, w) * e

        return qt, wt, et, p, Feff
    
    def thickKerrSteps(sim, qp, Wp, Ep, Ikl, fullStep, steps, M = None):
        ind = sim.recorded_data_indices
        for iStep in range(len(steps)):
            qp, Wp, Ep, pP, fP = thinKerr(qp, Wp, Ep, Ikl, steps[iStep] * fullStep, M if iStep == len(steps) - 1 else None)
            if (len(ind) > 0):
                wV = [Wp[i] for i in ind]
                pV = [pP[i] for i in ind]
                fV = [fP[i] for i in ind]
                sim.recorded_data_raw.append([iStep, pV, wV, fV])
        return qp, Wp, Ep

    qp, Wp, Ep = thickKerrSteps(sim, qp, Wp, Ep, Ikl, fullStep, steps, MRight)
    qp, Wp, Ep = thickKerrSteps(sim, qp, Wp, Ep, Ikl, fullStep, steps, MLeft)

    # #print(f"**** s- {qp[0]}")
    # for i in range(N - 1):
    #     #qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / N)
    #     qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / Nmid)
    #     #print(f"sa{i} {qp[0]}")

    # #M = distance(LCO / (2 * N)) @ MRight
    # #M = distance(LCO / Nedge) @ MRight
    # #qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / (2 * N), M)
    # qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / Nedge, distance(LCO / Nedge) @ MRight)
    # #print(f"saf {qp[0]}")

    # for i in range(N - 1):
    #     #qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / N)
    #     qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / Nmid)
    #     #print(f"sb{i} {qp[0]}")
    # #M = distance(LCO / (2 * N)) @ MLeft
    # #M = distance(LCO / Nedge) @ MLeft
    # #qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / (2 * N), M)
    # qp, Wp, Ep = thinKerr(qp, Wp, Ep, Ikl, LCO / Nedge, distance(LCO / Nedge) @ MLeft)
    # #print(f"sbf {qp[0]}")

    return qp, Wp, Ep


# def kerrInit(seed):
#     global n, bw, w, expW, dw, t, dt, cbuf, nbuf
#     global n2 ,L, kerr_par, N, Ikl, Is, Wp
#     global mirror_loss, spec_G_par, SNR, lambda_, delta, deltaPlane, disp_par, epsilon, num_rounds, D
#     global Ew, Et, It, phaseShift, ph2pi, R, waist, q, g0, W

#     np.random.seed(seed)
#     print('Seed - ', seed)

#     # Simulation initialization
#     n = 2 ** np.ceil(np.log2(2000)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
#     bw = n  # simulated bandwidth
#     n = n + 1  # to make the space between frequencies 1
#     w = np.linspace(-bw/2, bw/2, n)  # frequency is in units of reprate, time is in units of round-trip time
#     expW = np.exp(-1j * 2 * np.pi * w)

#     dw = bw / (n - 1)
#     t = np.linspace(-1/(2*dw), 1/(2*dw), n)
#     dt = 1 / (n * dw)

#     # Kerr lens parameters
#     n2 = 3e-20  # n2 of sapphire in m^2/W
#     L = 3e-3  # crystal length in meters
#     kerr_par = 4 * L * n2
#     N = 5  # number of NL lenses in the crystal
#     Ikl = kerr_par / N / 50 ## / np.pi ## 50
#     Is = 2.6 * n**2 * 500  # saturation power
#     Wp = 30e-6  # waist parameter in meters

#     # Units
#     mirror_loss = 0.95  # loss of the OC
#     spec_G_par = 200  # Gaussian linewidth parameter
#     SNR = 0e-3  # signal-to-noise ratio
#     lambda_ = 780e-9  # wavelength in meters
#     delta = 0.001  # how far we go into the stability gap

#     deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
#     disp_par = 0*1e-3 * 2 * np.pi / spec_G_par  # net dispersion
#     epsilon = 0.2  # small number to add to the linear gain
#     D = np.exp(-1j * disp_par * w**2)  # exp(-i phi(w)) dispersion

#     # Simulation parameters
#     num_rounds = 2  # number of simulated round-trips

#     cbuf = 0
#     nbuf = 1
#     # Function definitions
#     ##A = lambda w: np.random.rand(n)
#     ##theta = (-1 + 2 * np.random.rand(n))*np.pi
#     ##noise = SNR * A(w) * (np.cos(theta) + 1j * np.sin(theta))

#     # Initializations
#     Ew = np.zeros((num_rounds, n), dtype=complex)  # field in frequency
#     Et = np.zeros((num_rounds, n), dtype=complex)  # field in time
#     It = np.zeros((num_rounds, n))  # instantaneous intensity in time
#     phaseShift = np.zeros(n)
#     ph2pi = np.ones(n) * 2 * np.pi
#     R = np.zeros((num_rounds, n))  # instantaneous waist size
#     waist = np.zeros((num_rounds, n))  # instantaneous waist size
#     q = np.zeros((num_rounds, n), dtype=complex)  # instantaneous waist size

#     # Starting terms
#     Ew[cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(n) + 2j * np.random.rand(n) - 1j) / 2.0  # initialize field to noise
#     waist[cbuf, :] = np.ones(n) * 3.2e-5  # initial waist size probably
#     R[cbuf, :] = -np.ones(n) * 3.0e-2  # initial waist size probably
#     q[cbuf, :] = 1.0 / (1 / R[cbuf, :] - 1j * (lambda_ / (np.pi * waist[cbuf, :]**2)))

#     g0 = 1 / mirror_loss + epsilon  # linear gain

#     ##
#     W = 1 / (1 + (w / spec_G_par)**2)  # s(w) spectral gain function
#     Et[cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Ew[cbuf, :])))
#     return np.abs(Et[cbuf, :])**2, np.abs(Ew[cbuf, :]), waist[cbuf, :], np.angle(Ew[cbuf, :])

# def kerrStep(sim: CavityDataKerr):

#     phiKerr = lambda Itxx, Wxx: np.exp((1j * sim.Ikl * Itxx) / (sim.lambda_ * Wxx**2)) # non-linear instantenous phase accumulated due to Kerr effect

#     sim.It[sim.cbuf, :] = np.abs(sim.Et[sim.cbuf, :])**2

#     # Nonlinear effects calculated in time
#     sim.q[sim.nbuf, :], sim.waist[sim.nbuf, :], sim.Et[sim.nbuf, :] = MLSpatial_gain(sim.delta, sim.Et[sim.cbuf, :], sim.q[sim.cbuf, :], sim.waist[sim.cbuf, :], sim.Ikl, sim.L, sim.deltaPlane)
#     sd = NLloss(sim.waist[sim.cbuf, :], sim.Wp)
#     sim.Et[sim.nbuf, :] = phiKerr(sim.It[sim.cbuf, :], sim.waist[sim.nbuf, :]) * sd * sim.Et[sim.cbuf, :]

#     sim.Ew[sim.nbuf, :] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sim.Et[sim.nbuf, :])))

#     g = SatGain(sim.Ew[sim.cbuf, :], sim.waist[sim.cbuf, :], sim.g0, sim.Is, sim.Wp)
#     #D = np.exp(-1j * disp_par * w**2)  # exp(-i phi(w)) dispersion
#     G = g * sim.W * sim.D  # Overall gain
#     T = 0.5 * (1 + sim.mirror_loss * G * sim.expW) ##np.exp(-1j * 2 * np.pi * w))
#     sim.Ew[sim.nbuf, :] = T * sim.Ew[sim.nbuf, :]

#     sim.Et[sim.nbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(sim.Ew[sim.nbuf, :])))

#     sim.It[sim.nbuf, :] = np.abs(sim.Et[sim.nbuf, :])**2
#     am = np.argmax(sim.It[sim.nbuf, :])
#     sim.phaseShift = np.angle(sim.Ew[sim.nbuf, :])
#     if sim.It[sim.nbuf, :][am] > 14 * np.mean(sim.It[sim.nbuf, :]):
#         for ii in range(len(sim.phaseShift)):
#             sim.phaseShift[ii] += (am - 1024) / (326.0) * (ii - 1024)
#         sim.phaseShift = np.mod(sim.phaseShift, sim.ph2pi)

#     sim.cbuf = sim.nbuf
#     sim.nbuf = 1 - sim.nbuf

#     return sim.It[sim.cbuf, :], np.abs(sim.Ew[sim.cbuf, :]), sim.waist[sim.cbuf, :], sim.phaseShift
