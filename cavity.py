import numpy as np
from PIL import Image, ImageDraw
from fasthtml.common import *
from kerr import NLloss, SatGain, MLSpatial_gain
from chart import Chart
from component import *

class CavityData():
    def __init__(self, matlab = False):
        self.name = ""
        self.description = ""
        self.parameters = []        
        self.parts = []
        self.box = []
        self.hScale = 1.0
        self.vScale = 1.0
        self.hShift = 0.0
        self.vShift = 0.0
        self.gWidth = 100
        self.gHeight = 100
        self.beam_x_error = False
        self.beam_x = 0.0
        self.str_beam_x = "0.0"
        self.beam_theta_error = False
        self.beam_theta = math.radians(0.5)
        self.str_beam_theta = "0.5"
        self.matlab = matlab
        self.positionShift = 0.0
        self.step = 0
        self.recorded_data = [["Time", "Power", "Width", "Lensing Right", "Lensing Left", "Total Mat"]]
        self.recorded_data_indices = []
        self.finalized = False

        pass

    def point(self, p):
        return (p[0] * self.hScale + self.hShift, self.gHeight - 1 - (p[1] * self.vScale + self.vShift))

    def getParameter(self, id):
        for param in self.parameters:
            if param.id == id:
                return param, None
        for part in self.parts:
            for param in part.parameters:
                if param.id == id:
                    return param, part
        return None

    def getComponentIndexById(self, id):
        for i_part in range(len(self.parts)):
            if self.parts[i_part].id == id:
               return i_part
        return -1

    def removeComponentById(self, id):
        i = self.getComponentIndexById(id)
        if i >= 0:
            del self.parts[i]

    def addAfterById(self, id):
        i = self.getComponentIndexById(id)
        if i >= 0:
            self.parts.insert(i + 1, SimComponentInit())

    def addBeforeById(self, id):
        i = self.getComponentIndexById(id)
        if i >= 0:
            self.parts.insert(i, SimComponentInit())


    def replaceTypeById(self, id, tp):
        i = self.getComponentIndexById(id)
        if i >= 0:
            match tp:
                case "Lens":
                    self.parts[i] = Lens(focus = 100.0)
                case "Mirror":
                    self.parts[i] = Mirror(radius = 100.0)
                case "Propogation":
                    self.parts[i] = Propogation(distnance = 100.0)
                case "Ti-Sapp":
                    self.parts[i] = TiSapphs(length = 1.0, n2 = 1e-20)

    def build_beam_geometry(self):
        pass

    def finalize(self):
        pass

    def restart(self):
        self.step = 0
        pass

    def getPinnedParameters(self, tPin):
        return []

    def render(self):
        return Div(
            Input(type="text", value=self.name, id="simName", name="simName", placeholder="Name", style="width:120px;"),
            Input(type="text", value=self.description, id="simDesc", name="simDesc", placeholder="Description", style="width:390px;"),
            Div(
            Div(*[p.render() for p in self.parameters], style="padding:8px;"),
            Div(*[t.render() for t in self.parts], style="padding:8px; width:1300px;", cls="rowx"),
            cls="rowx"), id="cavity"
        )

    def get_state(self):
        return []
    
    def get_state_analysis(self):
        return {}

    def get_recorded_data(self):
        return self.recorded_data
    
    def get_record_steps(self, index):
        self.recorded_data = [["Time", "Power", "Width", "Lensing Right", "Lensing Left", "Total Mat"],
                              *[[str(x), "-", "-", "-", "-", "-"] for x in range(5)]]

  
    def simulation_step(self):
        self.step = self.step + 1
        pass
   
class CavityDataParts(CavityData):
    def __init__(self, matlab = False):
        super().__init__(matlab)
        self.parameters = [
            # SimParameterNumber("nbins", "number", "Number of bins", "Simulation", 2000),
            # SimParameterNumber("snr", "number", "SNR", "Simulation", 0.003),
            # SimParameterNumber("lambda", "number", "Center wave length", "Simulation", 780e-9),
            # SimParameterNumber("sat", "number", "Saturation power", "Simulation", 2600),
            # SimParameterNumber("wp", "number", "Pump waist (m)", "Simulation", 30e-6),
            # SimParameterNumber("delta", "number", "Delta (m)", "Simulation", 0.001),
        ]
        self.parts = [
            # Mirror(),
            # Propogation(distnance = 500.0),
            # Lens(focus = 75.0),
            # Propogation(distnance = 75.0),
            # TiSapphs(length = 3.0, n2 = 3e-20),
            # Propogation(distnance = 75.0),
            # Mirror(radius = 150.0, angleH = 30.0),
            # Propogation(distnance=200.0),
            # # Lens(focus = 75.0),
            # # Propogation(distnance = 75.0),
            # # TiSapphs(length = 3.0, n2 = 3e-20),
            # # Propogation(distnance = 75.0),
            # # Lens(focus = 75.0),
            # # Propogation(distnance=150.0),
            # # Mirror(angleH = 10.0),
            # # Propogation(distnance=150.0),
            # # Lens(focus = 75.0),
            # # Propogation(distnance=150.0),
            # # Mirror(angleH = - 10.0),
            # # Propogation(distnance=50.0),
            # Mirror(),
        ]
        self.build_beam_geometry()

    def build_beam_geometry(self, aligned = False):
        self.beams = [Beam(0.0, 0.0, np.pi, waist=self.beam_x, theta=self.beam_theta)]
        notBegin = False
        for part in self.parts:
            self.beams.append(part.stepCenterBeam(self.beams[-1], aligned and notBegin))
            notBegin = True
        
        self.box = [999999, 999999, -999999, -999999] # xmin, ymin, xmax, ymax
        for beam in self.beams:
            #print(str(beam))
            if self.box[0] > beam.x:
                self.box[0] = beam.x
            if self.box[1] > beam.y:
                self.box[1] = beam.y
            if self.box[2] < beam.x:
                self.box[2] = beam.x
            if self.box[3] < beam.y:
                self.box[3] = beam.y
        self.box[0] -= 20 * 0.001
        self.box[1] -= 20 * 0.001
        self.box[2] += 20 * 0.001
        self.box[3] += 20 * 0.001
    
    def draw_cavity(self, draw: ImageDraw, aligned = False, keep_aspect = True, zoom = 1.0):
        self.build_beam_geometry(aligned)

        self.gWidth = draw._image.width
        self.gHeight = draw._image.height

        self.hScale = self.gWidth / (self.box[2] - self.box[0])
        self.vScale = self.gHeight / (self.box[3] - self.box[1])
        if keep_aspect:
            if self.vScale > self.hScale:
                self.vScale = self.hScale
            else:
                self.hScale = self.vScale
        self.hShift = - self.box[0] * self.hScale
        self.vShift = - self.box[1] * self.vScale
        #print("Shifts ", self.hScale, self.vScale, self.hShift, self.vShift)

        for iComponent in range(len(self.parts)):
            self.parts[iComponent].draw(draw, self.point, self.beams[iComponent])

        self.draw_beam_geometry(draw)

    def draw_beam_geometry(self, draw: ImageDraw):

        for iBeam in range(len(self.beams)):
            if iBeam > 0:
                dp = rotAngle((0, self.beams[iBeam - 1].waist), self.beams[iBeam - 1].angle)
                dc = rotAngle((0, self.beams[iBeam].waist), self.beams[iBeam].angle)

                bp = (self.beams[iBeam - 1].x , self.beams[iBeam - 1].y)
                bc = (self.beams[iBeam].x , self.beams[iBeam].y)
                draw.line([self.point(bp), self.point(bc)], fill=["orange", "green", "blue"][np.mod(iBeam - 1, 3)], width=3)
                draw.line([self.point((bp[0] + dp[0], bp[1] + dp[1])), 
                                      self.point((bc[0] + dc[0], bc[1] + dc[1]))], fill="brown", width=1)
                draw.line([self.point((bp[0] - dp[0], bp[1] - dp[1])), 
                                      self.point((bc[0] - dc[0], bc[1] - dc[1]))], fill="brown", width=1)
    def getMirrorLoss(self):
        mirror_loss = 1.0
        for part in self.parts:
            if isinstance(part, Mirror):
                mirror_loss *= part.get_loss()
        return mirror_loss
    
    def getPartsOfClass(self, cls):
        ps = []
        for par in self.parts:
            if isinstance(par, cls):
                ps.append(par)
        return ps
    
    def getPartsByName(self, name):
        ps = []
        for par in self.parts:
            if par.name == name:
                ps.append(par)
        return ps

    def getPinnedParameters(self, tPin):
        pinned = []
        for param in self.parameters:
            if param.pinned == tPin:
                pinned.append(param)
        for part in self.parts:
            for param in part.parameters:
                if param.pinned == tPin:
                    pinned.append(param)

        return pinned

class CavityDataPartsKerr(CavityDataParts):
    def __init__(self, matlab = False):
        super().__init__(matlab = matlab)

        self.name = "Kerr modelock"
        self.description = "Kerr modelock by original simulation"
        self.parameters = [
            SimParameterNumber("nbins", "number", "Number of bins", "Simulation", 2000),
            SimParameterNumber("snr", "number", "SNR", "Simulation", 0.003),
            SimParameterNumber("lambda", "number", "Center wave length", "Simulation", 780e-9),
            SimParameterNumber("gainEpsilon", "number", "Gain Epsilon", "Simulation", 0.2),
            SimParameterNumber("dispersionFactor", "number", "Dispersion", "Simulation", 0.5),
           # SimParameterNumber("delta", "number", "Delta (m)", "Simulation", 0.000),
        ]

        self.parts = [
            Mirror(name="End Left"),
            Propogation(name = "L2", distnance = 900.0),
            Lens(name = "Lens", focus = 75.0),
            Propogation(name = "LensD", distnance = 81.81, delta = 1),
            TiSapphs(length = 3, n2 = 3e-20),
            Propogation(name = "MirrorD", distnance = 75.0),
            Mirror(name = "Mirror", radius = 150.0, angleH = 30.0),
            Propogation(name = "L1", distnance = 500.0),
            Mirror(name="End Right", reflection = 0.95),
        ]
        self.finalize()
        
        self.build_beam_geometry()

    def finalize(self):
        for part in self.parts:
            part.finalize()

        self.nbins = self.parameters[0].value
        self.SNR = self.parameters[1].value
        self.lambda_ = self.parameters[2].value
        self.epsilon = self.parameters[3].value
        self.dispersionFactor = self.parameters[4].value
        self.mirror_loss = self.getMirrorLoss()

        self.n = 2 ** np.ceil(np.log2(self.nbins)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
        self.bw = self.n  # simulated bandwidth
        self.n = self.n + 1  # to make the space between frequencies 1
        self.w = np.linspace(-self.bw/2, self.bw/2, self.n)[:-1]  # frequency is in units of reprate, time is in units of round-trip time
        self.expW = np.exp(-1j * 2 * np.pi * self.w)
        self.n = self.n - 1

        crystals = self.getPartsOfClass(TiSapphs)
        if len(crystals) == 1:
            sp = crystals[0]
            self.Ikl = sp.Ikl
            self.crystalLength = sp.length
            self.Wp = sp.Wp
            self.Is = sp.Is
            self.spec_G_par = sp.spec_G_par  # Gaussian linewidth parameter
            self.spectralGain = 1 / (1 + (self.w / self.spec_G_par)**2)  # s(w) spectral gain function
            self.nLenses = sp.nLenses
            self.positionShift = sp.positionShift

        lenses = self.getPartsByName("Lens")
        if len(lenses) == 1:
            self.FM = lenses[0].focus

        dists = self.getPartsByName("L1")
        if len(dists) == 1:
            self.L1 = dists[0].distance

        dists = self.getPartsByName("L2")
        if len(dists) == 1:
            self.L2 = dists[0].distance

        dists = self.getPartsByName("LensD")
        if len(dists) == 1:
            self.FMD = dists[0].distance

        dists = self.getPartsByName("MirrorD")
        if len(dists) == 1:
            self.RMD = dists[0].distance

        mirrors = self.getPartsByName("Mirror")
        if len(mirrors) == 1:
            self.RM = mirrors[0].radius


        self.dw = self.bw / (self.n)
        self.t = np.linspace(-1/(2*self.dw), 1/(2*self.dw), self.n + 1)[:-1]
        self.dt = 1 / (self.n * self.dw)
  
        #self.spec_G_par = 200  # Gaussian linewidth parameter
        #self.delta = 0.001  # how far we go into the stability gap (0.001)
        self.deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
        #self.deltaPlane = -0.6e-3  # position of crystal - distance from the "plane" lens focal
        self.disp_par = self.dispersionFactor * 1e-3 * 2 * np.pi / self.spec_G_par  # net dispersion
        #self.epsilon = self.epsilon  # 0.2 small number to add to the linear gain
        self.dispersion = np.exp(-1j * self.disp_par * self.w**2)  # exp(-i phi(w)) dispersion
        self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain



    def restart(self, seed):
        super().restart()

        self.cbuf = 0
        self.nbuf = 1
        
        # Initializations
        self.Ew = np.zeros((2, self.n), dtype=complex)  # field in frequency
        self.Et = np.zeros((2, self.n), dtype=complex)  # field in time
        self.It = np.zeros((2, self.n))  # instantaneous intensity in time
        self.ph2pi = np.ones(self.n) * 2 * np.pi
        self.waist = np.zeros((2, self.n))  # instantaneous waist size
        self.q = np.zeros((2, self.n), dtype=complex)  # beam parameter q

        # Starting terms
        np.random.seed(seed)
        self.Ew[self.cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(self.n) + 2j * np.random.rand(self.n) - 1j) / 2.0  # initialize field to noise

        self.waist[self.cbuf, :] = np.ones(self.n) * 3.2e-5  # initial waist size probably
        R = -np.ones(self.n) * 3.0e-2  # initial waist size probably
        self.q[self.cbuf, :] = 1.0 / (1 / R - 1j * (self.lambda_ / (np.pi * self.waist[self.cbuf, :]**2)))

        #self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain

        self.Et[self.cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.Ew[self.cbuf, :])))

        self.phaseShift = np.angle(self.Ew[self.cbuf, :])

        self.finalized = True


    def get_record_steps(self, index):
        powerChart = self.get_state()[0]
        index = np.argmax(powerChart.y) + index
        self.recorded_data = [["Time", "Step", "Power", "Width", "QR", "Qi", "Lensing Right", "Power", "Width", "QR", "Qi", "Lensing Left"],]
        self.recorded_data_indices = list(range(max(0, index - 4), min(self.n, index + 4), 1))
        self.recorded_data_raw = []
        self.simulation_step()
            
        for i in range(len(self.recorded_data_indices)):
            lens = [0, 0]
            for s in range(self.nLenses):
                v = [f"{self.recorded_data_indices[i]}", f"{s + 1}"]
                for side in range(2):
                    g = self.recorded_data_raw[s + self.nLenses * side]
                    qres = 1.0 / g[4][i]
                    r = 1.0 / qres.real
                    w2 = - self.lambda_ / qres.imag / np.pi
                    v = v + [f"{g[1][i]:.4e}", f"{g[2][i]:.4e}", f"{r:.4e}", f"{w2:.4e}", f"{g[3][i]:.4e}"]
                    lens[side] += 1.0 / g[3][i]
                self.recorded_data.append(v)
            v = [f"{self.recorded_data_indices[i]}", "T", "", "", "", "", f"{(1 / lens[0]):.4e}", "", "", "", "", f"{(1 / lens[1]):.4e}"]
            self.recorded_data.append(v)
            
        self.recorded_data_indices = []

    def simulation_step(self):
        super().simulation_step()
        sim = self

        phiKerr = lambda Itxx, Wxx: np.exp((1j * sim.Ikl * Itxx) / (sim.lambda_ * Wxx**2)) # non-linear instantenous phase accumulated due to Kerr effect

        # calculation in time including Nonlinear effects
        sim.It[sim.cbuf, :] = np.abs(sim.Et[sim.cbuf, :])**2
        sim.q[sim.nbuf, :], sim.waist[sim.nbuf, :], sim.Et[sim.nbuf, :] = MLSpatial_gain(sim)

        sd = NLloss(sim.waist[sim.cbuf, :], sim.Wp)
        sim.Et[sim.nbuf, :] = phiKerr(sim.It[sim.cbuf, :], sim.waist[sim.nbuf, :]) * sd * sim.Et[sim.cbuf, :]

        # calculation in frequency
        sim.Ew[sim.nbuf, :] = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(sim.Et[sim.nbuf, :])))

        g = SatGain(sim.Ew[sim.cbuf, :], sim.waist[sim.cbuf, :], sim.g0, sim.Is, sim.Wp)
        G = g * sim.spectralGain * sim.dispersion  # Overall gain
        T = 0.5 * (1 + sim.mirror_loss * G * sim.expW) ##np.exp(-1j * 2 * np.pi * w))
        sim.Ew[sim.nbuf, :] = T * sim.Ew[sim.nbuf, :]

        sim.Et[sim.nbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(sim.Ew[sim.nbuf, :])))

        sim.cbuf = sim.nbuf
        sim.nbuf = 1 - sim.nbuf   


    def get_state(self):
        print(f" in get_state finalized = {self.finalized}")
        if not self.finalized:
            return []
        self.It[self.nbuf, :] = np.abs(self.Et[self.nbuf, :])**2
        am = np.argmax(self.It[self.nbuf, :])
        self.phaseShift = np.angle(self.Ew[self.nbuf, :])
        self.phaseShiftEt = np.angle(self.Et[self.nbuf, :])
        if self.It[self.nbuf, :][am] > 14 * np.mean(self.It[self.nbuf, :]):
            for ii in range(len(self.phaseShift)):
                self.phaseShift[ii] += (am - 1024) / (326.0) * (ii - 1024)
            self.phaseShift = np.mod(self.phaseShift, self.ph2pi)

        charts = [
            Chart(self.t, np.abs(self.Et[self.cbuf, :])**2, "Power"),
            Chart(self.w, np.abs(self.Ew[self.cbuf, :]), "Spectrum"),
            Chart(self.t, self.waist[self.cbuf, :], "Waist"),
            Chart(self.t, (self.q[self.cbuf, :].imag / np.pi * self.lambda_) ** (1 / 2), "W0"),
            Chart(self.t, self.q[self.cbuf, :].real, "Z Q.real"),
            Chart(self.t, self.q[self.cbuf, :].imag, "Q.imaginary"),
            Chart(self.w, self.phaseShift, "Phase Ew"),
            Chart(self.t, self.phaseShiftEt, "Phase Et")
        ]

        return charts
    
    def get_state_analysis(self):
        if not self.finalized:
            return {}
        analysis = {}
        power = np.abs(self.Et[self.cbuf, :])**2
        total_power = np.sum(power)
        analysis['power'] = float(total_power)
        i_max = np.argmax(power)
        analysis["peakPower"] = power[i_max]
        near_power = np.sum(power[max(0, i_max - 20): min(i_max + 20, len(power) - 1)])
        #print("A ", total_power, near_power, total_power / (near_power + 0.1))
        if near_power * 1.1 > total_power:
            analysis['code'] = "1"
            analysis['nPulse'] = 1
            analysis['state'] = "One pulse"
            analysis['loc1'] = int(i_max)
            rel = power[max(0, i_max - 50): min(i_max + 50, len(power) - 1)]
            sum = np.sum(rel)
            analysis['p1'] = float(sum)
            rel = rel / sum
            l = len(rel)
            ord = np.linspace(0, l - 1, l)
            ord2 = ord ** 2
            width2 = np.sum(rel * ord2) - (np.sum(rel * ord)) ** 2
            analysis['w1'] = float(math.sqrt(width2))
        elif near_power * 2.2 > total_power:
            #print('------ enter two')
            rel = power[max(0, i_max - 50): min(i_max + 50, len(power) - 1)]
            sum_a = np.sum(rel)
            rel = rel / sum_a
            l = len(rel)
            ord_a = np.linspace(0, l - 1, l)
            ord_a2 = ord_a ** 2
            width_a2 = np.sum(rel * ord_a2) - (np.sum(rel * ord_a)) ** 2
            power[max(0, i_max - 50): min(i_max + 50, len(power) - 1)] = np.zeros(l)
            i_max_b = np.argmax(power)
            total_power = np.sum(power)
            near_power_b = np.sum(power[max(0, i_max_b - 20): min(i_max_b + 20, len(power) - 1)])
            #print("B ", total_power, near_power_b, total_power / (near_power_b + 0.1))
            if near_power_b * 1.1 > total_power:
                analysis['code'] = "2"
                analysis['nPulse'] = 2
                analysis['state'] = "Two pulses"
                analysis['loc1'] = int(i_max)
                rel = power[max(0, i_max_b - 50): min(i_max_b + 50, len(power) - 1)]
                sum_b = np.sum(rel)
                analysis['p1'] = float(sum_a)
                analysis['w1'] = float(math.sqrt(width_a2))
                rel = rel / sum_b
                l = len(rel)
                ord_b = np.linspace(0, l - 1, l)
                ord_b2 = ord_b ** 2
                width_b2 = np.sum(rel * ord_b2) - (np.sum(rel * ord_b)) ** 2
                analysis['loc2'] = int(i_max_b)
                analysis['p2'] = float(sum_b)
                analysis['w2'] = float(math.sqrt(width_b2))
            else:
                analysis['code'] = "."
                analysis['nPulse'] = 0
                analysis['state'] = "No pulse"
        else:
            analysis['code'] = "."
            analysis['nPulse'] = 0
            analysis['state'] = "No pulses"

        return analysis

# class CavityDataKerr(CavityData):
#     def __init__(self):
#         self.type = 1

#         self.SNR = 0e-3  # signal-to-noise ratio
#         self.lambda_ = 780e-9  # wavelength in meters

#         self.n = 2 ** np.ceil(np.log2(2000)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
#         self.bw = self.n  # simulated bandwidth
#         self.n = self.n + 1  # to make the space between frequencies 1
#         self.w = np.linspace(-self.bw/2, self.bw/2, self.n)  # frequency is in units of reprate, time is in units of round-trip time
#         self.expW = np.exp(-1j * 2 * np.pi * self.w)

#         self.dw = self.bw / (self.n - 1)
#         self.t = np.linspace(-1/(2*self.dw), 1/(2*self.dw), self.n)
#         self.dt = 1 / (self.n * self.dw)

#         #kerr-gain medium
#         self.n2 = 3e-20  # n2 of sapphire in m^2/W
#         self.crystalLength = 3e-3  # crystal length in meters
#         self.kerr_par = 4 * self.L * self.n2
#         self.N = 5  # number of NL lenses in the crystal
#         self.Ikl = self.kerr_par / self.N / 50

#         self.Is = 2.6 * self.n ** 2 * 500  # saturation power (2.6 * self.n ** 2 * 500)
#         self.Wp = 30e-6  # waist parameter in meters

#         self.mirror_loss = 0.95  # loss of the OC (0.95)
#         self.spec_G_par = 200  # Gaussian linewidth parameter
#         self.delta = 0.001  # how far we go into the stability gap (0.001)
#         self.deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
#         self.disp_par = 0*1e-3 * 2 * np.pi / self.spec_G_par  # net dispersion
#         self.epsilon = 0.2  # small number to add to the linear gain
#         self.D = np.exp(-1j * self.disp_par * self.w**2)  # exp(-i phi(w)) dispersion

#         self.cbuf = 0
#         self.nbuf = 1
        
#         # Initializations
#         self.Ew = np.zeros((2, self.n), dtype=complex)  # field in frequency
#         self.Et = np.zeros((2, self.n), dtype=complex)  # field in time
#         self.It = np.zeros((2, self.n))  # instantaneous intensity in time
#         self.ph2pi = np.ones(self.n) * 2 * np.pi
#         self.waist = np.zeros((2, self.n))  # instantaneous waist size
#         self.q = np.zeros((2, self.n), dtype=complex)  # instantaneous waist size

#         # Starting terms
#         self.Ew[self.cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(self.n) + 2j * np.random.rand(self.n) - 1j) / 2.0  # initialize field to noise
#         self.waist[self.cbuf, :] = np.ones(self.n) * 3.2e-5  # initial waist size probably
#         R = -np.ones(self.n) * 3.0e-2  # initial waist size probably
#         self.q[self.cbuf, :] = 1.0 / (1 / R - 1j * (self.lambda_ / (np.pi * self.waist[self.cbuf, :]**2)))

#         self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain

#         self.W = 1 / (1 + (self.w / self.spec_G_par)**2)  # s(w) spectral gain function
#         self.Et[self.cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.Ew[self.cbuf, :])))
#         self.phaseShift = np.angle(self.Ew[self.cbuf, :])

#     def get_state(self):
#         charts = [
#             Chart(self.t, np.abs(self.Et[self.cbuf, :])**2, "Power"),
#             Chart(self.w, np.abs(self.Ew[self.cbuf, :]), "Spectrum"),
#             Chart(self.w, self.waist[self.cbuf, :], "Waist"),
#             Chart(self.t, self.phaseShift, "Phase")
#         ]

#         return charts
    