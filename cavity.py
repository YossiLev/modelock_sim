import numpy as np
from PIL import Image, ImageDraw
from fasthtml.common import *
import uuid

def rotAngle(p, a):
    return retSC(p, np.sin(a), np.cos(a))
def retSC(p, s, c):
    return (p[0] * c - p[1] * s, p[1] * c + p[0] * s)
def multMat(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]

class Chart():
    def __init__(self, x, y, name):
        self.x = x.tolist()
        self.y = y.tolist()
        self.name = name
        pass

class Beam():
    def __init__(self, x, y, angle, waist = 5, theta = 0):
        self.x = x
        self.y = y
        self.waist = waist
        self.theta = theta

        self.angle = angle
        self.dx = np.cos(angle)
        self.dy = np.sin(angle)

    def __str__(self):
        return f"({self.x}, {self.y}) -> ({np.rad2deg(self.angle)} {self.dx}, {self.dy})"
    
class SimComponent():
    def __init__(self):
        self.id = uuid.uuid4()
        self.type = ""
        self.name = ""
        self.lineColor = (0, 0, 0)
        self.backColor = (255, 255, 255)
        self.parameters = []
    
    def finalize(self):
        pass

    def light(self, beam):
        return beam

    def transferBeamVec(self, q):
        pass

    def stepCenterBeam(self, beam: Beam):
        return Beam(beam.x, beam.y, beam.angle, beam.waist, beam.theta)

    def render(self):
        return Div(Div(Span(NotStr("&#9776;"), style="font-size:24px;line-height:16px;"), Span(self.type), cls="compType"), 
                   Div(*[p.render() for p in self.parameters], style="padding:8px;"),
                   cls="partBlock")
    
    def draw(self, draw, mapper, beam):
        p1 = (beam.x, beam.y + 10)
        p2 = (beam.x, beam.y - 10)
        pass

class LinearComponent(SimComponent):
    def __init__(self):
        super().__init__()
        self.M = [[1, 0], [0, 1]]

    def transferBeamVec(self, q):
        return (self.M[0][0] * q + self.M[0][1]) / (self.M[1][0] * q + self.M[1][1])
    
class Propogation(LinearComponent):
    def __init__(self, distnance = 0.0, refIndex = 1.0):
        super().__init__()
        self.type = "Propogation"
        self.parameters = [
            SimParameterNumber(f"{self.id}-propogationDistance", "number", "Distance", "General", distnance),
            SimParameterNumber(f"{self.id}-refractionIndex", "number", "Ref index", "General", refIndex),
        ]
        self.finalize()
        
    def finalize(self):
        self.distnance = self.parameters[0].get_value()
        self.refIndex = self.parameters[1].get_value()
        self.M = [[1, self.distnance], [0, 1]]

    def stepCenterBeam(self, beam: Beam):
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        #print(f"PROP TRANS {beam.waist} {beam.theta} -> {waist_theta[0]} {waist_theta[1]}")
        #print(f"stepCenterBeam Prop {self.distnance}")
        return Beam(beam.x + self.distnance * beam.dx, beam.y + self.distnance * beam.dy, beam.angle, *waist_theta)
    
class Mirror(LinearComponent):
    def __init__(self, radius = 0.0, reflection = 1.0, angleH = 0.0, angleV = 0.0):
        super().__init__()
        self.type = "Mirror"
        self.parameters = [
            SimParameterNumber(f"{self.id}-mirrorRadius", "number", "Radius (mm)", "General", radius),
            SimParameterNumber(f"{self.id}-reflection", "number", "Reflection", "General", reflection),
            SimParameterNumber(f"{self.id}-angleh", "number", "Angle (deg)", "General", angleH),
        ]  
        self.finalize()
        
    def finalize(self):
        self.radius = self.parameters[0].get_value()
        self.reflection = self.parameters[1].get_value()
        self.angleH = np.deg2rad(self.parameters[2].get_value())
        self.angleV = np.deg2rad(0.0)
        self.M = [[1, 0], [-2 / self.radius if self.radius != 0.0 else 0.0, 1]]

    def stepCenterBeam(self, beam: Beam):
        an = 2 * self.angleH + np.pi
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        return Beam(beam.x, beam.y, an + beam.angle, *waist_theta)

    def draw(self, draw, mapper, beam):
        d = rotAngle((0, 10), beam.angle + self.angleH)
        p1 = (beam.x + d[0], beam.y + d[1])
        p2 = (beam.x - d[0], beam.y - d[1])
        #print(f"mirror {p1} {p2}, {beam.dx} {beam.dy}" )
        draw.line([mapper(p1), mapper(p2)], fill="black", width=3)
        pass

class Lens(LinearComponent):
    def __init__(self, focus):
        super().__init__()
        self.type = "Lens"
        self.parameters = [
            SimParameterNumber(f"{self.id}-lensFocus", "number", "Focus", "General", focus),
        ]
        self.finalize()
        
    def finalize(self):
        self.focus = self.parameters[0].get_value()
        self.M = [[1, 0], [-1 / self.focus, 1]]

    def stepCenterBeam(self, beam: Beam):
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        #print(f"LENS TRANS {beam.waist} {beam.theta} -> {waist_theta[0]} {waist_theta[1]}")
        return Beam(beam.x, beam.y, beam.angle, *waist_theta)        

    def draw(self, draw, mapper, beam):
        d1 = rotAngle((0, 10), beam.angle)
        d2 = rotAngle((4, 6), beam.angle)
        d3 = rotAngle((-4, 6), beam.angle)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x - d1[0], beam.y - d1[1]))], fill="black", width=3)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x + d2[0], beam.y + d2[1]))], fill="black", width=3)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x + d3[0], beam.y + d3[1]))], fill="black", width=3)
        draw.line([mapper((beam.x - d1[0], beam.y - d1[1])), mapper((beam.x - d2[0], beam.y - d2[1]))], fill="black", width=3)
        draw.line([mapper((beam.x - d1[0], beam.y - d1[1])), mapper((beam.x - d3[0], beam.y - d3[1]))], fill="black", width=3)
        pass

class TiSapphs(SimComponent):
    def __init__(self, length, n2):
        super().__init__()
        self.type = "Ti-Sapp"
        self.parameters = [
            SimParameterNumber(f"{self.id}-n2", "number", "n2 (m^2/W)", "General", n2),
            SimParameterNumber(f"{self.id}-length", "number", "Length (mm)", "General", length),
        ]
        self.finalize()
        
    def finalize(self):
        self.n2 = self.parameters[0].get_value()
        self.length = self.parameters[1].get_value()
           
    def draw(self, draw, mapper, beam):
        d1 = rotAngle((10, 5), beam.angle)
        d2 = rotAngle((-10, 5), beam.angle)
        #print(f"beam angle {np.rad2deg(beam.angle)}, {d1}")
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x + d2[0], beam.y + d2[1]))], fill="red", width=1)
        draw.line([mapper((beam.x + d2[0], beam.y + d2[1])), mapper((beam.x - d1[0], beam.y - d1[1]))], fill="red", width=1)
        draw.line([mapper((beam.x - d1[0], beam.y - d1[1])), mapper((beam.x - d2[0], beam.y - d2[1]))], fill="red", width=1)
        draw.line([mapper((beam.x - d2[0], beam.y - d2[1])), mapper((beam.x + d1[0], beam.y + d1[1]))], fill="red", width=1)
    pass

class SimParameter():
    def __init__(self, id, type, name, group):
        self.id = id
        self.type = type
        self.group = group
        self.name = name
        self.value_error = False
        pass

    def render(self):
        pass

    def get_value(self):
        pass

    def set_value(self, value:str):
        pass
        
class SimParameterNumber(SimParameter):
    def __init__(self, id, type, name, group, value):
        #print("SimParameterNumber  ---. ",id, type, name, group, value)
        super().__init__(id, type, name, group)
        self.value = value
        self.strValue = str(value)

    def render(self):
        return Form(Div(self.name, cls="paramName"), 
                    Input(type="text", name="param", value=str(self.strValue), 
                          hx_post=f"/parnum/{self.id}", hx_vals='js:{localId: getLocalId()}', hx_trigger="keyup changed delay:1s",
                          hx_target="closest form", hx_swap="outerHTML",
                            style=f"margin-left:16px; width:50px;{'background: #f7e2e2; color: red;' if self.value_error else ''}"))

    def get_value(self):
        return self.value

    def set_value(self, value:str):
        try:
            self.strValue = value
            self.value = float(value)
            self.value_error = False
            print(f"new value {value} to parameter {self.id} ({self.value})")
            return True
        except ValueError as ve:
            print(f"*** Error set_value {value} ***")
            self.value_error = True
            return False


class CavityData():
    def __init__(self):
        self.name = ""
        self.description = ""
        self.parameters = []        
        self.parts = []
        self.box = []
        self.hScale = 1.0
        self.vScale = 1.0
        self.hShift = 0.0
        self.vShift = 0.0
        self.w = 100
        self.h = 100
        pass

    def point(self, p):
        return (p[0] * self.hScale + self.hShift, self.h - 1 - (p[1] * self.vScale + self.vShift))

    def getParameter(self, id):
        for param in self.parameters:
            if param.id == id:
                return param, None
        for part in self.parts:
            for param in part.parameters:
                if param.id == id:
                    return param, part
        return None

    def build_beam_geometry(self):
        pass

    def render(self):
        return Div(
            Div(*[p.render() for p in self.parameters], style="padding:8px;"),
            Div(*[t.render() for t in self.parts], style="padding:8px; width:85%;", cls="row"),
            cls="row"
        )

class CavityDataParts(CavityData):
    def __init__(self):
        super().__init__()
        self.parameters = [
            SimParameterNumber("nbins", "number", "Number of bins", "Simulation", 2000),
            SimParameterNumber("snr", "number", "SNR", "Simulation", 0.003),
            SimParameterNumber("lambda", "number", "Center wave length", "Simulation", 780e-9),
            SimParameterNumber("sat", "number", "Saturation power", "Simulation", 2600),
            SimParameterNumber("wp", "number", "Pump waist (m)", "Simulation", 30e-6),
            SimParameterNumber("delta", "number", "Delta (m)", "Simulation", 0.001),
        ]
        self.parts = [
            Mirror(),
            Propogation(distnance = 500.0),
            Lens(focus = 75.0),
            Propogation(distnance = 75.0),
            TiSapphs(length = 3.0, n2 = 3e-20),
            Propogation(distnance = 75.0),
            Mirror(radius = 150.0, angleH = 30.0),
            Propogation(distnance=200.0),
            Lens(focus = 75.0),
            Propogation(distnance = 75.0),
            TiSapphs(length = 3.0, n2 = 3e-20),
            Propogation(distnance = 75.0),
            Lens(focus = 75.0),
            Propogation(distnance=150.0),
            Mirror(angleH = 10.0),
            Propogation(distnance=150.0),
            Lens(focus = 75.0),
            Propogation(distnance=150.0),
            Mirror(angleH = - 10.0),
            Propogation(distnance=50.0),
            Mirror(),
        ]
        self.build_beam_geometry()

    def build_beam_geometry(self):
        self.beams = [Beam(0.0, 0.0, np.pi)]
        for part in self.parts:
            self.beams.append(part.stepCenterBeam(self.beams[-1]))
        
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
        self.box[0] -= 50
        self.box[1] -= 50
        self.box[2] += 50
        self.box[3] += 50
    
    def draw_cavity(self, draw: ImageDraw):
        self.build_beam_geometry()

        self.w = draw._image.width
        self.h = draw._image.height

        self.hScale = self.w / (self.box[2] - self.box[0])
        self.vScale = self.h / (self.box[3] - self.box[1])
        if self.vScale > self.hScale:
            self.vScale = self.hScale
        else:
            self.hScale = self.vScale
        self.hShift = - self.box[0] * self.hScale
        self.vShift = - self.box[1] * self.vScale
        #print("Shifts ", self.hScale, self.vScale, self.hShift, self.vShift)

        for iPart in range(len(self.parts)):
            self.parts[iPart].draw(draw, self.point, self.beams[iPart])

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

        #kerr-gain medium
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
        
        # Initializations
        self.Ew = np.zeros((2, self.n), dtype=complex)  # field in frequency
        self.Et = np.zeros((2, self.n), dtype=complex)  # field in time
        self.It = np.zeros((2, self.n))  # instantaneous intensity in time
        self.ph2pi = np.ones(self.n) * 2 * np.pi
        self.waist = np.zeros((2, self.n))  # instantaneous waist size
        self.q = np.zeros((2, self.n), dtype=complex)  # instantaneous waist size

        # Starting terms
        self.Ew[self.cbuf, :] = 1e2 * (-1 + 2 * np.random.rand(self.n) + 2j * np.random.rand(self.n) - 1j) / 2.0  # initialize field to noise
        self.waist[self.cbuf, :] = np.ones(self.n) * 3.2e-5  # initial waist size probably
        R = -np.ones(self.n) * 3.0e-2  # initial waist size probably
        self.q[self.cbuf, :] = 1.0 / (1 / R - 1j * (self.lambda_ / (np.pi * self.waist[self.cbuf, :]**2)))

        self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain

        self.W = 1 / (1 + (self.w / self.spec_G_par)**2)  # s(w) spectral gain function
        self.Et[self.cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.Ew[self.cbuf, :])))
        self.phaseShift = np.angle(self.Ew[self.cbuf, :])

    def get_state(self):
        charts = [
            Chart(self.t, np.abs(self.Et[self.cbuf, :])**2, "Power"),
            Chart(self.w, np.abs(self.Ew[self.cbuf, :]), "Spectrum"),
            Chart(self.w, self.waist[self.cbuf, :], "Waist"),
            Chart(self.t, self.phaseShift, "Phase")
        ]

        return charts
    