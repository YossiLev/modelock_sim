import numpy as np
from PIL import Image, ImageDraw
from fasthtml.common import *
import uuid
from kerr import NLloss, SatGain, MLSpatial_gain
from chart import Chart

def rotAngle(p, a):
    return retSC(p, np.sin(a), np.cos(a))
def retSC(p, s, c):
    return (p[0] * c - p[1] * s, p[1] * c + p[0] * s)
def multMat(M, v):
    return [M[0][0] * v[0] + M[0][1] * v[1], M[1][0] * v[0] + M[1][1] * v[1]]
   
class Beam():
    def __init__(self, x, y, angle, waist = 5 * 0.001, theta = 0):
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
    def __init__(self, name = ""):
        self.id = str(uuid.uuid4())
        self.type = ""
        self.name = name
        self.lineColor = (0, 0, 0)
        self.backColor = (255, 255, 255)
        self.parameters = []
    
    def finalize(self):
        pass

    def light(self, beam):
        return beam

    def transferBeamVec(self, q):
        pass

    def stepCenterBeam(self, beam: Beam, aligned = False):
        return Beam(beam.x, beam.y, beam.angle, beam.waist, beam.theta)

    def render(self):
        return Div(Div(Input(type="checkbox", style="z-index:10;position:absolute;opacity:0", cls="B"),
                       Span(NotStr("&#9776;"), style="font-size:24px;line-height:16px; position:relative; user-select: none;"),
                       Span(self.type), 
                       Div(
                           Div("Remove", cls="dropMenuItem", hx_post=f"/removeComp/{self.id}", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}', hx_confirm="Are you sure you wish to delete this component?"),
                           Div("Add After", cls="dropMenuItem", hx_post=f"/addAfter/{self.id}", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}'),
                           Div("Add before", cls="dropMenuItem", hx_post=f"/addBefore/{self.id}", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}'),
                             cls='dropMenu'),
                       cls="compType"), 
                   Div(*[Div(p.render(), cls="BBB") for p in self.parameters], cls="AAA"),
                   cls="partBlock")
    
    def draw(self, draw, mapper, beam):
        p1 = (beam.x, beam.y + 10 * 0.001)
        p2 = (beam.x, beam.y - 10 * 0.001)
        pass
    
class SimComponentInit(SimComponent):
    def __init__(self, name = ""):
        super().__init__(name = name)
        self.type = "Select type"

        self.M = [[1, 0], [0, 1]]

    def render(self):
        types = ['Lens', 'Mirror', 'Propogation', 'Ti-Sapp']
        return Div(Div(Input(type="checkbox", style="z-index:10;position:absolute;opacity:0", cls="B"),
                       Span(NotStr("&#9776;"), style="font-size:24px;line-height:16px; position:relative; user-select: none;"),
                       Span(self.type), 
                       Div(
                           Div("Remove", cls="dropMenuItem", hx_post=f"/removeComp/{self.id}", hx_target="#cavity", hx_vals='js:{localId: getLocalId()}'),
                           cls='dropMenu'
                       ),
                       cls="compType"), 
                   Div(*[Div(p, cls="dropMenuItem", hx_post=f"/setCompType/{self.id}/{p}", 
                             hx_target="#cavity", hx_vals='js:{localId: getLocalId()}') for p in types], style="padding:8px;"),
                   cls="partBlock")

    def transferBeamVec(self, q):
        return (self.M[0][0] * q + self.M[0][1]) / (self.M[1][0] * q + self.M[1][1])
    
class LinearComponent(SimComponent):
    def __init__(self, name = ""):
        super().__init__(name = name)
        self.M = [[1, 0], [0, 1]]

    def transferBeamVec(self, q):
        return (self.M[0][0] * q + self.M[0][1]) / (self.M[1][0] * q + self.M[1][1])
    
class Propogation(LinearComponent):
    def __init__(self, name = "", distnance = 0.0, refIndex = 1.0, delta = 0.0):
        super().__init__(name = name)
        self.type = "Propogation"
        self.parameters = [
            SimParameterNumber(f"{self.id}-propogationDistance", "number", "Distance (mm)", "General", distnance),
            SimParameterNumber(f"{self.id}-refractionIndex", "number", "Ref index", "General", refIndex),
            SimParameterNumber(f"{self.id}-delta", "number", "Delta (mm)", "General", delta),
        ]
        self.finalize()
        
    def finalize(self):
        self.distance = (self.parameters[0].get_value() + self.parameters[2].get_value()) * 0.001
        self.refIndex = self.parameters[1].get_value()
        self.M = [[1, self.distance], [0, 1]]

    def stepCenterBeam(self, beam: Beam, aligned = False):
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        return Beam(beam.x + self.distance * beam.dx, beam.y + self.distance * beam.dy, beam.angle, *waist_theta)
    
class Mirror(LinearComponent):
    def __init__(self, name = "", radius = 0.0, reflection = 1.0, angleH = 0.0, angleV = 0.0):
        super().__init__(name = name)
        self.type = "Mirror"
        self.parameters = [
            SimParameterNumber(f"{self.id}-mirrorRadius", "number", "Radius (mm)", "General", radius),
            SimParameterNumber(f"{self.id}-reflection", "number", "Reflection", "General", reflection),
            SimParameterNumber(f"{self.id}-angleh", "number", "Angle (deg)", "General", angleH),
        ]  
        self.finalize()
        
    def finalize(self):
        self.radius = self.parameters[0].get_value() * 0.001
        self.reflection = self.parameters[1].get_value()
        self.angleH = np.deg2rad(self.parameters[2].get_value())
        self.angleV = np.deg2rad(0.0)
        self.M = [[1, 0], [-2 / self.radius if self.radius != 0.0 else 0.0, 1]]

    def stepCenterBeam(self, beam: Beam, aligned = False):
        an = 0 if aligned else 2 * self.angleH + np.pi
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        return Beam(beam.x, beam.y, an + beam.angle, *waist_theta)

    def draw(self, draw, mapper, beam):
        d = rotAngle((0 * 0.001, 10 * 0.001), beam.angle + self.angleH)
        p1 = (beam.x + d[0], beam.y + d[1])
        p2 = (beam.x - d[0], beam.y - d[1])
        #print(f"mirror {p1} {p2}, {beam.dx} {beam.dy}" )
        draw.line([mapper(p1), mapper(p2)], fill="black", width=3)

    def get_loss(self):
        return self.reflection    

class Lens(LinearComponent):
    def __init__(self, focus, name = ""):
        super().__init__(name = name)
        self.type = "Lens"
        self.parameters = [
            SimParameterNumber(f"{self.id}-lensFocus", "number", "Focus (mm)", "General", focus),
        ]
        self.finalize()
        
    def finalize(self):
        self.focus = self.parameters[0].get_value() * 0.001
        self.M = [[1, 0], [-1 / self.focus, 1]]

    def stepCenterBeam(self, beam: Beam, aligned = False):
        waist_theta = multMat(self.M, [beam.waist, beam.theta])
        #print(f"LENS TRANS {beam.waist} {beam.theta} -> {waist_theta[0]} {waist_theta[1]}")
        return Beam(beam.x, beam.y, beam.angle, *waist_theta)        

    def draw(self, draw, mapper, beam):
        d1 = rotAngle((0 * 0.001, 10 * 0.001), beam.angle)
        d2 = rotAngle((4 * 0.001, 6 * 0.001), beam.angle)
        d3 = rotAngle((-4 * 0.001, 6 * 0.001), beam.angle)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x - d1[0], beam.y - d1[1]))], fill="black", width=3)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x + d2[0], beam.y + d2[1]))], fill="black", width=3)
        draw.line([mapper((beam.x + d1[0], beam.y + d1[1])), mapper((beam.x + d3[0], beam.y + d3[1]))], fill="black", width=3)
        draw.line([mapper((beam.x - d1[0], beam.y - d1[1])), mapper((beam.x - d2[0], beam.y - d2[1]))], fill="black", width=3)
        draw.line([mapper((beam.x - d1[0], beam.y - d1[1])), mapper((beam.x - d3[0], beam.y - d3[1]))], fill="black", width=3)
        pass

class TiSapphs(SimComponent):
    def __init__(self, length, n2, name = ""):
        super().__init__(name = name)
        self.type = "Ti-Sapp"
        self.parameters = [
            SimParameterNumber(f"{self.id}-n2", "number", "n2 (m^2/W)", "General", n2),
            SimParameterNumber(f"{self.id}-length", "number", "Length (mm)", "General", length),
            SimParameterNumber(f"{self.id}-sat", "number", "Saturation power", "General", 1300),
            SimParameterNumber(f"{self.id}-wp", "number", "Pump waist (m)", "General", 30e-6),
            SimParameterNumber(f"{self.id}-spec-g", "number", "Spectrum gain", "General", 200),
            SimParameterNumber(f"{self.id}-n-lens", "number", "Thin lenses", "General", 5),
        ]
        self.finalize()
        
    def finalize(self):
        self.n2 = self.parameters[0].get_value()
        self.length = self.parameters[1].get_value() * 0.001
        self.Is = self.parameters[2].get_value() * 2049.0 ** 2
        self.Wp = self.parameters[3].get_value()
        self.spec_G_par = self.parameters[4].get_value()
        self.nLenses = round(self.parameters[5].get_value())

        self.kerr_par = 4 * self.length * self.n2

        self.Ikl = self.kerr_par / self.nLenses / 50

    def draw(self, draw, mapper, beam):
        d1 = rotAngle((5 * 0.001, 3 * 0.001), beam.angle)
        d2 = rotAngle((-5 * 0.001, 3 * 0.001), beam.angle)
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
        self.pinned = False
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
        return Form(Div(self.name, Span(NotStr(f"<svg width='24' height='24' viewBox='0 0 24 24' fill={'#07be17' if self.pinned else 'lightgray'} xmlns='http://www.w3.org/2000/svg'><path transform='rotate(45 0 10)' d='M12 2C10.8954 2 10 2.89543 10 4V10H8V12H16V10H14V4C14 2.89543 13.1046 2 12 2ZM10 13H14L12 21L10 13Z'/></svg>"),   
                                       hx_post=f"parpinn/{self.id}",hx_vals='js:{localId: getLocalId()}', hx_target=f"#form{self.id}"), cls="paramName"), 
                    Input(type="text", name="param", value=str(self.strValue), 
                          hx_post=f"/parnum/{self.id}", hx_vals='js:{localId: getLocalId()}', hx_trigger="keyup changed delay:1s",
                          hx_target="closest form", hx_swap="outerHTML",
                            style=f"margin-left: 40px; margin-top: 5px; width:50px;{'background: #f7e2e2; color: red;' if self.value_error else ''}"),
                            id=f"form{self.id}", 
                            )

    def get_value(self):
        return self.value

    def set_value(self, value:str):
        try:
            self.strValue = value
            self.value = float(value)
            self.value_error = False
            #print(f"new value {value} to parameter {self.id} ({self.value})")
            return True
        except ValueError as ve:
            print(f"*** Error set_value {value} ***")
            self.value_error = True
            return False

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
        print(f"aa matlab {matlab}")
        self.matlab = matlab

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
        pass

    def getPinnedParameters(self):
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
    
    def simulation_step(self):
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

    def getPinnedParameters(self):
        pinned = []
        for param in self.parameters:
            if param.pinned:
                pinned.append(param)
        for part in self.parts:
            for param in part.parameters:
                if param.pinned:
                    pinned.append(param)

        return pinned

class CavityDataPartsKerr(CavityDataParts):
    def __init__(self, matlab = False):
        super().__init__(matlab = matlab)
        print(f"aa1matlab {matlab}")

        self.name = "Kerr modelock"
        self.description = "Kerr modelock by original simulation"
        self.parameters = [
            SimParameterNumber("nbins", "number", "Number of bins", "Simulation", 2000),
            SimParameterNumber("snr", "number", "SNR", "Simulation", 0.003),
            SimParameterNumber("lambda", "number", "Center wave length", "Simulation", 780e-9),
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
        self.mirror_loss = self.getMirrorLoss()

        self.n = 2 ** np.ceil(np.log2(self.nbins)).astype(int)  # number of simulated time-bins, power of 2 for FFT efficiency
        self.bw = self.n  # simulated bandwidth
        self.n = self.n + 1  # to make the space between frequencies 1
        self.w = np.linspace(-self.bw/2, self.bw/2, self.n)  # frequency is in units of reprate, time is in units of round-trip time
        self.expW = np.exp(-1j * 2 * np.pi * self.w)

        crystals = self.getPartsOfClass(TiSapphs)
        if len(crystals) == 1:
            sp = crystals[0]
            self.Ikl = sp.Ikl
            self.L = sp.length
            self.Wp = sp.Wp
            self.Is = sp.Is
            self.spec_G_par = sp.spec_G_par  # Gaussian linewidth parameter
            self.spectralGain = 1 / (1 + (self.w / self.spec_G_par)**2)  # s(w) spectral gain function
            self.nLenses = sp.nLenses

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


        self.dw = self.bw / (self.n - 1)
        self.t = np.linspace(-1/(2*self.dw), 1/(2*self.dw), self.n)
        self.dt = 1 / (self.n * self.dw)
  
        #self.spec_G_par = 200  # Gaussian linewidth parameter
        #self.delta = 0.001  # how far we go into the stability gap (0.001)
        self.deltaPlane = -0.75e-3  # position of crystal - distance from the "plane" lens focal
        #self.deltaPlane = -0.6e-3  # position of crystal - distance from the "plane" lens focal
        self.disp_par = 0*1e-3 * 2 * np.pi / self.spec_G_par  # net dispersion
        self.epsilon = 0.2  # small number to add to the linear gain
        self.dispersion = np.exp(-1j * self.disp_par * self.w**2)  # exp(-i phi(w)) dispersion

    def restart(self, seed):
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

        self.g0 = 1 / self.mirror_loss + self.epsilon  # linear gain

        self.Et[self.cbuf, :] = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(self.Ew[self.cbuf, :])))

        self.phaseShift = np.angle(self.Ew[self.cbuf, :])

    def simulation_step(self):
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
            Chart(self.w, self.phaseShift, "Phase"),
            Chart(self.t, self.phaseShiftEt, "Phase Et")
        ]

        return charts
    
    def get_state_analysis(self):
        analysis = {}
        power = np.abs(self.Et[self.cbuf, :])**2
        total_power = np.sum(power)
        analysis['power'] = float(total_power)
        i_max = np.argmax(power)
        near_power = np.sum(power[max(0, i_max - 20): min(i_max + 20, len(power) - 1)])
        #print("A ", total_power, near_power, total_power / (near_power + 0.1))
        if near_power * 1.1 > total_power:
            analysis['code'] = "1"
            analysis['state'] = "One pulse"
            analysis['loc'] = int(i_max)
            rel = power[max(0, i_max - 50): min(i_max + 50, len(power) - 1)]
            sum = np.sum(rel)
            analysis['p'] = float(sum)
            rel = rel / sum
            l = len(rel)
            ord = np.linspace(0, l - 1, l)
            ord2 = ord ** 2
            width2 = np.sum(rel * ord2) - (np.sum(rel * ord)) ** 2
            analysis['w'] = float(math.sqrt(width2))
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

                analysis['state'] = "No pulse"
        else:
            analysis['code'] = "."
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
#         self.L = 3e-3  # crystal length in meters
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
    