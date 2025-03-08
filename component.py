import uuid
import numpy as np
from fasthtml.common import *

   
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
            SimParameterNumber(f"{self.id}-position-shift", "number", "Pos Shift (mm)", "General", 0),
        ]
        self.finalize()
        
    def finalize(self):
        self.n2 = self.parameters[0].get_value()
        self.length = self.parameters[1].get_value() * 0.001
        self.Is = self.parameters[2].get_value() * 2049.0 ** 2
        self.Wp = self.parameters[3].get_value()
        self.spec_G_par = self.parameters[4].get_value()
        self.nLenses = round(self.parameters[5].get_value())
        self.positionShift = self.parameters[6].get_value()  * 0.001

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
        self.pinned = 0
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
        color = ['#D3D3D3', '#07be17', '#6747be'][self.pinned]
        return Form(Div(self.name, Span(NotStr(f"<svg width='24' height='24' viewBox='0 0 24 24' fill={color} xmlns='http://www.w3.org/2000/svg'><path transform='rotate(45 0 10)' d='M12 2C10.8954 2 10 2.89543 10 4V10H8V12H16V10H14V4C14 2.89543 13.1046 2 12 2ZM10 13H14L12 21L10 13Z'/></svg>"),   
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

