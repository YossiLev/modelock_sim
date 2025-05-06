try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
    else:
        import numpy as np
except ImportError:
    import numpy as np
import re
from fasthtml.common import *
from controls import *
from multi_mode import cget, cylindrical_fresnel_prepare

def MMult(M1, M2):
    res = [[
        M1[0][0] * M2[0][0] + M1[0][1] * M2[1][0],
        M1[0][0] * M2[0][1] + M1[0][1] * M2[1][1]
    ], [
        M1[1][0] * M2[0][0] + M1[1][1] * M2[1][0],
        M1[1][0] * M2[0][1] + M1[1][1] * M2[1][1]
    ]]
    print(res)

    return res

def MInv(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]]

def to_meters(s):
    match = re.fullmatch(r'\s*([0-9]*\.?[0-9]+)\s*(mm|cm|m)\s*', s)
    if not match:
        raise ValueError(f"Invalid input format: '{s}'")

    value, unit = match.groups()
    value = float(value)

    if unit == 'mm':
        return value / 1000
    elif unit == 'cm':
        return value / 100
    elif unit == 'm':
        return value
    else:
        raise ValueError(f"Unknown unit: '{unit}'")

class CalculatorData:
    def __init__(self):
        print("CalculatorData init")
        self.M1 = [[1, 0], [0, 1]]
        self.M2 = [[1, 0], [0, 1]]
        self.M3 = [[1, 0], [0, 1]]

        self.cavity_text = "p 0.15mm\np 81.81818181mm\nl 75mm\np 0.9m\np 0.9m\nl 75mm\np 81.81818181mm\np 0.15mm"
        self.cavity_mat = [[1, 0], [0, 1]]

        self.M5 = [[1, 0], [0, 1]]
        self.fresnel_mat = [[1, 0], [0, 1]]
        self.fresnel_dx_in = 0.000001
        self.fresnel_dx_out = 0.00001
        self.fresnel_N = 256
        self.fresnel_waist = 0.000030
        self.x = []
        self.x_out = []
        self.vf_in = []
        self.vf_out = []
    '''
        position_lens = -0.00015 + crystal_shift  # -0.00015 shift needed due to conclusions from single lens usage in original simulation
        m_long = m_mult_v(m_dist(position_lens), m_dist(0.081818181), m_lens(0.075), m_dist(0.9),
                            m_dist(0.9), m_lens(0.075), m_dist(0.081818181), m_dist(position_lens))
        m_short = m_mult_v(m_dist(0.001 - position_lens), m_dist(0.075), m_lens(0.075), m_dist(0.5),
                            m_dist(0.5), m_lens(0.075), m_dist(0.075), m_dist(0.001 - position_lens))

    '''
    def set(self, params):
        print(params)
        for key, value in params.items():
            if hasattr(self, key):
                if type(value) is list:
                    setattr(self, key, np.asarray(value))
                else:
                    setattr(self, key, value)

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
        return params
    
    def doCalcCommand(self, cmd, params):
        match (cmd):
            case "mult":
                match params:
                    case "M1-M2-M3":
                        self.M3 = MMult(self.M1, self.M2)
                    case "M3-M2i-M1":
                        self.M1 = MMult(self.M3, MInv(self.M2))
            case "cavity":
                self.cavity_mat = [[1, 0], [0, 1]]
                print(f"cavity_text={self.cavity_text}")
                coms = self.cavity_text.split("\n")
                print(f"coms={coms}")
                for com in coms:
                    self.exec_cavity_command(com.strip().lower())
            case "fresnel":
                #vec = (np.arange(self.fresnel_N) - np.asarray(self.fresnel_N / 2)) if self.beam_type == 0 else (np.arange(self.fresnel_N) + 0.5)
                if (params == "calcrad"):
                    vec = np.arange(self.fresnel_N) + 0.5
                else:
                    vec = np.arange(self.fresnel_N) - np.asarray(self.fresnel_N / 2) + 0.5
                self.x = vec * np.asarray(self.fresnel_dx_in)
                self.x_out = vec * np.asarray(self.fresnel_dx_out)
                #print(f"x={self.x}")
                print(f"fresnel_dx_in={self.fresnel_dx_in}")
                print(f"fresnel_N={self.fresnel_N}")
                print(f"fresnel_mat={self.fresnel_mat}")
                kernel = cylindrical_fresnel_prepare(self.x, self.x_out, 0.000000780, self.fresnel_mat)
                waist = self.fresnel_waist
                front_exp = - np.square(self.x / waist)
                self.vf_in = np.exp(front_exp)
                self.vf_out = kernel @ self.vf_in
                # print(f"vf_in={self.vf_in}")
                # print(f"vf_out={self.vf_out}")
                # print(f"kernel shape={kernel.shape}")
                
    def exec_cavity_command(self, com):
        print(f"exec_cavity_command: {com}")
        if com.startswith("p"):
            d = to_meters(com[1:])
            M = [[1, d], [0, 1]]
        elif com.startswith("l"):
            f = to_meters(com[1:])
            M = [[1, 0], [- 1.0 / f, 1]]
        else:
            print(f"Unknown command: {com}")
            return
        self.cavity_mat = MMult(self.cavity_mat, M)
                
def InputCalcS(id, title, value, step=0.01, width = 150):
    return Input(type="number", id=id, title=title, 
                 value=value, step=f"{step}", 
                 hx_trigger="input changed delay:1s, ", hx_post="/clUpdate", hx_include="#calcForm *", 
                 hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px; margin:2px;")

def ABCDMatControl(name, M):
    return Div(
        Div(
            Div(
                Img(src="/static/copy.png", title="Copy", width=20, height=20, onclick=f"AbcdMatCopy('{name}')"),
                Img(src="/static/paste.png", title="Paste", width=20, height=20, onclick=f"AbcdMatPaste('{name}')"),
                cls="floatRight"
            ),
            Span(name),
        ),
        Div(
            InputCalcS(f'{name}_A', "A", f'{M[0][0]}', width = 180),
            InputCalcS(f'{name}_B', "B", f'{M[0][1]}', width = 180),
        ),
        Div(
            InputCalcS(f'{name}_C', "C", f'{M[1][0]}', width = 180),
            InputCalcS(f'{name}_D', "D", f'{M[1][1]}', width = 180),
        ),
        cls="ABCDMatControl"
    )

def generate_calc(data_obj, tab, offset = 0):
    print(f"tab={tab} offset={offset}")
    if data_obj is None:
        return Div()
    
    if "calcData" not in data_obj or data_obj["calcData"] is None:
        data_obj["calcData"] = CalculatorData()
        a = CalculatorData()

    calcData = data_obj["calcData"]
    added = Div()

    match tab:
        case 1: # Matrix
            added = Div(
                Div(
                    #Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                    Button("M3=M1xM2", escapse=False, hx_post=f'/doCalc/1/mult/M1-M2-M3', hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("M1=M3xM2^-1", escapse=False, hx_post=f'/doCalc/1/mult/M3-M2i-M1', hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                Div(
                    ABCDMatControl("M1", calcData.M1),
                    ABCDMatControl("M2", calcData.M2),
                    ABCDMatControl("M3", calcData.M3),
                ),
            )
        case 2: # Cavity
            added = Div(
                Div(
                    #Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                    Button("Cavity", escapse=False, hx_post=f'/doCalc/2/cavity/a', hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                Textarea(calcData.cavity_text, id="cavityText", style="min-height: 400px;",
                            hx_trigger="input changed delay:1s", hx_post="/clUpdate", hx_vals='js:{localId: getLocalId()}', hx_include="#calcForm *",),
                ABCDMatControl("MCavity", calcData.cavity_mat),
            )
            
        case 3: # Fresnel
            added = Div(
                Div(
                    #Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                    Button("Calc Radial", escapse=False, hx_post=f'/doCalc/3/fresnel/calcrad', hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Calc 1D", escapse=False, hx_post=f'/doCalc/3/fresnel/calc1d', hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                ABCDMatControl("MFresnel", calcData.fresnel_mat),
                Div(
                    InputCalcS(f'FresnelN', "N", f'{calcData.fresnel_N}', width = 80),
                    InputCalcS(f'FresnelDX', "DX", f'{calcData.fresnel_dx_in}', width = 80),
                    InputCalcS(f'FresnelDXOut', "DX_Out", f'{calcData.fresnel_dx_out}', width = 80),
                    InputCalcS(f'FresnelWaist', "Waist", f'{calcData.fresnel_waist}', width = 80),
                ),
                Div(
                    Div(
                        generate_chart([cget(calcData.x).tolist()], [cget(np.square(np.abs(calcData.vf_in))).tolist()], [""], "In"), 
                        generate_chart([cget(calcData.x_out).tolist()], [cget(np.square(np.abs(calcData.vf_out))).tolist()], [""], "In"), 
                        cls="box", style="background-color: #008080;"
                    ) 
                ) if len(calcData.vf_out) > 0 else Div(),



            )
        case 4: 
            pass
        case 5:
            pass

    return Div(
        Div(
            TabMaker("Matrix", "/tabcalc/1", tab == 1, target="#gen_calc"),
            TabMaker("Cavity", "/tabcalc/2", tab == 2, target="#gen_calc"),
            TabMaker("Fresnel", "/tabcalc/3", tab == 3, target="#gen_calc"),
            TabMaker("TBD3", "/tabcalc/4", tab == 4, target="#gen_calc"),
            TabMaker("TBD4", "/tabcalc/5", tab == 5, target="#gen_calc"),
        ),
        Div(added, id="calcForm"),

        id="gen_calc"
    )

    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


