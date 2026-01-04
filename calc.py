import numpy as np
from scipy.signal import fftconvolve
import re
from fasthtml.common import *
from controls import *
from multi_mode import cget, cylindrical_fresnel_prepare, prepare_linear_fresnel_calc_data, prepare_linear_fresnel_straight_calc_data, linear_fresnel_propogate
from calc_diode import *
from calc_dispersion import *

def MMult(M1, M2):
    res = [[
        M1[0][0] * M2[0][0] + M1[0][1] * M2[1][0],
        M1[0][0] * M2[0][1] + M1[0][1] * M2[1][1]
    ], [
        M1[1][0] * M2[0][0] + M1[1][1] * M2[1][0],
        M1[1][0] * M2[0][1] + M1[1][1] * M2[1][1]
    ]]

    return res

def MInv(M):
    det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    return [[M[1][1] / det, -M[0][1] / det], [-M[1][0] / det, M[0][0] / det]]

def FixMethod1(M, t):
    print(f"FixMethod1 t={t} M={M}")
    FM = [[(M[0][0] + 1) / t, t * M[0][1] / (M[0][0] + 1)], [M[1][1] * (M[0][0] + 1) / (t * M[0][1]), t * (M[1][1] - M[1][0] * M[0][1] / (M[0][0] + 1))]]
    return FM
 
def to_meters(s):
    match = re.fullmatch(r'\s*(-?[0-9]*\.?[0-9]+)\s*(mm|cm|m)\s*', s)
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
        self.M1 = [[1, 0], [0, 1]]
        self.M2 = [[1, 0], [0, 1]]
        self.M3 = [[1, 0], [0, 1]]
        self.t_fixer = 1.0

        self.fixed_cavitis = [
            "Left arm;p 0.1mm\np -0.15mm\np 81.81818181mm\nl 75mm\np 0.9m\np 0.9m\nl 75mm\np 81.81818181mm\np -0.15mm\np 0.1mm",
            "Right arm;p 1mm\np -0.1mm\np 0.15mm\np 75mm\nl 75mm\np 0.5m\np 0.5m\nl 75mm\np 75mm\np 0.15mm\np -0.1mm\np 1mm",
        ]
        self.cavity_text = "p 0.1mm\np -0.15mm\np 81.81818181mm\nl 75mm\np 0.9m\np 0.9m\nl 75mm\np 81.81818181mm\np -0.15mm\np 0.1mm"
        self.cavity_mat = [[1, 0], [0, 1]]

        self.M5 = [[1, 0], [0, 1]]
        self.fresnel_mat = [[1, 0], [0, 1]]
        self.fresnel_dx_in = 0.000001
        self.fresnel_dx_out = 0.00001
        self.fresnel_N = 256
        self.fresnel_factor = 1.0
        self.fresnel_waist = 0.000030
        self.x_in = []
        self.x_out = []
        self.vf_in = []
        self.vf_out = []
        self.distance_shifts = [-0.0002, -0.00015, -0.0001, -0.00005, 0.0000, 0.00005, 0.0001, 0.00015, 0.0002]
        self.select_front = "Gaussian"

        self.harmony = 2

        self.diode = diode_calc()

        self.dispersion = dispersion_calc()

    def set(self, params):
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

    def doCalcCommand(self, cmd, params, dataObj):
        match (cmd):
            case "mult":
                match params:
                    case "M1-M2-M3":
                        self.M3 = MMult(self.M1, self.M2)
                    case "M3-M2i-M1":
                        self.M1 = MMult(self.M3, MInv(self.M2))
                    case "M1i-M3-M2":
                        self.M2 = MMult(MInv(self.M1), self.M3)
                    case "fixM3-M1":
                        self.M1 = FixMethod1(self.M3, self.t_fixer)
            case "cavity":
                match params:
                    case "calc":
                        self.cavity_mat = [[1, 0], [0, 1]]
                        coms = self.cavity_text.split("\n")
                        for com in coms:
                            self.exec_cavity_command(com.strip().lower())
                    case "0":
                        self.cavity_text = self.fixed_cavitis[0].split(";")[1]
                    case "1":
                        self.cavity_text = self.fixed_cavitis[1].split(";")[1]
                    case "2":
                        self.cavity_text = self.fixed_cavitis[2].split(";")[1]
            case "fresnel":
                print(f"select_front={self.select_front}")
                match self.select_front:
                    case "Gaussian":
                        N = int(self.fresnel_N * self.fresnel_factor)
                        if (params == "calcrad"):
                            vec = np.arange(N) + 0.5
                        elif (params == "calc1d"):
                            vec = np.arange(N) - np.asarray(N / 2) + 0.5
                        else:
                            vec = np.arange(N) - np.asarray(N / 2) + 0.5
                        self.x_in = vec * np.asarray(self.fresnel_dx_in / self.fresnel_factor)
                        self.x_out = vec * np.asarray(self.fresnel_dx_out / self.fresnel_factor)    
                        waist = self.fresnel_waist
                        front_exp = - np.square(self.x_in / waist)
                        self.vf_in = np.exp(front_exp)
                    case "Live Front":
                        print("Live Front")
                        if not hasattr(dataObj, 'mmData'):
                            print("No mmData")
                            return
                        mmData = dataObj.mmData
                        self.vf_in = np.asarray(mmData.get_x_values_full(0))
                        print(f"vf_in={len(self.vf_in)} dx0={mmData.dx0}")
                        if (params == "calcrad"):
                            vec = np.arange(len(self.vf_in)) + 0.5
                        elif (params == "calc1d"):
                            vec = np.arange(len(self.vf_in)) - np.asarray(len(self.vf_in) / 2) + 0.5
                        else:
                            vec = np.arange(len(self.vf_in)) - np.asarray(len(self.vf_in) / 2) + 0.5
                        self.x_in = vec * np.asarray(mmData.dx0)
                        self.x_out = vec * np.asarray(mmData.dx0)
                    case "From Output":
                        print("From Output")
                        self.x_in = self.x_out
                        self.vf_in = np.copy(self.vf_out[(len(self.vf_out) - 1) // 2])
                        N = len(self.vf_in)
                        if params == "calcrad":
                            vec = np.arange(N) + 0.5
                        elif (params == "calc1d"):
                            vec = np.arange(N) - np.asarray(N / 2) + 0.5
                        else:
                            vec = np.arange(N) - np.asarray(N / 2) + 0.5
                        self.x_out = vec * np.asarray(self.fresnel_dx_out)    

                self.vf_out = []
                for shift in self.distance_shifts:
                    MShift = [[1, shift], [0, 1]]
                    local_fresnel_mat = MMult(MShift, self.fresnel_mat)
                    if params == "calcrad":
                        self.kernel, self.j0 = cylindrical_fresnel_prepare(self.x_in, self.x_out, 0.000000780, local_fresnel_mat, True)
                        res = self.kernel @ self.vf_in
                    elif (params == "calc1d"):
                        dx0 = np.asarray(self.fresnel_dx_in / self.fresnel_factor)
                        fresnel_data = prepare_linear_fresnel_calc_data(local_fresnel_mat, dx0, len(self.x_in), 0.000000780, 1)
                        [res] = linear_fresnel_propogate(fresnel_data, np.asarray([self.vf_in]))
                    else:
                        dx0 = np.asarray(self.fresnel_dx_in / self.fresnel_factor)
                        fresnel_data = prepare_linear_fresnel_straight_calc_data(local_fresnel_mat, dx0, len(self.x_in), 0.000000780, 1)
                        [res] = linear_fresnel_propogate(fresnel_data, np.asarray([self.vf_in]))
                    self.vf_out.append(res)

            case "diode":
                self.diode.doCalcCommand(params)

            case "dispersion":
                self.dispersion.doCalcCommand(params)

    def exec_cavity_command(self, com):
        if len(com) == 0:
            return
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

def gain_function(Ga, N):
    xh1 = Ga * 4468377122.5 * 0.46 * 16.5
    xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * np.exp(0.000000000041*14E+10)
    gGain = xh1 - xh2 * np.exp(-0.000000000041 * N)
    return gGain

def loss_function(Gb, N0b, N):
    gAbs = Gb * 0.02 * (N - N0b)
    return gAbs

def shrink_with_max(arrp, max_size, fromX=-1, toX=-1):
    if (fromX >= 0 and toX > fromX and toX <= arrp.size):
        arr = arrp[fromX:toX]
    else:
        arr = arrp
    if arr.size <= 40000:
        return arr
    
    rc = arr.reshape(max_size, arr.size // max_size).max(axis=1)
    return rc

def generate_calc(data_obj, tab, offset = 0):

    if data_obj is None:
        return Div()
    
    data_obj.assure("calcData")

    calcData = data_obj.calcData
    added = Div()

    match tab:
        case 1: # Matrix
            added = Div(
                Div(
                    #Input(type="number", id=f'el{s}length', placeholder="0", step="0.01", style="width:50px;", value=f'{par[1]}'),
                    Button("M3=M1xM2", escapse=False, hx_post=f'/doCalc/1/mult/M1-M2-M3', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("M1=M3xM2^-1", escapse=False, hx_post=f'/doCalc/1/mult/M3-M2i-M1', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("M2=M1^-1xM3", escapse=False, hx_post=f'/doCalc/1/mult/M1i-M3-M2', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("M1=Fix(M3)", escapse=False, hx_post=f'/doCalc/1/mult/fixM3-M1', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'),
                    InputCalcS(f'MatFixer', "Fixer", f'{calcData.t_fixer}', width = 80),

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
                    Button("Cavity into ABCD mat", escapse=False, hx_post=f'/doCalc/2/cavity/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                FlexN([
                    Textarea(calcData.cavity_text, id="cavityText", style="min-height: 400px;", spellcheck="false",
                            hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}', hx_include="#calcForm *",),
                    Div(
                        Div(*[Div(Button(f"Fixed {x[1].split(';')[0]}", escapse=False, 
                                        hx_post=f'/doCalc/2/cavity/{x[0]}', hx_target="#gen_calc", 
                                        hx_vals='js:{localId: getLocalId()}')) for x in enumerate(calcData.fixed_cavitis)]),
                        ABCDMatControl("MCavity", calcData.cavity_mat),
                    ),
                ]),
            )
            
        case 3: # Fresnel
            internal_data = False
            try:
                if len(calcData.vf_out) > 0 and len(calcData.vf_out[0]) > 0:
                    s = calcData.kernel.shape[0]
                    skip = int(64 * calcData.fresnel_factor)
                    internal_data = True
            except:
                pass

            added = Div(
                Div(
                    SelectCalcS(tab, f'CalcSelectFront', "Initial Front", ["Gaussian", "Live Front", "From Output"], calcData.select_front, width = 150),
                    Button("Calc Radial", hx_post=f'/doCalc/3/fresnel/calcrad', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Calc 1D", hx_post=f'/doCalc/3/fresnel/calc1d', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Calc 1D Straight", hx_post=f'/doCalc/3/fresnel/calc1dstraight', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                ABCDMatControl("MFresnel", calcData.fresnel_mat),
                Div(
                    InputCalcS(tab, f'FresnelN', "N samples", f'{calcData.fresnel_N}', width = 80),
                    InputCalcS(tab, f'FresnelFactor', "Factor", f'{calcData.fresnel_factor}', width = 80),
                    InputCalcS(tab, f'FresnelDXIn', "dx input", f'{calcData.fresnel_dx_in}', width = 140),
                    InputCalcS(tab, f'FresnelDXOut', "dx ouput", f'{calcData.fresnel_dx_out}', width = 140),
                    InputCalcS(tab, f'FresnelWaist', "Waist", f'{calcData.fresnel_waist}', width = 80),
                ),
                Div(
                    Div(
                        generate_chart([cget(calcData.x_in).tolist()], [cget(np.square(np.abs(calcData.vf_in))).tolist()], [""], "In", color="#227700", marker="."),
                        *[generate_chart([cget(calcData.x_out).tolist()], [cget(np.square(np.abs(calcData.vf_out[x[0]]))).tolist()], [""], 
                                         f"Out {x[1]}", color="#774400", marker=".") for x in enumerate(calcData.distance_shifts)],
                        cls="box", style="background-color: #008080;"
                    ),
                    Div(
                        *[generate_chart([cget(calcData.x_in).tolist()], [cget(np.abs(calcData.kernel[min(i, s - 1)])).tolist()], [""], f"kernel {min(i, s - 1)}", color="#227722", marker=".") 
                            for i in range(0, s + 1, skip)],
                        *[generate_chart([cget(calcData.x_in).tolist()], [cget(calcData.j0[min(i, s - 1)]).tolist()], [""], f"j0 {min(i, s - 1)}", color="#770022", marker=".") 
                            for i in range(0, s + 1, skip)],
                        cls="box", style="background-color: #008080;"
                    ) if internal_data else Div(),
                ) if (len(calcData.vf_out) > 0 and len(calcData.vf_out[0]) > 0) else Div(),
            )
        case 4: # split pulse
            added = Div(
                FlexN(
                    (Div(
                        InputCalcS(f'pulseHarmony', "Harmony", f'{calcData.harmony}', width = 80, ),
                        Button("Calc", escapse=False, onclick=f"drawPulseGraph();", hx_vals='js:{localId: getLocalId()}'), 
                    ),
                    Canvas(id=f"pulsesplit", width=800, height=1500, style="background: blue;",
                    #    **{'onmousemove':f"mainCanvasMouseMove(event);",
                    #       'onmousedown':f"mainCanvasMouseDown(event);",
                    #       'onmouseup':f"mainCanvasMouseUp(event);",
                    #       }
                    ),
                    )
                ),
            )
        case 5: # "Diode Dynamics"
            added = calcData.diode.generate_calc()
        case 6: # Dispersion
            added = Div(
                Div(
                    Button("Calc Dispersion", hx_post=f'/doCalc/6/dispersion/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                ABCDMatControl("MDispersion", calcData.dispersion_mat),
                Div(
                    InputCalcS(f'DispersionN', "N samples", f'{calcData.dispersion_N}', width = 80),
                    InputCalcS(f'DispersionDX', "dx", f'{calcData.dispersion_dx}', width = 140),
                    InputCalcS(f'DispersionWaist', "Waist", f'{calcData.dispersion_waist}', width = 80),
                ),
            )
    return Div(
        Div(
            TabMaker("Matrix", "/tabcalc/1", tab == 1, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Cavity", "/tabcalc/2", tab == 2, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Fresnel", "/tabcalc/3", tab == 3, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Split Pulse", "/tabcalc/4", tab == 4, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Diode Dynamics", "/tabcalc/5", tab == 5, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Dispersion", "/tabcalc/6", tab == 6, target="#gen_calc", inc="#calcForm *"),
        ),
        Div(added, id="calcForm"),

        id="gen_calc"
    )
    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


