# try:
#     import cupy
#     if cupy.cuda.is_available():
#         np = cupy
#         from cupyx.scipy.signal import fftconvolve
#     else:
#         import numpy as np
#         from scipy.signal import fftconvolve
# except ImportError:
import numpy as np
from scipy.signal import fftconvolve

import re
from fasthtml.common import *
from controls import *
from multi_mode import cget, cylindrical_fresnel_prepare, prepare_linear_fresnel_calc_data, prepare_linear_fresnel_straight_calc_data, linear_fresnel_propogate
import ctypes
import platform

lib_suffix = {
    "Linux": "so",
    "Darwin": "dylib",
    "Windows": "dll"
}.get(platform.system(), "so") 

lib_diode = ctypes.CDLL(os.path.abspath(f"./cfuncs/libs/libdiode.{lib_suffix}"))

lib_diode.diode_gain.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.diode_gain.restype = None

lib_diode.diode_loss.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double, 
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.diode_loss.restype = None

lib_diode.diode_round_trip.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.diode_round_trip.restype = None

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
        print("CalculatorData init")
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

        self.diode_intensity = "Pulse"
        self.calculation_rounds = 1

        self.diode_pulse_width = 100.0
        self.diode_alpha = 0.01
        self.diode_gamma0 = 5.0
        self.diode_saturation = 2000
        self.loss_shift = 2048
        self.gain_distance = 3
        self.oc_shift = 1500
        self.oc_val = 0.02

        self.start_gain = 7.44E+10
        self.start_absorber = 0.0 #18200000000.0
        self.initial_photons = 1E+07
        self.gain_saturation = 5E+07
        self.gain_length = 0.46
        self.loss_length = 0.04
        self.gain_factor = 1.0
        self.Ta = 3000    # in pico seconds
        self.Tb = 300 # in pico seconds
        self.Pa = 2.48e+19 #2.8e+23
        self.Pb = 0
        self.Ga = 5.024E-09#2.2379489747279815e-10 # 2.0 * np.log(100.0) / 7.44E+10 #2E-16
        self.Gb = 8.07e-10
        self.N0a = 20000000000.0 #0.0 # 1.6E+18
        self.N0b = 30000000000.0

        self.dt = 1e-12  # 1 pico second
        self.volume = 1 #0.46 * 0.03 * 2E-05
        self.cavity_loss = 4.5
        self.diode_update_pulse = "Update Pulse"
        self.h = 0.1

        # diode dynamics parameters
        self.diode_t_list = []
        self.diode_pulse = []
        self.diode_pulse_original = []
        self.diode_pulse_after = []
        self.diode_accum_pulse = []
        self.diode_accum_pulse_after = []
        self.diode_gain = np.array([1])
        self.diode_loss = np.array([1])
        self.diode_gain_value = np.array([1])
        self.diode_loss_value = np.array([1])

        # diode summary parameters
        self.summary_photons_before = 0.0
        self.summary_photons_after_gain = 0.0
        self.summary_photons_after_absorber = 0.0
        self.summary_photons_after_cavity_loss = 0.0

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
                N = 4096

                self.diode_t_list = np.arange(N, dtype=np.float64)
                self.diode_gain_value = np.full_like(self.diode_t_list, 0.0)
                self.diode_loss_value = np.full_like(self.diode_t_list, 0.0)
                smooth = np.asarray([1, 6, 15, 20, 15, 6, 1], dtype=np.float32) / 64.0         

                # Conditions for self-sustained pulsation and bistability in semiconductor lasers - Masayasu Ueno and Roy Lang
                # d Na / dt = - Na / Ta - Ga(Na - N0a) * N + Pa
                # d Nb / dt = - Nb / Tb - Gb(Nb - N0b) * N + Pb
                # d N  / dt = [(1 - h) * Ga(Na - N0a) + h * Gb(Nb - N0b) - GAMMA] * N

                #print(f"Round {i + 1} of {self.calculation_rounds}")
                match params:
                    case "calc":
                        pusleVal = 60000 / self.dt / self.volume
                        match self.diode_intensity:
                            case "Pulse":
                                self.diode_pulse = pusleVal * np.exp(-np.square(self.diode_t_list - 1000.0) / (self.diode_pulse_width * self.diode_pulse_width))
                                self.diode_accum_pulse = np.add.accumulate(self.diode_pulse) * self.dt * self.volume
                                pulse_photons = self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, self.initial_photons / pulse_photons)
                                self.diode_pulse = np.multiply(self.diode_pulse, self.initial_photons / pulse_photons)
                            case "Noise":
                                self.diode_pulse = np.random.random(self.diode_t_list.shape)
                                self.diode_accum_pulse = np.add.accumulate(self.diode_pulse) * self.dt * self.volume
                                pulse_photons = self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, self.initial_photons / pulse_photons)
                                self.diode_pulse = np.multiply(self.diode_pulse, self.initial_photons / pulse_photons)
                            case "CW":
                                self.diode_pulse = np.full_like(self.diode_t_list, 1)
                                self.diode_accum_pulse = np.add.accumulate(self.diode_pulse) * self.dt * self.volume
                                pulse_photons = self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, self.initial_photons / pulse_photons)
                                self.diode_pulse = np.multiply(self.diode_pulse, self.initial_photons / pulse_photons)
                            case "Flat":
                                self.diode_pulse = np.full_like(self.diode_t_list, 0)
                        self.diode_pulse_original = np.copy(self.diode_pulse)
                        self.diode_gain = np.full_like(self.diode_pulse, self.start_gain)
                        self.diode_loss = np.full_like(self.diode_pulse, self.start_absorber)

                    case "recalc":
                        if self.diode_update_pulse == "Update Pulse":
                            self.diode_pulse = np.copy(self.diode_pulse_after)

                #self.diode_round_trip_old()
                self.diode_round_trip_new()

    def diode_round_trip_new(self):
        N = len(self.diode_t_list)
        self.oc_val = np.exp(- self.cavity_loss)
        print(f"oc_val: {self.oc_val}")
        self.diode_pulse_after = np.copy(self.diode_pulse)


        c_pulse = self.diode_pulse.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_gain = self.diode_gain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_gain_value = self.diode_gain_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_loss = self.diode_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_loss_value = self.diode_loss_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_pulse_after = self.diode_pulse_after.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        lib_diode.diode_round_trip(c_gain, c_loss, c_gain_value, c_loss_value,
                                    c_pulse, c_pulse_after,
                                    self.calculation_rounds, N, self.loss_shift, self.oc_shift, self.gain_distance,
                                    self.dt, self.Pa, self.Ta, self.Ga, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val)

        self.diode_accum_pulse_after = np.add.accumulate(self.diode_pulse_after) * self.dt * self.volume


    def diode_round_trip_old(self):
        N = 4096

        self.diode_pulse_after = np.copy(self.diode_pulse)

        for i in range(self.calculation_rounds):

            if self.diode_update_pulse == "Update Pulse":
                self.diode_pulse = np.copy(self.diode_pulse_after)

            self.diode_accum_pulse = np.add.accumulate(self.diode_pulse) * self.dt * self.volume
            self.summary_photons_before = self.diode_accum_pulse[-1]

            self.gain_factor = 0.46 
            self.loss_factor = 0.010 # rrrrrr

            #xh1 = self.Ga * 4468377122.5 * self.gain_factor * 16.5
            #xh2 = self.Ga * 4468377122.5 * self.gain_factor * 0.32 * np.exp(0.000000000041*14E+10)
            # gain medium calculations
            c_pulse = self.diode_pulse.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_gain = self.diode_gain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_gain_value = self.diode_gain_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_loss = self.diode_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_loss_value = self.diode_loss_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            c_pulse_after = self.diode_pulse_after.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            # Call the C function
            lib_diode.diode_gain(c_pulse, c_gain, c_gain_value, c_pulse_after, 
                                N, self.dt, self.Pa, self.Ta, self.Ga, self.gain_factor)

            # for i in range(N):
            #     iN = i + 1 if i < N - 1 else 0
            #     #gGain = self.Ga * self.gain_factor * (self.diode_gain[i] - self.N0a) * self.diode_pulse[i]
            #     #gGain = self.Ga * 4468377122.5 * self.gain_factor * (16.5-0.32*np.exp(-0.000000000041*(self.diode_gain[i]-14E+10)))
            #     gGain = xh1 - xh2 * np.exp(-0.000000000041 * self.diode_gain[i])
            #     #print(f"i={i}, gGain={gGain}, gGaint={gGaint}")
            #     self.diode_gain_value[i] = 1 + gGain
            #     gGain *= self.diode_pulse[i]
            #     self.diode_pulse_after[i] = self.diode_pulse[i] + gGain
            #     self.diode_gain[iN] = self.diode_gain[i] + self.dt * (- gGain + self.Pa - self.diode_gain[i] / (self.Ta * 1E-12))

            self.summary_photons_after_gain = np.sum(self.diode_pulse_after) * self.dt * self.volume

            # absorber calculations

            lib_diode.diode_loss(c_loss, c_loss_value, c_pulse_after,
                                N, self.dt, self.Pb, self.Tb, self.Gb, self.N0b)
            # for i in range(N):
            #     iN = i + 1 if i < N - 1 else 0
            #     gAbs = self.Gb * self.loss_factor * (self.diode_loss[i] - self.N0b)
            #     self.diode_loss_value[i] = 1 + gAbs
            #     gAbs *= self.diode_pulse_after[i]
            #     self.diode_loss[iN] = self.diode_loss[i] + self.dt * (- gAbs + self.Pb - self.diode_loss[i] / (self.Tb * 1E-12))
            #     self.diode_pulse_after[i] += gAbs

            self.summary_photons_after_absorber = np.sum(self.diode_pulse_after) * self.dt * self.volume

            #cavity loss
            self.diode_pulse_after *= np.exp(- self.cavity_loss)
            self.summary_photons_after_cavity_loss = np.sum(self.diode_pulse_after) * self.dt * self.volume

            self.diode_accum_pulse_after = np.add.accumulate(self.diode_pulse_after) * self.dt * self.volume

            #pulse_photons_after = self.diode_accum_pulse_after[-1] 
            #print(f"Pulse photons: {pulse_photons}, after: {pulse_photons_after}, {'{:.3e}'.format(pulse_photons_after - pulse_photons)} Gain factor: {self.gain_factor}")

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

def generate_calc(data_obj, tab, offset = 0):
                        
    def InputCalcS(id, title, value, step=0.01, width = 150):
        return Div(
                Div(title, cls="floatRight", style="font-size: 10px; top:-1px; right:10px; padding: 0px 4px; background: #e7f0f0;"),
                Input(type="number", id=id, title=title,
                    value=value, step=f"{step}", 
                    # hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
                    # hx_vals='js:{localId: getLocalId()}',
                    style=f"width:{width}px; margin:2px;"),
                style="display: inline-block; position: relative;"
        )
    def InputCalcM(id, title, value, step=0.01, width = 150):
        return Div(
                Div(title, cls="floatRight", style="font-size: 10px; top:-3px; right:10px;background: #e7edb8;"),
                Input(type="number", id=id, title=title,
                    value=value, step=f"{step}", 
#                    hx_trigger="input changed delay:1s", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", 
#                    hx_vals='js:{localId: getLocalId()}',
                    style=f"width:{width}px; margin:2px;",
                    **{'onkeyup':f"validateMat(event);",
                       'onpaste':f"validateMat(event);",}),
                style="display: inline-block; position: relative;"
        )

    def SelectCalcS(id, title, options, selected, width = 150):
        return Select(*[Option(o) if o != selected else Option(o, selected="1") for o in options], id=id,
                    hx_trigger="input changed", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", hx_include="#calcForm *", 
                    hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;")

    def ABCDMatControl(name, M):
        det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
        msg = f'&#9888; det={det}'
        return Div(
            Div(
                Div(
                    Img(src="/static/eigen.png", title="Copy", width=20, height=20, onclick=f"AbcdMatEigenValuesCalc('{name}');"),
                    Img(src="/static/copy.png", title="Copy", width=20, height=20, onclick=f"AbcdMatCopy('{name}');"),
                    Img(src="/static/paste.png", title="Paste", width=20, height=20, onclick=f"AbcdMatPaste('{name}');"),
                    cls="floatRight"
                ),
                Span(name), 
                Span(NotStr(msg), id=f"{name}_msg", 
                     style=f"visibility: {'hidden' if abs(det - 1.0) < 0.000001 else 'visible'}; color: yellow; background-color: red; padding: 1px; border-radius: 4px; margin-left: 30px; ") if len(msg) > 0 else "",
            ),
            Div(
                InputCalcM(f'{name}_A', "A", f'{M[0][0]}', width = 180),
                InputCalcM(f'{name}_B', "B", f'{M[0][1]}', width = 180),
            ),
            Div(
                InputCalcM(f'{name}_C', "C", f'{M[1][0]}', width = 180),
                InputCalcM(f'{name}_D', "D", f'{M[1][1]}', width = 180),
            ),
            Div("", id=f"{name}_eigen", style="visibility: hidden;"),
            cls="ABCDMatControl"
        )

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
                    SelectCalcS(f'CalcSelectFront', "Initial Front", ["Gaussian", "Live Front", "From Output"], calcData.select_front, width = 150),
                    Button("Calc Radial", hx_post=f'/doCalc/3/fresnel/calcrad', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Calc 1D", hx_post=f'/doCalc/3/fresnel/calc1d', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Calc 1D Straight", hx_post=f'/doCalc/3/fresnel/calc1dstraight', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                ),
                ABCDMatControl("MFresnel", calcData.fresnel_mat),
                Div(
                    InputCalcS(f'FresnelN', "N samples", f'{calcData.fresnel_N}', width = 80),
                    InputCalcS(f'FresnelFactor', "Factor", f'{calcData.fresnel_factor}', width = 80),
                    InputCalcS(f'FresnelDXIn', "dx input", f'{calcData.fresnel_dx_in}', width = 140),
                    InputCalcS(f'FresnelDXOut', "dx ouput", f'{calcData.fresnel_dx_out}', width = 140),
                    InputCalcS(f'FresnelWaist', "Waist", f'{calcData.fresnel_waist}', width = 80),
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
            colors = ["#ff0000", "#ff8800", "#aaaa00", "#008800", "#0000ff", "#ff00ff", "#110011"]
            minN = 2E+10
            maxN = 7E+10
            xGain = [minN, maxN]
            xLoss = [minN, maxN / 2]
            yGain = list(map(lambda x : calcData.Ga * (x - calcData.N0a), xGain))
            yLoss = list(map(lambda x : calcData.Gb * (x - calcData.N0b), xLoss))
            xGainRange = [cget(np.min(calcData.diode_gain)) * calcData.volume, cget(np.max(calcData.diode_gain)) * calcData.volume]
            yGainRange = list(map(lambda x : calcData.Ga * (x - calcData.N0a), xGainRange))
            xLossRange = [cget(np.min(calcData.diode_loss)) * calcData.volume, cget(np.max(calcData.diode_loss)) * calcData.volume]
            yLossRange = list(map(lambda x : calcData.Gb * (x - calcData.N0b), xLossRange))
            xVec = [xGain, xLoss, xGainRange, xLossRange]
            yVec = [yGain, yLoss, yGainRange, yLossRange]
            min_gain = np.min(calcData.diode_gain) * calcData.volume
            max_gain = np.max(calcData.diode_gain) * calcData.volume
            min_loss = np.min(calcData.diode_loss) * calcData.volume# * 0.04 / 0.46
            max_loss = np.max(calcData.diode_loss) * calcData.volume# * 0.04 / 0.46
            output_photons = calcData.summary_photons_after_absorber - calcData.summary_photons_after_cavity_loss
            energy_of_1064_photon = 1.885E-19 # Joule
            # if len(calcData.diode_pulse) > 0:
            #     print(np.shape(calcData.diode_pulse_after))
            #     diode_pulse_fftr = np.fft.rfft(np.sqrt(np.asarray(calcData.diode_pulse_after)))
            #     diode_pulse_fft = np.concatenate((diode_pulse_fftr[::-1][1:- 1], diode_pulse_fftr))

            added = Div(FlexN(
                (Div(
                    Div(
                        SelectCalcS(f'CalcDiodeSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], calcData.diode_intensity, width = 150),
                        Button("Calculate", hx_post=f'/doCalc/5/diode/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                        Button("Recalculate", hx_post=f'/doCalc/5/diode/recalc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                        InputCalcS(f'DiodeRounds', "Rounds", f'{calcData.calculation_rounds}', width = 80),
                    ),
                    Div(
                        InputCalcS(f'DiodePulseWidth', "Width", f'{calcData.diode_pulse_width}', width = 80),
                        InputCalcS(f'DiodeAlpha', "Alpha", f'{calcData.diode_alpha}', width = 80),
                        InputCalcS(f'DiodeGamma0', "Gamma", f'{calcData.diode_gamma0}', width = 80),
                        InputCalcS(f'DiodeSaturation', "U-Sat", f'{calcData.diode_saturation}', width = 80),
                    ),
                    Div(
                        InputCalcS(f'Ta', "Gain Half-life (ps)", f'{calcData.Ta}', width = 100),
                        InputCalcS(f'Pa', "Gain current", f'{calcData.Pa}', width = 100),
                        InputCalcS(f'Ga', "Gain diff gain (cm^2)", f'{calcData.Ga}', width = 100),
                        InputCalcS(f'N0a', "Gain N0(tr) (cm^-3)", f'{calcData.N0a}', width = 100),
                        InputCalcS(f'start_gain', "Gain start val", f'{calcData.start_gain}', width = 100),

                    ),
                    Div(
                        InputCalcS(f'Tb', "Abs half-life (ps)", f'{calcData.Tb}', width = 100),
                        InputCalcS(f'Pb', "Abs current", f'{calcData.Pb}', width = 100),
                        InputCalcS(f'Gb', "Abs diff gain (cm^2)", f'{calcData.Gb}', width = 100),
                        InputCalcS(f'N0b', "Abs N0(tr) (cm^2)", f'{calcData.N0b}', width = 100),
                        InputCalcS(f'start_absorber', "Abs start val", f'{calcData.start_absorber}', width = 100),
                    ),
                    Div(
                        InputCalcS(f'dt', "dt (sec)", f'{calcData.dt}', width = 60),
                        InputCalcS(f'volume', "Volume cm^3", f'{calcData.volume}', width = 120),
                        InputCalcS(f'initial_photons', "Initial Photns", f'{calcData.initial_photons}', width = 120),
                        InputCalcS(f'cavity_loss', "OC (Cavity loss)", f'{calcData.cavity_loss}', width = 100),
                        SelectCalcS(f'CalcDiodeUpdatePulse', "UpdatePulse", ["Update Pulse", "Unchanged Pulse"], calcData.diode_update_pulse, width = 120),
                        InputCalcS(f'h', "Abs ratio", f'{calcData.h}', width = 50),
                    ),
                ),

                Div(
                    Table(
                        Tr(Td(""), Td("Value"), Td("Change")), 
                        Tr(Td("Begore gain"), Td(f"{calcData.summary_photons_before:.3e}"), Td("")), 
                        Tr(Td("After gain"), Td(f"{calcData.summary_photons_after_gain:.3e}"), Td(f"{(calcData.summary_photons_after_gain - calcData.summary_photons_before):.3e}")), 
                        Tr(Td("After absorber"), Td(f"{calcData.summary_photons_after_absorber:.3e}"), Td(f"{(calcData.summary_photons_after_absorber - calcData.summary_photons_before):.3e}")),
                        Tr(Td("After OC"), Td(f"{calcData.summary_photons_after_cavity_loss:.3e}"), Td(f"{(calcData.summary_photons_after_cavity_loss - calcData.summary_photons_before):.3e}")),
                        Tr(Td("Output"), Td(f"{output_photons:.3e}"), Td(f"{(output_photons * energy_of_1064_photon * 10E+9):.3e}nJ")),
                        ),
                ))),

                Div(
                    Div(
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_pulse_original).tolist(), cget(np.log(calcData.diode_pulse_after)).tolist()], [""], 
                                       "Original Pulse and Pulse after (photons/sec)", h=2, color=colors, marker=None, twinx=True),

                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_pulse).tolist()], [""], 
                                       "Pulse in (photons/sec)", h=2, color=colors, marker=None, twinx=True),
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_pulse_after).tolist()], [""], 
                                       "Pulse out (photons/sec)", h=2, color=colors, marker=None, twinx=True),
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_accum_pulse).tolist(), cget(calcData.diode_accum_pulse_after).tolist()], [""], 
                                       f"Accumulate Pulse AND after (photons) [difference: {(calcData.diode_accum_pulse_after[-1] - calcData.diode_accum_pulse[-1]):.2e}]", 
                                       h=2, color=colors, marker=None, twinx=True),
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_gain).tolist(), cget(calcData.diode_gain_value).tolist()], [""], 
                                       f"Gain carriers (1/cm^3) [{(max_gain - min_gain):.2e} = {max_gain:.4e} - {min_gain:.4e}] and Gain (cm^-1)", 
                                       color=["black", "green"], h=2, marker=None, twinx=True),
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(calcData.diode_loss).tolist(), cget(calcData.diode_loss_value).tolist()], [""], 
                                       f"Abs carrs (cm^-3) [{(max_loss - min_loss):.2e} = {max_loss:.3e} - {min_loss:.3e}] and Loss (cm^-1)", 
                                       color=["black", "red"], h=2, marker=None, twinx=True),
                        generate_chart([cget(calcData.diode_t_list).tolist()], 
                                       [cget(np.exp(- calcData.cavity_loss) * np.multiply(calcData.diode_gain_value, calcData.diode_loss_value)).tolist(),
                                        cget(calcData.diode_pulse).tolist()], [""],
                                       "Net gain", color=["blue", "red"], h=2, marker=None, twinx=True),
                        generate_chart(xVec, yVec, [""], "Gain By Pop", h=4, color=["green", "red", "black", "black"], marker="."),

                        cls="box", style="background-color: #008080;"
                  ),
                ) if (len(calcData.diode_pulse) > 0 and len(calcData.diode_gain) > 0) else Div(),
            )
    return Div(
        Div(
            TabMaker("Matrix", "/tabcalc/1", tab == 1, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Cavity", "/tabcalc/2", tab == 2, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Fresnel", "/tabcalc/3", tab == 3, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Split Pulse", "/tabcalc/4", tab == 4, target="#gen_calc", inc="#calcForm *"),
            TabMaker("Diode Dynamics", "/tabcalc/5", tab == 5, target="#gen_calc", inc="#calcForm *"),
        ),
        Div(added, id="calcForm"),

        id="gen_calc"
    )
    #return Img(src=f'data:image/jpg;base64,{str(my_base64_jpgData, "utf-8")}')


