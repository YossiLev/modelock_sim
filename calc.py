# try:
#     import cupy
#     if cupy.cuda.is_available():
#         np = cupy
#         from cupyx.scipy.signal import fftconvolve
#     else:
#         import numpy as np
#         from scipy.signal import fftconvolve
# except ImportError:
import json
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

lib_diode.cmp_diode_gain.argtypes = [
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
lib_diode.cmp_diode_gain.restype = None

lib_diode.cmp_diode_loss.argtypes = [
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
lib_diode.cmp_diode_loss.restype = None

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
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.diode_round_trip.restype = None

lib_diode.cmp_diode_round_trip.argtypes = [
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
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.cmp_diode_round_trip.restype = None

lib_diode.mb_diode_round_trip.argtypes = [
    ctypes.POINTER(ctypes.c_double), 
    ctypes.POINTER(ctypes.c_double), 
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
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.mb_diode_round_trip.restype = None

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

def intens(arr):
    if len(arr) == 0 or arr.dtype != np.complex128:
        return arr
    if (np.isnan(arr.real)).any():
        print("NaN in real part **** could be an error")
        print(f"arr={arr[0]},{arr[1]},{arr[2]},{arr[3]}, ")
        raise Exception("NaN in real part **** could be an error")
    return np.square(arr.real) + np.square(arr.imag)

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

        self.diode_cavity_type = "Ring"
        self.diode_mode = "Amplitude"
        self.diode_sampling = "4096"
        self.diode_pulse_dtype = np.complex128

        self.diode_cavity_time = 3.95138389E-09 #4E-09
        self.diode_N = 4096 # * 4
        self.diode_dt = self.diode_cavity_time / self.diode_N
        self.diode_intensity = "Pulse"
        self.calculation_rounds = 1
        self.calculation_rounds_done = 0
        self.diode_pulse_width = 100.0

        # actual shift parameters in millimeters
        self.diode_absorber_shift = 0.0
        self.diode_gain_shift = 11.0
        self.diode_output_coupler_shift = 130.0

        self.loss_shift = self.diode_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
        self.gain_distance = self.mm_to_unit_shift(self.diode_gain_shift)
        self.oc_shift = self.mm_to_unit_shift(self.diode_output_coupler_shift)
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
        self.gain_width = 7.1 # in THz
        self.Ga = 5.024E-09#2.2379489747279815e-10 # 2.0 * np.log(100.0) / 7.44E+10 #2E-16
        self.Gb = 8.07e-10
        self.N0a = 20000000000.0 #0.0 # 1.6E+18
        self.N0b = 30000000000.0

        self.volume = 1 #0.46 * 0.03 * 2E-05
        self.cavity_loss = 4.5
        self.diode_update_pulse = "Update Pulse"
        self.h = 0.1

        # diode dynamics parameters
        self.diode_t_list = np.array([1], dtype=np.float64)
        self.diode_pulse = np.array([], dtype=self.diode_pulse_dtype)
        self.diode_pulse_original = np.array([], dtype=self.diode_pulse_dtype)
        self.diode_pulse_after = np.array([], dtype=self.diode_pulse_dtype)
        self.diode_accum_pulse = []
        self.diode_accum_pulse_after = []
        self.diode_gain = np.array([1], dtype=np.float64)
        self.diode_loss = np.array([1], dtype=np.float64)
        self.diode_gain_polarization = np.array([1], dtype=np.complex128)
        self.diode_loss_polarization = np.array([1], dtype=np.complex128)
        self.diode_gain_value = np.array([1], dtype=np.float64)
        self.diode_loss_value = np.array([1], dtype=np.float64)

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

    def mm_to_unit_shift(self, mm):
        shift = int(mm / 1E+03 / (self.diode_dt * 3E+08))
        print(f"mm_to_unit_shift mm={mm} shift={shift}")
        return shift
    
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

                smooth = np.asarray([1, 6, 15, 20, 15, 6, 1], dtype=np.float32) / 64.0         

                # Conditions for self-sustained pulsation and bistability in semiconductor lasers - Masayasu Ueno and Roy Lang
                # d Na / dt = - Na / Ta - Ga(Na - N0a) * N + Pa
                # d Nb / dt = - Nb / Tb - Gb(Nb - N0b) * N + Pb
                # d N  / dt = [(1 - h) * Ga(Na - N0a) + h * Gb(Nb - N0b) - GAMMA] * N

                #print(f"Round {i + 1} of {self.calculation_rounds}")
                match params:
                    case "calc":

                        self.diode_N = int(self.diode_sampling)
                        self.diode_dt = self.diode_cavity_time / self.diode_N
                        self.loss_shift = self.diode_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
                        self.gain_distance = self.mm_to_unit_shift(self.diode_gain_shift)
                        self.oc_shift = self.mm_to_unit_shift(self.diode_output_coupler_shift)
                        print(f"diode_N={self.diode_N} dt={self.diode_dt} loss_shift={self.loss_shift} gain_distance={self.gain_distance} oc_shift={self.oc_shift}")
 
                        gg = 20
                        ll = 0.4
                        oo = np.exp(- self.cavity_loss)
                        self.diode_levels_x = [0, self.gain_distance - 1, 
                                               self.gain_distance, self.oc_shift - 1, 
                                               self.oc_shift, self.loss_shift - self.gain_distance - 1, 
                                               self.loss_shift - self.gain_distance, self.loss_shift - 1, 
                                               self.loss_shift, self.diode_N]
                        #self.diode_levels_x.reverse()
                        self.diode_levels_y = [1, 1, 1 / gg, 1 / gg, 
                                               1 / gg / oo, 1 / gg / oo, 1 / gg / oo / gg, 1 / gg / oo / gg, 
                                               1 / gg / oo / gg / ll, 1 / gg / oo / gg / ll]

                        self.diode_t_list = np.arange(self.diode_N, dtype=np.float64)
                        self.diode_gain_value = np.full_like(self.diode_t_list, 0.0)
                        self.diode_loss_value = np.full_like(self.diode_t_list, 0.0)
                        self.diode_pulse_dtype = np.complex128 if self.diode_mode != "Intensity" else np.float64 
                        self.diode_pulse = np.array([], dtype=self.diode_pulse_dtype)
                        self.diode_pulse_original = np.array([], dtype=self.diode_pulse_dtype)
                        self.diode_pulse_after = np.array([], dtype=self.diode_pulse_dtype)
                    
                        pulseVal = np.array([60000 / self.diode_dt / self.volume], dtype=self.diode_pulse_dtype)
                        match self.diode_intensity:
                            case "Pulse":
                                w2 = (self.diode_pulse_width * 1.41421356237) if self.diode_pulse_dtype == np.complex128 else self.diode_pulse_width 
                                self.diode_pulse = pulseVal * np.exp(-np.square(self.diode_t_list - 1000.0) / (2 * w2 * w2))
                                self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
                                pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                                if self.diode_pulse_dtype == np.complex128:
                                    pulse_ratio = np.sqrt(pulse_ratio)
                                self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                            case "Noise":
                                self.diode_pulse = np.random.random(self.diode_t_list.shape).astype(self.diode_pulse_dtype)
                                self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
                                pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                                if self.diode_pulse_dtype == np.complex128:
                                    pulse_ratio = np.sqrt(pulse_ratio)
                                self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                            case "CW":
                                self.diode_pulse = np.full(self.diode_N, 1.0, dtype=self.diode_pulse_dtype)
                                self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
                                pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                                if self.diode_pulse_dtype == np.complex128:
                                    pulse_ratio = np.sqrt(pulse_ratio)
                                self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                            case "Flat":
                                self.diode_pulse = np.full_like(self.diode_t_list, 0).astype(self.diode_pulse_dtype)
                                self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
                               
                        if self.diode_pulse_dtype == np.complex128:
                            lambda_ = 1064E-09
                            omega0 = 2.0 * np.pi * 3E+08 / lambda_
                            phase = self.diode_t_list * (-1.j * omega0 * self.diode_dt)
                            self.diode_pulse = self.diode_pulse * np.exp(phase)
                            
                        self.diode_pulse_original = np.copy(self.diode_pulse)
                        self.diode_gain = np.full_like(self.diode_t_list, self.start_gain)
                        self.diode_loss = np.full_like(self.diode_t_list, self.start_absorber)
                        shape = self.diode_t_list.shape
                        self.diode_gain_polarization = np.full_like(self.diode_t_list, 0.j, dtype=np.complex128)
                        self.diode_loss_polarization = np.full_like(self.diode_t_list, 0.j, dtype=np.complex128)
                        self.calculation_rounds_done = 0
                    case "recalc":
                        if self.diode_cavity_type == "Ring" and self.diode_update_pulse == "Update Pulse":
                            self.diode_pulse = np.copy(self.diode_pulse_after)

                if self.diode_cavity_type == "Ring":
                    self.diode_round_trip_old()
                else:
                    self.diode_round_trip_new()

                self.calculation_rounds_done += self.calculation_rounds

    def diode_round_trip_new(self):
        self.oc_val = np.exp(- self.cavity_loss)
        self.diode_pulse_after = np.copy(self.diode_pulse)

        c_pulse = self.diode_pulse.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_gain = self.diode_gain.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_gain_polarization = self.diode_gain_polarization.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_gain_value = self.diode_gain_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_loss = self.diode_loss.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_loss_polarization = self.diode_loss_polarization.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_loss_value = self.diode_loss_value.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_pulse_after = self.diode_pulse_after.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        print(f"diode_round_trip_new: diode_mode={self.diode_mode} diode_pulse_dtype={self.diode_pulse_dtype}")
        if self.diode_mode == "MB":
            round_trip_func = lib_diode.mb_diode_round_trip
            round_trip_func(c_gain, c_gain_polarization, c_loss, c_loss_polarization, c_gain_value, c_loss_value,
                            c_pulse, c_pulse_after,
                            self.calculation_rounds, self.diode_N, self.loss_shift, self.oc_shift, self.gain_distance,
                            self.diode_dt, self.gain_width, self.Pa, self.Ta, self.Ga, self.N0a, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val)
            print("MB round trip done")
            k = 0
            for i in range(self.diode_pulse_after.shape[0]):
                if np.isnan(self.diode_pulse_after[i].real) or np.isnan(self.diode_pulse_after[i].imag):
                    k = k + 1
                    print(f"NAN index {i}: val=({self.diode_pulse_after[i].real}, {self.diode_pulse_after[i].imag})\n")
                if k > 100:
                    break
            self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

            print("MB accumulation done")
            return
        
        round_trip_func = lib_diode.cmp_diode_round_trip if self.diode_pulse_dtype == np.complex128 else lib_diode.diode_round_trip

        round_trip_func(c_gain, c_loss, c_gain_value, c_loss_value,
                        c_pulse, c_pulse_after,
                        self.calculation_rounds, self.diode_N, self.loss_shift, self.oc_shift, self.gain_distance,
                        self.diode_dt, self.gain_width, self.Pa, self.Ta, self.Ga, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val)

        self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

    def diode_round_trip_old(self):

        self.diode_pulse_after = np.copy(self.diode_pulse)

        for i in range(self.calculation_rounds):

            if self.diode_update_pulse == "Update Pulse":
                self.diode_pulse = np.copy(self.diode_pulse_after)

            self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
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

            # gain calculations
            gain_func = lib_diode.cmp_diode_gain if self.diode_pulse_dtype == np.complex128 else lib_diode.diode_gain
            gain_func(c_pulse, c_gain, c_gain_value, c_pulse_after, 
                                self.diode_N, self.diode_dt, self.Pa, self.Ta, self.Ga, self.gain_factor)

            self.summary_photons_after_gain = np.sum(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

            # absorber calculations
            loss_func = lib_diode.cmp_diode_loss if self.diode_pulse_dtype == np.complex128 else lib_diode.diode_loss
            loss_func(c_loss, c_loss_value, c_pulse_after,
                                self.diode_N, self.diode_dt, self.Pb, self.Tb, self.Gb, self.N0b)

            self.summary_photons_after_absorber = np.sum(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

            #cavity loss
            cavity_loss = self.cavity_loss if self.diode_pulse_dtype == np.float64 else self.cavity_loss / 2.0
            self.diode_pulse_after *= np.exp(- cavity_loss)
            self.summary_photons_after_cavity_loss = np.sum(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

            self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.diode_dt * self.volume

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

    def serialize_diode_graphs_data(self):
        graphs = []
        graphs.append({
            "id": "diode_pulse_chart",
            "title": "Pulse in (photons/sec)",
            "x_label": "Time (s)",
            "y_label": "Intensity",
            "lines": [
                {"color": "orange", "values": cget(intens(self.diode_pulse)).tolist(), "text": f"rrrrrrrrrr"}
            ],
            # "x_values": cget(self.diode_t_list).tolist(),
            # "y_values": cget(intens(self.diode_pulse)).tolist(),
        })

        return graphs
    
    def collectDiodeData(self, delay=20, more=False):
        data = {
            "type": "diode",
            "delay": delay,
            "more": more,
            "title": "Pulse in (photons/sec)",
            "graphs": self.serialize_diode_graphs_data(),
        }
        try:
            s = json.dumps(data)
        except TypeError as e:
            print("error: json exeption", data)
            s = ""
        return s

def gain_function(Ga, N):
    xh1 = Ga * 4468377122.5 * 0.46 * 16.5
    xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * np.exp(0.000000000041*14E+10)
    gGain = xh1 - xh2 * np.exp(-0.000000000041 * N)
    return gGain

def loss_function(Gb, N0b, N):
    gAbs = Gb * 0.02 * (N - N0b)
    return gAbs

def shrink_with_max(arr, max_size):
    if arr.size <= 40000:
        return arr
    
    rc = arr.reshape(max_size, arr.size // max_size).max(axis=1)
    return rc

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

    def SelectCalcS(id, title, options, selected, width = 150):
        return Select(*[Option(o) if o != selected else Option(o, selected="1") for o in options], id=id,
                    hx_trigger="input changed", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", hx_include="#calcForm *", 
                    hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;")

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
            # xGain = [minN, maxN]
            # xLoss = [minN, maxN / 2]
            xGain = np.linspace(4E+10, 9.0E+10, 50).tolist()
            xLoss = np.linspace(0.1E+10, 5.0E+10, 50).tolist()
            yGain = list(map(lambda x : gain_function(calcData.Ga, x), xGain))
            yLoss = list(map(lambda x : loss_function(calcData.Gb, calcData.N0b, x), xLoss))

            xGainRange = np.linspace(cget(np.min(calcData.diode_gain)) * calcData.volume, cget(np.max(calcData.diode_gain)) * calcData.volume, 10).tolist()
            yGainRange = list(map(lambda x : gain_function(calcData.Ga, x), xGainRange))
            xLossRange = np.linspace(cget(np.min(calcData.diode_loss)) * calcData.volume, cget(np.max(calcData.diode_loss)) * calcData.volume, 10).tolist()
            yLossRange = list(map(lambda x : loss_function(calcData.Gb, calcData.N0b, x), xLossRange))
            xVec = [xGainRange, xLossRange, xGain, xLoss ]
            yVec = [yGainRange, yLossRange, yGain, yLoss ]
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
            pulse = intens(calcData.diode_pulse)
            pulse_after = intens(calcData.diode_pulse_after)
            pulse_original = intens(calcData.diode_pulse_original)

            t_list = cget(shrink_with_max(calcData.diode_t_list, 1024)).tolist()

            added = Div(
                Div(
                        Button("Calculate", hx_post=f'/doCalc/5/diode/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                        Button("Recalculate", hx_post=f'/doCalc/5/diode/recalc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                        InputCalcS(f'DiodeRounds', "Rounds", f'{calcData.calculation_rounds}', width = 80),
                        Div(
                            Button("Save Parameters", onclick="saveMultiTimeParametersProcess()"),
                            Button("Restore Parameters", onclick="restoreMultiTimeParametersProcess('diodeDynamicsOptionsForm')"),
                            Div(Div("Give a name to saved parameters"),
                                Div(Input(type="text", id=f'parametersName', placeholder="Descibe", style="width:450px;", value="")),
                                Button("Save", onclick="saveMultiTimeParameters(1, 'diodeDynamicsOptionsForm')"),
                                Button("Cancel", onclick="saveMultiTimeParameters(0, 'diodeDynamicsOptionsForm')"),
                                id="saveParametersDialog", cls="pophelp", style="position: absolute; visibility: hidden"),
                            Div(Div("Select the parameters set"),
                                Div("", id="restoreParametersList"),
                                Div("", id="copyParametersList"),
                                Button("Cancel", onclick="restoreMultiTimeParameters(-1, 'diodeDynamicsOptionsForm')"),
                                Button("Export", onclick="exportMultiTimeParameters()"),
                                Button("Import", onclick="importMultiTimeParameters()"),
                                id="restoreParametersDialog", cls="pophelp", style="position: absolute; visibility: hidden"),
                            style="display: inline-block; position: relative;"
                        ),
                    ),
                FlexN(
                (
                Div(
                    Div(
                        SelectCalcS(f'DiodeSelectSampling', "Sampling", ["4096", "8192", "16384", "32768", "1048576"], calcData.diode_sampling, width = 100),
                        SelectCalcS(f'CalcDiodeCavityType', "Cavity Type", ["Ring", "Linear"], calcData.diode_cavity_type, width = 80),
                        SelectCalcS(f'CalcDiodeSelectMode', "mode", ["Intensity", "Amplitude", "MB"], calcData.diode_mode, width = 120),
                        InputCalcS(f'DiodePulseWidth', "Pulse width", f'{calcData.diode_pulse_width}', width = 80),
                        SelectCalcS(f'CalcDiodeSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], calcData.diode_intensity, width = 80),
                    ),
                    Div(
                        InputCalcS(f'DiodeAbsorberShift', "Absorber Shift (mm)", f'{calcData.diode_absorber_shift}', width = 100),
                        InputCalcS(f'DiodeGainShift', "Gain Shift (mm)", f'{calcData.diode_gain_shift}', width = 100),
                        InputCalcS(f'DiodeOutputCouplerShift', "OC Shift (mm)", f'{calcData.diode_output_coupler_shift}', width = 100),
                    ),
                    Div(
                        InputCalcS(f'Ta', "Gain Half-life (ps)", f'{calcData.Ta}', width = 80),
                        InputCalcS(f'gain_width', "Gain Width (THz)", f'{calcData.gain_width}', width = 80),
                        InputCalcS(f'Pa', "Gain current", f'{calcData.Pa}', width = 80),
                        InputCalcS(f'Ga', "Gain diff gain (cm^2)", f'{calcData.Ga}', width = 100),
                        InputCalcS(f'N0a', "Gain N0(tr) (cm^-3)", f'{calcData.N0a}', width = 100),
                        InputCalcS(f'start_gain', "Gain start val", f'{calcData.start_gain}', width = 100),

                    ),
                    Div(
                        InputCalcS(f'Tb', "Abs half-life (ps)", f'{calcData.Tb}', width = 80),
                        InputCalcS(f'Pb', "Abs current", f'{calcData.Pb}', width = 80),
                        InputCalcS(f'Gb', "Abs diff gain (cm^2)", f'{calcData.Gb}', width = 100),
                        InputCalcS(f'N0b', "Abs N0(tr) (cm^2)", f'{calcData.N0b}', width = 100),
                        InputCalcS(f'start_absorber', "Abs start val", f'{calcData.start_absorber}', width = 100),
                    ),
                    Div(
                        InputCalcS(f'dt', "dt (ps)", f'{format(calcData.diode_dt * 1E+12, ".4f")}', width = 80),
                        InputCalcS(f'volume', "Volume cm^3", f'{calcData.volume}', width = 80),
                        InputCalcS(f'initial_photons', "Initial Photns", f'{calcData.initial_photons}', width = 100),
                        InputCalcS(f'cavity_loss', "OC (Cavity loss)", f'{calcData.cavity_loss}', width = 80),
                        SelectCalcS(f'CalcDiodeUpdatePulse', "UpdatePulse", ["Update Pulse", "Unchanged Pulse"], calcData.diode_update_pulse, width = 120),
                        #InputCalcS(f'h', "Abs ratio", f'{calcData.h}', width = 50),
                    ),
                    id="diodeDynamicsOptionsForm"
                ),

                Div(
                    Table(
                        Tr(Td(f"{calcData.calculation_rounds_done}"), Td("Value"), Td("Change")), 
                        Tr(Td("Before gain"), Td(f"{calcData.summary_photons_before:.3e}"), Td("")), 
                        Tr(Td("After gain"), Td(f"{calcData.summary_photons_after_gain:.3e}"), Td(f"{(calcData.summary_photons_after_gain - calcData.summary_photons_before):.3e}")), 
                        Tr(Td("After absorber"), Td(f"{calcData.summary_photons_after_absorber:.3e}"), Td(f"{(calcData.summary_photons_after_absorber - calcData.summary_photons_before):.3e}")),
                        Tr(Td("After OC"), Td(f"{calcData.summary_photons_after_cavity_loss:.3e}"), Td(f"{(calcData.summary_photons_after_cavity_loss - calcData.summary_photons_before):.3e}")),
                        Tr(Td("Output"), Td(f"{output_photons:.3e}"), Td(f"{(output_photons * energy_of_1064_photon * 10E+9):.3e}nJ")),
                        ),
                ))),

                Div(
                    Div(
                        Frame_chart("fc1", [t_list], 
                                       [cget(shrink_with_max(pulse_original, 1024)).tolist(), cget(shrink_with_max(np.log(pulse_after+ 0.000000001), 1024)).tolist()], [""], 
                                       "Original Pulse and Pulse after (photons/sec)", h=2, color=colors, marker=None, twinx=True),

                        generate_chart([cget(calcData.diode_t_list).tolist(), calcData.diode_levels_x], 
                                       [cget(pulse).tolist(), calcData.diode_levels_y], [""], 
                                       "Pulse in (photons/sec)", h=2, color=["red", "black"], marker=None, twinx=True),
                        # Div(
                        #    Div(cls="handle", draggable="true"),
                        #        FlexN([graphCanvas(id="diode_pulse_chart", width=1100, height=300, options=False, mode = 2), 
                        #        ]), cls="container"
                        # ),
                        generate_chart([t_list], 
                                       [cget(shrink_with_max(pulse_after, 1024)).tolist()], [""], 
                                       "Pulse out (photons/sec)", h=2, color=colors, marker=None, twinx=True),
                        generate_chart([t_list], 
                                       [cget(shrink_with_max(calcData.diode_accum_pulse, 1024)).tolist(), cget(shrink_with_max(calcData.diode_accum_pulse_after, 1024)).tolist()], [""], 
                                       f"Accumulate Pulse AND after (photons) [difference: {(calcData.diode_accum_pulse_after[-1] - calcData.diode_accum_pulse[-1]):.2e}]", 
                                       h=2, color=colors, marker=None, twinx=True),
                        generate_chart([t_list], 
                                       [cget(shrink_with_max(calcData.diode_gain, 1024)).tolist(), cget(shrink_with_max(calcData.diode_gain_value, 1024)).tolist()], [""], 
                                       f"Gain carriers (1/cm^3) [{(max_gain - min_gain):.2e} = {max_gain:.4e} - {min_gain:.4e}] and Gain (cm^-1)", 
                                       color=["black", "green"], h=2, marker=None, twinx=True),
                        generate_chart([t_list], 
                                        [cget(np.angle(shrink_with_max(calcData.diode_gain_polarization, 1024))).tolist(), 
                                         cget(np.absolute(shrink_with_max(calcData.diode_gain_polarization, 1024))).tolist()], [""], 
                                        "Gain Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                        generate_chart([t_list], 
                                       [cget(shrink_with_max(calcData.diode_loss, 1024)).tolist(), cget(shrink_with_max(calcData.diode_loss_value, 1024)).tolist()], [""], 
                                       f"Abs carrs (cm^-3) [{(max_loss - min_loss):.2e} = {max_loss:.3e} - {min_loss:.3e}] and Loss (cm^-1)", 
                                       color=["black", "red"], h=2, marker=None, twinx=True),
                        generate_chart([t_list], 
                                        [cget(np.angle(shrink_with_max(calcData.diode_loss_polarization, 1024))).tolist(), 
                                         cget(np.absolute(shrink_with_max(calcData.diode_loss_polarization, 1024))).tolist()], [""], 
                                        "Loss Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                        generate_chart([t_list], 
                                       [cget(np.exp(- calcData.cavity_loss) * np.multiply(shrink_with_max(calcData.diode_gain_value, 1024),
                                                                                           shrink_with_max(calcData.diode_loss_value, 1024))).tolist(),
                                        cget(shrink_with_max(pulse, 1024)).tolist()], [""],
                                       "Net gain", color=["blue", "red"], h=2, marker=None, twinx=True),
                        generate_chart(xVec, yVec, [""], "Gain By Pop", h=4, color=["black", "black", "green", "red"], marker=".", lw=[5, 5, 1, 1]),

                        #Div(calcData.collectDiodeData(), id="numData"),

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


