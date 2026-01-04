# try:
#     import cupy
#     if cupy.cuda.is_available():
#         np = cupy
#         from cupyx.scipy.signal import fftconvolve
#     else:
#         import numpy as np
#         from scipy.signal import fftconvolve
# except ImportError:
import ctypes
import platform
import json
import numpy as np
from fasthtml.common import *
from controls import *

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
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double
]
lib_diode.mb_diode_round_trip.restype = None



def intens(arr):
    if len(arr) == 0 or arr.dtype != np.complex128:
        return arr
    if (np.isnan(arr.real)).any():
        print("NaN in real part **** could be an error")
        print(f"arr={arr[0]},{arr[1]},{arr[2]},{arr[3]}, ")
        raise Exception("NaN in real part **** could be an error")
    return np.square(arr.real) + np.square(arr.imag)

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

def cget(x):
    return x.get() if hasattr(x, "get") else x

                        
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

def SelectCalcS(tab, id, title, options, selected, width = 150):
    return Select(*[Option(o) if o != selected else Option(o, selected="1") for o in options], id=id,
                hx_trigger="input changed", hx_post=f"/clUpdate/{tab}", hx_target="#gen_calc", hx_include="#calcForm *", 
                hx_vals='js:{localId: getLocalId()}', style=f"width:{width}px;")

class diode_calc:
    def __init__(self):
        self.diode_view_from = -1
        self.diode_view_to = -1

        self.diode_cavity_type = "Ring"
        self.diode_mode = "Amplitude"
        self.diode_sampling = "4096"
        self.diode_sampling_x = "32"
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

        self.left_arm_mat = [[1, 0], [0, 1]]
        self.right_arm_mat = [[1, 0], [0, 1]]
        self.left_arm_cavity = ""
        self.right_arm_cavity = ""
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

        self.rand_factor_seed = 0.0000000005
        self.kappa = 3.0E07
        self.C_loss = 95.0E+06
        self.C_gain = 300.0E+05
        self.coupling_out_loss =-5000E+06
        self.coupling_out_gain = 2800E+05

        # diode dynamics parameters
        self.diode_t_list = np.array([1], dtype=np.float64)
        self.diode_pulse = np.array([], dtype=self.diode_pulse_dtype)
        self.diode_pulse_init = np.array([], dtype=self.diode_pulse_dtype)
        self.diode_pulse_save = np.array([], dtype=self.diode_pulse_dtype)
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
    
    def zoom_view(self, factor):
        if (self.diode_view_from == -1) or (self.diode_view_to == -1):
            self.diode_view_from = 0
            self.diode_view_to = self.diode_N
        center = (self.diode_view_from + self.diode_view_to) / 2
        half_range = (self.diode_view_to - self.diode_view_from) / 2 / factor
        self.diode_view_from = int(center - half_range)
        self.diode_view_to = int(center + half_range)
        print(f"zoom_view factor={factor} from={self.diode_view_from} to={self.diode_view_to}")

    def shift_view(self, shift_factor):
        if (self.diode_view_from == -1) or (self.diode_view_to == -1):
            self.diode_view_from = 0
            self.diode_view_to = self.diode_N
        center = (self.diode_view_from + self.diode_view_to) / 2
        half_range = (self.diode_view_to - self.diode_view_from) / 2
        shift = int(half_range * shift_factor)
        self.diode_view_from = int(center - half_range + shift)
        self.diode_view_to = int(center + half_range + shift)
        print(f"shift_view factor={shift_factor} from={self.diode_view_from} to={self.diode_view_to}")


    def doCalcCommand(self, params):

                smooth = np.asarray([1, 6, 15, 20, 15, 6, 1], dtype=np.float32) / 64.0         

                # Conditions for self-sustained pulsation and bistability in semiconductor lasers - Masayasu Ueno and Roy Lang
                # d Na / dt = - Na / Ta - Ga(Na - N0a) * N + Pa
                # d Nb / dt = - Nb / Tb - Gb(Nb - N0b) * N + Pb
                # d N  / dt = [(1 - h) * Ga(Na - N0a) + h * Gb(Nb - N0b) - GAMMA] * N

                #print(f"Round {i + 1} of {self.calculation_rounds}")
                match params:
                    case "view":
                        return
                    case "zoomin":
                        self.zoom_view(2.0)
                        return
                    case "zoomout":
                        self.zoom_view(0.5)
                        return
                    case "shiftright":
                        self.shift_view(-0.5)
                        return
                    case "shiftleft":
                        self.shift_view(0.5)
                        return
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
                                w2 = (self.diode_pulse_width * 1.0E-12 /self.diode_dt * 1.41421356237) if self.diode_pulse_dtype == np.complex128 else self.diode_pulse_width 
                                self.diode_pulse = pulseVal * np.exp(-np.square(self.diode_t_list - self.diode_N / 2) / (2 * w2 * w2))
                                self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.diode_dt * self.volume
                                pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                                self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                                if self.diode_pulse_dtype == np.complex128:
                                    pulse_ratio = np.sqrt(pulse_ratio)
                                self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                                for i in range(990, 1010):
                                    print(f"diode_pulse[{i}]={self.diode_pulse[i]}, angle={np.angle(self.diode_pulse[i])}")
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
                               
                        # if self.diode_pulse_dtype == np.complex128:
                        #     lambda_ = 1064E-09
                        #     omega0 = 2.0 * np.pi * 3E+08 / lambda_
                        #     phase = self.diode_t_list * (-1.j * omega0 * self.diode_dt)
                        #     self.diode_pulse = self.diode_pulse * np.exp(phase)
                            
                        self.diode_pulse_original = np.copy(self.diode_pulse)
                        # for i in range(990, 1010):
                        #     print(f"diode_pulse_original: pulse[{i}] = ({self.diode_pulse_original[i].real}, {self.diode_pulse_original[i].imag}) angle={np.angle(self.diode_pulse_original[i])}") 
                        self.diode_gain = np.full_like(self.diode_t_list, self.start_gain)
                        self.diode_loss = np.full_like(self.diode_t_list, self.start_absorber)
                        shape = self.diode_t_list.shape
                        self.diode_gain_polarization = np.full_like(self.diode_t_list, 0.j, dtype=np.complex128)
                        self.diode_loss_polarization = np.full_like(self.diode_t_list, 0.j, dtype=np.complex128)
                        self.calculation_rounds_done = 0
                        if self.diode_mode == "MB":
                            self.diode_pulse_init = np.copy(self.diode_pulse)
                            self.diode_pulse = np.full_like(self.diode_t_list, 0.0 + 0.0j,dtype=np.complex128)

                    case "recalc":
                        if self.diode_cavity_type == "Ring" and self.diode_update_pulse == "Update Pulse":
                            self.diode_pulse = np.copy(self.diode_pulse_after)
                        if self.diode_mode == "MB" and self.diode_update_pulse != "Update Pulse":
                            self.diode_pulse = np.copy(self.diode_pulse_save)

                if self.diode_cavity_type == "Ring":
                    self.diode_round_trip_old()
                else:
                    self.diode_round_trip_new()

                self.calculation_rounds_done += self.calculation_rounds

    def diode_round_trip_new(self):
        self.oc_val = np.exp(- self.cavity_loss)
        self.diode_pulse_after = np.copy(self.diode_pulse)
        self.diode_pulse_save = np.copy(self.diode_pulse)

        c_pulse = self.diode_pulse.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        c_pulse_init = self.diode_pulse_init.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # for ii in range(990, 1010):
        #     print(f"diode_pulse_init: pulse[{ii}] = ({self.diode_pulse_init[ii].real}, {self.diode_pulse_init[ii].imag})")
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
                            c_pulse_init, c_pulse, c_pulse_after,
                            self.calculation_rounds, self.diode_N, self.loss_shift, self.oc_shift, self.gain_distance,
                            self.diode_dt, self.gain_width, self.Pa, self.Ta, self.Ga, self.N0a, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val,
                            self.rand_factor_seed, self.kappa, self.C_loss, self.C_gain, self.coupling_out_loss, self.coupling_out_gain)
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

    def generate_calc(self):
        tab = 5
        colors = ["#ff0000", "#ff8800", "#aaaa00", "#008800", "#0000ff", "#ff00ff", "#110011"]
        minN = 2E+10
        maxN = 7E+10
        # xGain = [minN, maxN]
        # xLoss = [minN, maxN / 2]
        xGain = np.linspace(4E+10, 9.0E+10, 50).tolist()
        xLoss = np.linspace(0.1E+10, 5.0E+10, 50).tolist()
        yGain = list(map(lambda x : gain_function(self.Ga, x), xGain))
        yLoss = list(map(lambda x : loss_function(self.Gb, self.N0b, x), xLoss))

        xGainRange = np.linspace(cget(np.min(self.diode_gain)) * self.volume, cget(np.max(self.diode_gain)) * self.volume, 10).tolist()
        yGainRange = list(map(lambda x : gain_function(self.Ga, x), xGainRange))
        xLossRange = np.linspace(cget(np.min(self.diode_loss)) * self.volume, cget(np.max(self.diode_loss)) * self.volume, 10).tolist()
        yLossRange = list(map(lambda x : loss_function(self.Gb, self.N0b, x), xLossRange))
        xVec = [xGainRange, xLossRange, xGain, xLoss ]
        yVec = [yGainRange, yLossRange, yGain, yLoss ]
        min_gain = np.min(self.diode_gain) * self.volume
        max_gain = np.max(self.diode_gain) * self.volume
        min_loss = np.min(self.diode_loss) * self.volume# * 0.04 / 0.46
        max_loss = np.max(self.diode_loss) * self.volume# * 0.04 / 0.46
        output_photons = self.summary_photons_after_absorber - self.summary_photons_after_cavity_loss
        energy_of_1064_photon = 1.885E-19 # Joule
        # if len(self.diode_pulse) > 0:
        #     print(np.shape(self.diode_pulse_after))
        #     diode_pulse_fftr = np.fft.rfft(np.sqrt(np.asarray(self.diode_pulse_after)))
        #     diode_pulse_fft = np.concatenate((diode_pulse_fftr[::-1][1:- 1], diode_pulse_fftr))
        pulse = intens(self.diode_pulse)
        pulse_after = intens(self.diode_pulse_after)
        pulse_original = intens(self.diode_pulse_original)

        t_list = cget(shrink_with_max(self.diode_t_list, 1024, self.diode_view_from, self.diode_view_to)).tolist()

        added = Div(
            Div(
                Div(
                    Button("Calculate", hx_post=f'/doCalc/5/diode/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Recalculate", hx_post=f'/doCalc/5/diode/recalc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    InputCalcS(f'DiodeRounds', "Rounds", f'{self.calculation_rounds}', width = 80),
                    Button("View", hx_post=f'/doCalc/5/diode/view', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("ZIN", hx_post=f'/doCalc/5/diode/zoomin', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("ZOUT", hx_post=f'/doCalc/5/diode/zoomout', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("S>", hx_post=f'/doCalc/5/diode/shiftright', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("S<", hx_post=f'/doCalc/5/diode/shiftleft', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 

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
                        SelectCalcS(tab, f'DiodeSelectSampling', "Sampling", ["4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576"], self.diode_sampling, width = 100),
                        SelectCalcS(tab, f'DiodeSelectSamplingX', "Sampling X", ["32", "64", "128", "256"], self.diode_sampling_x, width = 80),
                        SelectCalcS(tab, f'CalcDiodeCavityType', "Cavity Type", ["Ring", "Linear"], self.diode_cavity_type, width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectMode', "mode", ["Intensity", "Amplitude", "MB", "MBGPU"], self.diode_mode, width = 120),
                        InputCalcS(f'DiodePulseWidth', "Pulse width (ps)", f'{self.diode_pulse_width}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], self.diode_intensity, width = 80),
                        InputCalcS(f'DiodeViewFrom', "View from", f'{self.diode_view_from}', width = 80),
                        InputCalcS(f'DiodeViewTo', "View to", f'{self.diode_view_to}', width = 80),
                    ),
                    Div(
                        InputCalcS(f'DiodeAbsorberShift', "Absorber Shift (mm)", f'{self.diode_absorber_shift}', width = 100),
                        InputCalcS(f'DiodeGainShift', "Gain Shift (mm)", f'{self.diode_gain_shift}', width = 100),
                        InputCalcS(f'DiodeOutputCouplerShift', "OC Shift (mm)", f'{self.diode_output_coupler_shift}', width = 100),
                    ),
                    Div(
                        InputCalcS(f'Ta', "Gain Half-life (ps)", f'{self.Ta}', width = 80),
                        InputCalcS(f'gain_width', "Gain Width (THz)", f'{self.gain_width}', width = 80),
                        InputCalcS(f'Pa', "Gain current", f'{self.Pa}', width = 80),
                        InputCalcS(f'Ga', "Gain diff gain (cm^2)", f'{self.Ga}', width = 100),
                        InputCalcS(f'N0a', "Gain N0(tr) (cm^-3)", f'{self.N0a}', width = 100),
                        InputCalcS(f'start_gain', "Gain start val", f'{self.start_gain}', width = 100),

                    ),
                    Div(
                        InputCalcS(f'Tb', "Abs half-life (ps)", f'{self.Tb}', width = 80),
                        InputCalcS(f'Pb', "Abs current", f'{self.Pb}', width = 80),
                        InputCalcS(f'Gb', "Abs diff gain (cm^2)", f'{self.Gb}', width = 100),
                        InputCalcS(f'N0b', "Abs N0(tr) (cm^2)", f'{self.N0b}', width = 100),
                        InputCalcS(f'start_absorber', "Abs start val", f'{self.start_absorber}', width = 100),
                    ),
                    Div(
                        InputCalcS(f'dt', "dt (ps)", f'{format(self.diode_dt * 1E+12, ".4f")}', width = 80),
                        InputCalcS(f'volume', "Volume cm^3", f'{self.volume}', width = 80),
                        InputCalcS(f'initial_photons', "Initial Photns", f'{self.initial_photons}', width = 100),
                        InputCalcS(f'cavity_loss', "OC (Cavity loss)", f'{self.cavity_loss}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeUpdatePulse', "UpdatePulse", ["Update Pulse", "Unchanged Pulse"], self.diode_update_pulse, width = 120),
                        #InputCalcS(f'h', "Abs ratio", f'{self.h}', width = 50),
                    ),

                    Div(
                        InputCalcS(f'rand_factor_seed', "rand seed", f'{self.rand_factor_seed}', width = 60),
                        InputCalcS(f'kappa', "kappa", f'{self.kappa}', width = 100),
                        InputCalcS(f'C_loss', "C_loss", f'{self.C_loss}', width = 100),
                        InputCalcS(f'C_gain', "C_gain", f'{self.C_gain}', width = 100),
                        InputCalcS(f'coupling_out_loss', "coupling_out_loss", f'{self.coupling_out_loss}', width = 120 ),
                        InputCalcS(f'coupling_out_gain', "coupling_out_gain", f'{self.coupling_out_gain}', width = 120)
                    ),
                    FlexN((ABCDMatControl("Left Arm", self.left_arm_mat, self.left_arm_cavity),
                        ABCDMatControl("Right Arm", self.right_arm_mat, self.right_arm_cavity)),
                    ),
                    id="diodeDynamicsOptionsForm"
                                    
                ),

                Div(
                    Table(
                        Tr(Td(f"{self.calculation_rounds_done}"), Td("Value"), Td("Change")), 
                        Tr(Td("Before gain"), Td(f"{self.summary_photons_before:.3e}"), Td("")), 
                        Tr(Td("After gain"), Td(f"{self.summary_photons_after_gain:.3e}"), Td(f"{(self.summary_photons_after_gain - self.summary_photons_before):.3e}")), 
                        Tr(Td("After absorber"), Td(f"{self.summary_photons_after_absorber:.3e}"), Td(f"{(self.summary_photons_after_absorber - self.summary_photons_before):.3e}")),
                        Tr(Td("After OC"), Td(f"{self.summary_photons_after_cavity_loss:.3e}"), Td(f"{(self.summary_photons_after_cavity_loss - self.summary_photons_before):.3e}")),
                        Tr(Td("Output"), Td(f"{output_photons:.3e}"), Td(f"{(output_photons * energy_of_1064_photon * 10E+9):.3e}nJ")),
                        ),
                )
                )),
                style="position:sticky; top:0px; background:#f0f8f8;"
            ),
            Div(
                Div(
                    Frame_chart("fc1", [t_list], 
                                    [cget(shrink_with_max(pulse_original, 1024, self.diode_view_from, self.diode_view_to)).tolist(), 
                                    cget(shrink_with_max(np.log(pulse_after+ 0.000000001), 1024, self.diode_view_from, self.diode_view_to)).tolist()], [""], 
                                    "Original Pulse and Pulse after (photons/sec)", h=2, color=colors, marker=None, twinx=True),

                    generate_chart([cget(self.diode_t_list).tolist(), self.diode_levels_x], 
                                    [cget(pulse).tolist(), self.diode_levels_y], [""], 
                                    "Pulse in (photons/sec)", h=2, color=["red", "black"], marker=None, twinx=True),
                    # Div(
                    #    Div(cls="handle", draggable="true"),
                    #        FlexN([graphCanvas(id="diode_pulse_chart", width=1100, height=300, options=False, mode = 2), 
                    #        ]), cls="container"
                    # ),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(pulse_after, 1024, self.diode_view_from, self.diode_view_to)).tolist()], [""], 
                                    "Pulse out (photons/sec)", h=2, color=colors, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.diode_pulse_after, 1024, self.diode_view_from, self.diode_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.diode_pulse_after, 1024, self.diode_view_from, self.diode_view_to))).tolist()], [""], 
                                    "E", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.diode_accum_pulse, 1024, self.diode_view_from, self.diode_view_to)).tolist(), cget(shrink_with_max(self.diode_accum_pulse_after, 1024, self.diode_view_from, self.diode_view_to)).tolist()], [""], 
                                    f"Accumulate Pulse AND after (photons) [difference: {(self.diode_accum_pulse_after[-1] - self.diode_accum_pulse[-1]):.2e}]", 
                                    h=2, color=colors, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.diode_gain, 1024, self.diode_view_from, self.diode_view_to)).tolist(), cget(shrink_with_max(self.diode_gain_value, 1024, self.diode_view_from, self.diode_view_to )).tolist()], [""], 
                                    f"Gain carriers (1/cm^3) [{(max_gain - min_gain):.2e} = {max_gain:.4e} - {min_gain:.4e}] and Gain (cm^-1)", 
                                    color=["black", "green"], h=2, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.diode_gain_polarization, 1024, self.diode_view_from, self.diode_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.diode_gain_polarization, 1024, self.diode_view_from, self.diode_view_to))).tolist()], [""], 
                                    "Gain Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.diode_loss, 1024, self.diode_view_from, self.diode_view_to)).tolist(), 
                                    cget(shrink_with_max(self.diode_loss_value, 1024, self.diode_view_from, self.diode_view_to)).tolist()], [""], 
                                    f"Abs carrs (cm^-3) [{(max_loss - min_loss):.2e} = {max_loss:.3e} - {min_loss:.3e}] and Loss (cm^-1)", 
                                    color=["black", "red"], h=2, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.diode_loss_polarization, 1024, self.diode_view_from, self.diode_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.diode_loss_polarization, 1024, self.diode_view_from, self.diode_view_to))).tolist()], [""], 
                                    "Loss Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.exp(- self.cavity_loss) * 
                                            np.multiply(shrink_with_max(self.diode_gain_value, 1024, self.diode_view_from, self.diode_view_to),
                                            shrink_with_max(self.diode_loss_value, 1024, self.diode_view_from, self.diode_view_to))).tolist(),
                                    cget(shrink_with_max(pulse, 1024, self.diode_view_from, self.diode_view_to)).tolist()], [""],
                                    "Net gain", color=["blue", "red"], h=2, marker=None, twinx=True),
                    generate_chart(xVec, yVec, [""], "Gain By Pop", h=4, color=["black", "black", "green", "red"], marker=".", lw=[5, 5, 1, 1]),

                    #Div(self.collectDiodeData(), id="numData"),

                    cls="box", style="background-color: #008080;"
                ),
            ) if (len(self.diode_pulse) > 0 and len(self.diode_gain) > 0) else Div(),
        )
        return added