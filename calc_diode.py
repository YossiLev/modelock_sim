# try:
#     import cupy
#     if cupy.cuda.is_available():
#         np = cupy
#         from cupyx.scipy.signal import fftconvolve
#     else:
#         import numpy as np
#         from scipy.signal import fftconvolve
# except ImportError:
import time
import ctypes
import numpy as np
from fasthtml.common import *
from controls import *
from calc_common import *
from calc_diode_capi import *

def gain_function(Ga, N):
    xh1 = Ga * 4468377122.5 * 0.46 * 16.5
    xh2 = Ga * 4468377122.5 * 0.46 * 0.32 * np.exp(0.000000000041*14E+10)
    gGain = xh1 - xh2 * np.exp(-0.000000000041 * N)
    return gGain

def loss_function(Gb, N0b, N):
    gAbs = Gb * 0.02 * (N - N0b)
    return gAbs

class diode_calc(CalcCommonBeam):
    def __init__(self):
        super().__init__()

        self.beam_view_from = -1
        self.beam_view_to = -1

        self.diode_cavity_type = "Linear"
        self.diode_mode = "MBGPU"
        self.beam_sampling = "4096"
        self.beam_sampling_x = "32"
        self.diode_pulse_dtype = np.complex128

        self.beam_time = 3.95138389E-09 #4E-09
        self.beam_N = 4096 # * 4
        self.beam_N_x = 32 # * 4
        self.beam_dt = self.beam_time / self.beam_N
        self.diode_intensity = "Pulse"
        self.calculation_rounds = 1
        self.calculation_rounds_done = 0
        self.diode_pulse_width = 100.0

        self.target_slice_length = 1024

        # actual shift parameters in millimeters
        self.diode_absorber_shift = 0.0
        self.diode_gain_shift = 11.0
        self.diode_output_coupler_shift = 130.0

        self.loss_shift = self.beam_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
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

        self.gain_position = [30, 40, 870, 860]
        self.loss_position = [0, 5, 900, 895]
        self.output_coupler_position = 2000
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
        self.diode_gain_polarization_dir1 = np.array([1], dtype=np.complex128)
        self.diode_gain_polarization_dir2 = np.array([1], dtype=np.complex128)
        self.diode_loss_polarization_dir1 = np.array([1], dtype=np.complex128)
        self.diode_loss_polarization_dir2 = np.array([1], dtype=np.complex128)
        self.diode_gain_value = np.array([1], dtype=np.float64)
        self.diode_loss_value = np.array([1], dtype=np.float64)

        self.ext_beam_in = np.array([1], dtype=np.complex128)
        self.ext_beam_out = np.array([1], dtype=np.complex128)
        self.ext_gain_N = np.array([1], dtype=np.float64)
        self.ext_gain_polarization_dir1 = np.array([1], dtype=np.complex128)
        self.ext_gain_polarization_dir2 = np.array([1], dtype=np.complex128)
        self.ext_loss_N = np.array([1], dtype=np.float64)
        self.ext_loss_polarization_dir1 = np.array([1], dtype=np.complex128)
        self.ext_loss_polarization_dir2 = np.array([1], dtype=np.complex128)
        # diode summary parameters
        self.summary_photons_before = 0.0
        self.summary_photons_after_gain = 0.0
        self.summary_photons_after_absorber = 0.0
        self.summary_photons_after_cavity_loss = 0.0

        self.gpu_memory = None
        self.gpu_memory_exists = False

    def mm_to_unit_shift(self, mm):
        shift = int(mm / 1E+03 / (self.beam_dt * 3E+08))
        return shift
    
    def calcDiodeLocations(self):
        gainLengthMm = 4.0
        lossLengthMm = 1.0
        # self.beam_time = 3.95138389E-09 #4E-09
        # self.loss_shift = self.beam_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
        # self.gain_distance = self.mm_to_unit_shift(self.diode_gain_shift)
        # self.oc_shift = self.mm_to_unit_shift(self.diode_output_coupler_shift)

        abs_length = self.mm_to_unit_shift(lossLengthMm)
        abs_shift = self.beam_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift)
        self.loss_position = [0, abs_length, abs_shift - abs_length, abs_shift]
        
        gain_length = self.mm_to_unit_shift(gainLengthMm)
        gain_shift = self.mm_to_unit_shift(self.diode_gain_shift)

        self.gain_position = [abs_length + gain_shift, abs_length + gain_shift + gain_length, 
                              abs_shift - gain_shift, abs_shift - gain_shift - gain_length] 
        self.output_coupler_position = self.mm_to_unit_shift(self.diode_output_coupler_shift)

        print(f"absorber" , self.loss_position[0], self.loss_position[1], self.loss_position[2], self.loss_position[3])
        print(f"Gain    ", self.gain_position[0], self.gain_position[1], self.gain_position[2], self.gain_position[3])
        print(f"OC ", self.output_coupler_position)
        print("Length of diode ", self.loss_position[1] - self.loss_position[0] + self.gain_position[1] - self.gain_position[0] + 1)

    def pack_diode_params(self):
        params = DiodeParams()
        params.n_cavity_bits = int(np.log2(self.beam_N))
        params.n_x_bits = int(np.log2(self.beam_N))
        params.n_rounds = self.calculation_rounds
        params.target_slice_length = self.target_slice_length
        params.target_slice_start = 0
        params.target_slice_end = self.beam_N
        params.start_round = self.calculation_rounds_done
        params.N = self.beam_N
        params.N_x = self.beam_N_x
        params.beam_init_type = ["Pulse", "Noise", "CW", "Flat"].index(self.diode_intensity)
        params.beam_init_parameter = self.diode_pulse_width
        params.diode_length = 2
        params.gain_position = (ctypes.c_int * 4)(*[self.gain_position[0], self.gain_position[1], self.gain_position[2], self.gain_position[3]])
        params.loss_position = (ctypes.c_int * 4)(*[self.loss_position[0], self.loss_position[1], self.loss_position[2], self.loss_position[3]])
        params.output_coupler_position = self.output_coupler_position
        params.dt = self.beam_dt
        params.tGain = self.Ta * 1E-12
        params.tLoss = self.Tb * 1E-12
        params.C_gain = self.C_gain
        params.C_loss = self.C_loss
        params.N0b = self.N0b
        params.Pa = self.Pa
        params.kappa = self.kappa
        params.alpha = self.h
        params.one_minus_alpha_div_a = (1.0 - self.h) / self.Ga
        params.noise_val = 1.0E-26 * self.rand_factor_seed
        params.coupling_out_gain = self.coupling_out_gain
        params.oc_val = np.exp(- self.cavity_loss)
        params.left_linear_cavity = (ctypes.c_double * 4)(*[self.left_arm_mat[0][0], self.left_arm_mat[0][1],
                                                            self.left_arm_mat[1][0], self.left_arm_mat[1][1]])
        params.right_linear_cavity = (ctypes.c_double * 4)(*[self.right_arm_mat[0][0], self.right_arm_mat[0][1],
                                                             self.right_arm_mat[1][0], self.right_arm_mat[1][1]])
        params.ext_len = self.ext_beam_in.shape[0]  
        params.ext_beam_in = self.ext_beam_in.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        params.ext_beam_out = self.ext_beam_out.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        params.ext_gain_N = self.ext_gain_N.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        params.ext_gain_polarization_dir1 = self.ext_gain_polarization_dir1.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        params.ext_gain_polarization_dir2 = self.ext_gain_polarization_dir2.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        params.ext_loss_N = self.ext_loss_N.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        params.ext_loss_polarization_dir1 = self.ext_loss_polarization_dir1.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        params.ext_loss_polarization_dir2 = self.ext_loss_polarization_dir2.ctypes.data_as(ctypes.POINTER(cuDoubleComplex))
        
        return params

    def doCalcCommand(self, params):
        if super().doCalcCommand(params) == 1:
            return 1
        # Conditions for self-sustained pulsation and bistability in semiconductor lasers - Masayasu Ueno and Roy Lang
        # d Na / dt = - Na / Ta - Ga(Na - N0a) * N + Pa
        # d Nb / dt = - Nb / Tb - Gb(Nb - N0b) * N + Pb
        # d N  / dt = [(1 - h) * Ga(Na - N0a) + h * Gb(Nb - N0b) - GAMMA] * N

        match params:
            case "calc":

                self.beam_N = int(self.beam_sampling)
                self.beam_N_x = int(self.beam_sampling_x)
                self.beam_dt = self.beam_time / self.beam_N
                self.loss_shift = self.beam_N // 2 + self.mm_to_unit_shift(self.diode_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
                self.gain_distance = self.mm_to_unit_shift(self.diode_gain_shift)
                self.oc_shift = self.mm_to_unit_shift(self.diode_output_coupler_shift)
                print(f"beam_N={self.beam_N} dt={self.beam_dt} loss_shift={self.loss_shift} gain_distance={self.gain_distance} oc_shift={self.oc_shift}")

                gg = 20
                ll = 0.4
                oo = np.exp(- self.cavity_loss)
                self.diode_levels_x = [0, self.gain_distance - 1, 
                                        self.gain_distance, self.oc_shift - 1, 
                                        self.oc_shift, self.loss_shift - self.gain_distance - 1, 
                                        self.loss_shift - self.gain_distance, self.loss_shift - 1, 
                                        self.loss_shift, self.beam_N]
                self.diode_levels_y = [1, 1, 1 / gg, 1 / gg, 
                                        1 / gg / oo, 1 / gg / oo, 1 / gg / oo / gg, 1 / gg / oo / gg, 
                                        1 / gg / oo / gg / ll, 1 / gg / oo / gg / ll]

                self.diode_t_list = np.arange(self.beam_N, dtype=np.float64)
                self.diode_gain_value = np.full_like(self.diode_t_list, 0.0)
                self.diode_loss_value = np.full_like(self.diode_t_list, 0.0)

                self.diode_pulse_dtype = np.complex128 if self.diode_mode != "Intensity" else np.float64 
                
                self.diode_pulse = np.array([], dtype=self.diode_pulse_dtype)
                self.diode_pulse_original = np.array([], dtype=self.diode_pulse_dtype)
                self.diode_pulse_after = np.array([], dtype=self.diode_pulse_dtype)
            
                pulseVal = np.array([60000 / self.beam_dt / self.volume], dtype=self.diode_pulse_dtype)
                match self.diode_intensity:
                    case "Pulse":
                        w2 = (self.diode_pulse_width * 1.0E-12 /self.beam_dt * 1.41421356237) if self.diode_pulse_dtype == np.complex128 else self.diode_pulse_width 
                        self.diode_pulse = pulseVal * np.exp(-np.square(self.diode_t_list - self.beam_N / 2) / (2 * w2 * w2))
                        self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                        self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                        if self.diode_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                    case "Noise":
                        self.diode_pulse = np.random.random(self.diode_t_list.shape).astype(self.diode_pulse_dtype)
                        self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                        self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                        if self.diode_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                    case "CW":
                        self.diode_pulse = np.full(self.beam_N, 1.0, dtype=self.diode_pulse_dtype)
                        self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.diode_accum_pulse[-1]
                        self.diode_accum_pulse = np.multiply(self.diode_accum_pulse, pulse_ratio)
                        if self.diode_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.diode_pulse = np.multiply(self.diode_pulse, pulse_ratio)
                    case "Flat":
                        self.diode_pulse = np.full_like(self.diode_t_list, 0).astype(self.diode_pulse_dtype)
                        self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.beam_dt * self.volume
                        
                # if self.diode_pulse_dtype == np.complex128:
                #     lambda_ = 1064E-09
                #     omega0 = 2.0 * np.pi * 3E+08 / lambda_
                #     phase = self.diode_t_list * (-1.j * omega0 * self.beam_dt)
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

                if (self.diode_mode == "MBGPU"):
                    self.calcDiodeLocations()

                    self.ext_beam_in = np.empty((self.target_slice_length,), dtype=np.complex128)
                    self.ext_beam_out = np.empty((self.target_slice_length,), dtype=np.complex128)
                    self.ext_gain_N = np.empty((self.target_slice_length,), dtype=np.float64)
                    self.ext_gain_polarization_dir1 = np.empty((self.target_slice_length,), dtype=np.complex128)
                    self.ext_gain_polarization_dir2 = np.empty((self.target_slice_length,), dtype=np.complex128)
                    self.ext_loss_N = np.empty((self.target_slice_length,), dtype=np.float64)
                    self.ext_loss_polarization_dir1 = np.empty((self.target_slice_length,), dtype=np.complex128)
                    self.ext_loss_polarization_dir2 = np.empty((self.target_slice_length,), dtype=np.complex128)
                    if self.gpu_memory_exists:
                        mbg_diode_cavity_destroy(self.gpu_memory)
                    self.gpu_memory = mbg_diode_cavity_build(self.pack_diode_params())
                    self.gpu_memory_exists = True

                self.keep_current_run_parameters([self.beam_sampling, self.beam_sampling_x, self.diode_mode])

            case "recalc":
                if not self.verify_current_run_parameters([self.beam_sampling, self.beam_sampling_x, self.diode_mode]):
                    print("Parameters changed, recalculation not possible")
                    return 1
                if self.diode_cavity_type == "Ring" and self.diode_update_pulse == "Update Pulse":
                    self.diode_pulse = np.copy(self.diode_pulse_after)
                if self.diode_mode == "MB" and self.diode_update_pulse != "Update Pulse":
                    self.diode_pulse = np.copy(self.diode_pulse_save)

        if self.diode_cavity_type == "Ring":
            self.diode_round_trip_old()
        elif self.diode_mode == "MBGPU":
            self.diode_round_trip_wide()
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
                            self.calculation_rounds, self.beam_N, self.loss_shift, self.oc_shift, self.gain_distance,
                            self.beam_dt, self.gain_width, self.Pa, self.Ta, self.Ga, self.N0a, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val,
                            self.rand_factor_seed, self.kappa, self.C_loss, self.C_gain, self.coupling_out_loss, self.coupling_out_gain)
            print("MB round trip done")
            k = 0
            for i in range(self.diode_pulse_after.shape[0]):
                if np.isnan(self.diode_pulse_after[i].real) or np.isnan(self.diode_pulse_after[i].imag):
                    k = k + 1
                    print(f"NAN index {i}: val=({self.diode_pulse_after[i].real}, {self.diode_pulse_after[i].imag})\n")
                if k > 100:
                    break
            self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

            print("MB accumulation done")
            return
        
        round_trip_func = lib_diode.cmp_diode_round_trip if self.diode_pulse_dtype == np.complex128 else lib_diode.diode_round_trip

        round_trip_func(c_gain, c_loss, c_gain_value, c_loss_value,
                        c_pulse, c_pulse_after,
                        self.calculation_rounds, self.beam_N, self.loss_shift, self.oc_shift, self.gain_distance,
                        self.beam_dt, self.gain_width, self.Pa, self.Ta, self.Ga, self.Pb, self.Tb, self.Gb, self.N0b, self.oc_val)

        self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

    def diode_round_trip_wide(self):

        print("diode_round_trip_wide GPU calculation started")
        mbg_diode_cavity_prepare(ctypes.byref(self.pack_diode_params()), self.gpu_memory)
        print("diode_round_trip_wide: prepared GPU memory")

        start_time = time.perf_counter()

        mbg_diode_cavity_run(self.gpu_memory)

        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds. ({(1000.0 * elapsed_time / self.calculation_rounds):.1f}ms per round)")

        print("diode_round_trip_wide: GPU run done")
        mbg_diode_cavity_extract(self.gpu_memory)
        print("diode_round_trip_wide GPU calculation done")

        return

    def diode_round_trip_old(self):

        self.diode_pulse_after = np.copy(self.diode_pulse)

        for i in range(self.calculation_rounds):

            if self.diode_update_pulse == "Update Pulse":
                self.diode_pulse = np.copy(self.diode_pulse_after)

            self.diode_accum_pulse = np.add.accumulate(intens(self.diode_pulse)) * self.beam_dt * self.volume
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
                                self.beam_N, self.beam_dt, self.Pa, self.Ta, self.Ga, self.gain_factor)

            self.summary_photons_after_gain = np.sum(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

            # absorber calculations
            loss_func = lib_diode.cmp_diode_loss if self.diode_pulse_dtype == np.complex128 else lib_diode.diode_loss
            loss_func(c_loss, c_loss_value, c_pulse_after,
                                self.beam_N, self.beam_dt, self.Pb, self.Tb, self.Gb, self.N0b)

            self.summary_photons_after_absorber = np.sum(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

            #cavity loss
            cavity_loss = self.cavity_loss if self.diode_pulse_dtype == np.float64 else self.cavity_loss / 2.0
            self.diode_pulse_after *= np.exp(- cavity_loss)
            self.summary_photons_after_cavity_loss = np.sum(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

            self.diode_accum_pulse_after = np.add.accumulate(intens(self.diode_pulse_after)) * self.beam_dt * self.volume

    def generate_charts_mbgpu(self):
        t_list = self.shrink_list(self.diode_t_list)

        return Div(
            generate_chart_complex_log(t_list, self.ext_beam_in, "Amplitude in"),
            generate_chart_complex(t_list, self.ext_beam_out, "Amplitude out"),
            generate_chart([t_list], [cget(self.ext_gain_N).tolist()], "Gain carriers (1/cm^3)"),
            generate_chart_complex(t_list, cget(self.ext_gain_polarization_dir1).tolist(), "Gain Polarization 1"),
            generate_chart_complex(t_list, cget(self.ext_gain_polarization_dir2).tolist(), "Gain Polarization 2"),
            generate_chart([t_list], [cget(self.ext_loss_N).tolist()], "Abs carriers (1/cm^3)"),
            generate_chart_complex(t_list, cget(self.ext_loss_polarization_dir1).tolist(), "Loss Polarization 1"),
            generate_chart_complex(t_list, cget(self.ext_loss_polarization_dir2).tolist(), "Loss Polarization 2"),
            cls="box", style="background-color: #008080;"
        )
    
    def generate_charts_default(self):
        if self.diode_mode == "MBGPU":
            return self.generate_charts_mbgpu()

        # xGain = np.linspace(4E+10, 9.0E+10, 50).tolist()
        # xLoss = np.linspace(0.1E+10, 5.0E+10, 50).tolist()
        # yGain = list(map(lambda x : gain_function(self.Ga, x), xGain))
        # yLoss = list(map(lambda x : loss_function(self.Gb, self.N0b, x), xLoss))

        # xGainRange = np.linspace(cget(np.min(self.diode_gain)) * self.volume, cget(np.max(self.diode_gain)) * self.volume, 10).tolist()
        # yGainRange = list(map(lambda x : gain_function(self.Ga, x), xGainRange))
        # xLossRange = np.linspace(cget(np.min(self.diode_loss)) * self.volume, cget(np.max(self.diode_loss)) * self.volume, 10).tolist()
        # yLossRange = list(map(lambda x : loss_function(self.Gb, self.N0b, x), xLossRange))
        # xVec = [xGainRange, xLossRange, xGain, xLoss ]
        # yVec = [yGainRange, yLossRange, yGain, yLoss ]
        min_gain = np.min(self.diode_gain) * self.volume
        max_gain = np.max(self.diode_gain) * self.volume
        min_loss = np.min(self.diode_loss) * self.volume# * 0.04 / 0.46
        max_loss = np.max(self.diode_loss) * self.volume# * 0.04 / 0.46

        pulse = intens(self.diode_pulse)
        pulse_after = intens(self.diode_pulse_after)
        pulse_original = intens(self.diode_pulse_original)

        t_list = self.shrink_list(self.diode_t_list)

        return Div(
            Frame_chart("fc1", [t_list], self.shrink_lists([pulse_original, ]), 
                            "Original Pulse and Pulse after (photons/sec)", twinx=True),

            generate_chart([cget(self.diode_t_list).tolist(), self.diode_levels_x], [cget(pulse).tolist(), self.diode_levels_y],  
                            "Pulse in (photons/sec)", color=["red", "black"], twinx=True),

            generate_chart([t_list], [self.shrink_list(pulse_after)], "Pulse out (photons/sec)", twinx=True),
            generate_chart_complex(t_list, self.shrink_def(self.diode_pulse_after), "E"),
            generate_chart([t_list], self.shrink_lists([self.diode_accum_pulse, self.diode_accum_pulse_after]), 
                            f"Accumulate Pulse AND after (photons) [difference: {(self.diode_accum_pulse_after[-1] - self.diode_accum_pulse[-1]):.2e}]", twinx=True),
            generate_chart([t_list], self.shrink_lists([self.diode_gain, self.diode_gain_value]), 
                            f"Gain carriers (1/cm^3) [{(max_gain - min_gain):.2e} = {max_gain:.4e} - {min_gain:.4e}] and Gain (cm^-1)", 
                            color=["black", "green"], twinx=True),
            generate_chart_complex(t_list, self.shrink_def(self.diode_gain_polarization), "Gain Polarization"),
            generate_chart([t_list], self.shrink_lists([self.diode_loss, self.diode_loss_value]), 
                            f"Abs carrs (cm^-3) [{(max_loss - min_loss):.2e} = {max_loss:.3e} - {min_loss:.3e}] and Loss (cm^-1)", color=["black", "red"], twinx=True),
            generate_chart_complex(t_list, self.shrink_def(self.diode_loss_polarization), "Loss Polarization"),
            generate_chart([t_list], [cget(np.exp(- self.cavity_loss) * 
                            np.multiply(self.shrink_def(self.diode_gain_value), self.shrink_def(self.diode_loss_value))).tolist(),
                            self.shrink_list(pulse)], "Net gain", color=["blue", "red"], twinx=True),
            #generate_chart(xVec, yVec, "Gain By Pop", h=4, color=["black", "black", "green", "red"], marker=".", lw=[5, 5, 1, 1]),

            # experimental new type of grpah manage by JS
            # Div(
            #    Div(cls="handle", draggable="true"),
            #        FlexN([graphCanvas(id="diode_pulse_chart", width=1100, height=300, options=False, mode = 2), 
            #        ]), cls="container"
            # ),
            # Div(self.collectCommonData(), id="numData"),

            cls="box", style="background-color: #008080;"
        )

    def generate_calc(self):
        tab = 5

        output_photons = self.summary_photons_after_absorber - self.summary_photons_after_cavity_loss
        energy_of_1064_photon = 1.885E-19 # Joule

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
                    Button("><", hx_post=f'/doCalc/5/diode/center', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 


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
                        SelectCalcS(tab, f'DiodeSelectSampling', "Sampling", ["4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576"], self.beam_sampling, width = 100),
                        SelectCalcS(tab, f'DiodeSelectSamplingX', "Sampling X", ["32", "64", "128", "256"], self.beam_sampling_x, width = 80),
                        SelectCalcS(tab, f'CalcDiodeCavityType', "Cavity Type", ["Ring", "Linear"], self.diode_cavity_type, width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectMode', "mode", ["Intensity", "Amplitude", "MB", "MBGPU"], self.diode_mode, width = 120),
                        InputCalcS(f'DiodePulseWidth', "Pulse width (ps)", f'{self.diode_pulse_width}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], self.diode_intensity, width = 80),
                        InputCalcS(f'DiodeViewFrom', "View from", f'{self.beam_view_from}', width = 80),
                        InputCalcS(f'DiodeViewTo', "View to", f'{self.beam_view_to}', width = 80),
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
                        InputCalcS(f'dt', "dt (ps)", f'{format(self.beam_dt * 1E+12, ".4f")}', width = 80),
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
            Div(self.generate_charts_default()
            ) if (len(self.diode_pulse) > 0 and len(self.diode_gain) > 0) else Div(),
        )
        return added