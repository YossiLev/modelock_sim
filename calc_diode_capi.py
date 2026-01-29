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
import os
import platform
from cffi import FFI

ffi = FFI()

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


# ---------------------------

mbg_diode_cavity_destroy = lib_diode.mbg_diode_cavity_destroy
mbg_diode_cavity_destroy.argtypes = [ctypes.c_void_p]
mbg_diode_cavity_destroy.restype = None

class cuDoubleComplex(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]

class DiodeParams(ctypes.Structure):
    _fields_ = [
        ("n_cavity_bits", ctypes.c_int),
        ("n_x_bits", ctypes.c_int),
        ("n_rounds", ctypes.c_int),
        ("target_slice_length", ctypes.c_int),
        ("target_slice_start", ctypes.c_int),
        ("target_slice_end", ctypes.c_int),

        ("N", ctypes.c_int),
        ("N_x", ctypes.c_int),
        ("diode_length", ctypes.c_int), # number of locations in the diode (including positions for gain, loss and output coupler
        ("gain_position", ctypes.c_double * 4), # ranges on beam 1 (ltr) and beam 2 (rtl) of the positions of the gain part of the diode
        ("loss_position", ctypes.c_double * 4), # ranges on beam 1 (ltr) and beam 2 (rtl) of the positions of the loss part of the diode
        ("output_coupler_position", ctypes.c_double), #single position on beam for the output coupler

        ("dt", ctypes.c_double),

        ("beam_init_type", ctypes.c_int), # time of beam in the cavity at t=0
        ("beam_init_parameter", ctypes.c_double), #tuning parameter for the initial beam (wifth of pulse, etc)

        ("tGain", ctypes.c_double),
        ("tLoss", ctypes.c_double),
        ("C_gain", ctypes.c_double),
        ("C_loss", ctypes.c_double),
        ("N0b", ctypes.c_double),
        ("Pa", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("alpha", ctypes.c_double), 
        ("one_minus_alpha_div_a", ctypes.c_double),
        ("coupling_out_gain", ctypes.c_double),

        ("left_linear_cavity", ctypes.c_double * 4), # left cavity ABCD parameters
        ("right_linear_cavity", ctypes.c_double * 4), # right cavity ABCD parameters

        ("ext_len", ctypes.c_int),
        ("ext_beam_in", ctypes.POINTER(cuDoubleComplex)), # the beam amlitude inside the cavity
        ("ext_beam_out", ctypes.POINTER(cuDoubleComplex)), # the beam amplitude as it comes out of the cavity
    ]

print("Python ctypes.sizeof(DiodeParams)", ctypes.sizeof(DiodeParams))

mbg_diode_cavity_build = lib_diode.mbg_diode_cavity_build
mbg_diode_cavity_build.argtypes = [ctypes.POINTER(DiodeParams)]
mbg_diode_cavity_build.restype = ctypes.c_void_p

mbg_diode_cavity_prepare = lib_diode.mbg_diode_cavity_prepare
mbg_diode_cavity_prepare.argtypes = [ctypes.POINTER(DiodeParams), ctypes.c_void_p]
mbg_diode_cavity_prepare.restype = ctypes.c_int

mbg_diode_cavity_run = lib_diode.mbg_diode_cavity_run
mbg_diode_cavity_run.argtypes = [ctypes.c_void_p]
mbg_diode_cavity_run.restype = ctypes.c_int

mbg_diode_cavity_extract = lib_diode.mbg_diode_cavity_extract
mbg_diode_cavity_extract.argtypes = [ctypes.c_void_p]
mbg_diode_cavity_extract.restype = ctypes.c_int
