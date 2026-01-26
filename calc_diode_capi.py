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
        ("diode_length", ctypes.c_int),
        ("dt", ctypes.c_double),
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
        ("I1", ctypes.c_double),
        ("left_linear_cavity", ctypes.c_double * 4),
        ("right_linear_cavity", ctypes.c_double * 4),
    ]

mbg_diode_cavity_build = lib_diode.mbg_diode_cavity_build
mbg_diode_cavity_build.argtypes = [ctypes.POINTER(DiodeParams)]
mbg_diode_cavity_build.restype = ctypes.c_void_p

mbg_diode_cavity_prepare = lib_diode.mbg_diode_cavity_prepare
mbg_diode_cavity_prepare.argtypes = [ctypes.POINTER(DiodeParams), ctypes.c_void_p]
mbg_diode_cavity_prepare.restype = ctypes.c_int

mbg_diode_cavity_run = lib_diode.mbg_diode_cavity_run
mbg_diode_cavity_run.argtypes = [ctypes.c_void_p]
mbg_diode_cavity_run.restype = ctypes.c_int

# mbg_diode_cavity_extract = lib_diode.mbg_diode_cavity_extract
# mbg_diode_cavity_extract.argtypes = [ctypes.c_void_p]
# mbg_diode_cavity_extract.restype = ctypes.c_int
