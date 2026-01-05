import ctypes
import platform
import json
import numpy as np
from fasthtml.common import *
from controls import *
from calc_common import *

class dispersion_calc(CalcCommonBeam):
    def __init__(self):
        super().__init__()

        self.beam_view_from = -1
        self.beam_view_to = -1

        self.dispersion_mode = "Amplitude"
        self.beam_sampling = "4096"
        self.dispersion_pulse_dtype = np.complex128

        self.beam_time = 3.95138389E-09 #4E-09
        self.beam_N = 4096 # * 4
        self.beam_dt = self.beam_time / self.beam_N
        self.dispersion_intensity = "Pulse"
        self.calculation_rounds = 1
        self.calculation_rounds_done = 0
        self.dispersion_pulse_width = 100.0

        # actual shift parameters in millimeters
        self.dispersion_absorber_shift = 0.0
        self.dispersion_gain_shift = 11.0
        self.dispersion_output_coupler_shift = 130.0


        self.initial_photons = 1E+07

        self.volume = 1 #0.46 * 0.03 * 2E-05
        self.cavity_loss = 4.5
        self.dispersion_update_pulse = "Update Pulse"
        self.h = 0.1

        # diode dynamics parameters
        self.dispersion_t_list = np.array([1], dtype=np.float64)
        self.dispersion_pulse = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_calculated_fft = np.array([], dtype=np.float64)
        self.dispersion_calculated_fd = np.array([], dtype=np.float64)
        self.dispersion_calculated_fd_s5 = np.array([], dtype=np.float64)
        self.dispersion_calculated_fd2 = np.array([], dtype=np.float64)

    def mm_to_unit_shift(self, mm):
        shift = int(mm / 1E+03 / (self.beam_dt * 3E+08))
        print(f"mm_to_unit_shift mm={mm} shift={shift}")
        return shift
    
    def doCalcCommand(self, params):
        if super().doCalcCommand(params) == 1:
            return 1
        
        match params:

            case "calc":


                self.beam_N = int(self.beam_sampling)
                self.beam_dt = self.beam_time / self.beam_N
                self.loss_shift = self.beam_N // 2 + self.mm_to_unit_shift(self.dispersion_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
                self.gain_distance = self.mm_to_unit_shift(self.dispersion_gain_shift)
                self.oc_shift = self.mm_to_unit_shift(self.dispersion_output_coupler_shift)
                print(f"beam_N={self.beam_N} dt={self.beam_dt} loss_shift={self.loss_shift} gain_distance={self.gain_distance} oc_shift={self.oc_shift}")

                self.dispersion_t_list = np.arange(self.beam_N, dtype=np.float64)
                print(f"Generated time list of size {self.dispersion_t_list.size} dt={self.beam_N}")
                self.dispersion_pulse_dtype = np.complex128 if self.dispersion_mode != "Intensity" else np.float64 
                self.dispersion_pulse = np.array([], dtype=self.dispersion_pulse_dtype)
                self.dispersion_calculated_fd = np.array([], dtype=self.dispersion_pulse_dtype)
                self.dispersion_calculated_fd_s5 = np.array([], dtype=self.dispersion_pulse_dtype)
                self.dispersion_calculated_fd2 = np.array([], dtype=self.dispersion_pulse_dtype)
            
                pulseVal = np.array([60000 / self.beam_dt / self.volume], dtype=self.dispersion_pulse_dtype)
                match self.dispersion_intensity:
                    case "Pulse":
                        print("Generating Pulse", self.dispersion_intensity)

                        w2 = (self.dispersion_pulse_width * 1.0E-12 /self.beam_dt * 1.41421356237) if self.dispersion_pulse_dtype == np.complex128 else self.dispersion_pulse_width 
                        print(w2)
                        self.dispersion_pulse = pulseVal * np.exp(-np.square(self.dispersion_t_list - self.beam_N / 2) / (2 * w2 * w2))
                        self.dispersion_accum_pulse = np.add.accumulate(intens(self.dispersion_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.dispersion_accum_pulse[-1]
                        self.dispersion_accum_pulse = np.multiply(self.dispersion_accum_pulse, pulse_ratio)
                        if self.dispersion_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.dispersion_pulse = np.multiply(self.dispersion_pulse, pulse_ratio)
                    case "Noise":
                        self.dispersion_pulse = np.random.random(self.dispersion_t_list.shape).astype(self.dispersion_pulse_dtype)
                        self.dispersion_accum_pulse = np.add.accumulate(intens(self.dispersion_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.dispersion_accum_pulse[-1]
                        self.dispersion_accum_pulse = np.multiply(self.dispersion_accum_pulse, pulse_ratio)
                        if self.dispersion_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.dispersion_pulse = np.multiply(self.dispersion_pulse, pulse_ratio)
                    case "CW":
                        self.dispersion_pulse = np.full(self.beam_N, 1.0, dtype=self.dispersion_pulse_dtype)
                        self.dispersion_accum_pulse = np.add.accumulate(intens(self.dispersion_pulse)) * self.beam_dt * self.volume
                        pulse_ratio = self.initial_photons / self.dispersion_accum_pulse[-1]
                        self.dispersion_accum_pulse = np.multiply(self.dispersion_accum_pulse, pulse_ratio)
                        if self.dispersion_pulse_dtype == np.complex128:
                            pulse_ratio = np.sqrt(pulse_ratio)
                        self.dispersion_pulse = np.multiply(self.dispersion_pulse, pulse_ratio)
                    case "Flat":
                        self.dispersion_pulse = np.full_like(self.dispersion_t_list, 0).astype(self.dispersion_pulse_dtype)
                        self.dispersion_accum_pulse = np.add.accumulate(intens(self.dispersion_pulse)) * self.beam_dt * self.volume
                        
                # if self.dispersion_pulse_dtype == np.complex128:
                #     lambda_ = 1064E-09
                #     omega0 = 2.0 * np.pi * 3E+08 / lambda_
                #     phase = self.dispersion_t_list * (-1.j * omega0 * self.beam_dt)
                #     self.dispersion_pulse = self.dispersion_pulse * np.exp(phase)
                    
                # for i in range(990, 1010):
                shape = self.dispersion_t_list.shape
                self.calculation_rounds_done = 0

                beta2 = 6E-23  # s^2/m
                dz = 5.0E-3    # m
                dt = self.beam_dt
                self.dispersion_calculated_fft = apply_dispersion(self.dispersion_pulse, dt, L=dz, beta2=beta2, beta3=0.0, beta4=0.0)
                df_diode_cells = np.floor(0.004 / (dt * 3E+08 / 3)).astype(np.int32)
                alpha = 1j * beta2 * (dz / df_diode_cells) / (2 * dt**2)
                self.dispersion_calculated_fd = np.copy(self.dispersion_pulse)
                self.dispersion_calculated_fd2 = np.copy(self.dispersion_pulse)
                p = self.beam_N // 2 - df_diode_cells - 500
                print(f"FD dispersion with alpha={alpha} p={p} df_diode_cells={df_diode_cells}, beam_n={self.beam_N} alpha={alpha}")
                for start in range(0, df_diode_cells + 1000, 2):
                    end = start + df_diode_cells
                    #print(f"FD dispersion step from {p + start} to {p + end} with alpha={alpha} p={p} df_diode_cells={df_diode_cells}, beam_n={self.beam_N}")
                    dispersion_fd_step(self.dispersion_calculated_fd, self.dispersion_calculated_fd2, p + start, p + end, alpha, 3)
                    dispersion_fd_step(self.dispersion_calculated_fd2, self.dispersion_calculated_fd, p + start + 1, p + end + 1, alpha, 3)

                self.dispersion_calculated_fd_s5 = np.copy(self.dispersion_pulse)
                self.dispersion_calculated_fd2 = np.copy(self.dispersion_pulse)
                for start in range(0, df_diode_cells + 1000, 2):
                    end = start + df_diode_cells
                    #print(f"FD dispersion step from {p + start} to {p + end} with alpha={alpha} p={p} df_diode_cells={df_diode_cells}, beam_n={self.beam_N}")
                    dispersion_fd_step(self.dispersion_calculated_fd_s5, self.dispersion_calculated_fd2, p + start, p + end, alpha, 5)
                    dispersion_fd_step(self.dispersion_calculated_fd2, self.dispersion_calculated_fd_s5, p + start + 1, p + end + 1, alpha, 5)

                #self.dispersion_calculated_cn = propagate_dispersion_CN(self.dispersion_pulse, self.beam_dt, 6E-23, 1.0E-2/200, 200)


    def generate_calc(self):
        tab = 6
        colors = ["#ff0000", "#ff8800", "#aaaa00", "#008800", "#0000ff", "#ff00ff", "#110011"]

        pulse = intens(self.dispersion_pulse)
        t_list = cget(shrink_with_max(self.dispersion_t_list, 1024, self.beam_view_from, self.beam_view_to)).tolist()

        print(self.beam_view_from, self.beam_view_to, len(self.dispersion_t_list), len(t_list))

        added = Div(
            Div(
                Div(
                    Button("Calculate", hx_post=f'/doCalc/6/dispersion/calc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("Recalculate", hx_post=f'/doCalc/6/dispersion/recalc', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    InputCalcS(f'DiodeRounds', "Rounds", f'{self.calculation_rounds}', width = 80),
                    Button("View", hx_post=f'/doCalc/6/dispersion/view', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("ZIN", hx_post=f'/doCalc/6/dispersion/zoomin', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("ZOUT", hx_post=f'/doCalc/6/dispersion/zoomout', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("S>", hx_post=f'/doCalc/6/dispersion/shiftright', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("S<", hx_post=f'/doCalc/6/dispersion/shiftleft', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 
                    Button("><", hx_post=f'/doCalc/6/dispersion/center', hx_include="#calcForm *", hx_target="#gen_calc", hx_vals='js:{localId: getLocalId()}'), 

                    Div(
                        Button("Save Parameters", onclick="saveMultiTimeParametersProcess()"),
                        Button("Restore Parameters", onclick="restoreMultiTimeParametersProcess('dispersionDynamicsOptionsForm')"),
                        Div(Div("Give a name to saved parameters"),
                            Div(Input(type="text", id=f'parametersName', placeholder="Descibe", style="width:450px;", value="")),
                            Button("Save", onclick="saveMultiTimeParameters(1, 'dispersionDynamicsOptionsForm')"),
                            Button("Cancel", onclick="saveMultiTimeParameters(0, 'dispersionDynamicsOptionsForm')"),
                            id="saveParametersDialog", cls="pophelp", style="position: absolute; visibility: hidden"),
                        Div(Div("Select the parameters set"),
                            Div("", id="restoreParametersList"),
                            Div("", id="copyParametersList"),
                            Button("Cancel", onclick="restoreMultiTimeParameters(-1, 'dispersionDynamicsOptionsForm')"),
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
                        SelectCalcS(tab, f'DispersionSelectSampling', "Sampling", ["4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576"], self.beam_sampling, width = 100),
                        SelectCalcS(tab, f'CalcDispersionSelectMode', "mode", ["Intensity", "Amplitude", "MB", "MBGPU"], self.dispersion_mode, width = 120),
                        InputCalcS(f'DispersionPulseWidth', "Pulse width (ps)", f'{self.dispersion_pulse_width}', width = 80),
                        SelectCalcS(tab, f'CalcDispersionSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], self.dispersion_intensity, width = 80),
                        InputCalcS(f'DispersionViewFrom', "View from", f'{self.beam_view_from}', width = 80),
                        InputCalcS(f'DispersionViewTo', "View to", f'{self.beam_view_to}', width = 80),
                    ),
                    Div(
                        InputCalcS(f'DiodeAbsorberShift', "Absorber Shift (mm)", f'{self.dispersion_absorber_shift}', width = 100),
                        InputCalcS(f'DiodeGainShift', "Gain Shift (mm)", f'{self.dispersion_gain_shift}', width = 100),
                        InputCalcS(f'DiodeOutputCouplerShift', "OC Shift (mm)", f'{self.dispersion_output_coupler_shift}', width = 100),
                    ),

                    Div(
                        InputCalcS(f'dt', "dt (ps)", f'{format(self.beam_dt * 1E+12, ".4f")}', width = 80),
                        InputCalcS(f'volume', "Volume cm^3", f'{self.volume}', width = 80),
                        InputCalcS(f'initial_photons', "Initial Photns", f'{self.initial_photons}', width = 100),
                        InputCalcS(f'cavity_loss', "OC (Cavity loss)", f'{self.cavity_loss}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeUpdatePulse', "UpdatePulse", ["Update Pulse", "Unchanged Pulse"], self.dispersion_update_pulse, width = 120),
                        #InputCalcS(f'h', "Abs ratio", f'{self.h}', width = 50),
                    ),

                    id="dispersionDynamicsOptionsForm"
                                    
                ),
                )),
                style="position:sticky; top:0px; background:#f0f8f8;"
            ),
            Div(
                Div(


                    generate_chart([cget(self.dispersion_t_list).tolist()], 
                                    [cget(pulse).tolist()], [""], 
                                    "Pulse in (photons/sec)", h=2, color=["red"], marker=None, twinx=True),
                    # Div(
                    #    Div(cls="handle", draggable="true"),
                    #        FlexN([graphCanvas(id="dispersion_pulse_chart", width=1100, height=300, options=False, mode = 2), 
                    #        ]), cls="container"
                    # ),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_pulse, 1024, self.beam_view_from, self.beam_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_pulse, 1024, self.beam_view_from, self.beam_view_to))).tolist()], [""], 
                                    "E", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_calculated_fft, 1024, self.beam_view_from, self.beam_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_calculated_fft, 1024, self.beam_view_from, self.beam_view_to))).tolist()], [""], 
                                    "Using FFT", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_calculated_fd, 1024, self.beam_view_from, self.beam_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_calculated_fd, 1024, self.beam_view_from, self.beam_view_to))).tolist()], [""], 
                                    "Using FD S3", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_calculated_fd_s5, 1024, self.beam_view_from, self.beam_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_calculated_fd_s5, 1024, self.beam_view_from, self.beam_view_to))).tolist()], [""], 
                                    "Using FD S5", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),

                    #Div(self.collectDiodeData(), id="numData"),

                    cls="box", style="background-color: #008080;"
                ),
            ) if (len(self.dispersion_pulse) > 0) else Div(),
        )
        return added
    
import numpy as np

def apply_dispersion(
    A_t,
    dt,
    L,
    beta2=0.0,
    beta3=0.0,
    beta4=0.0
):
    """
    code from chatGPT

    Apply linear dispersion to a complex time-domain envelope.

    Parameters
    ----------
    A_t : np.ndarray (complex)
        Complex envelope in time domain
    dt : float
        Time step [s]
    L : float
        Propagation length [m]
    beta2 : float
        Second-order dispersion [s^2 / m]
    beta3 : float
        Third-order dispersion [s^3 / m]
    beta4 : float
        Fourth-order dispersion [s^4 / m]

    Returns
    -------
    A_out : np.ndarray (complex)
        Dispersed complex envelope in time domain
    """

    N = A_t.size

    # Angular frequency grid (rad/s), centered
    omega = 2 * np.pi * np.fft.fftfreq(N, dt)

    # Forward FFT
    A_w = np.fft.fft(A_t)

    # Dispersion phase
    phase = (
        0.5 * beta2 * omega**2 +
        (1.0/6.0) * beta3 * omega**3 +
        (1.0/24.0) * beta4 * omega**4
    ) * L

    # Apply dispersion
    A_w *= np.exp(1j * phase)

    # Inverse FFT
    A_out = np.fft.ifft(A_w)

    return A_out

import numpy as np

import numpy as np

def dispersion_fd_step(
    A_in: np.ndarray,
    A_out: np.ndarray,
    n_start: int,
    n_end: int,
    alpha: complex,
    n_stencil: int
):
    """
    One explicit finite-difference dispersion step.

    Parameters
    ----------
    A_in : complex ndarray
        Input field (length N, cyclic).
    A_out : complex ndarray
        Output field (same shape, already initialized as copy of A_in).
    n_start, n_end : int
        Index range [n_start, n_end) where dispersion is applied.
    alpha : complex
        Dispersion coefficient:
        alpha = 1j * beta2 * dz / (2 * dt**2)
    """
    N = A_in.shape[0]

    if n_stencil == 3:
        for n in range(n_start, n_end):
            nm = (n - 1)
            np_ = (n + 1)
            lap = A_in[np_] - 2.0 * A_in[n] + A_in[nm]
            A_out[n] = A_in[n] + alpha * lap
    if n_stencil == 5:
        for n in range(n_start, n_end):
            lap = (-A_in[n+2] - A_in[n-2] + 16.0*(A_in[n+1]+ A_in[n-1]) - 30.0*A_in[n]) / 12.0
            A_out[n] = A_in[n] + alpha * lap

def propagate_dispersion_CN(A_t, dt, beta2, dz, n_steps):
    """
    Time-domain Crank–Nicolson propagation for pure dispersion.

    Parameters
    ----------
    A_t : np.ndarray (complex)
        Initial complex envelope A(t)
    dt : float
        Time step [s]
    beta2 : float
        GVD [s^2 / m]
    dz : float
        Propagation step [m]
    n_steps : int
        Number of propagation steps

    Returns
    -------
    A_t : np.ndarray (complex)
        Envelope after propagation
    """

    N = A_t.size
    alpha = 1j * beta2 * dz / (4 * dt**2)

    # Tridiagonal matrix coefficients
    main_diag = (1 + 2*alpha) * np.ones(N, dtype=complex)
    off_diag  = -alpha * np.ones(N-1, dtype=complex)

    # Assemble LHS matrix (with periodic BCs)
    A_mat = np.diag(main_diag) \
          + np.diag(off_diag,  1) \
          + np.diag(off_diag, -1)

    A_mat[0, -1] = -alpha
    A_mat[-1, 0] = -alpha

    # Precompute inverse once (matrix is constant)
    A_mat_inv = np.linalg.inv(A_mat)

    for _ in range(n_steps):
        # RHS
        rhs = (
            (1 - 2*alpha) * A_t
            + alpha * np.roll(A_t,  1)
            + alpha * np.roll(A_t, -1)
        )

        # Solve
        A_t = A_mat_inv @ rhs

    return A_t
