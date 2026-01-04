import ctypes
import platform
import json
import numpy as np
from fasthtml.common import *
from controls import *
from calc_common import *

class dispersion_calc(CalcCommon):
    def __init__(self):
        self.dispersion_view_from = -1
        self.dispersion_view_to = -1

        self.dispersion_cavity_type = "Ring"
        self.dispersion_mode = "Amplitude"
        self.dispersion_sampling = "4096"
        self.dispersion_sampling_x = "32"
        self.dispersion_pulse_dtype = np.complex128

        self.dispersion_cavity_time = 3.95138389E-09 #4E-09
        self.dispersion_N = 4096 # * 4
        self.dispersion_dt = self.dispersion_cavity_time / self.dispersion_N
        self.dispersion_intensity = "Pulse"
        self.calculation_rounds = 1
        self.calculation_rounds_done = 0
        self.dispersion_pulse_width = 100.0

        # actual shift parameters in millimeters
        self.dispersion_absorber_shift = 0.0
        self.dispersion_gain_shift = 11.0
        self.dispersion_output_coupler_shift = 130.0

        self.loss_shift = self.dispersion_N // 2 + self.mm_to_unit_shift(self.dispersion_absorber_shift) # zero shift means that the absorber is in the middle of the cavity
        self.gain_distance = self.mm_to_unit_shift(self.dispersion_gain_shift)
        self.oc_shift = self.mm_to_unit_shift(self.dispersion_output_coupler_shift)
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
        self.dispersion_update_pulse = "Update Pulse"
        self.h = 0.1

        self.rand_factor_seed = 0.0000000005
        self.kappa = 3.0E07
        self.C_loss = 95.0E+06
        self.C_gain = 300.0E+05
        self.coupling_out_loss =-5000E+06
        self.coupling_out_gain = 2800E+05

        # diode dynamics parameters
        self.dispersion_t_list = np.array([1], dtype=np.float64)
        self.dispersion_pulse = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_pulse_init = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_pulse_save = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_pulse_original = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_pulse_after = np.array([], dtype=self.dispersion_pulse_dtype)
        self.dispersion_accum_pulse = []
        self.dispersion_accum_pulse_after = []
        self.dispersion_gain = np.array([1], dtype=np.float64)
        self.dispersion_loss = np.array([1], dtype=np.float64)
        self.dispersion_gain_polarization = np.array([1], dtype=np.complex128)
        self.dispersion_loss_polarization = np.array([1], dtype=np.complex128)
        self.dispersion_gain_value = np.array([1], dtype=np.float64)
        self.dispersion_loss_value = np.array([1], dtype=np.float64)

        # diode summary parameters
        self.summary_photons_before = 0.0
        self.summary_photons_after_gain = 0.0
        self.summary_photons_after_absorber = 0.0
        self.summary_photons_after_cavity_loss = 0.0

    def doCalcCommand(self, params):
        pass

    def generate_calc(self):
        tab = 6
        t_list = cget(shrink_with_max(self.diode_t_list, 1024, self.diode_view_from, self.diode_view_to)).tolist()

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
                        SelectCalcS(tab, f'DiodeSelectSampling', "Sampling", ["4096", "8192", "16384", "32768", "65536", "131072", "262144", "524288", "1048576"], self.dispersion_sampling, width = 100),
                        SelectCalcS(tab, f'DiodeSelectSamplingX', "Sampling X", ["32", "64", "128", "256"], self.dispersion_sampling_x, width = 80),
                        SelectCalcS(tab, f'CalcDiodeCavityType', "Cavity Type", ["Ring", "Linear"], self.dispersion_cavity_type, width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectMode', "mode", ["Intensity", "Amplitude", "MB", "MBGPU"], self.dispersion_mode, width = 120),
                        InputCalcS(f'DiodePulseWidth', "Pulse width (ps)", f'{self.dispersion_pulse_width}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeSelectIntensity', "Intensity", ["Pulse", "Noise", "CW", "Flat"], self.dispersion_intensity, width = 80),
                        InputCalcS(f'DiodeViewFrom', "View from", f'{self.dispersion_view_from}', width = 80),
                        InputCalcS(f'DiodeViewTo', "View to", f'{self.dispersion_view_to}', width = 80),
                    ),
                    Div(
                        InputCalcS(f'DiodeAbsorberShift', "Absorber Shift (mm)", f'{self.dispersion_absorber_shift}', width = 100),
                        InputCalcS(f'DiodeGainShift', "Gain Shift (mm)", f'{self.dispersion_gain_shift}', width = 100),
                        InputCalcS(f'DiodeOutputCouplerShift', "OC Shift (mm)", f'{self.dispersion_output_coupler_shift}', width = 100),
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
                        InputCalcS(f'dt', "dt (ps)", f'{format(self.dispersion_dt * 1E+12, ".4f")}', width = 80),
                        InputCalcS(f'volume', "Volume cm^3", f'{self.volume}', width = 80),
                        InputCalcS(f'initial_photons', "Initial Photns", f'{self.initial_photons}', width = 100),
                        InputCalcS(f'cavity_loss', "OC (Cavity loss)", f'{self.cavity_loss}', width = 80),
                        SelectCalcS(tab, f'CalcDiodeUpdatePulse', "UpdatePulse", ["Update Pulse", "Unchanged Pulse"], self.dispersion_update_pulse, width = 120),
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
                                    [cget(shrink_with_max(pulse_original, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist(), 
                                    cget(shrink_with_max(np.log(pulse_after+ 0.000000001), 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist()], [""], 
                                    "Original Pulse and Pulse after (photons/sec)", h=2, color=colors, marker=None, twinx=True),

                    generate_chart([cget(self.dispersion_t_list).tolist(), self.dispersion_levels_x], 
                                    [cget(pulse).tolist(), self.dispersion_levels_y], [""], 
                                    "Pulse in (photons/sec)", h=2, color=["red", "black"], marker=None, twinx=True),
                    # Div(
                    #    Div(cls="handle", draggable="true"),
                    #        FlexN([graphCanvas(id="dispersion_pulse_chart", width=1100, height=300, options=False, mode = 2), 
                    #        ]), cls="container"
                    # ),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(pulse_after, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist()], [""], 
                                    "Pulse out (photons/sec)", h=2, color=colors, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_pulse_after, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_pulse_after, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist()], [""], 
                                    "E", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.dispersion_accum_pulse, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist(), cget(shrink_with_max(self.dispersion_accum_pulse_after, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist()], [""], 
                                    f"Accumulate Pulse AND after (photons) [difference: {(self.dispersion_accum_pulse_after[-1] - self.dispersion_accum_pulse[-1]):.2e}]", 
                                    h=2, color=colors, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.dispersion_gain, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist(), cget(shrink_with_max(self.dispersion_gain_value, 1024, self.dispersion_view_from, self.dispersion_view_to )).tolist()], [""], 
                                    f"Gain carriers (1/cm^3) [{(max_gain - min_gain):.2e} = {max_gain:.4e} - {min_gain:.4e}] and Gain (cm^-1)", 
                                    color=["black", "green"], h=2, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_gain_polarization, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_gain_polarization, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist()], [""], 
                                    "Gain Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(shrink_with_max(self.dispersion_loss, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist(), 
                                    cget(shrink_with_max(self.dispersion_loss_value, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist()], [""], 
                                    f"Abs carrs (cm^-3) [{(max_loss - min_loss):.2e} = {max_loss:.3e} - {min_loss:.3e}] and Loss (cm^-1)", 
                                    color=["black", "red"], h=2, marker=None, twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.angle(shrink_with_max(self.dispersion_loss_polarization, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist(), 
                                        cget(np.absolute(shrink_with_max(self.dispersion_loss_polarization, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist()], [""], 
                                    "Loss Polarization", color=["green", "red"], h=2, marker=None, lw=[1, 3], twinx=True),
                    generate_chart([t_list], 
                                    [cget(np.exp(- self.cavity_loss) * 
                                            np.multiply(shrink_with_max(self.dispersion_gain_value, 1024, self.dispersion_view_from, self.dispersion_view_to),
                                            shrink_with_max(self.dispersion_loss_value, 1024, self.dispersion_view_from, self.dispersion_view_to))).tolist(),
                                    cget(shrink_with_max(pulse, 1024, self.dispersion_view_from, self.dispersion_view_to)).tolist()], [""],
                                    "Net gain", color=["blue", "red"], h=2, marker=None, twinx=True),
                    generate_chart(xVec, yVec, [""], "Gain By Pop", h=4, color=["black", "black", "green", "red"], marker=".", lw=[5, 5, 1, 1]),

                    #Div(self.collectDiodeData(), id="numData"),

                    cls="box", style="background-color: #008080;"
                ),
            ) if (len(self.dispersion_pulse) > 0 and len(self.dispersion_gain) > 0) else Div(),
        )
        return added        