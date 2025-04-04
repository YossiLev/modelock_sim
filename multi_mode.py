import numpy as np
from numpy.fft import fftshift
from controls import random_lcg_set_seed, random_lcg
import json

def m_dist(d):
    return [[1, d], [0, 1]]

def m_lens(f):
    return [[1, 0], [-1 / f, 1]]

def m_mult_v(*Ms):
    result = [[1.0, 0.0], [0.0, 1.0]]
    for m in Ms:
        result = m_mult(m, result)
    return result

def m_mult(m, m2):
    a = m[0][0] * m2[0][0] + m[0][1] * m2[1][0]
    b = m[0][0] * m2[0][1] + m[0][1] * m2[1][1]
    c = m[1][0] * m2[0][0] + m[1][1] * m2[1][0]
    d = m[1][0] * m2[0][1] + m[1][1] * m2[1][1]
    return [[a, b], [c, d]]

def m_mult_inv(m, m2):
    a = m[0][0] * m2[1][1] - m[0][1] * m2[1][0]
    b = -m[0][0] * m2[0][1] + m[0][1] * m2[0][0]
    c = m[1][0] * m2[1][1] - m[1][1] * m2[1][0]
    d = -m[1][0] * m2[0][1] + m[1][1] * m2[0][0]
    return [[a, b], [c, d]]

def m_inv(m):
    return [[m[1][1], -m[0][1]], [-m[1][0], m[0][0]]]

def calc_original_sim_matrices():
    position_lens = -0.00015
    m_long = m_mult_v(m_dist(position_lens), m_dist(0.081818181), m_lens(0.075), m_dist(0.9),
                        m_dist(0.9), m_lens(0.075), m_dist(0.081818181), m_dist(position_lens))
    m_short = m_mult_v(m_dist(0.001 - position_lens), m_dist(0.075), m_lens(0.075), m_dist(0.5),
                        m_dist(0.5), m_lens(0.075), m_dist(0.075), m_dist(0.001 - position_lens))

    mat_side = [m_short, m_long]
    print(f"MShort = {m_short}")
    print(f"MLong = {m_long}")

    return mat_side

def serialize_fronts(fs):
    return [[f"{np.real(val):.2f},{np.imag(val):.2f}" if np.abs(val) > 0.001 else "" for val in row] for row in fs]

class MultiModeSimulation:
    def __init__(self):
        self.lambda_ = 0.000000780
        self.initial_range = 0.01047
        self.multi_ranges = [[], []]
        self.n_samples = 256
        self.n_max_matrices = 1000
        self.n_rounds = 0

        self.steps_counter = 0
        self.n_time_samples = 1024
        self.multi_time_fronts = []
        self.multi_frequency_fronts = []
        self.multi_time_fronts_saves = [[], [], [], [], [], []]

        self.intensity_saturation_level = 400000000000000.0
        self.intensity_total_by_ix = []
        self.factor_gain_by_ix = []
        self.ikl = 0.02
        self.ikl_times_i = 1j * self.ikl * 160 * 0.000000006
        self.range_w = []
        self.spectral_gain = []
        self.modulation_gain_factor = 0.1
        self.modulator_gain = []
        self.dispersion = []
        self.sum_power_ix = []
        self.gain_reduction = []
        self.gain_reduction_with_origin = []
        self.gain_reduction_after_aperture = []
        self.gain_reduction_after_diffraction = []
        self.gain_factor = 0.50
        self.epsilon = 0.2
        self.dispersion_factor = 1.0
        self.lensing_factor = 1.0
        self.is_factor = 200 * 352000
        self.pump_gain0 = []
        self.multi_time_aperture = []
        self.aperture = 0.000056
        self.multi_time_diffraction = []
        self.multi_time_diffraction_val = 0.000030
        self.frequency_total_mult_factor = []
        self.mirror_loss = 0.95
        self.fresnel_data = []
        self.total_range = 0.001000
        self.dx0 = self.total_range / self.n_samples
        self.scalar_one = 1 + 0j
        self.n_time_samples_ones = [self.scalar_one] * self.n_time_samples
        self.n_samples_ones = [self.scalar_one] * self.n_samples
        self.n_samples_ones_r = [1.0] * self.n_samples
        self.ps = [[], []]
        self.view_on_stage = ["1", "1"]
        self.view_on_amp_freq = ["Amp", "Amp"]
        self.view_on_abs_phase = ["Abs", "Abs"]
        self.view_on_x = self.n_time_samples // 2
        self.view_on_y = self.n_samples // 2
        self.view_on_sample = 0

        self.mat_side = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  # right
                         [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]]    # left

        self.mat_side = calc_original_sim_matrices()

    def printSamples(self):
        print("----------------------------------")
        print(f"{self.multi_time_fronts[128][63]}")

    def set(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, key)
        return params

    def vectors_for_fresnel(self, M, N, dx0, gain, is_back):
        A, B = M[0]
        C, D = M[1]
        #print(A, B, C, D, self.lambda_)
        dxf = B * self.lambda_ / (N * dx0)
        factor = 1j * gain * np.sqrt(-1j / (B * self.lambda_))
        if is_back:
            factor *= 1j * gain
        co0 = -np.pi * dx0 * dx0 * A / (B * self.lambda_)
        cof = -np.pi * dxf * dxf * D / (B * self.lambda_)

        vec0 = [np.exp(1j * co0 * (i - N / 2) ** 2) for i in range(N)]
        vecF = [factor * np.exp(1j * cof * (i - N / 2) ** 2) for i in range(N)]

        return {'dx': dx0, 'vecs': [vec0, vecF]}

    def spectral_gain_dispersion(self):
        self.multi_frequency_fronts = fftshift(np.fft.fft(np.fft.ifftshift(self.multi_time_fronts, axes=1), axis=1), axes=1) * self.frequency_total_mult_factor
        self.multi_time_fronts = fftshift(np.fft.ifft(np.fft.ifftshift(self.multi_frequency_fronts, axes=1), axis=1), axes=1)

    def modulator_gain_multiply(self):
        self.multi_time_fronts = self.multi_time_fronts * self.modulator_gain

    def prepare_gain_pump(self):
        pump_width = 0.000030 * 0.5
        g0 = 1 / self.mirror_loss + self.epsilon
        self.pump_gain0 = []
        for ix in range(self.n_samples):
            x = (ix - self.n_samples / 2) * self.dx0
            xw = x / pump_width
            self.pump_gain0.append(g0 * np.exp(-xw * xw))

    def prepare_aperture(self):
        self.multi_time_aperture = []
        aperture_width = self.aperture * 0.5
        for ix in range(self.n_samples):
            x = (ix - self.n_samples / 2) * self.dx0
            xw = x / aperture_width
            self.multi_time_aperture.append(np.exp(-xw * xw))

        self.multi_time_diffraction = []
        diffraction_width = self.multi_time_diffraction_val
        for ix in range(self.n_samples):
            x = (ix - self.n_samples / 2) * self.dx0
            xw = x / diffraction_width
            self.multi_time_diffraction.append(np.exp(-xw * xw))

    def prepare_linear_fresnel_help_data(self):
        self.fresnel_data = []
        dx = self.total_range / self.n_samples

        for index_side, side_m in enumerate(self.mat_side):
            A, B, C, D = side_m[0][0], side_m[0][1], side_m[1][0], side_m[1][1]
            if A > 0:
                M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]]
                M1 = [[1, B / (A + 1)], [0, 1]]
            else:
                M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]]
                M1 = [[-1, B / (-A + 1)], [0, -1]]

            fresnel_side_data = []
            for index, M in enumerate([M1, M2]):
                loss = self.mirror_loss if index == 0 and index_side == 1 else 1
                fresnel_side_data.append(self.vectors_for_fresnel(M, self.n_samples, dx, loss, M[0][0] < 0))
                dx = M[0][1] * self.lambda_ / (self.n_samples * dx)

            self.fresnel_data.append(fresnel_side_data)

    def total_ix_power(self):
        self.intensity_total_by_ix = []
        for ix in range(self.n_samples):
            self.intensity_total_by_ix.append(np.sum(np.multiply(self.multi_time_fronts[ix], np.conj(self.multi_time_fronts[ix]))))

    def init_gain_by_frequency(self):
        spec_gain = 200
        disp_par = self.dispersion_factor * 0.5e-3 * 2 * np.pi / spec_gain
        self.range_w = np.array([complex(v) for v in np.arange(-self.n_time_samples / 2, self.n_time_samples / 2)])
        ones = np.ones_like(self.range_w, dtype=complex)
        mid = self.range_w / spec_gain
        self.spectral_gain = ones / (mid ** 2 + 1)
        self.dispersion = np.exp(-1j * disp_par * self.range_w ** 2)
        exp_w = np.exp(-1j * 2 * np.pi * self.range_w)
        self.frequency_total_mult_factor = 0.5 * (1.0 + exp_w * self.spectral_gain * self.dispersion)
        self.modulator_gain = [1.0 + self.modulation_gain_factor * np.cos(2 * np.pi * w / self.n_time_samples) for w in self.range_w]
        print(f"modulator_gain = {self.modulator_gain[80]} {self.modulator_gain[180]} ")

    def get_init_front(self, p_par=-1):
        vf = []
        self.initial_range = 0.00024475293  # Example value, replace with actual value
        waist0 = p_par if p_par > 0.0 else 0.00003  # Example value, replace with actual value
        beam_dist = 0.0  # Example value, replace with actual value
        RayleighRange = np.pi * waist0 * waist0 / self.lambda_
        theta = 0 if abs(beam_dist) < 0.000001 else np.pi / (self.lambda_ * beam_dist)
        waist = waist0 * np.sqrt(1 + beam_dist / RayleighRange)
        dx = self.initial_range / self.n_samples
        x0 = (self.n_samples - 1) / 2 * dx

        for i in range(self.n_samples):
            px = i * dx
            x = px - x0
            xw = x / waist
            f_val = np.exp(complex(-xw * xw, -theta * x * x)) * (1.0 + 0.3 * random_lcg())
            vf.append(f_val)

        return vf

    def update_helpData(self):
        self.prepare_linear_fresnel_help_data()
        self.prepare_gain_pump()
        self.prepare_aperture()
        self.init_gain_by_frequency()
    
    def init_multi_time(self):
        self.multi_time_fronts = [[] for _ in range(self.n_samples)]
        self.multi_frequency_fronts = [[] for _ in range(self.n_samples)]
        self.multi_time_fronts_saves = [[], [], [], [], [], []]

        random_lcg_set_seed(1323)
        for i_time in range(self.n_time_samples):
            #rnd = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            rnd = (random_lcg() * 2 - 1) + 1j * (random_lcg() * 2 - 1)
            fr = np.multiply(rnd, self.get_init_front())
            for i in range(self.n_samples):
                self.multi_time_fronts[i].append(fr[i])
                self.multi_frequency_fronts[i].append(0 + 0j)

        self.multi_time_fronts_saves[0] = np.copy(self.multi_time_fronts)

        self.update_helpData()

    def phase_change_during_kerr(self):
        self.sum_power_ix = []
        total_kerr_lensing = np.multiply(self.lensing_factor, self.ikl_times_i)

        bin2 = np.abs(self.multi_time_fronts) ** 2
        self.sum_power_ix = np.sum(bin2, axis=1).tolist()
        phase_shift1 = total_kerr_lensing * bin2
        self.multi_time_fronts *= np.exp(phase_shift1)
        self.ps[self.side] = phase_shift1[:, self.view_on_x].imag.tolist()
        # for ix in range(self.n_samples):
        #     bin = self.multi_time_fronts[ix]
        #     bin2 = np.abs(np.multiply(bin, np.conj(bin)))
        #     self.sum_power_ix.append(np.sum(bin2))
        #     phase_shift1 = np.multiply(total_kerr_lensing, bin2)
        #     self.multi_time_fronts[ix] = np.multiply(bin, np.exp(phase_shift1))
        #     self.ps1.append(phase_shift1[0].imag)

        self.multi_time_fronts_saves[self.side * 3 + 0] = np.copy(self.multi_time_fronts)

        multi_time_fronts_trans = self.multi_time_fronts.T
        p_fr_before = np.sum(np.abs(multi_time_fronts_trans)**2, axis=1, keepdims=True)
        fr_after = multi_time_fronts_trans * self.multi_time_aperture
        p_fr_after = np.sum(np.abs(fr_after)**2, axis=1, keepdims=True)
        multi_time_fronts_trans = fr_after * np.sqrt(p_fr_before / p_fr_after)
        self.multi_time_fronts = multi_time_fronts_trans.T

        # for i_time in range(self.n_time_samples):
        #     fr = multi_time_fronts_trans[i_time]
        #     p_fr_before = np.sum(np.multiply(fr, np.conj(fr)))
        #     fr_after = np.multiply(fr, self.multi_time_aperture)
        #     p_fr_after = np.sum(np.multiply(fr_after, np.conj(fr_after)))
        #     fr = np.multiply(fr_after, np.sqrt(p_fr_before / p_fr_after))
        #     multi_time_fronts_trans[i_time] = fr
        # self.multi_time_fronts = np.transpose(multi_time_fronts_trans)
    
    def linear_cavity_one_side(self):
        self.multi_time_fronts_saves[self.side * 3 + 1] = np.copy(self.multi_time_fronts)

        Is = self.is_factor
        self.gain_reduction = np.real(np.multiply(self.pump_gain0, np.divide(self.n_samples_ones, 1 + np.divide(self.sum_power_ix, Is * self.n_time_samples))))
        self.gain_reduction_with_origin = np.multiply(self.gain_factor, 1 + self.gain_reduction)
        self.gain_reduction_after_diffraction = np.multiply(self.gain_reduction_with_origin, self.multi_time_diffraction)

        multi_time_fronts_trans = self.multi_time_fronts.T
        gain_factors = self.gain_reduction_after_diffraction
        multi_time_fronts_trans = gain_factors * multi_time_fronts_trans
        for fresnel_side_data in self.fresnel_data[self.side]:
            vec0 = fresnel_side_data['vecs'][0]
            vecF = fresnel_side_data['vecs'][1]
            dx = fresnel_side_data['dx']
            
            multi_time_fronts_trans = vec0 * multi_time_fronts_trans
            multi_time_fronts_trans = fftshift(np.fft.fft(fftshift(multi_time_fronts_trans, axes=1), axis=1), axes=1) * dx
            multi_time_fronts_trans = vecF * multi_time_fronts_trans
        self.multi_time_fronts = multi_time_fronts_trans.T

        # multi_time_fronts_trans = np.transpose(self.multi_time_fronts)
        # for i_time in range(self.n_time_samples):
        #     fr = multi_time_fronts_trans[i_time]
        #     fr = np.multiply(fr, self.gain_reduction_after_diffraction)
        #     for fresnel_side_data in self.fresnel_data[self.side]:
        #         fr = np.multiply(fr, fresnel_side_data['vecs'][0])
        #         fr = fftshift(np.fft.fft(fftshift(fr))) * fresnel_side_data['dx']
        #         fr = np.multiply(fr, fresnel_side_data['vecs'][1])
        #     multi_time_fronts_trans[i_time] = fr
        # self.multi_time_fronts = np.transpose(multi_time_fronts_trans)

        self.multi_time_fronts_saves[self.side * 3 + 2] = np.copy(self.multi_time_fronts)

    def multi_time_round_trip(self):
        #if self.i_count % 10 == 0:
        # fs = np.abs(self.multi_time_fronts)
        # fs = np.multiply(fs, fs)
        # mean_v = np.mean(fs, axis=0)
        # mean_mean = np.mean(mean_v)
        self.n_rounds += 1

        for self.side in [0, 1]:
            self.phase_change_during_kerr()

            self.spectral_gain_dispersion()

            if self.side == 1:
                self.modulator_gain_multiply()

            self.linear_cavity_one_side()

    def get_x_values(self):

        target = int(self.view_on_stage[self.view_on_sample]) - 1
        stage_data = np.copy(self.multi_time_fronts_saves[target])
        if (self.view_on_amp_freq[self.view_on_sample] == "Frq"):
            stage_data = fftshift(np.fft.fft(np.fft.ifftshift(stage_data, axes=1), axis=1), axes=1)
        if (self.view_on_abs_phase[self.view_on_sample] == "Abs"):
            stage_data = np.abs(stage_data).T
        else:
            stage_data = np.angle(stage_data).T

        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_x:
            fr = stage_data[self.view_on_x].tolist()
            return {"values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})"}
        return {}
    
    def get_y_values(self):

        target = int(self.view_on_stage[self.view_on_sample]) - 1
        stage_data = np.copy(self.multi_time_fronts_saves[target])
        if (self.view_on_amp_freq[self.view_on_sample] == "Frq"):
            stage_data = fftshift(np.fft.fft(np.fft.ifftshift(stage_data, axes=1), axis=1), axes=1)
        if (self.view_on_abs_phase[self.view_on_sample] == "Abs"):
            stage_data = np.abs(stage_data)
        else:
            stage_data = np.angle(stage_data)
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_y:
            fr = stage_data[self.view_on_y].tolist()
            return {"values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})"}
        return {}
    
    def select_source(self, target):
        stage_data = self.multi_time_fronts_saves[int(self.view_on_stage[target]) - 1]
        if (self.view_on_amp_freq[target] == "Frq"):
            stage_data = fftshift(np.fft.fft(np.fft.ifftshift(stage_data, axes=1), axis=1), axes=1)
        if (self.view_on_abs_phase[target] == "Abs"):
            stage_data = np.abs(stage_data)
        else:
            stage_data = np.angle(stage_data)
        return stage_data
    
    def serialize_mm_data(self, more):
        print("in serialize")
        s = json.dumps({
            "more": more,
            "rounds": self.n_rounds,
            "pointer": [self.view_on_sample, self.view_on_x, self.view_on_y],
            "samples": 
                [
                    {"name": "funCanvasSample1", "samples": serialize_fronts(self.select_source(0))},
                    {"name": "funCanvasSample2", "samples": serialize_fronts(self.select_source(1))}
                ],
            "graphs": 
                [
                    {"name": "gr1", "lines": []},
                    {"name": "gr2", "lines": []},
                    {"name": "gr3", "lines": [{"color": "red", **self.get_x_values()}]},
                    {"name": "gr4", "lines": [{"color": "blue", "values": self.ps[0], "text": f"M{max(self.ps[0]):.4f}({self.ps[0].index(max(self.ps[0]))})"},
                                            {"color": "green", "values": self.ps[1], "text": f"M{max(self.ps[1]):.4f}({self.ps[1].index(max(self.ps[1]))})"} ] 
                                            if len(self.ps[0]) > 10 and len(self.ps[1]) > 10 else []},    
                    {"name": "gr5", "lines": [{"color": "red", **self.get_y_values()}]}
                ],
            "view_buttons": 
                {
                    "view_on_stage": self.view_on_stage,
                    "view_on_amp_freq": self.view_on_amp_freq,
                    "view_on_abs_phase": self.view_on_abs_phase,
                }    
            
         })
        return s
    def serialize_mm_graphs(self):
        s = json.dumps({
            "pointer": [self.view_on_sample, self.view_on_x, self.view_on_y],
            "graphs": [
                {"name": "gr1", "lines": []},
                {"name": "gr2", "lines": []},
                {"name": "gr3", "lines": [{"color": "red", **self.get_x_values()}]},
                {"name": "gr4", "lines": [{"color": "blue", "values": self.ps[0], "text": f"M{max(self.ps[0]):.4f}({self.ps[0].index(max(self.ps[0]))})"},
                                          {"color": "green", "values": self.ps[1], "text": f"M{max(self.ps[1]):.4f}({self.ps[1].index(max(self.ps[1]))})"} ] 
                                          if len(self.ps[0]) > 10 and len(self.ps[1]) > 10 else []},    
                {"name": "gr5", "lines": [{"color": "red", **self.get_y_values()}]},
            ]
            })
        return s