import platform

if platform.system() == "Linux":
    try:
        import cupy as np
        import numpy as nump
        print("Using CuPy for Linux")
    except ImportError:
        import numpy as np
        print("CuPy not available, falling back to NumPy")
else:
    import numpy as np
    print("Using NumPy for non-Linux OS")
# import numpy as np
from numpy.fft import fftshift
from controls import random_lcg_set_seed, random_lcg
import json

def cget(x):
    return x.get() if hasattr(x, "get") else x

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

def calc_original_sim_matrices(crystal_shift=0.0):
    position_lens = -0.00015 + crystal_shift  # -0.00015 shift needed due to conclusions from single lens usage in original simulation
    m_long = m_mult_v(m_dist(position_lens), m_dist(0.081818181), m_lens(0.075), m_dist(0.9),
                        m_dist(0.9), m_lens(0.075), m_dist(0.081818181), m_dist(position_lens))
    m_short = m_mult_v(m_dist(0.001 - position_lens), m_dist(0.075), m_lens(0.075), m_dist(0.5),
                        m_dist(0.5), m_lens(0.075), m_dist(0.075), m_dist(0.001 - position_lens))

    mat_side = [m_short, m_long]
    print(f"MShort = {m_short}")
    print(f"MLong = {m_long}")

    return mat_side

def list_mean_and_std(weights):
    indices = np.arange(len(weights))
    mean_index = np.average(indices, weights=weights)
    variance = np.average((indices - mean_index)**2, weights=weights)
    std_index = np.sqrt(variance)
    return mean_index, std_index

def serialize_fronts(fs):
    getfs = cget(fs)
    s = [[f"{val:.2f}" if abs(val) > 0.001 else "" for val in row] for row in getfs]
    return s

class MultiModeSimulation:
    def __init__(self):

        self.modulation_gain_factor = np.asarray(0.1)
        self.gain_factor = np.asarray(0.50)
        self.epsilon = np.asarray(0.2)
        self.dispersion_factor = np.asarray(1.0)
        self.lensing_factor = np.asarray(1.0)
        self.is_factor = np.asarray(200 * 352000)
        self.crystal_shift = np.asarray(0.0)
        self.aperture = np.asarray(0.000056)

        self.lambda_ = 0.000000780
        self.initial_range = 0.00024475293
        self.multi_ranges = [[], []]
        self.n_samples = 256
        self.n_max_matrices = 1000
        self.n_rounds = 0

        self.steps_counter = 0
        self.n_time_samples = 1024
        self.multi_time_fronts = []
        self.multi_time_fronts_saves = [[], [], [], [], [], []]

        self.intensity_saturation_level = 400000000000000.0
        self.intensity_total_by_ix = []
        self.factor_gain_by_ix = []
        self.ikl = 0.02
        self.ikl_times_i = np.asarray(1j * self.ikl * 160 * 0.000000006)
        self.range_w = []
        self.spectral_gain = []
        self.modulator_gain = []
        self.dispersion = []
        self.sum_power_ix = []
        self.gain_reduction = []
        self.gain_reduction_with_origin = []
        self.gain_reduction_after_aperture = []
        self.gain_reduction_after_diffraction = []

        self.pump_gain0 = []
        self.multi_time_aperture = []
        self.multi_time_diffraction = []
        self.multi_time_diffraction_val = np.asarray(0.000030)
        self.frequency_total_mult_factor = []
        self.mirror_loss = np.asarray(0.95)
        self.fresnel_data = []
        self.total_range = 0.001000
        self.dx0 = self.total_range / self.n_samples
        self.scalar_one = 1 + 0j
        self.n_time_samples_ones = np.asarray([self.scalar_one] * self.n_time_samples)
        self.n_samples_ones = np.asarray([self.scalar_one] * self.n_samples)
        self.n_samples_ones_r = np.asarray([1.0] * self.n_samples)
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
        print(f"{cget(self.multi_time_fronts)[128][63]}")

    def set(self, params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, np.asarray(value))

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
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

        vec0 = np.asarray([np.exp(1j * co0 * (i - N / 2) ** 2) for i in range(N)])
        vecF = np.asarray([factor * np.exp(1j * cof * (i - N / 2) ** 2) for i in range(N)])

        return {'dx': np.asarray(dx0), 'vecs': [vec0, vecF]}

    def spectral_gain_dispersion(self):
        multi_frequency_fronts = fftshift(np.fft.fft(np.fft.ifftshift(self.multi_time_fronts, axes=1), axis=1), axes=1) * self.frequency_total_mult_factor
        self.multi_time_fronts = fftshift(np.fft.ifft(np.fft.ifftshift(multi_frequency_fronts, axes=1), axis=1), axes=1)

    def modulator_gain_multiply(self):
        self.multi_time_fronts = self.multi_time_fronts * self.modulator_gain

    def prepare_gain_pump(self):
        pump_width = np.asarray(0.000030 * 0.5)
        g0 = np.asarray(1 / self.mirror_loss + self.epsilon)
        x = (np.arange(self.n_samples) - np.asarray(self.n_samples / 2)) * np.asarray(self.dx0)
        xw = x / pump_width
        self.pump_gain0 = g0 * np.exp(-xw * xw)

    def prepare_aperture(self):
        aperture_width = np.asarray(self.aperture * 0.5)
        x = (np.arange(self.n_samples) - np.asarray(self.n_samples / 2)) * np.asarray(self.dx0)
        xw = x / aperture_width
        self.multi_time_aperture = np.exp(-xw * xw)

        diffraction_width = self.multi_time_diffraction_val
        xw = x / diffraction_width
        self.multi_time_diffraction = np.exp(-xw * xw)

    def prepare_linear_fresnel_help_data(self):
        self.mat_side = calc_original_sim_matrices(self.crystal_shift)

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

    # def total_ix_power(self):
    #     self.intensity_total_by_ix = []
    #     for ix in range(self.n_samples):
    #         self.intensity_total_by_ix.append(np.sum(np.multiply(self.multi_time_fronts[ix], np.conj(self.multi_time_fronts[ix]))))

    def init_gain_by_frequency(self):

        spec_gain = np.asarray(400)
        disp_par = self.dispersion_factor * 0.5e-3 * 2 * np.pi / spec_gain

        self.range_w = np.array([complex(v) for v in np.arange(-self.n_time_samples / 2, self.n_time_samples / 2)])
        ones = np.ones_like(self.range_w, dtype=complex)
        mid = self.range_w / spec_gain
        self.spectral_gain = ones / (mid ** 2 + 1)
        self.dispersion = np.exp(-1j * disp_par * self.range_w ** 2)
        exp_w = np.exp(-1j * 2 * np.pi * self.range_w)
        self.frequency_total_mult_factor = 0.5 * (1.0 + exp_w * self.spectral_gain * self.dispersion)
        self.modulator_gain = np.asarray([1.0 + self.modulation_gain_factor * np.cos(2 * np.pi * w / self.n_time_samples) for w in self.range_w])
        # print(f"modulator_gain = {self.modulator_gain[80]} {self.modulator_gain[180]} ")

    def get_init_front(self, p_par=-1):
        self.initial_range = 0.00024475293  # Example value, replace with actual value
        waist0 = p_par if p_par > 0.0 else 0.00003  # Example value, replace with actual value
        beam_dist = 0.0  # Example value, replace with actual value
        RayleighRange = np.pi * waist0 * waist0 / self.lambda_
        theta = np.asarray(0 if abs(beam_dist) < 0.000001 else np.pi / (self.lambda_ * beam_dist))
        waist = np.asarray(waist0 * np.sqrt(1 + beam_dist / RayleighRange))
        dx = np.asarray(self.initial_range / self.n_samples)
        x0 = np.asarray((self.n_samples - 1) / 2 * dx)

        x = np.arange(self.n_samples) * dx - x0
        xw = x / waist
        random_values = np.asarray(1.0) + np.asarray(0.3) * np.random.rand(self.n_samples)
        val_complex = -xw * xw - 1j * theta * x * x
        vf = np.exp(val_complex) * random_values

        return vf

    def update_helpData(self):
        self.prepare_linear_fresnel_help_data()
        self.prepare_gain_pump()
        self.prepare_aperture()
        self.init_gain_by_frequency()
    
    def init_multi_time(self):
        self.multi_time_fronts_saves = [[], [], [], [], [], []]
        self.n_rounds = 0
        self.steps_counter = 0

        random_lcg_set_seed(888) #1323
        multi_time_fronts_tr = np.empty((self.n_time_samples, self.n_samples), dtype=complex)
        for i_time in range(self.n_time_samples):
            #rnd = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            rnd = np.asarray((random_lcg() * 2 - 1) + 1j * (random_lcg() * 2 - 1))
            fr = np.multiply(rnd, self.get_init_front())
            multi_time_fronts_tr[i_time] = fr


        self.multi_time_fronts = multi_time_fronts_tr.T
        self.multi_time_fronts_saves[0] = np.copy(self.multi_time_fronts)

        self.update_helpData()

    def phase_change_during_kerr(self):
        total_kerr_lensing = np.multiply(self.lensing_factor, self.ikl_times_i)

        bin2 = np.square(np.abs(self.multi_time_fronts))
        self.sum_power_ix = np.sum(bin2, axis=1)
        phase_shift1 = total_kerr_lensing * bin2
        self.multi_time_fronts *= np.exp(phase_shift1)
        self.ps[self.side] = phase_shift1[:, self.view_on_x].imag

        self.multi_time_fronts_saves[self.side * 3 + 0] = np.copy(self.multi_time_fronts)

        multi_time_fronts_trans = self.multi_time_fronts.T
        p_fr_before = np.sum(np.square(np.abs(multi_time_fronts_trans)), axis=1, keepdims=True)
        fr_after = multi_time_fronts_trans * self.multi_time_aperture
        p_fr_after = np.sum(np.abs(fr_after)**2, axis=1, keepdims=True)
        multi_time_fronts_trans = fr_after * np.sqrt(p_fr_before / p_fr_after)
        self.multi_time_fronts = multi_time_fronts_trans.T


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
            multi_time_fronts_trans = vecF * fftshift(np.fft.fft(fftshift(vec0 * multi_time_fronts_trans, axes=1), axis=1), axes=1) * dx
        self.multi_time_fronts = multi_time_fronts_trans.T

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

    def get_x_values(self, sample):
        stage_data = self.select_source(sample).T
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_x:
            fr = cget(stage_data)[self.view_on_x].tolist()
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})"}
        return {}
    
    def get_y_values(self, sample):
        stage_data = self.select_source(sample)
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_y:
            fr = cget(stage_data)[self.view_on_y].tolist()
            fr_mean_and_std = list_mean_and_std(fr)
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})({fr_mean_and_std[0]:.4f}±{fr_mean_and_std[1]:.4f})"}
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
    
    def get_kerr_influence(self, batch, direction):
        if self.n_rounds < 10:
            return []
        if len(self.multi_time_fronts_saves[batch].T) <= self.view_on_x:
            return []
        fr_original = self.multi_time_fronts_saves[batch].T[self.view_on_x]
       
        total_kerr_lensing = np.multiply(self.lensing_factor, self.ikl_times_i)
        phase_shift1 = total_kerr_lensing * (np.abs(fr_original) ** 2)
        fr_with_kerr = fr_original * np.exp(phase_shift1)

        p_fr_original_before1 = np.sum(np.abs(fr_original)**2)
        fr_original_after1 = fr_original * self.multi_time_aperture
        p_fr_original_after1 = np.sum(np.abs(fr_original_after1)**2)
        fr_original1 = fr_original_after1 * np.sqrt(p_fr_original_before1 / p_fr_original_after1)
        p_fr_with_kerr_before1 = np.sum(np.abs(fr_with_kerr)**2)
        fr_with_kerr_after1 = fr_with_kerr * self.multi_time_aperture
        p_fr_with_kerr_after1 = np.sum(np.abs(fr_with_kerr_after1)**2)
        fr_with_kerr1 = fr_with_kerr_after1 * np.sqrt(p_fr_with_kerr_before1 / p_fr_with_kerr_after1)

        fr_after = []
        for fr in [fr_original1, fr_with_kerr1]:
            fr_next = np.copy(fr)
            for fresnel_side_data in self.fresnel_data[direction]:
                vec0 = fresnel_side_data['vecs'][0]
                vecF = fresnel_side_data['vecs'][1]
                dx = fresnel_side_data['dx']
                fr_next = vecF * fftshift(np.fft.fft(fftshift(vec0 * fr_next))) * dx

            fr_after.append(cget(np.abs(fr_next)).tolist())
        
        return [{"color": "black", "values": cget(np.abs(fr_original)).tolist(), "text": f"Start"},
                {"color": "purple", "values": cget(np.abs(fr_original1)).tolist(), "text": f"squeeze({max(np.abs(fr_original1)):.2f})"},
                {"color": "green", "values": fr_after[1], "text": f"with Kerr({max(fr_after[1]):.2f})"},
                {"color": "red", "values": fr_after[0], "text": f"without Kerr({max(fr_after[0]):.2f})"}]
        

    def serialize_mm_graphs_data(self):
        if self.n_rounds < 10:
            return []
        sample = self.view_on_sample
        ps = [cget(self.ps[0]), cget(self.ps[1])]
        psr = ps[sample]
        psb = ps[1 - sample]
        # if isinstance(psr, list):
        #     print(f"psr type {type(psr)} {len(psr)}")
        #     print(f"psb type {type(psb)} {len(psb)}")
        if isinstance(psr, nump.ndarray if platform.system() == "Linux" else np.ndarray):
            # print(f"psr type {type(psr)} {psr.shape}")
            # print(f"psb type {type(psb)} {psb.shape}")
            psr = cget(psr).tolist()
            psb = cget(psb).tolist()
        psr_mean_and_std = list_mean_and_std(psr)
        psb_mean_and_std = list_mean_and_std(psb)
        # print(f"psr type {type(psr)} {len(psr)}")
        # print(f"psb type {type(psb)} {len(psb)}")
        s = [
                {"name": "gr1", "lines": self.get_kerr_influence(0, 0)},
                {"name": "gr2", "lines": self.get_kerr_influence(3, 1)},
                {"name": "gr3", "lines": [self.get_x_values(sample),
                                          self.get_x_values(1 - sample)]},
                {"name": "gr4", "lines": [{"color": ["red", "blue"][sample], "values": psr, 
                                           "text": f"M{max(psr):.4f}({psr.index(max(psr))})"},
                                          {"color": ["red", "blue"][1 - sample], "values": psb, 
                                           "text": f"M{max(psb):.4f}({psb.index(max(psb))})"} ] 
                                          if len(ps[0]) > 10 and len(ps[1]) > 10 else []},    
                {"name": "gr5", "lines": [self.get_y_values(sample),
                                          self.get_y_values(1 - sample)]},
            ]
        return s

    def serialize_mm_data(self, more):
        s = json.dumps({
            "more": more,
            "rounds": self.n_rounds,
            "pointer": [self.view_on_sample, self.view_on_x, self.view_on_y],
            "samples": 
                [
                    {"name": "funCanvasSample1", "samples": serialize_fronts(self.select_source(0))},
                    {"name": "funCanvasSample2", "samples": serialize_fronts(self.select_source(1))}
                ],
            "graphs": self.serialize_mm_graphs_data(),
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
            "graphs": self.serialize_mm_graphs_data(),
            })
        return s