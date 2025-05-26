# try:
#     import cupy
#     if cupy.cuda.is_available():
#         np = cupy
#     else:
#         import numpy as np
# except ImportError:
#     import numpy as np

import torch

# Set device (CPU or CUDA if available)
if torch.backends.mps.is_available():
    device = torch.device("cpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

from controls import random_lcg_set_seed, random_lcg
import random
import json

from scipy.special import j0  # No direct torch equivalent

def fftshift(x, dim=-1):
    n = x.shape[dim]
    p2 = (n + 1) // 2
    return torch.roll(x, shifts=p2, dims=dim)

def ifftshift(x, dim=-1):
    n = x.shape[dim]
    p2 = n // 2
    return torch.roll(x, shifts=-p2, dims=dim)

def vectors_for_linear_fresnel(lambda_, M, N, dx0, gain, is_back):
    A, B = M[0]
    C, D = M[1]
    dxf = B * lambda_ / (N * dx0)
    factor = 1j * gain * torch.sqrt(torch.tensor(-1j / (B * lambda_), dtype=torch.complex64, device=device))
    if is_back:
        factor *= 1j * gain
    co0 = -torch.pi * dx0 * dx0 * A / (B * lambda_)
    cof = -torch.pi * dxf * dxf * D / (B * lambda_)

    idx = torch.arange(N, device=device)
    vec0 = torch.exp(1j * co0 * (idx - N / 2) ** 2)
    vecF = factor * torch.exp(1j * cof * (idx - N / 2) ** 2)

    return {'dx': torch.tensor(dx0, device=device), 'vecs': [vec0, vecF]}

def prepare_linear_fresnel_calc_data(mat, dx0, n_samples, lambda_, loss):
    A, B, C, D = mat[0][0], mat[0][1], mat[1][0], mat[1][1]
    if A > 0:
        M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]]
        M1 = [[1, B / (A + 1)], [0, 1]]
    else:
        M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]]
        M1 = [[-1, B / (-A + 1)], [0, -1]]

    print(f"M1 = {M1[0][0]:11.6f}, {M1[0][1]:11.6f}, {M1[1][0]:11.6f}, {M1[1][1]:11.6f} dx = {M1[0][1] * lambda_ / (n_samples * dx0)} dx0 = {dx0}")
    print(f"M2 = {M2[0][0]:11.6f}, {M2[0][1]:11.6f}, {M2[1][0]:11.6f}, {M2[1][1]:11.6f}")
    fresnel_data = []
    dx = dx0
    for index, M in enumerate([M1, M2]):
        fresnel_data.append(vectors_for_linear_fresnel(lambda_, M, n_samples, dx, loss if index == 0 else 1.0, M[0][0] < 0))
        dx = M[0][1] * lambda_ / (n_samples * dx)
        print(f"---- dx = {dx} dx0 = {dx0}")

    return fresnel_data

def prepare_linear_fresnel_straight_calc_data(mat, dx0, n_samples, lambda_, loss):
    fresnel_data = []
    dx = dx0
    fresnel_data.append(vectors_for_linear_fresnel(lambda_, mat, n_samples, dx, 1.0, mat[0][0] < 0))
    dx = mat[0][1] * lambda_ / (n_samples * dx)
    print(f"---- dx = {dx} dx0 = {dx0}")

    return fresnel_data

def linear_fresnel_propogate(fresnel_data, multi_time_fronts_trans):
    for fresnel_step_data in fresnel_data:
        vec0 = fresnel_step_data['vecs'][0]
        vecF = fresnel_step_data['vecs'][1]
        dx = fresnel_step_data['dx']
        # Broadcasting vec0 and vecF if needed
        if multi_time_fronts_trans.ndim == 2:
            vec0 = vec0.unsqueeze(0)
            vecF = vecF.unsqueeze(0)
        multi_time_fronts_trans = vecF * fftshift(torch.fft.fft(fftshift(vec0 * multi_time_fronts_trans, dim=1), dim=1), dim=1) * dx
    return multi_time_fronts_trans

def cylindrical_fresnel_prepare(r_in, r_out, wavelength, M):
    # r_in, r_out: torch tensors
    A, B = M[0]
    C, D = M[1]
    k = torch.tensor(2 * torch.pi / wavelength, device=device, dtype=torch.float32)

    dr = r_out[1] - r_out[0]
    r1 = r_in.reshape(1, -1).cpu().numpy()
    r2 = r_out.reshape(-1, 1).cpu().numpy()
    kernel = j0(k.item() * r1 * r2 / B).T  # Use k.item() to get the float value
    kernel = torch.from_numpy(kernel).to(device=device, dtype=torch.complex64)
    phase_input = torch.exp(1j * k * A * r_in / (2 * B) * r_in)
    factor_input = phase_input * r_in
    phase_output = torch.exp(1j * k * D * r_out / (2 * B) * r_out)
    factor_output = 2 * torch.pi * dr * phase_output / (1j * wavelength * B)

    new_kernel = torch.diag(factor_output) @ kernel @ torch.diag(factor_input)
    return new_kernel, kernel

def cylindrical_fresnel_preparekeep(r, r_out, wavelength, M):
    A, B = M[0]
    C, D = M[1]
    k = 2 * torch.pi / wavelength

    dr = r[1] - r[0]
    r1 = r.reshape(1, -1).cpu().numpy()
    r2 = r_out.reshape(-1, 1).cpu().numpy()
    kernel = j0(k.cpu().numpy() * r1 * r2 / B).T
    kernel = torch.from_numpy(kernel).to(device)

    phase_input = torch.exp(1j * k * A * r / (2 * B) * r)
    factor_input = phase_input * r
    phase_output = torch.exp(1j * k * D * r / (2 * B) * r)
    factor_output = 2 * torch.pi * dr * phase_output / (1j * wavelength * B)

    new_kernel = torch.diag(factor_output) @ kernel @ torch.diag(factor_input)
    return new_kernel, kernel

def cylindrical_fresnel_propogate(fronts, kernel):
    return kernel @ fronts

def cget(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else x

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
    position_lens = -0.00015 + crystal_shift
    m_long = m_mult_v(m_dist(position_lens), m_dist(0.081818181), m_lens(0.075), m_dist(0.9),
                        m_dist(0.9), m_lens(0.075), m_dist(0.081818181), m_dist(position_lens))
    m_short = m_mult_v(m_dist(0.001 - position_lens), m_dist(0.075), m_lens(0.075), m_dist(0.5),
                        m_dist(0.5), m_lens(0.075), m_dist(0.075), m_dist(0.001 - position_lens))

    mat_side = [m_short, m_long]
    print(f"MShort = {m_short[0][0]:11.6f}, {m_short[0][1]:11.6f}, {m_short[1][0]:11.6f}, {m_short[1][1]:11.6f}")
    print(f" MLong = {m_long[0][0]:11.6f}, {m_long[0][1]:11.6f}, {m_long[1][0]:11.6f}, {m_long[1][1]:11.6f}")

    return mat_side

def list_mean_and_std(weights):
    indices = torch.arange(len(weights), device=device, dtype=torch.float32)
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    mean_index = torch.sum(indices * weights) / torch.sum(weights)
    variance = torch.sum(weights * (indices - mean_index) ** 2) / torch.sum(weights)
    std_index = torch.sqrt(variance)
    return float(mean_index.cpu()), float(std_index.cpu())

def serialize_fronts(fs):
    getfs = cget(fs)
    s = [[f"{val:.2f}" if abs(val) > 0.001 else "" for val in row] for row in getfs]
    return s

class MultiModeSimulation:
    def __init__(self):
        self.beam_type = 1
        self.modulation_gain_factor = torch.tensor(0.0, device=device)
        self.gain_factor = torch.tensor(0.50, device=device)
        self.epsilon = torch.tensor(5.0, device=device)
        self.dispersion_factor = torch.tensor(1.5, device=device)
        self.lensing_factor = torch.tensor(0.8, device=device)
        self.is_factor = torch.tensor(10000, device=device)
        self.crystal_shift = torch.tensor(0.00009, device=device)
        self.aperture = torch.tensor(0.000156, device=device)
        self.diffraction_waist = torch.tensor(0.000060, device=device)
        self.initial_range = 0.0001

        self.n_rounds_per_full = 1
        self.lambda_ = 0.000000780
        self.multi_ranges = [[], []]
        self.n_samples = 256
        self.n_max_matrices = 1000
        self.n_rounds = 0
        self.seed = 0
        self.used_seed = 0

        self.steps_counter = 0
        self.n_time_samples = 1024
        self.multi_time_fronts = []
        self.multi_time_fronts_saves = []

        self.intensity_saturation_level = 400000000000000.0
        self.intensity_total_by_ix = []
        self.factor_gain_by_ix = []
        self.ikl = 0.02
        self.ikl_times_i = torch.tensor(1j * self.ikl * 160 * 0.000000006, device=device)
        self.x = []
        self.range_w = []
        self.spectral_gain = []
        self.modulator_gain = []
        self.dispersion = []
        self.sum_power_ix = [torch.zeros(self.n_samples, device=device), torch.zeros(self.n_samples, device=device)]
        self.two_sided_sum_power_ix = torch.zeros(self.n_samples, device=device)

        self.gain_reduction = []
        self.gain_reduction_with_origin = []
        self.gain_reduction_after_aperture = []
        self.gain_reduction_after_diffraction = []

        self.pump_gain0 = []
        self.multi_time_aperture = []
        self.multi_time_diffraction = []
        self.frequency_total_mult_factor = []
        self.mirror_loss = torch.tensor(0.95, device=device)
        self.fresnel_data = []
        self.dx0 = self.initial_range / self.n_samples
        self.scalar_one = torch.tensor(1 + 0j, device=device)
        self.n_time_samples_ones = torch.ones(self.n_time_samples, dtype=torch.complex64, device=device)
        self.n_samples_ones = torch.ones(self.n_samples, dtype=torch.complex64, device=device)
        self.n_samples_ones_r = torch.ones(self.n_samples, dtype=torch.float32, device=device)
        self.ps = [[], []]
        self.view_on_stage = ["1", "1"]
        self.view_on_amp_freq = ["Amp", "Amp"]
        self.view_on_abs_phase = ["Abs", "Abs"]
        self.view_on_x = self.n_time_samples // 2
        self.view_on_y = self.n_samples // 2
        self.view_on_sample = 0

        self.mat_side = calc_original_sim_matrices()
        self.prepare_x()

    def printSamples(self, name = "", sample = None):
        if sample is None:
            sample = self.multi_time_fronts
        print("----------------------------------", name, "----------------------------------")
        power = self.front_power(sample.T)
        sample = cget(sample)
        print(f"{sample[0][63]}     --- abs = {abs(sample[0][63])} --- power = {power[63]}") 
        print(f"{sample[0][163]}    --- abs = {abs(sample[0][163])} --- power = {power[163]}")
        print(f"{sample[0][263]}    --- abs = {abs(sample[0][263])} --- power = {power[263]}")

    def set(self, params):
        print(params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, torch.tensor(value, device=device))

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
        return params

    def spectral_gain_dispersion(self):
        multi_frequency_fronts = fftshift(torch.fft.fft(ifftshift(self.multi_time_fronts, dim=1), dim=1), dim=1) * self.frequency_total_mult_factor
        self.multi_time_fronts = fftshift(torch.fft.ifft(ifftshift(multi_frequency_fronts, dim=1), dim=1), dim=1)
        self.multi_time_fronts_saves[self.side * 7 + 3] = self.multi_time_fronts.clone()

    def modulator_gain_multiply(self):
        if self.side == 1:
            self.multi_time_fronts = self.multi_time_fronts * self.modulator_gain
        self.multi_time_fronts_saves[self.side * 7 + 4] = self.multi_time_fronts.clone()

    def prepare_x(self):
        self.dx0 = self.initial_range / self.n_samples
        print(f"++++ B dx0 = {self.dx0}")

        if self.beam_type == 0:
            vec = torch.arange(self.n_samples, device=device) - (self.n_samples / 2)
        else:
            vec = torch.arange(self.n_samples, device=device) + 0.5

        self.x = vec * self.dx0

    def prepare_gain_pump(self):
        pump_width = 0.000030 * 0.5
        g0 = 1 / self.mirror_loss + self.epsilon
        xw = self.x / pump_width
        self.pump_gain0 = g0 * torch.exp(-xw * xw)

    def prepare_aperture(self):
        aperture_width = self.aperture * 0.5
        self.multi_time_aperture = torch.exp(- (self.x / aperture_width) ** 2)

        diffraction_width = self.diffraction_waist
        self.multi_time_diffraction = torch.exp(- (self.x / diffraction_width) ** 2)

    def prepare_linear_fresnel_help_data(self):
        self.mat_side = calc_original_sim_matrices(float(self.crystal_shift))

        self.fresnel_data = []

        for index_side, side_m in enumerate(self.mat_side):
            dx = self.dx0

            fresnel_side_data = prepare_linear_fresnel_calc_data(side_m, dx, self.n_samples, self.lambda_, 
                                             float(self.mirror_loss) if index_side == 1 else 1)
            
            self.fresnel_data.append(fresnel_side_data)

    def prepare_cylindrical_fresnel_help_data(self):
        self.mat_side = calc_original_sim_matrices(float(self.crystal_shift))

        self.fresnel_data, _ = map(list, zip(*[cylindrical_fresnel_prepare(self.x, self.x, self.lambda_, mat) for mat in self.mat_side]))

    def init_gain_by_frequency(self):
        spec_gain = 400
        disp_par = float(self.dispersion_factor) * 0.5e-3 * 2 * torch.pi / spec_gain

        self.range_w = torch.arange(-self.n_time_samples // 2, self.n_time_samples // 2, device=device, dtype=torch.float32)
        ones = torch.ones_like(self.range_w, dtype=torch.complex64)
        mid = self.range_w / spec_gain
        self.spectral_gain = ones / (mid ** 2 + 1)
        self.dispersion = torch.exp(-1j * disp_par * self.range_w ** 2)
        exp_w = torch.exp(-1j * 2 * torch.pi * self.range_w)
        self.frequency_total_mult_factor = 0.5 * (1.0 + exp_w * self.spectral_gain * self.dispersion)
        self.modulator_gain = torch.tensor([1.0 + float(self.modulation_gain_factor) * torch.cos(2 * torch.pi * w / self.n_time_samples) for w in self.range_w], device=device)

    def get_init_front(self, p_par=-1):
        waist0 = p_par if p_par > 0.0 else 0.00003
        beam_dist = 0.0
        RayleighRange = torch.pi * waist0 * waist0 / self.lambda_
        theta = 0 if abs(beam_dist) < 0.000001 else torch.pi / (self.lambda_ * beam_dist)
        waist = waist0 * torch.sqrt(torch.tensor(1 + beam_dist / RayleighRange))
        random_values = torch.tensor(1.0, device=device) + torch.tensor(0.3, device=device) * torch.rand(self.n_samples, device=device)
        val_complex = - (self.x / waist) ** 2 - 1j * theta * (self.x ** 2)
        vf = torch.exp(val_complex) * random_values
        return vf

    def update_helpData(self):
        if self.beam_type == 0:
            self.prepare_linear_fresnel_help_data()
        else:
            self.prepare_cylindrical_fresnel_help_data()

        self.prepare_gain_pump()
        self.prepare_aperture()
        self.init_gain_by_frequency()
    
    def init_multi_time(self):
        self.prepare_x()

        self.multi_time_fronts_saves = [[]] * 14
        self.n_rounds = 0
        self.steps_counter = 0

        if self.seed != -1:
            self.used_seed = self.seed
        else:
            self.used_seed = random.randint(1, 1000000)
        random_lcg_set_seed(self.used_seed)
        multi_time_fronts_tr = torch.empty((self.n_time_samples, self.n_samples), dtype=torch.complex64, device=device)
        for i_time in range(self.n_time_samples):
            rnd = torch.tensor((random_lcg() * 2 - 1) + 1j * (random_lcg() * 2 - 1), dtype=torch.complex64, device=device)
            fr = rnd * self.get_init_front()
            multi_time_fronts_tr[i_time] = fr

        self.multi_time_fronts = multi_time_fronts_tr.T
        self.multi_time_fronts_saves[0] = self.multi_time_fronts.clone()

        self.update_helpData()

    def front_power(self, bin_field):
        bin_intencity = torch.abs(bin_field) ** 2
        if (self.beam_type == 1):
            bin_intencity = bin_intencity * self.x
        return torch.sum(bin_intencity, dim=1, keepdim=True)
    
    def phase_change_during_kerr(self):
        self.multi_time_fronts_saves[self.side * 7 + 0] = self.multi_time_fronts.clone()

        total_kerr_lensing = self.lensing_factor * self.ikl_times_i

        bin2 = torch.abs(self.multi_time_fronts) ** 2
        self.sum_power_ix[self.side] = torch.sum(bin2, dim=1)
        phase_shift1 = total_kerr_lensing * bin2
        self.multi_time_fronts *= torch.exp(phase_shift1)
        self.ps[self.side] = phase_shift1[:, self.view_on_x].imag.cpu().numpy()

        self.multi_time_fronts_saves[self.side * 7 + 1] = self.multi_time_fronts.clone()

        multi_time_fronts_trans = self.multi_time_fronts.T
        p_fr_before = self.front_power(multi_time_fronts_trans)
        fr_after = multi_time_fronts_trans * self.multi_time_aperture
        p_fr_after = self.front_power(fr_after)
        multi_time_fronts_trans = fr_after * torch.sqrt(p_fr_before / p_fr_after)
        self.multi_time_fronts = multi_time_fronts_trans.T
        self.multi_time_fronts_saves[self.side * 7 + 2] = self.multi_time_fronts.clone()

    def linear_cavity_one_side(self):
        if True:
            self.two_sided_sum_power_ix = self.sum_power_ix[self.side]
        else:
            self.two_sided_sum_power_ix = (self.sum_power_ix[0] + self.sum_power_ix[1]) * 0.5

        Is = self.is_factor
        self.gain_reduction = torch.real(self.pump_gain0 * (self.n_samples_ones / (1 + (self.two_sided_sum_power_ix / (Is * self.n_time_samples)))))
        
        if True:
            self.gain_reduction_with_origin = self.gain_factor * (1 + self.gain_reduction)
            self.gain_reduction_after_diffraction = self.gain_reduction_with_origin * self.multi_time_diffraction
        else:
            self.gain_reduction_after_diffraction = self.gain_reduction * self.multi_time_diffraction
            self.gain_reduction_with_origin = 1 + self.gain_factor * self.gain_reduction_after_diffraction

        multi_time_fronts_trans = self.multi_time_fronts.T
        gain_factors = self.gain_reduction_after_diffraction
        multi_time_fronts_trans = gain_factors * multi_time_fronts_trans
        self.multi_time_fronts_saves[self.side * 7 + 5] = multi_time_fronts_trans.T.clone()

        if self.beam_type == 0:
            self.multi_time_fronts = linear_fresnel_propogate(self.fresnel_data[self.side], multi_time_fronts_trans).T
        else:
            self.multi_time_fronts = cylindrical_fresnel_propogate(multi_time_fronts_trans.T, self.fresnel_data[self.side])

        self.multi_time_fronts_saves[self.side * 7 + 6] = self.multi_time_fronts.clone()

    def multi_time_round_trip(self):
        self.n_rounds += 1

        for self.side in [0, 1]:
            self.phase_change_during_kerr()
            self.spectral_gain_dispersion()
            self.modulator_gain_multiply()
            self.linear_cavity_one_side()

    def center_multi_time(self):
        a = torch.sum(torch.abs(self.multi_time_fronts) ** 2, dim=0)
        print(f"length of a = {len(a)}")
        index = torch.argmax(a).item()
        print(f"index = {index}")
        roll = -index + self.n_time_samples // 2
        self.multi_time_fronts = torch.roll(self.multi_time_fronts, shifts=roll, dims=1)
        for i in range(len(self.multi_time_fronts_saves)):
            self.multi_time_fronts_saves[i] = torch.roll(self.multi_time_fronts_saves[i], shifts=roll, dims=1)

    def get_x_values(self, sample):
        source = self.select_source(sample)
        if source is None:
            return {}
        stage_data = source.T
        if isinstance(stage_data, torch.Tensor) and len(stage_data) > self.view_on_x:
            fr = cget(stage_data)[self.view_on_x].tolist()
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})"}
        return {}
    
    def get_y_values(self, sample):
        source = self.select_source(sample)
        if source is None:
            return {}
        stage_data = source
        if isinstance(stage_data, torch.Tensor) and len(stage_data) > self.view_on_y:
            fr = cget(stage_data)[self.view_on_y].tolist()
            fr_mean_and_std = list_mean_and_std(fr)
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})({fr_mean_and_std[0]:.4f}±{fr_mean_and_std[1]:.4f})"}
        return {}
    
    def get_x_values_full(self, sample):
        source = self.select_source(sample, True)
        if source is None:
            return []
        stage_data = source.T
        if isinstance(stage_data, torch.Tensor) and len(stage_data) > self.view_on_x:
            fr = cget(stage_data)[self.view_on_x]
            return fr
        return []
    
    def select_source(self, target, original=False):
        fronts_index = int(self.view_on_stage[target]) - 1
        if (fronts_index >= len(self.multi_time_fronts_saves)):
            return None
        stage_data = self.multi_time_fronts_saves[fronts_index]
        if (len(stage_data) == 0):
            return None
        if not original:
            if (self.view_on_amp_freq[target] == "Frq"):
                stage_data = fftshift(torch.fft.fft(ifftshift(stage_data, dim=1), dim=1), dim=1)
            if (self.view_on_abs_phase[target] == "Abs"):
                stage_data = torch.abs(stage_data)
            elif (self.view_on_abs_phase[target] == "Pow"):
                stage_data = torch.abs(stage_data) ** 2
            else:
                stage_data = torch.angle(stage_data)
        return stage_data
    
    def focus_front(self, fr):
        if (self.beam_type == 0):
            q = len(fr) // 4
            x_original = torch.linspace(0, 1, len(fr) // 2, device=device)
            x_new = torch.linspace(0, 1, len(fr), device=device)
            return torch.interp(x_new, x_original, fr[q:3*q])
        else:
            q = len(fr) // 2
            a = fr[:q]
            return torch.cat((a.flip(0), a))
        
    def get_kerr_influence(self, batch, direction):
        if (self.beam_type != 0):
            return []
        if self.n_rounds < 10:
            return []
        if len(self.multi_time_fronts_saves[batch].T) <= self.view_on_x:
            return []
        fr_original = self.multi_time_fronts_saves[batch].T[self.view_on_x]
       
        total_kerr_lensing = self.lensing_factor * self.ikl_times_i
        phase_shift1 = total_kerr_lensing * (torch.abs(fr_original) ** 2)
        fr_with_kerr = fr_original * torch.exp(phase_shift1)

        p_fr_original_before1 = torch.sum(torch.abs(fr_original)**2)
        fr_original_after1 = fr_original * self.multi_time_aperture
        p_fr_original_after1 = torch.sum(torch.abs(fr_original_after1)**2)
        fr_original1 = fr_original_after1 * torch.sqrt(p_fr_original_before1 / p_fr_original_after1)
        p_fr_with_kerr_before1 = torch.sum(torch.abs(fr_with_kerr)**2)
        fr_with_kerr_after1 = fr_with_kerr * self.multi_time_aperture
        p_fr_with_kerr_after1 = torch.sum(torch.abs(fr_with_kerr_after1)**2)
        fr_with_kerr1 = fr_with_kerr_after1 * torch.sqrt(p_fr_with_kerr_before1 / p_fr_with_kerr_after1)

        fr_after = []
        for fr in [fr_original1, fr_with_kerr1]:
            fr_next = fr.clone()
            for fresnel_side_data in self.fresnel_data[direction]:
                vec0 = fresnel_side_data['vecs'][0]
                vecF = fresnel_side_data['vecs'][1]
                dx = fresnel_side_data['dx']
                fr_next = vecF * fftshift(torch.fft.fft(fftshift(vec0 * fr_next)), dim=0) * dx

            fr_after.append(cget(torch.abs(fr_next)).tolist())
        
        return [{"color": "black", "values": cget(torch.abs(fr_original)).tolist(), "text": f"Start"},
                {"color": "purple", "values": cget(torch.abs(fr_original1)).tolist(), "text": f"squeeze({max(torch.abs(fr_original1)).item():.2f})"},
                {"color": "green", "values": fr_after[1], "text": f"with Kerr({max(fr_after[1]):.2f})"},
                {"color": "red", "values": fr_after[0], "text": f"without Kerr({max(fr_after[0]):.2f})"}]
        
    def get_saturation_graph_data(self):
        return [{"color": "red", "values": cget(self.focus_front(self.pump_gain0)).tolist(), "text": f"pump_gain0"},
                {"color": "blue", "values": cget(self.focus_front(self.gain_reduction)).tolist(), "text": f"gain_reduction"},
        ]

    def get_diffraction_graph_data(self):
        return [{"color": "red", "values": cget(self.focus_front(self.gain_reduction_with_origin)).tolist(), "text": f"before diffraction"},
                {"color": "blue", "values": cget(self.focus_front(self.gain_reduction_after_diffraction)).tolist(), "text": f"after diffraction"},
        ]

    def serialize_mm_graphs_data(self):
        if self.n_rounds < 1:
            print("No graph data to serialize")
            return []
        sample = self.view_on_sample
        ps = [cget(self.ps[0]), cget(self.ps[1])]
        psr = ps[sample]
        psb = ps[1 - sample]
        psr = cget(psr).tolist()
        psb = cget(psb).tolist()
        s = [
                {"name": "gr1", "lines": self.get_saturation_graph_data()},
                {"name": "gr2", "lines": self.get_diffraction_graph_data()},
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

    def serialize_mm_fronts_data(self):
        samples = []
        for i in range(2):
            source = self.select_source(i)
            if source is not None:
                sample = serialize_fronts(source)
                samples.append({"name": f"funCanvasSample{i + 1}", "samples": sample})
        return samples

    def serialize_mm_data(self, delay, more):
        data = {
            "delay": delay,
            "more": more,
            "rounds": self.n_rounds,
            "pointer": [self.view_on_sample, self.view_on_x, self.view_on_y],
            "samples": self.serialize_mm_fronts_data(),
            "graphs": self.serialize_mm_graphs_data(),
            "view_buttons": 
                {
                    "view_on_stage": self.view_on_stage,
                    "view_on_amp_freq": self.view_on_amp_freq,
                    "view_on_abs_phase": self.view_on_abs_phase,
                }    
            
        }
        try:
            s = json.dumps(data)
        except TypeError as e:
            print("error: json exeption", data)
            s = ""
        return s
    def serialize_mm_graphs(self):
        s = json.dumps({
            "pointer": [self.view_on_sample, self.view_on_x, self.view_on_y],
            "graphs": self.serialize_mm_graphs_data(),
            })
        return s
