try:
    import cupy
    if cupy.cuda.is_available():
        np = cupy
    else:
        import numpy as np
except ImportError:
    import numpy as np

from controls import random_lcg_set_seed, random_lcg
import random
import json

from scipy.special import j0


# rrrr need fix (skip)
def vectors_for_linear_fresnel(lambda_, M, N, dx0, gain, is_back):
    A, B = M[0]
    C, D = M[1]
    #print(A, B, C, D, self.lambda_)
    dxf = B * lambda_ / (N * dx0)
    factor = 1j * gain * np.sqrt(-1j / (B * lambda_))
    if is_back:
        factor *= 1j * gain
    co0 = -np.pi * dx0 * dx0 * A / (B * lambda_)
    cof = -np.pi * dxf * dxf * D / (B * lambda_)

    vec0 = np.asarray([np.exp(1j * co0 * (i - N / 2) ** 2) for i in range(N)])
    vecF = np.asarray([factor * np.exp(1j * cof * (i - N / 2) ** 2) for i in range(N)])

    return {'dx': np.asarray(dx0), 'vecs': [vec0, vecF]}

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

def linear_fresnel_propogate(fresnel_data, multi_time_fronts_trans):

    for fresnel_step_data in fresnel_data:
        vec0 = fresnel_step_data['vecs'][0]
        vecF = fresnel_step_data['vecs'][1]
        dx = fresnel_step_data['dx']
        multi_time_fronts_trans = vecF * np.fft.fftshift(np.fft.fft(np.fft.fftshift(vec0 * multi_time_fronts_trans, axes=1), axis=1), axes=1) * dx
    
    return multi_time_fronts_trans

def cylindrical_fresnel_prepare(r_in, r_out, wavelength, M):
    A, B = M[0]
    C, D = M[1]
    k = 2 * np.pi / wavelength

    # Assume linear spacing of r
    dr = r_out[1] - r_out[0]
    
    # Precompute kernel matrix: J0(k r1 r2 / B)
    r1 = r_in.reshape(1, -1)
    r2 = r_out.reshape(-1, 1)
    kernel = j0(k * r1 * r2 / B).T  # shape (N_r, N_r)

    # Precompute phase and prefactor
    phase_input = np.exp(1j * k * A * r_in / (2 * B) * r_in)        # shape (N_r,)
    factor_input = phase_input * r_in
    phase_output = np.exp(1j * k * D * r_out / (2 * B) * r_out)        # shape (N_r,)
    factor_output = 2 * np.pi * dr * phase_output / (1j * wavelength * B)           # shape (N_r,)
    # print(f"factor_input = {factor_input}")
    # print(f"factor_output = {factor_output}")

    new_kernel = np.diag(factor_output) @ kernel @ np.diag(factor_input)  # shape (N_r, N_r)
    # print(f"new_kernel = {new_kernel}")
    #return np.asarray(kernel), np.asarray(factor_input), np.asarray(factor_output)
    return np.asarray(new_kernel), np.asarray(kernel)

def cylindrical_fresnel_preparekeep(r, r_out, wavelength, M):
    A, B = M[0]
    C, D = M[1]
    k = 2 * np.pi / wavelength

    # Assume linear spacing of r
    dr = r[1] - r[0]
    
    # Precompute kernel matrix: J0(k r1 r2 / B)
    r1 = r.reshape(1, -1)
    r2 = r_out.reshape(-1, 1)
    kernel = j0(k * r1 * r2 / B).T  # shape (N_r, N_r)

    # Precompute phase and prefactor
    phase_input = np.exp(1j * k * A * r / (2 * B) * r)        # shape (N_r,)
    factor_input = phase_input * r
    phase_output = np.exp(1j * k * D * r / (2 * B) * r)        # shape (N_r,)
    factor_output = 2 * np.pi * dr * phase_output / (1j * wavelength * B)           # shape (N_r,)
    # print(f"factor_input = {factor_input}")
    # print(f"factor_output = {factor_output}")

    new_kernel = np.diag(factor_output) @ kernel @ np.diag(factor_input)  # shape (N_r, N_r)
    # print(f"new_kernel = {new_kernel}")
    #return np.asarray(kernel), np.asarray(factor_input), np.asarray(factor_output)
    return np.asarray(new_kernel), np.asarray(kernel)

def cylindrical_fresnel_propogate(fronts, kernel):
    #kernel, factor_input, factor_output = params
    
    # U1_weighted = fronts * factor_input                 # shape (N_profiles, N_r)

    # # Matrix multiplication: (N_profiles, N_r) @ (N_r, N_r)
    # U2_batch = U1_weighted @ kernel                     # shape (N_profiles, N_r)

    # # Multiply each row of result by factor_output
    # return U2_batch * factor_output


    return kernel @ fronts

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
    print(f"MShort = {m_short[0][0]:11.6f}, {m_short[0][1]:11.6f}, {m_short[1][0]:11.6f}, {m_short[1][1]:11.6f}")
    print(f" MLong = {m_long[0][0]:11.6f}, {m_long[0][1]:11.6f}, {m_long[1][0]:11.6f}, {m_long[1][1]:11.6f}")

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

        self.beam_type = 0 # 0 - 1D line, 1 - radial
        self.modulation_gain_factor = np.asarray(0.0)
        self.gain_factor = np.asarray(0.50)
        self.epsilon = np.asarray(1.8)
        self.dispersion_factor = np.asarray(0.45)
        self.lensing_factor = np.asarray(1.0)
        self.is_factor = np.asarray(15000)
        self.crystal_shift = np.asarray(0.0001)
        self.aperture = np.asarray(0.000056)
        self.diffraction_waist = np.asarray(0.000030)

        self.n_rounds_per_full = 1
        self.lambda_ = 0.000000780
        self.initial_range = 0.001 # 0.00024475293
        self.multi_ranges = [[], []]
        # rrrr need fix (ok)
        #self.n_samples = 256 if self.beam_type == 0 else 128
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
        self.ikl_times_i = np.asarray(1j * self.ikl * 160 * 0.000000006)
        self.x = []
        self.range_w = []
        self.spectral_gain = []
        self.modulator_gain = []
        self.dispersion = []
        self.sum_power_ix = [np.asarray([0] * self.n_samples), np.asarray([0] * self.n_samples)]
        self.two_sided_sum_power_ix = np.asarray([0] * self.n_samples)

        self.gain_reduction = []
        self.gain_reduction_with_origin = []
        self.gain_reduction_after_aperture = []
        self.gain_reduction_after_diffraction = []

        self.pump_gain0 = []
        self.multi_time_aperture = []
        self.multi_time_diffraction = []
        self.frequency_total_mult_factor = []
        self.mirror_loss = np.asarray(0.95)
        self.fresnel_data = []
        #self.total_range = 0.001000
        self.dx0 = self.initial_range / self.n_samples
        print(f"++++ A dx0 = {self.dx0}")
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

        #self.mat_side = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  # right
        #                 [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]]    # left

        self.mat_side = calc_original_sim_matrices()
        self.prepare_x()

    def printSamples(self, name = "", sample = None):
        if sample is None:
            sample = self.multi_time_fronts
        print("----------------------------------", name, "----------------------------------")
        power = self.front_power(sample.T)
        sample = cget(sample)
        print(f"{sample[0][63]}     --- np.abs = {np.abs(sample[0][63])} --- power = {power[63]}") 
        print(f"{sample[0][163]}    --- np.abs = {np.abs(sample[0][163])} --- power = {power[163]}")
        print(f"{sample[0][263]}    --- np.abs = {np.abs(sample[0][263])} --- power = {power[263]}")

    def set(self, params):
        print(params)
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, np.asarray(value))

    def get(self, params):
        for key in params.keys():
            if hasattr(self, key):
                params[key] = getattr(self, cget(key)[0])
        return params

    def spectral_gain_dispersion(self):
        multi_frequency_fronts = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(self.multi_time_fronts, axes=1), axis=1), axes=1) * self.frequency_total_mult_factor
        self.multi_time_fronts = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(multi_frequency_fronts, axes=1), axis=1), axes=1)
        self.multi_time_fronts_saves[self.side * 7 + 3] = np.copy(self.multi_time_fronts)

    def modulator_gain_multiply(self):
        if self.side == 1:
            self.multi_time_fronts = self.multi_time_fronts * self.modulator_gain
        self.multi_time_fronts_saves[self.side * 7 + 4] = np.copy(self.multi_time_fronts)

    # rrrr need fix (ok)
    def prepare_x(self):
        self.dx0 = self.initial_range / self.n_samples
        print(f"++++ B dx0 = {self.dx0}")

        vec = (np.arange(self.n_samples) - np.asarray(self.n_samples / 2)) if self.beam_type == 0 else (np.arange(self.n_samples) + 0.5)

        self.x = vec * np.asarray(self.dx0)

        # dx = np.asarray(self.initial_range / self.n_samples)
        # x0 = np.asarray(((self.n_samples) / 2 - 0.5) * dx)
        # x = (np.arange(self.n_samples) - np.asarray(((self.n_samples) / 2 - 0.5)) * dx

    # need fix (ok)
    def prepare_gain_pump(self):
        pump_width = np.asarray(0.000030 * 0.5)
        g0 = np.asarray(1 / self.mirror_loss + self.epsilon)
        #x = (np.arange(self.n_samples) - np.asarray(self.n_samples / 2)) * np.asarray(self.dx0)
        xw = self.x / pump_width
        self.pump_gain0 = g0 * np.exp(-xw * xw)

    # need fix (ok)
    def prepare_aperture(self):
        aperture_width = np.asarray(self.aperture * 0.5)
        #x = (np.arange(self.n_samples) - np.asarray(self.n_samples / 2)) * np.asarray(self.dx0)
        self.multi_time_aperture = np.exp(- np.square(self.x / aperture_width))

        diffraction_width = self.diffraction_waist
        self.multi_time_diffraction = np.exp(- np.square(self.x / diffraction_width))


    # need fix (skip)
    def prepare_linear_fresnel_help_data(self):
        self.mat_side = calc_original_sim_matrices(self.crystal_shift)

        self.fresnel_data = []

        for index_side, side_m in enumerate(self.mat_side):
            dx = self.dx0

            fresnel_side_data = prepare_linear_fresnel_calc_data(side_m, dx, self.n_samples, self.lambda_, 
                                             self.mirror_loss if index_side == 1 else 1)
            
            # A, B, C, D = side_m[0][0], side_m[0][1], side_m[1][0], side_m[1][1]
            # if A > 0:
            #     M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]]
            #     M1 = [[1, B / (A + 1)], [0, 1]]
            # else:
            #     M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]]
            #     M1 = [[-1, B / (-A + 1)], [0, -1]]

            # print(f"M1 = {M1[0][0]:11.6f}, {M1[0][1]:11.6f}, {M1[1][0]:11.6f}, {M1[1][1]:11.6f} dx = {M1[0][1] * self.lambda_ / (self.n_samples * dx)} dx0 = {dx}")
            # print(f"M2 = {M2[0][0]:11.6f}, {M2[0][1]:11.6f}, {M2[1][0]:11.6f}, {M2[1][1]:11.6f}")
            # fresnel_side_data = []
            # for index, M in enumerate([M1, M2]):
            #     loss = self.mirror_loss if index == 0 and index_side == 1 else 1
            #     fresnel_side_data.append(self.vectors_for_linear_fresnel(M, self.n_samples, dx, loss, M[0][0] < 0))
            #     dx = M[0][1] * self.lambda_ / (self.n_samples * dx)
            #     print(f"---- dx = {dx} dx0 = {self.dx0}")

            self.fresnel_data.append(fresnel_side_data)

    def prepare_cylindrical_fresnel_help_data(self):
        self.mat_side = calc_original_sim_matrices(self.crystal_shift)

        self.fresnel_data, _ = map(list, zip(*[cylindrical_fresnel_prepare(self.x, self.x, self.lambda_, mat) for mat in self.mat_side]))

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

    # rrrr need fix (ok)
    def get_init_front(self, p_par=-1):

        waist0 = p_par if p_par > 0.0 else 0.00003  # Example value, replace with actual value
        beam_dist = 0.0  # Example value, replace with actual value
        RayleighRange = np.pi * waist0 * waist0 / self.lambda_
        theta = np.asarray(0 if abs(beam_dist) < 0.000001 else np.pi / (self.lambda_ * beam_dist))
        waist = np.asarray(waist0 * np.sqrt(1 + beam_dist / RayleighRange))

        random_values = np.asarray(1.0) + np.asarray(0.3) * np.random.rand(self.n_samples)
        val_complex = - np.square(self.x / waist) - 1j * theta * np.square(self.x)
        vf = np.exp(val_complex) * random_values

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
        multi_time_fronts_tr = np.empty((self.n_time_samples, self.n_samples), dtype=complex)
        for i_time in range(self.n_time_samples):
            #rnd = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            rnd = np.asarray((random_lcg() * 2 - 1) + 1j * (random_lcg() * 2 - 1))
            fr = np.multiply(rnd, self.get_init_front())
            multi_time_fronts_tr[i_time] = fr

        self.multi_time_fronts = multi_time_fronts_tr.T
        self.multi_time_fronts_saves[0] = np.copy(self.multi_time_fronts)

        self.update_helpData()

    def front_power(self, bin_field):
        bin_intencity = np.square(np.abs(bin_field))
        if (self.beam_type == 1):
            bin_intencity = np.multiply(bin_intencity, self.x)
        return np.sum(bin_intencity, axis=1, keepdims=True)
    
    def phase_change_during_kerr(self):
        self.multi_time_fronts_saves[self.side * 7 + 0] = np.copy(self.multi_time_fronts)

        total_kerr_lensing = np.multiply(self.lensing_factor, self.ikl_times_i)

        bin2 = np.square(np.abs(self.multi_time_fronts))
        self.sum_power_ix[self.side] = np.sum(bin2, axis=1)
        phase_shift1 = total_kerr_lensing * bin2
        self.multi_time_fronts *= np.exp(phase_shift1)
        self.ps[self.side] = phase_shift1[:, self.view_on_x].imag

        self.multi_time_fronts_saves[self.side * 7 + 1] = np.copy(self.multi_time_fronts)

        multi_time_fronts_trans = self.multi_time_fronts.T
        p_fr_before = self.front_power(multi_time_fronts_trans)
        fr_after = multi_time_fronts_trans * self.multi_time_aperture
        p_fr_after = self.front_power(fr_after)
        multi_time_fronts_trans = fr_after * np.sqrt(p_fr_before / p_fr_after)
        self.multi_time_fronts = multi_time_fronts_trans.T
        self.multi_time_fronts_saves[self.side * 7 + 2] = np.copy(self.multi_time_fronts)

    # # need fix (ok)
    # def fresnel_progress(self, multi_time_fronts_trans):

    #     #linear_fresnel_propogate(self.fresnel_data[self.side], multi_time_fronts_trans)
    #     for fresnel_step_data in self.fresnel_data[self.side]:
    #         vec0 = fresnel_step_data['vecs'][0]
    #         vecF = fresnel_step_data['vecs'][1]
    #         dx = fresnel_step_data['dx']
    #         multi_time_fronts_trans = vecF * np.fft.fftshift(np.fft.fft(np.fft.fftshift(vec0 * multi_time_fronts_trans, axes=1), axis=1), axes=1) * dx
        
    #     return multi_time_fronts_trans

    # need fix (ok)
    def linear_cavity_one_side(self):

        #rrrrr choice
        if True:
            self.two_sided_sum_power_ix = self.sum_power_ix[self.side]
        else:
            self.two_sided_sum_power_ix = (self.sum_power_ix[0] + self.sum_power_ix[1]) * 0.5

        Is = self.is_factor
        self.gain_reduction = np.real(np.multiply(self.pump_gain0, np.divide(self.n_samples_ones, 1 + np.divide(self.two_sided_sum_power_ix, Is * self.n_time_samples))))
        
        #rrrrr choice
        if True:
            self.gain_reduction_with_origin = np.multiply(self.gain_factor, 1 + self.gain_reduction)
            self.gain_reduction_after_diffraction = np.multiply(self.gain_reduction_with_origin, self.multi_time_diffraction)
        else:
            self.gain_reduction_after_diffraction = np.multiply(self.gain_reduction, self.multi_time_diffraction)
            self.gain_reduction_with_origin = 1 + np.multiply(self.gain_factor, self.gain_reduction_after_diffraction)

        multi_time_fronts_trans = self.multi_time_fronts.T
        gain_factors = self.gain_reduction_after_diffraction
        multi_time_fronts_trans = gain_factors * multi_time_fronts_trans
        self.multi_time_fronts_saves[self.side * 7 + 5] = np.copy(multi_time_fronts_trans.T)


        if self.beam_type == 0:
            #self.multi_time_fronts = self.fresnel_progress(multi_time_fronts_trans).T
            self.multi_time_fronts = linear_fresnel_propogate(self.fresnel_data[self.side], multi_time_fronts_trans).T
        else:
            self.multi_time_fronts = cylindrical_fresnel_propogate(multi_time_fronts_trans.T, self.fresnel_data[self.side])

        self.multi_time_fronts_saves[self.side * 7 + 6] = np.copy(self.multi_time_fronts)

    def multi_time_round_trip(self):
        self.n_rounds += 1

        for self.side in [0, 1]:
            self.phase_change_during_kerr()

            self.spectral_gain_dispersion()

            self.modulator_gain_multiply()

            self.linear_cavity_one_side()

    def center_multi_time(self):
        a = np.sum(np.square(np.abs(self.multi_time_fronts)), 0)
        print(f"length of a = {len(a)}")
        index = np.argmax(a)
        print(f"index = {index}")
        roll = -index + self.n_time_samples // 2
        self.multi_time_fronts = np.roll(self.multi_time_fronts, roll, axis=1)
        for i in range(len(self.multi_time_fronts_saves)):
            self.multi_time_fronts_saves[i] = np.roll(self.multi_time_fronts_saves[i], roll, axis=1)

    def get_x_values(self, sample):
        source = self.select_source(sample)
        if source is None:
            return {}
        stage_data = source.T
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_x:
            fr = cget(stage_data)[self.view_on_x].tolist()
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})"}
        return {}
    
    def get_y_values(self, sample):
        source = self.select_source(sample)
        if source is None:
            return {}
        stage_data = source
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_y:
            fr = cget(stage_data)[self.view_on_y].tolist()
            fr_mean_and_std = list_mean_and_std(fr)
            return {"color": ["red", "blue"][sample], "values": fr, "text": f"M{max(fr):.2f}({fr.index(max(fr))})({fr_mean_and_std[0]:.4f}±{fr_mean_and_std[1]:.4f})"}
        return {}
    
    def get_x_values_full(self, sample):
        source = self.select_source(sample, True)
        if source is None:
            return []
        stage_data = source.T
        if isinstance(stage_data, np.ndarray) and len(stage_data) > self.view_on_x:
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
                stage_data = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(stage_data, axes=1), axis=1), axes=1)
            if (self.view_on_abs_phase[target] == "Abs"):
                stage_data = np.abs(stage_data)
            else:
                stage_data = np.angle(stage_data)
        return stage_data
    
    def focus_front(self, fr):
        if (self.beam_type == 0):
            q = len(fr) // 4
            x_original = np.linspace(0, 1, len(fr) // 2)
            x_new = np.linspace(0, 1, len(fr))
            return np.interp(x_new, x_original, fr[q:3*q])
        else:
            q = len(fr) // 2
            a = fr[:q]
            #b = a[::-1]
            #print(f"shapes = {a.shape[0]}, {b.shape[0]}")
            return np.concatenate((a[::-1], a))
        
    def get_kerr_influence(self, batch, direction):
        if (self.beam_type != 0):
            return []
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
                fr_next = vecF * np.fft.fftshift(np.fft.fft(np.fft.fftshift(vec0 * fr_next))) * dx

            fr_after.append(cget(np.abs(fr_next)).tolist())
        
        return [{"color": "black", "values": cget(np.abs(fr_original)).tolist(), "text": f"Start"},
                {"color": "purple", "values": cget(np.abs(fr_original1)).tolist(), "text": f"squeeze({max(np.abs(fr_original1)):.2f})"},
                {"color": "green", "values": fr_after[1], "text": f"with Kerr({max(fr_after[1]):.2f})"},
                {"color": "red", "values": fr_after[0], "text": f"without Kerr({max(fr_after[0]):.2f})"}]
        
    def get_saturation_graph_data(self):
        return [{"color": "red", "values": cget(self.focus_front(self.pump_gain0)).tolist(), "text": f"pump_gain0"},
                {"color": "blue", "values": cget(self.focus_front(self.gain_reduction)).tolist(), "text": f"gain_reduction"},
                # {"color": "green", "values": cget(self.gain_reduction_with_origin), "text": f"gain_reduction_with_origin"},
                # {"color": "purple", "values": cget(self.gain_reduction_after_aperture), "text": f"gain_reduction_after_aperture"},
                # {"color": "black", "values": cget(self.gain_reduction_after_diffraction), "text": f"gain_reduction_after_diffraction"}
        ]

    def get_diffraction_graph_data(self):
        return [{"color": "red", "values": cget(self.focus_front(self.gain_reduction_with_origin)).tolist(), "text": f"before diffraction"},
                {"color": "blue", "values": cget(self.focus_front(self.gain_reduction_after_diffraction)).tolist(), "text": f"after diffraction"},
                # {"color": "green", "values": cget(self.gain_reduction_with_origin), "text": f"gain_reduction_with_origin"},
                # {"color": "purple", "values": cget(self.gain_reduction_after_aperture), "text": f"gain_reduction_after_aperture"},
                # {"color": "black", "values": cget(self.gain_reduction_after_diffraction), "text": f"gain_reduction_after_diffraction"}
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
                # {"name": "gr1", "lines": self.get_kerr_influence(0, 0)},
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
