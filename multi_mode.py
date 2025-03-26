import numpy as np
import json

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
        self.multi_time_aperture_val = 0.000056
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
        self.ps1 = []

        self.mat_side = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  # right
                         [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]]    # left

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
        self.multi_frequency_fronts = np.fft.fft(self.multi_time_fronts[self.n_samples // 2], norm='ortho')

        for ix in range(self.n_samples):
            self.multi_frequency_fronts[ix] = np.multiply(self.multi_frequency_fronts[ix], self.frequency_total_mult_factor)

        self.multi_time_fronts = np.fft.ifft(self.multi_frequency_fronts, norm='ortho')

    def modulator_gain_multiply(self):
        for ix in range(self.n_samples):
            self.multi_time_fronts[ix] = np.multiply(self.multi_time_fronts[ix], self.modulator_gain)

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
        aperture_width = self.multi_time_aperture_val * 0.5
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

    def get_init_front(self, p_par=-1):
        vf = []
        self.initial_range = 0.01047  # Example value, replace with actual value
        waist0 = p_par if p_par > 0.0 else 0.0005  # Example value, replace with actual value
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
            f_val = np.exp(complex(-xw * xw, -theta * x * x))
            vf.append(f_val)

        return vf

    def init_multi_time(self):
        self.multi_time_fronts = [[] for _ in range(self.n_samples)]
        self.multi_frequency_fronts = [[] for _ in range(self.n_samples)]
        self.multi_time_fronts_saves = [[], [], [], [], [], []]

        for i_time in range(self.n_time_samples):
            rnd = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
            fr = np.multiply(rnd, self.get_init_front())
            for i in range(self.n_samples):
                self.multi_time_fronts[i].append(fr[i])
                self.multi_frequency_fronts[i].append(0 + 0j)

        self.prepare_linear_fresnel_help_data()
        self.prepare_gain_pump()
        self.init_gain_by_frequency()

    def phase_change_during_kerr(self):
        self.sum_power_ix = []
        self.ps1 = []
        total_kerr_lensing = np.multiply(self.lensing_factor, self.ikl_times_i)

        for ix in range(self.n_samples):
            bin = self.multi_time_fronts[ix]
            bin2 = np.abs(np.multiply(bin, np.conj(bin)))
            self.sum_power_ix.append(np.sum(bin2))
            phase_shift1 = np.multiply(total_kerr_lensing, bin2)
            self.multi_time_fronts[ix] = np.multiply(bin, np.exp(phase_shift1))
            self.ps1.append(phase_shift1[0].imag)

        self.multi_time_fronts_saves[self.side * 3 + 0] = np.copy(self.multi_time_fronts)

        multi_time_fronts_trans = np.transpose(self.multi_time_fronts)
        for i_time in range(self.n_time_samples):
            fr = multi_time_fronts_trans[i_time]
            p_fr_before = np.sum(np.multiply(fr, np.conj(fr)))
            fr_after = np.multiply(fr, self.multi_time_aperture)
            p_fr_after = np.sum(np.multiply(fr_after, np.conj(fr_after)))
            fr = np.multiply(fr_after, np.sqrt(p_fr_before / p_fr_after))
            multi_time_fronts_trans[i_time] = fr

        self.multi_time_fronts = np.transpose(multi_time_fronts_trans)

    def linear_cavity_one_side(self):
        self.multi_time_fronts_saves[self.side * 3 + 1] = np.copy(self.multi_time_fronts)

        Is = self.is_factor
        self.gain_reduction = np.real(np.multiply(self.pump_gain0, np.divide(self.n_samples_ones, 1 + np.divide(self.sum_power_ix, Is * self.n_time_samples))))
        self.gain_reduction_with_origin = np.multiply(self.gain_factor, 1 + self.gain_reduction)
        self.gain_reduction_after_diffraction = np.multiply(self.gain_reduction_with_origin, self.multi_time_diffraction)

        multi_time_fronts_trans = np.transpose(self.multi_time_fronts)

        for i_time in range(self.n_time_samples):
            fr = multi_time_fronts_trans[i_time]
            fr = np.multiply(fr, self.gain_reduction_after_diffraction)
            for fresnel_side_data in self.fresnel_data[self.side]:
                fr = np.multiply(fr, fresnel_side_data['vecs'][0])
                fr = np.fft.fft(fr, norm='ortho') * fresnel_side_data['dx']
                fr = np.multiply(fr, fresnel_side_data['vecs'][1])
            multi_time_fronts_trans[i_time] = fr

        self.multi_time_fronts = np.transpose(multi_time_fronts_trans)
        self.multi_time_fronts_saves[self.side * 3 + 2] = np.copy(self.multi_time_fronts)

    def multi_time_round_trip(self):
        if self.i_count % 10 == 0:
            fs = np.abs(self.multi_time_fronts)
            fs = np.multiply(fs, fs)
            mean_v = np.mean(fs, axis=0)
            mean_mean = np.mean(mean_v)

        for self.side in [0, 1]:
            self.phase_change_during_kerr()
            self.spectral_gain_dispersion()
            if self.side == 1:
                self.modulator_gain_multiply()
            self.linear_cavity_one_side()

    def serialize_data(self):
        return json.dumps({"multi_time_fronts": "1.0"})
