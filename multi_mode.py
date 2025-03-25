import numpy as np

lambda_ = 0.000000780
initial_range = 0.01047
multi_ranges = [[], []]
n_samples = 256
n_max_matrices = 1000
n_rounds = 0

steps_counter = 0
n_time_samples = 1024
multi_time_fronts = []
multi_frequency_fronts = []
multi_time_fronts_saves = [[], [], [], [], [], []]

intensity_saturation_level = 400000000000000.0
intensity_total_by_ix = []
factor_gain_by_ix = []
ikl = 0.02
ikl_times_i = 1j * ikl * 160 * 0.000000006
range_w = []
spectral_gain = []
modulation_gain_factor = 0.1
modulator_gain = []
dispersion = []
sum_power_ix = []
gain_reduction = []
gain_reduction_with_origin = []
gain_reduction_after_aperture = []
gain_reduction_after_diffraction = []
gain_factor = 0.50
epsilon = 0.2
dispersion_factor = 1.0
lensing_factor = 1.0
is_factor = 200 * 352000
pump_gain0 = []
multi_time_aperture = []
multi_time_aperture_val = 0.000056
multi_time_diffraction = []
multi_time_diffraction_val = 0.000030
frequency_total_mult_factor = []
mirror_loss = 0.95
fresnel_data = []
total_range = 0.001000
dx0 = total_range / n_samples
scalar_one = 1 + 0j
n_time_samples_ones = [scalar_one] * n_time_samples
n_samples_ones = [scalar_one] * n_samples
n_samples_ones_r = [1.0] * n_samples
kerr_focal_length = 0.0075
ps1 = []
ps2 = []
time_content_option = 0
freq_content_option = 0
time_content_view = 0
freq_content_view = 0
content_option_vals = ["F", "1", "2", "3", "4", "5", "6", "M"]
content_view_vals = ["Am", "ph"]
matrices = []

mat_side = [[[-1.2947E+00, 4.8630E-03], [1.5111E+02, -1.3400E+00]],  # right
            [[1.1589E+00, 8.2207E-04], [2.9333E+02, 1.0709E+00]]]    # left

def vectors_for_fresnel(M, N, dx0, gain, is_back):
    A, B = M[0]
    C, D = M[1]
    dxf = B * lambda_ / (N * dx0)
    factor = 1j * gain * np.sqrt(-1 / (B * lambda_))
    if is_back:
        factor *= 1j * gain
    co0 = -np.pi * dx0 * dx0 * A / (B * lambda_)
    cof = -np.pi * dxf * dxf * D / (B * lambda_)

    vec0 = [np.exp(1j * co0 * (i - N / 2) ** 2) for i in range(N)]
    vecF = [factor * np.exp(1j * cof * (i - N / 2) ** 2) for i in range(N)]

    return {'dx': dx0, 'vecs': [vec0, vecF]}

def spectral_gain_dispersion():
    global multi_time_fronts, multi_frequency_fronts, n_samples, frequency_total_mult_factor, sum_power_ix

    multi_frequency_fronts = np.fft.fft(multi_time_fronts[n_samples // 2], norm='ortho')

    for ix in range(n_samples):
        multi_frequency_fronts[ix] = np.multiply(multi_frequency_fronts[ix], frequency_total_mult_factor)

    multi_time_fronts = np.fft.ifft(multi_frequency_fronts, norm='ortho')


def modulator_gain_multiply():
    global multi_time_fronts, n_samples, modulator_gain
    for ix in range(n_samples):
        multi_time_fronts[ix] = np.multiply(multi_time_fronts[ix], modulator_gain)

    return multi_time_fronts

def prepare_gain_pump():
    global n_samples, dx0, mirror_loss, epsilon, pump_gain0
    pump_width = 0.000030 * 0.5
    g0 = 1 / mirror_loss + epsilon
    pump_gain0 = []
    for ix in range(n_samples):
        x = (ix - n_samples / 2) * dx0
        xw = x / pump_width
        pump_gain0.append(g0 * np.exp(-xw * xw))
    return pump_gain0

def prepare_aperture(n_samples, dx0, multi_time_aperture_val, multi_time_diffraction_val):
    multi_time_aperture = []
    aperture_width = multi_time_aperture_val * 0.5
    for ix in range(n_samples):
        x = (ix - n_samples / 2) * dx0
        xw = x / aperture_width
        multi_time_aperture.append(np.exp(-xw * xw))

    multi_time_diffraction = []
    diffraction_width = multi_time_diffraction_val
    for ix in range(n_samples):
        x = (ix - n_samples / 2) * dx0
        xw = x / diffraction_width
        multi_time_diffraction.append(np.exp(-xw * xw))
    
    return multi_time_aperture, multi_time_diffraction

def prepare_linear_fresnel_help_data():
    global n_samples, total_range, dx0, lambda_, mat_side, mirror_loss, fresnel_data

    fresnel_data = []
    dx = total_range / n_samples

    for index_side, side_m in enumerate(mat_side):
        A, B, C, D = side_m[0][0], side_m[0][1], side_m[1][0], side_m[1][1]
        if A > 0:
            M2 = [[A, B / (A + 1)], [C, D - C * B / (A + 1)]]
            M1 = [[1, B / (A + 1)], [0, 1]]
        else:
            M2 = [[-A, -B / (-A + 1)], [-C, -D - C * B / (-A + 1)]]
            M1 = [[-1, B / (-A + 1)], [0, -1]]

        fresnel_side_data = []
        for index, M in enumerate([M1, M2]):
            loss = mirror_loss if index == 0 and index_side == 1 else 1
            fresnel_side_data.append(vectors_for_fresnel(M, n_samples, dx, loss, M[0][0] < 0))
            dx = M[0][1] * lambda_ / (n_samples * dx)
        
        fresnel_data.append(fresnel_side_data)
    
    return fresnel_data

def total_ix_power(multi_time_fronts, n_samples):
    intensity_total_by_ix = []
    for ix in range(n_samples):
        intensity_total_by_ix.append(np.sum(np.multiply(multi_time_fronts[ix], np.conj(multi_time_fronts[ix]))))
    return intensity_total_by_ix

def init_gain_by_frequency():
    global spectral_gain, dispersion, range_w, frequency_total_mult_factor, modulator_gain

    spec_gain = 200
    disp_par = dispersion_factor * 0.5e-3 * 2 * np.pi / spec_gain
    range_w = np.array([complex(v) for v in np.arange(-n_time_samples / 2, n_time_samples / 2)])
    ones = np.ones_like(range_w, dtype=complex)
    mid = range_w / spec_gain
    spectral_gain = ones / (mid ** 2 + 1)
    dispersion = np.exp(-1j * disp_par * range_w ** 2)
    exp_w = np.exp(-1j * 2 * np.pi * range_w)
    frequency_total_mult_factor = 0.5 * (1.0 + exp_w * spectral_gain * dispersion)
    modulator_gain = [1.0 + modulation_gain_factor * np.cos(2 * np.pi * w / n_time_samples) for w in range_w]

def get_init_front(p_par=-1):
    global initial_range, n_samples, lambda_

    vf = []
    initial_range = 0.01047  # Example value, replace with actual value
    waist0 = p_par if p_par > 0.0 else 0.0005  # Example value, replace with actual value
    beam_dist = 0.0  # Example value, replace with actual value
    RayleighRange = np.pi * waist0 * waist0 / lambda_
    theta = 0 if abs(beam_dist) < 0.000001 else np.pi / (lambda_ * beam_dist)
    waist = waist0 * np.sqrt(1 + beam_dist / RayleighRange)
    dx = initial_range / n_samples
    x0 = (n_samples - 1) / 2 * dx

    for i in range(n_samples):
        px = i * dx
        x = px - x0
        xw = x / waist
        f_val = np.exp(complex(-xw * xw, -theta * x * x))
        vf.append(f_val)

    return vf

def init_multi_time():
    global multi_time_fronts, multi_frequency_fronts, multi_time_fronts_saves

    multi_time_fronts = [[] for _ in range(n_samples)]
    multi_frequency_fronts = [[] for _ in range(n_samples)]
    multi_time_fronts_saves = [[], [], [], [], [], []]

    for i_time in range(n_time_samples):
        rnd = np.random.uniform(-1, 1) + 1j * np.random.uniform(-1, 1)
        fr = rnd * get_init_front()
        for i in range(n_samples):
            multi_time_fronts[i].append(fr[i])
            multi_frequency_fronts[i].append(0 + 0j)

    #gain_factor_changed(False)
    #is_factor_changed(False)
    #n_rounds_changed()

    #update_content_options()

    prepare_linear_fresnel_help_data()

    prepare_gain_pump()
    init_gain_by_frequency()

    #multi_time_aperture_changed(False)

    #fft_to_frequency()
    #ifft_to_time()

    #draw_multi_mode()

def phase_change_during_kerr():
    global side, multi_time_fronts, n_samples, lensing_factor, ikl_times_i, dx0, lambda_, multi_time_aperture, n_time_samples
    
    sum_power_ix = []
    ps1 = []
    ps2 = []
    total_kerr_lensing = np.multiply(lensing_factor, ikl_times_i)

    for ix in range(n_samples):
        bin = multi_time_fronts[ix]
        bin2 = np.abs(np.multiply(bin, np.conj(bin)))
        sum_power_ix.append(np.sum(bin2))
        phase_shift1 = np.multiply(total_kerr_lensing, bin2)
        multi_time_fronts[ix] = np.multiply(bin, np.exp(phase_shift1))
        ps1.append(phase_shift1[0].imag)

    multi_time_fronts_saves[side * 3 + 0] = np.copy(multi_time_fronts)

    multi_time_fronts_trans = np.transpose(multi_time_fronts)
    for i_time in range(n_time_samples):
        fr = multi_time_fronts_trans[i_time]
        p_fr_before = np.sum(np.multiply(fr, np.conj(fr)))
        fr_after = np.multiply(fr, multi_time_aperture)
        p_fr_after = np.sum(np.multiply(fr_after, np.conj(fr_after)))
        fr = np.multiply(fr_after, np.sqrt(p_fr_before / p_fr_after))
        multi_time_fronts_trans[i_time] = fr

    multi_time_fronts = np.transpose(multi_time_fronts_trans)
    multi_time_fronts, sum_power_ix, ps1, ps2

def linear_cavity_one_side():
    global side, multi_time_fronts, multi_time_fronts_saves, pump_gain0, n_samples_ones, sum_power_ix, Is_factor, n_time_samples, gain_factor, multi_time_diffraction, fresnel_data

    multi_time_fronts_saves[side * 3 + 1] = np.copy(multi_time_fronts)

    Is = Is_factor
    gain_reduction = np.real(np.multiply(pump_gain0, np.divide(n_samples_ones, 1 + np.divide(sum_power_ix, Is * n_time_samples))))
    gain_reduction_with_origin = np.multiply(gain_factor, 1 + gain_reduction)
    gain_reduction_after_diffraction = np.multiply(gain_reduction_with_origin, multi_time_diffraction)

    multi_time_fronts_trans = np.transpose(multi_time_fronts)

    for i_time in range(n_time_samples):
        fr = multi_time_fronts_trans[i_time]
        fr = np.multiply(fr, gain_reduction_after_diffraction)
        for fresnel_side_data in fresnel_data[side]:
            fr = np.multiply(fr, fresnel_side_data['vecs'][0])
            fr = np.fft.fft(fr, norm='ortho') * fresnel_side_data['dx']
            fr = np.multiply(fr, fresnel_side_data['vecs'][1])
        multi_time_fronts_trans[i_time] = fr

    multi_time_fronts = np.transpose(multi_time_fronts_trans)
    multi_time_fronts_saves[side * 3 + 2] = np.copy(multi_time_fronts)

def multi_time_round_trip():
    global i_count, start_time, multi_time_fronts, update_steps_counter
    
    if i_count % 10 == 0:

        fs = np.abs(multi_time_fronts)
        fs = np.multiply(fs, fs)
        mean_v = np.mean(fs, axis=0)
        mean_mean = np.mean(mean_v)

    for side in [0, 1]:
        phase_change_during_kerr()

        multi_time_fronts = spectral_gain_dispersion()
        if side == 1:
            multi_time_fronts = modulator_gain_multiply()

        linear_cavity_one_side()

