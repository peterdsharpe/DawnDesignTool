import aerosandbox.numpy as np
import aerosandbox as asb


### Payload Module
opti = asb.Opti()

c = 299792458 # [m/s] speed of light
k_b = 1.38064852E-23 # [m2 kg s-2 K-1]
required_resolution = opti.parameter(value=2) # 1-2 meters required from conversation with Brent on 2/18/22
required_snr = opti.parameter(value=20)  # 6 dB min and 20 dB ideally from conversation w Brent on 2/18/22
center_wavelength = opti.parameter(value=0.226) # meters from GAMMA Remote Sensing Doc
groundspeed = opti.parameter(value=30) # average groudspeed
T = opti.parameter(value=216)
y = opti.parameter(value=12000)
sigma0_db = opti.parameter(value=-10)

radar_length = opti.variable(
    init_guess=0.1,
    scale=1,
    category='des',
    lower_bound=0,
)
radar_width = opti.variable(
    init_guess=0.03,
    scale=0.1,
    category='des',
    lower_bound=0,
)
bandwidth = opti.variable(
    init_guess=1e6,
    scale=1e4,
    lower_bound=0,
    category='ops'
)  # Hz
peak_power = opti.variable(
    init_guess=1000,
    scale=100,
    lower_bound=0,
    category='ops'
)  # Watts
pulse_rep_freq = opti.variable(
    init_guess=10000,
    scale=1000,
    lower_bound=0,
    category='ops'
)
power_out_payload = opti.variable(
    init_guess=1,
    scale=0.1,
    lower_bound=0,
    category='ops'
)
# define key radar parameters
radar_area = radar_width * radar_length
look_angle = opti.parameter(value=45)
dist = y / np.cosd(look_angle)
grazing_angle = 90 - look_angle
swath_azimuth = center_wavelength * dist / radar_length
swath_range = center_wavelength * dist / (radar_width * np.cosd(look_angle))
max_length_synth_ap = center_wavelength * dist / radar_length
ground_area = swath_range * swath_azimuth * np.pi / 4
radius = (swath_azimuth + swath_range) / 4
ground_imaging_offset = np.sin(look_angle) * dist
sigma0 = 10 ** (sigma0_db / 10)
scattering_cross_sec = sigma0 # Todo change from db to ratio
# scattering_cross_sec = np.pi * radius ** 2radar_offset_length =
antenna_gain = 4 * np.pi * radar_area * 0.7 / center_wavelength ** 2
pulse_duration = 1 / bandwidth

# constrain SAR resolution to required value
range_resolution = c * pulse_duration / (2 * np.sind(look_angle))
azimuth_resolution = radar_length / 2
opti.subject_to([
    range_resolution <= required_resolution,
    azimuth_resolution <= required_resolution,
])

# account for snr
noise_power_density = k_b * T * bandwidth / (center_wavelength ** 2)
power_trans = peak_power * pulse_duration
power_received = power_trans * antenna_gain * radar_area * scattering_cross_sec / ((4 * np.pi) ** 2 * dist ** 4)
power_out_payload = power_trans * pulse_rep_freq
snr = power_received / noise_power_density
snr_db = 10 * np.log10(snr)
opti.subject_to([
    required_snr <= snr_db,
    pulse_rep_freq >= 2 * groundspeed / radar_length,
    pulse_rep_freq <= c / (2 * swath_azimuth),
    radar_width <= 0.4,
    radar_length <= 0.4,
])

minimize = "power_out_payload"
objective = eval(minimize)
opti.minimize(objective)
sol = opti.solve(max_iter=1000)