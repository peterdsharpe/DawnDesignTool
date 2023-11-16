import aerosandbox.numpy as np
import aerosandbox as asb


### Payload Module
opti = asb.Opti()

c = 299792458  # [m/s] speed of light
k_b = 1.38064852E-23  # [m2 kg s-2 K-1]
required_resolution = opti.parameter(value=2)  # meters from conversation with Brent on 2/18/22
required_snr = opti.parameter(value=6)  # 6 dB min and 20 dB ideally from conversation w Brent on 2/18/22
center_wavelength = opti.parameter(value=0.024)  # meters
scattering_cross_sec_db = opti.parameter(value=0)  # meters ** 2 ranges from -20 to 0 db according to Charles in 4/19/22 email
groundspeed = opti.parameter(value=5) # average groundspeed
T = opti.parameter(value=216)
y = opti.parameter(value=14000)
radar_length = 1
radar_width = 0.3
# radar_length = opti.variable(
#     init_guess=0.1,
#     scale=1,
#     category='des',
#     lower_bound=0.1,
#     upper_bound=1,
# ) # meters
# radar_width = opti.variable(
#     init_guess=0.03,
#     scale=0.1,
#     category='des',
#     lower_bound=0,
# ) # meters
bandwidth = opti.variable(
    init_guess=1e8,
    scale=1e6,
    lower_bound=0,
    category='des'
)  # Hz
peak_power = opti.variable(
    init_guess=1e+4,
    scale=1e3,
    lower_bound=0,
    category='des'
)  # Watts
pulse_rep_freq = opti.variable(
    init_guess=10,
    scale=1,
    lower_bound=0,
    category='des'
)
power_out_payload = opti.variable(
    init_guess=200,
    scale=100,
    lower_bound=0,
    category='des'
)
# pulse_duration = opti.variable(
#     init_guess=0.03,
#     scale=1,
#     lower_bound=0,
#     category='des'
# )
power_trans = opti.variable(
    init_guess = 1e6,
    scale=1e5,
    lower_bound=0,
    upper_bound=1e8,
    category='des',
) # watts
# # define key radar parameters
radar_area = radar_width * radar_length # meters ** 2
look_angle = opti.parameter(value=45) # degrees
dist = y / np.cosd(look_angle) # meters
grazing_angle = 90 - look_angle # degrees
swath_azimuth = center_wavelength * dist / radar_length # meters
swath_range = center_wavelength * dist / (radar_width * np.cosd(look_angle)) # meters
max_length_synth_ap = center_wavelength * dist / radar_length # meters
ground_area = swath_range * swath_azimuth * np.pi / 4 # meters ** 2
radius = (swath_azimuth + swath_range) / 4 # meters
ground_imaging_offset = np.tand(look_angle) * y # meters
scattering_cross_sec = 10 ** (scattering_cross_sec_db / 10)
sigma0 = scattering_cross_sec / ground_area
sigma0_db = 10 * np.log(sigma0)
antenna_gain = 4 * np.pi * radar_area * 0.7 / center_wavelength ** 2
a_hs = 0.88 # aperture-illumination taper factor associated with the synthetic aperture (value from Ulaby and Long)
F = 4 # receiver noise figure (somewhat randomly chosen value from Ulaby and Long)
a_B = 1 # pulse-taper factor to relate bandwidth and pulse duration
# doppler_bandwidth = 2 * groundspeed * horz_beamwidth / (wavelength * y)
#
# # constrain SAR resolution to required value
pulse_duration = a_B / bandwidth
range_resolution = c * pulse_duration / (2 * np.sind(look_angle))
azimuth_resolution = radar_length / 2
opti.subject_to([
    range_resolution <= required_resolution,
    azimuth_resolution <= required_resolution,
])
#
# account for snr
# noise_power_density = k_b * T * bandwidth / (wavelength ** 2)
# power_trans = peak_power * pulse_duration
# power_received = power_trans * antenna_gain * radar_area * sigma0 / ((4 * np.pi) ** 2 * dist ** 4)
# power_received = power_trans * antenna_gain ** 2 * wavelength ** 2 * sigma0 * azimuth_resolution * range_resolution / ((4 * np.pi) ** 3 * dist ** 4)
# power_received = power_trans * wavelength ** 2 * antenna_gain ** 2 * radar_length * c * pulse_duration * sigma0 / (4 * (4 * np.pi) ** 3 * dist ** 4 * np.sind(look_angle))
# power_out_payload = power_trans * pulse_rep_freq
# snr = power_received / noise_power_density

# use SAR specific equations from Ulaby and Long
power_out_payload = power_trans * pulse_rep_freq * pulse_duration
snr = power_out_payload * antenna_gain ** 2 * center_wavelength ** 3 * a_hs * sigma0 * range_resolution / ((2 * 4 * np.pi) ** 3 * dist ** 3 * k_b * T * F * groundspeed * a_B)

snr_db = 10 * np.log(snr)
opti.subject_to([
    required_snr <= snr_db,
    pulse_rep_freq >= 2 * groundspeed / radar_length,
    pulse_rep_freq <= c / (2 * swath_azimuth),
])

minimize = "power_out_payload"
objective = eval(minimize)
opti.minimize(objective)
sol = opti.solve(max_iter=3000)