import aerosandbox.numpy as np
import aerosandbox as asb


### Payload Module
opti = asb.Opti()
minimize = "power_out_payload"  # any "eval-able" expression

c = 299792458 # [m/s] speed of light
k_b = 1.38064852E-23 # [m2 kg s-2 K-1]
required_resolution = opti.parameter(value=5) # meters from conversation with Brent on 2/18/22
required_snr = opti.parameter(value=6)  # dB from conversation w Brent on 2/18/22
radar_length = opti.parameter(value=0.1) # meter from GAMMA remote sensing doc
radar_width = opti.parameter(value=0.03) # meter from GAMMA remote sensing doc
scattering_cross_sec = opti.parameter(value=1) # TODO check this
center_wavelength = opti.parameter(value=0.226) # meters
groundspeed = opti.parameter(value=30)
T = opti.parameter(value=216)
y = opti.parameter(value=12000)

radar_area = radar_width * radar_length
look_angle = opti.parameter(value=45)
dist = y / np.cosd(look_angle)

antenna_gain = 4 * np.pi * radar_area * 0.7 / center_wavelength ** 2
bandwidth = opti.variable(
    init_guess=2E8,
    scale=1E6,
    category='des'
) #Hz
peak_power = opti.variable(
    init_guess=500,
    scale=100,
    category='des'
) # Watts
pulse_rep_freq = opti.variable(
    init_guess=10,
    scale=1,
    category='des'
)
power_out_payload = opti.variable(
    init_guess=100,
    scale=10,
    category='des'
)

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
    peak_power >= 0,
    bandwidth >= 0,
    pulse_rep_freq >= 2 * groundspeed / radar_length,
])
objective = eval(minimize)
opti.minimize(objective)
sol = opti.solve(max_iter=1000)