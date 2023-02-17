import aerosandbox as asb
from aerosandbox.library import winds as lib_winds
import aerosandbox.numpy as np
import pathlib
from aerosandbox.modeling.interpolation import InterpolatedModel

path = str(
    pathlib.Path(__file__).parent.absolute()
)

##### Section: Initialize Optimization
opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='design'
)
des = dict(category="design")
tra = dict(category="trajectory")

##### optimization assumptions
make_plots = False

##### Section: Parameters

# Mission Operating Parameters
latitude = opti.parameter(value=-75)  # degrees, the location the sizing occurs
day_of_year = opti.parameter(vlaue=45)  # Julian day, the day of the year the sizing occurs
mission_length = opti.parameter(value=45)  # days, the length of the mission without landing to download data
strat_offset_value = opti.parameter(value=1000)  # meters, margin above the stratosphere height the aircraft is required to stay above
min_cruise_altitude = lib_winds.tropopause_altitude(latitude, day_of_year) + strat_offset_value
climb_opt = False  # are we optimizing for the climb as well?
hold_cruise_altitude = True # must we hold the cruise altitude (True) or can we altitude cycle (False)?

# Trajectory Parameters
sample_area_height = opti.parameter(value=150000)  # meters, the height of the area the aircraft must sample
sample_area_width = opti.parameter(value=100000)  # meters, the width of the area the aircraft must sample
required_headway_per_day = opti.parameter(value=0)  # meters, the minimum distance the aircraft must cover in the sizing day
trajectory = 1 # value to determine the particular trajectory
required_revisit_rate = opti.parameter(value=0) # How many times must the aircraft fully cover the sample area in the sizing day?
swath_overlap = opti.parameter(value=0.1) # What fraction of the adjacent swaths must overlap? Typically ranges from 0.1 to 0.5

# Aircraft Parameters
battery_specific_energy_Wh_kg = opti.parameter(value=400) # cell level specific energy of the battery
battery_pack_cell_percentage = opti.parameter(value=0.75)  # What percent of the battery pack consists of the module, by weight?
variable_pitch = False # Do we assume the propeller is variable pitch?
structural_load_factor = opti.parameter(value=3)  # over static
mass_payload_base = opti.parameter(value=10)
tail_panels = True # Do we assume we can mount solar cells on the vertical tail?
wing_cells = "sunpower"  # select cells for wing, options include ascent_solar, sunpower, and microlink
vertical_cells = "sunpower"  # select cells for vtail, options include ascent_solar, sunpower, and microlink
use_propulsion_fits_from_FL2020_1682_undergrads = True  # Warning: Fits not yet validated
# fits for propeller and motors to derive motor and propeller efficiencies

# Instrument Parameters
required_resolution = opti.parameter(value=2)  # meters from conversation with Brent on 2/18/22
required_snr = opti.parameter(value=6)  # 6 dB min and 20 dB ideally from conversation w Brent on 2/18/22
center_wavelength = opti.parameter(value=0.024)  # meters given from Brent based on the properties of the ice sampled by the radar
scattering_cross_sec_db = opti.parameter(value=-10)  # meters ** 2 ranges from -20 to 0 db according to Charles in 4/19/22 email
radar_length = opti.parameter(value=1)  # meters, given from existing Gamma Remote Sensing instrument
radar_width = opti.parameter(value=0.3)  # meters, given from existing Gamma Remote Sensing instrument
look_angle = opti.parameter(value=45)  # degrees

# Margins
structural_mass_margin_multiplier = opti.parameter(value=1.25) # A value greater than 1 represents the structural components as sized are
energy_generation_margin = opti.parameter(value=1.05) # A value greater than 1 represents aircraft must generate said fractional surplus of energy
allowable_battery_depth_of_discharge = opti.parameter(value=0.95)  # How much of the battery can you actually use? # updated according to Matthew Berk discussion 10/21/21 # TODO reduce ?
q_ne_over_q_max = opti.parameter(value=2)  # Chosen on the basis of a paper read by Trevor Long about Helios, 1/16/21 TODO re-evaluate?

##### Section: Time Discretization
n_timesteps_per_segment = 180  # number of timesteps in the 25 hour sizing period

if climb_opt:  # roughly 1-day-plus-climb window, starting at ground. Periodicity enforced for last 24 hours.
    time_start = opti.variable(init_guess=-12 * 3600,
                               scale=3600,
                               upper_bound=0,
                               lower_bound=-24 * 3600
                               **tra)
    time_end = 36 * 3600

    time_periodic_window_start = time_end - 24 * 3600
    time = np.concatenate((
        np.linspace(
            time_start,
            time_periodic_window_start,
            n_timesteps_per_segment
        ),
        np.linspace(
            time_periodic_window_start,
            time_end,
            n_timesteps_per_segment
        )
    ))
    time_periodic_start_index = time.shape[0] - n_timesteps_per_segment
    time_periodic_end_index = time.shape[0] - 1

else:  # Normal mode: 24-hour periodic window, starting at altitude.
    time_start = 0 * 3600
    time_end = 24 * 3600

    time = np.linspace(
        time_start,
        time_end,
        n_timesteps_per_segment
    )
    time_periodic_start_index = 0
    time_periodic_end_index = time.shape[0] - 1

n_timesteps = time.shape[0]
hour = time / 3600

##### Section: Vehicle definition

# overall layout wing layout
boom_location = 0.80  # as a fraction of the half-span
break_location = 0.67  # as a fraction of the half-span

# Wing
wing_span = opti.variable(
    init_guess=30,
    scale=10,
    lower_bound=0.01,
    **des
)

boom_offset = boom_location * wing_span / 2  # in real units (meters)

cl_array = np.load(path + '/data/cl_function.npy')
cd_array = np.load(path + '/data/cd_function.npy')
cm_array = np.load(path + '/data/cm_function.npy')
alpha_array = np.load(path + '/data/alpha.npy')
reynolds_array = np.load(path + '/data/reynolds.npy')
cl_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(np.array(reynolds_array)), },
                                cl_array, "bspline")
cd_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(np.array(reynolds_array))},
                                cd_array, "bspline")
cm_function = InterpolatedModel({"alpha": alpha_array, "reynolds": np.log(np.array(reynolds_array))},
                                cm_array, "bspline")

wing_airfoil = asb.geometry.Airfoil(
    name="HALE_03",
    coordinates=r"studies/airfoil_optimizer/HALE_03.dat",
    CL_function=cl_function,
    CD_function=cd_function,
    CM_function=cm_function)

wing_root_chord = opti.variable(
    init_guess=1.8,
    scale=1,
    lower_bound=0.01,
    **des
)

wing_x_quarter_chord = opti.variable(
    init_guess=0,
    scale=0.01,
    lower_bound=0.01,
    **des
)

wing_y_taper_break = break_location * wing_span / 2

wing_taper_ratio = 0.5  # TODO analyze this more
wing_tip_chord = wing_root_chord * wing_taper_ratio

wing_incidence = opti.variable(
    init_guess=0,
    lower_bound=-15,
    upper_bound=15,
    freeze=True,
    **des
)

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([-wing_root_chord / 4, 0, 0]),
            chord=wing_root_chord,
            twist=wing_incidence,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Break
            xyz_le=np.array([-wing_root_chord / 4, wing_y_taper_break, 0]),
            chord=wing_root_chord,
            twist=wing_incidence,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([-wing_root_chord * wing_taper_ratio / 4, wing_span / 2, 0]),
            chord=wing_tip_chord,
            twist=wing_incidence,
            airfoil=wing_airfoil,
        ),
    ]
).translate(np.array([wing_x_quarter_chord, 0, 0]))

# center fuselage
center_boom_length = opti.variable(
    init_guess=3,
    scale=1,
    lower_bound=0.1,
    **des
)

# outboard_fuselage
outboard_boom_length = opti.variable(
    init_guess=3,
    scale=1,
    lower_bound=wing_root_chord * 3/4
    **des
)

# values roughly to match the demonstrator fuselage and payload pod
boom_diameter = 0.2  # meters
payload_pod_length = 2.5  # meters
payload_pod_diameter = 0.5  # meters

# tail section
tail_airfoil = asb.Airfoil("naca0008")
tail_airfoil.generate_polars(
    cache_filename="cache/naca0008.json",
    include_compressibility_effects=False,
    make_symmetric_polars=True
)

# center vstab
center_vstab_span = opti.variable(
    init_guess=3,
    scale=2,
    lower_bound=0.1
    **des
)

center_vstab_chord = opti.variable(
    init_guess=1,
    scale=1,
    lower_bound=0.1,
    **des
)

# center hstab
center_hstab_span = opti.variable(
    init_guess=2,
    scale=2,
    lower_bound=0.1,
    upper_bound=wing_span/6,
    **des
)
center_hstab_chord = opti.variable(
    init_guess=0.8,
    scale=2,
    lower_bound=0.1
)
center_hstab_incidence = opti.variable(
    init_guess=0,
    lower_bound=-15,
    upper_bound=15,
    freeze=True,
    **des
)
center_hstab = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            chord=center_hstab_chord,
            twist=center_hstab_incidence,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, center_hstab_span / 2, 0]),
            chord=center_hstab_chord,
            twist=center_hstab_incidence,
            airfoil=tail_airfoil,
        ),
    ]
).translate(np.array([center_boom_length - center_vstab_chord * 0.75 - center_hstab_chord, 0, 0.1]))

opti.subject_to([
    center_boom_length - center_vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

