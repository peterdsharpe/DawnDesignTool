import sys
sys.path.append("C:\\Users\\AnnickDewald\\PycharmProjects\\AeroSandbox")
import aerosandbox as asb
import aerosandbox.library.aerodynamics as aero_lib
from aerosandbox.atmosphere import Atmosphere as atmo
import aerosandbox.library.aerodynamics as aero
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.library import winds as lib_winds
from aerosandbox.library.airfoils import naca0008, flat_plate
import aerosandbox.tools.units as u
from aerosandbox.optimization.opti import Opti
import aerosandbox.numpy as np
from aerosandbox.numpy.integrate_discrete import integrate_discrete_squared_curvature
import plotly.express as px
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from design_opt_utilities.fuselage import make_fuselage, aero_payload_pod
from typing import Union, List
from aerosandbox.modeling.interpolation import InterpolatedModel
import pathlib
from aerosandbox.tools.pretty_plots import plt, sns, mpl, show_plot

path = str(
    pathlib.Path(__file__).parent.absolute()
)

sns.set(font_scale=1)

##### Section: Initialize Optimization
opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='design'
)
des = dict(category="design")
ops = dict(category="operations")

##### optimization assumptions
minimize = ('wingspan_optimization_scaling_term * wing_span / 25 '
            '+ azimuth_optimization_scaling_term * strain_azimuth_resolution / 500 '
            '- coverage_optimization_scaling_term * coverage_area / 1e+10')
make_plots = True

##### Debug flags
draw_initial_guess_config = False

##### end section

##### Section: Input Parameters

# Objective Function Scaling Parameters
wingspan_optimization_scaling_term = opti.parameter(value=1) # scale from 0 to 1 to adjust the relative importance of wingspan in the objective function
azimuth_optimization_scaling_term = opti.parameter(value=0) # scale from 0 to 1 to adjust the relative importance of spatial resolution in the objective function
coverage_optimization_scaling_term = opti.parameter(value=0) # scale from 0 to 1 to adjust the relative importance of spatial coverage in the objective function

# Aircraft Parameters
battery_specific_energy_Wh_kg = 390  # cell level specific energy of the battery
battery_pack_cell_percentage = 0.85  # What percent of the battery pack consists of the module, by weight?
# these roughly correspond to the value for cells we are planning for near-term
variable_pitch = False  # Do we assume the propeller is variable pitch?
structural_load_factor = 3  # over static
tail_panels = True  # Do we assume we can mount solar cells on the vertical tail?
max_wing_solar_area_fraction = 0.8
max_vstab_solar_area_fraction = 0.8
use_propulsion_fits_from_FL2020_1682_undergrads = True  # Warning: Fits not yet validated
# fits for propeller and motors to derive motor and propeller efficiencies
# todo validate fits from FL2020 1682 undergrads or replace with better propulsion model

# Mission Operating Parameters
latitude = -73  # degrees, the location the sizing occurs
day_of_year = 60  # Julian day, the day of the year the sizing occurs
mission_length = 80  # days, the length of the mission without landing to download data
strat_offset_value = 1000  # meters, margin above the stratosphere height the aircraft is required to stay above
min_cruise_altitude = lib_winds.tropopause_altitude(latitude, day_of_year) + strat_offset_value
climb_opt = False  # are we optimizing for the climb as well?
hold_cruise_altitude = True  # must we hold the cruise altitude (True) or can we altitude cycle (False)?

# Trajectory Parameters
min_speed = 0.5 # specify a minimum groundspeed (bad convergence if less than 0.5 m/s)
wind_direction = 45 # degrees, the direction the wind is blowing from 0 being North and aligned with the x-axis
run_with_95th_percentile_wind_condition = False # do we want to run the sizing with the 95th percentile wind condition?
required_strain_temporal_resolution = opti.parameter(value=6) # hours

# todo finalize trajectory parameterization
# trajectory = 'straight'   # do we want to assume a straight line trajectory?
required_headway_per_day = 100000
vehicle_heading = opti.parameter(value=0) # degrees

# trajectory = 'circular' # do we want to assume a circular trajectory?
# temporal_resolution = opti.variable(init_guess=6, scale=1, lower_bound=0.5, category='des')  # hours

trajectory = 'racetrack'

# trajectory = 'lawnmower'  # do we want to assume a lawnmower trajectory?

# Instrument Parameters
mass_payload_base = 5 # kg, does not include data storage or aperture mass
payload_volume = 0.023 * 1.5  # assuming payload mass from gamma remote sensing with 50% margin on volume
tb_per_day = 4 # terabytes per day, the amount of data the payload collects per day, to account for storage
strain_range_resolution = opti.variable(init_guess=0.5, scale=1, lower_bound=0.015, category='des') # meters
strain_azimuth_resolution = opti.variable(init_guess=0.5, scale=1, lower_bound=0.015, category='des') # meters
strain_temporal_resolution = opti.variable(init_guess=4, scale=1, category='des') # hours
# required_snr = 20  # 6 dB min and 20 dB ideally from conversation w Brent on 2/18/22
required_strain_precision = 1E-4 # 1/yr, the required precision of the strain measurement
# meters given from Brent based on the properties of the ice sampled by the radar
scattering_cross_sec_db = -10
# meters ** 2 ranges from -20 to 0 db according to Charles in 4/19/22 email

# Margins
structural_mass_margin_multiplier = 1.25
# A value greater than 1 represents the structural components as sized are
energy_generation_margin = 1.05
# A value greater than 1 represents aircraft must generate said fractional surplus of energy
allowable_battery_depth_of_discharge = 0.95
# How much of the battery can you actually use? # updated according to Matthew Berk discussion 10/21/21 # TODO reduce?
q_ne_over_q_max = 2
# Chosen on the basis of a paper read by Trevor Long about Helios, 1/16/21 TODO re-evaluate?

##### end section

##### Time Discretization
n_timesteps_per_segment = 180
# Quick convergence testing indicates you can get bad analyses below 150 or so...
# todo scale up on timestep count as we explore more complex trajectories
if climb_opt:  # roughly 1-day-plus-climb window, starting at ground. Periodicity enforced for last 24 hours.
    time_start = opti.variable(init_guess=-12 * 3600, scale=3600, category="ops")
    # optimizer selects most optimal start time
    opti.subject_to([
        time_start / 3600 < 0,
        time_start / 3600 > -24,
    ])
    time_end = 36 * 3600 # time ends

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

# endregion

##### Section: Vehicle Overall Specs
mass_total = opti.variable(
    init_guess=230,
    scale=100,
    lower_bound=0,
    **des
)

power_in_after_panels_max = opti.variable(
    init_guess=5000,
    lower_bound=0,
    scale=1000,
    **des
)

power_out_propulsion_max = opti.variable(
    init_guess=1683,
    lower_bound=10,
    scale=1e3,
    **des
)

##### Initialize design optimization variables (all units in base SI or derived units)
# start with defining payload pod variables
payload_pod_length = opti.variable(
    init_guess=5,
    scale=1,
    lower_bound=0.5,
    # upper_bound=5,
    category="des",
) # meters
payload_pod_diameter = opti.variable(
    init_guess=0.25,
    scale=0.1,
    lower_bound=0.2,
    upper_bound=1,
    category="des",
)  # meters

payload_pod_y_offset = 1.5  # meters # this matches the demonstrator vehicle configuration
x_payload_pod = opti.variable(
    init_guess=-0.5,
    scale=0.1,
    category="des",
    lower_bound=0 - payload_pod_length,
    upper_bound=0 + payload_pod_length,
) # x location of payload pod constrained to be no more or less than the length of the pod from the wing LE

payload_pod_nose_length = 0.5
payload_pod = aero_payload_pod(
    total_length=payload_pod_length,
    nose_length=payload_pod_nose_length,
    tail_length=1,
    fuse_diameter=payload_pod_diameter,
).translate([x_payload_pod, 0, -payload_pod_y_offset]) # nose and tail length are guesses for now todo reconsider?

payload_pod_shell_thickness = 0.003 # meters
# account for the shell thickness to find the internal volume
payload_pod_volume = payload_pod.volume() - payload_pod_shell_thickness * payload_pod.area_wetted()
payload_pod_structure_volume = payload_pod_volume * 0.20  # 20% of the volume is structure a guess for now

# overall layout
boom_location = 0.80  # as a fraction of the half-span
break_location = 0.67  # as a fraction of the half-span

# wing
wing_span = opti.variable(
    init_guess=30,
    scale=6,
    category="des"
)

boom_offset = boom_location * wing_span / 2  # in real units (meters)

opti.subject_to([wing_span > 1])

wing_root_chord = opti.variable(
    init_guess=1.8,
    scale=0.1,
    category="des",
    lower_bound=1,
)
wing_x_quarter_chord = opti.variable(
    init_guess=0,
    scale=0.01,
    category="des"
)
wing_y_taper_break = break_location * wing_span / 2
wing_taper_ratio = 0.5  # TODO analyze this more

# center hstab
center_hstab_span = opti.variable(
    init_guess=5.3,
    scale=1,
    category="des",
    lower_bound=0.1,
)
opti.subject_to(center_hstab_span < wing_span / 6)

center_hstab_chord = opti.variable(
    init_guess=1,
    scale=0.1,
    category="des",
    lower_bound=0.1,
)

center_hstab_twist = opti.variable(
    init_guess=0,
    scale=2,
    category="ops",
    lower_bound=-15,
    upper_bound=0,
)

# center hstab
outboard_hstab_span = opti.variable(
    init_guess=2,
    scale=1,
    category="des",
    lower_bound=2, # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20

)
opti.subject_to(outboard_hstab_span < wing_span / 6)

outboard_hstab_chord = opti.variable(
    init_guess=0.8,
    scale=0.1,
    category="des",
    lower_bound=0.8, # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20
)

outboard_hstab_twist = opti.variable(
    init_guess=-0.3,
    scale=0.1,
    lower_bound=-15,
    upper_bound=0,
    category="ops"
)

# center_vstab
center_vstab_span = opti.variable(
    init_guess=4,
    scale=2,
    category="des",
    lower_bound=0.1,
)

center_vstab_chord = opti.variable(
    init_guess=1.6,
    scale=0.5,
    category="des",
    lower_bound=0.1,
)

# center_fuselage
center_boom_length = opti.variable(
    init_guess=7.2,
    scale=2,
    category="des"
)
opti.subject_to([
    center_boom_length - center_vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

# outboard_fuselage
outboard_boom_length = opti.variable(
    init_guess=2.3,
    scale=2,
    category="des"
)
opti.subject_to([
    outboard_boom_length > wing_root_chord,
    outboard_boom_length < center_boom_length,
    # outboard_boom_length < 3.5, # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20
])

# Propeller
propeller_diameter = opti.variable(
    init_guess=1,
    scale=1,
    category="des",
    upper_bound=10,
    lower_bound=1
)

n_propellers = opti.parameter(value=2)

# import airfoil information
wing_airfoil = asb.Airfoil(
    name='HALE_03',
    coordinates="studies/airfoil_optimizer/HALE_03.dat"
)
wing_airfoil.generate_polars(
    cache_filename="HALE_03.json",
    include_compressibility_effects=False,
)
# todo wing airfoil polars change sizing by ~5 meters against previous polars
tail_airfoil = naca0008

# construct the wing
wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([-wing_root_chord / 4, 0, 0]),
            chord=wing_root_chord,
            twist=0,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Break
            xyz_le=np.array([-wing_root_chord / 4, wing_y_taper_break, 0]),
            chord=wing_root_chord,
            twist=0,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([-wing_root_chord * wing_taper_ratio / 4, wing_span / 2, 0]),
            chord=wing_root_chord * wing_taper_ratio,
            twist=0,
            airfoil=wing_airfoil,
        ),
    ]
).translate(np.array([wing_x_quarter_chord, 0, 0]))

# move the taileron relative to the wing
outboard_hstab_x_location = wing_x_quarter_chord + outboard_boom_length - outboard_hstab_chord * 0.75
center_boom_diameter = 0.2  # meters

# build a taileron
right_hstab = asb.Wing(
    name="Taileron",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([0, 0, 0]),
            chord=outboard_hstab_chord,
            twist=-3,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, outboard_hstab_span / 2, 0]),
            chord=outboard_hstab_chord,
            twist=-3,
            airfoil=tail_airfoil,
        ),
    ]
).translate(np.array([
    outboard_boom_length - outboard_hstab_chord * 0.75,
    boom_offset,
    center_boom_diameter / 2]))

# copy right taileron to a left taileron
left_hstab = right_hstab.translate([
    0,
    -boom_offset * 2,
    0])

# define vstab location
center_vstab_x_location = wing_x_quarter_chord + center_boom_length - center_vstab_chord * 0.25

# build a vertical stabilizer
center_vstab = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([0, 0, 0]),
            chord=center_vstab_chord,
            twist=0,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, 0, center_vstab_span]),
            chord=center_vstab_chord,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ]
).translate(
    np.array([center_boom_length - center_vstab_chord * 0.75,
              0,
              -center_vstab_span * 0.35])
)

# define hstab location
center_hstab_x_location = center_vstab_x_location - center_hstab_chord

# build a horizontal stabilizer
center_hstab = asb.Wing(
    name="Horizontal Stabilizer",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([0, 0, 0]),
            chord=center_hstab_chord,
            twist=-3,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, center_hstab_span / 2, 0]),
            chord=center_hstab_chord,
            twist=-3,
            airfoil=tail_airfoil,
        ),
    ]
).translate(
    np.array([center_boom_length - center_vstab_chord * 0.75 - center_hstab_chord,
              0,
              center_boom_diameter / 2])
)

# build center boom
center_boom = asb.Fuselage(
    name="Center Boom",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0, 0, 0],
            radius=center_boom_diameter / 2,
        ),
        asb.FuselageXSec(
            xyz_c=[center_boom_length, 0, 0],
            radius=center_boom_diameter / 2,
        )
    ]
).translate(np.array([wing_x_quarter_chord, 0, 0]))

opti.subject_to([
    center_boom_length - center_vstab_chord - center_hstab_chord > wing_root_chord, #todo review this constraint
    center_vstab.area() < 0.1 * wing.area(),
])

# build outboard booms
outboard_boom_diameter = 0.1  # meters
right_boom = asb.Fuselage(
    name="Outboard Boom",
    xsecs=[
        asb.FuselageXSec(
            xyz_c=[0, 0, 0],
            radius=outboard_boom_diameter / 2,
        ),
        asb.FuselageXSec(
            xyz_c=[outboard_boom_length, 0, 0],
            radius=outboard_boom_diameter / 2,
        )
    ]
)
right_boom = right_boom.translate(np.array([wing_x_quarter_chord, boom_offset, 0]))
left_boom = right_boom.translate(np.array([0, -2 * boom_offset, 0]))

# define length of payload pod struts as function of the payload pod location
# to be used to bookkeep the strut mass and aerodynamics
forward_payload_pod_strut_x_location = x_payload_pod + 0.2 * payload_pod_length + 0.5
rear_payload_pod_strut_x_location = x_payload_pod + payload_pod_length * 0.8 + 0.5
payload_pod_strut_diameter = 0.05

# assume payload pod strut is a NACA 0008 airfoil like the stabilizers todo review this
strut_airfoil = naca0008

# make wing for the payload strut fairing
payload_pod_forward_strut = asb.Wing(
    name="Payload Pod Forward Strut",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([wing_x_quarter_chord, 0, 0]),
            chord=payload_pod_strut_diameter * 2,
            twist=0,  # degrees
            airfoil=strut_airfoil,  # Airfoils are blended between a given XSec and the next one.
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([forward_payload_pod_strut_x_location, 0, -payload_pod_y_offset]),
            chord=payload_pod_strut_diameter * 2,
            twist=0,
            airfoil=strut_airfoil,
        ),
        ]
)

# make wing for the payload strut fairing
payload_pod_rear_strut = asb.Wing(
    name="Payload Pod Rear Strut",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([wing_x_quarter_chord, 0, 0]),
            chord=payload_pod_strut_diameter * 2,
            twist=0,  # degrees
            airfoil=strut_airfoil,  # Airfoils are blended between a given XSec and the next one.
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([rear_payload_pod_strut_x_location, 0, -payload_pod_y_offset]),
            chord=payload_pod_strut_diameter * 2,
            twist=0,
            airfoil=strut_airfoil,
        ),
        ]
)
# define lengths of payload pod struts to use for mass bookkeeping
payload_pod_rear_strut_length = payload_pod_rear_strut.span()
payload_pod_forward_strut_length = payload_pod_forward_strut.span()

# Assemble the airplane
airplane = asb.Airplane(
    name="Solar1",
    xyz_ref=np.array([0, 0, 0]),
    wings=[
        wing,
        center_hstab,
        right_hstab,
        left_hstab,
        center_vstab,
        payload_pod_forward_strut,
        payload_pod_rear_strut,
    ],
    fuselages=[
        center_boom,
        right_boom,
        left_boom,
        payload_pod
    ],
)


##### Section: Internal Geometry and Weights
mass_props = {}

wing_n_ribs = opti.variable(
    init_guess=100,
    scale=50,
    lower_bound=1,
    upper_bound=500,
    **des
)

# wing primary mass model is interpolated model from Axalp study in Fall 2021
mass_wing_primary = lib_mass_struct.mass_wing_spar(
    span=wing.span(),  # effective span
    mass_supported=mass_total,
    # technically the spar doesn't really have to support its own weight (since it's roughly spanloaded), so this is conservative
    ultimate_load_factor=structural_load_factor,
    n_booms=1
) * 11.382 / 9.222  # scaling factor taken from Daedalus weights to account for real-world effects, non-cap mass, etc.


# wing secondary mass model from DAE11 Juan Cruz paper
def estimate_mass_wing_secondary(
        span,
        chord,
        n_ribs,  # You should optimize on this, there's a trade between rib weight and LE sheeting weight!
        skin_density,
        n_wing_sections=1,  # defaults to a single-section wing (be careful: can you disassemble/transport this?)
        t_over_c=0.128,  # default from DAE11
        # Should we include the mass of the spar? Useful if you want to do your own primary structure calculations.
        scaling_factor=1.0  # Scale-up factor for masses that haven't yet totally been pinned down experimentally
):
    """
    Finds the mass of the wing structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    :param span: wing span [m]
    :param chord: wing mean chord [m]
    :param vehicle_mass: aircraft gross weight [kg]
    :param n_ribs: number of ribs in the wing
    :param n_wing_sections: number of wing sections or panels (for disassembly?)
    :param ultimate_load_factor: ultimate load factor [unitless]
    :param type: Type of bracing: "cantilevered", "one-wire", "multi-wire"
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :param include_spar: Should we include the mass of the spar? Useful if you want to do your own primary structure calculations. [boolean]
    :return: Wing structure mass [kg]
    """
    ### Secondary structure
    # ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord
    n_end_ribs = n_wing_sections + 1 + 4 # four for the wing break and one for the center rib # todo check
    area = span * chord

    # Rib mass
    W_wr = n_ribs * (chord ** 2 * t_over_c * 5.50e-2 + chord * 1.91e-3) * 1.3
    # x1.3 scales to estimates from structures subteam

    # End rib mass
    W_wer = n_end_ribs * (chord ** 2 * t_over_c * 6.62e-1 + chord * 6.57e-3)

    # LE sheeting mass
    # W_wLE = 0.456/2 * (span ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # Skin Panel Mass
    W_wsp = area * skin_density * 1.05  # assumed constant thickness from 0.9c around LE to 0.15c

    # TE mass
    W_wTE = span * 2.77e-2

    # Covering
    W_wc = area * 0.033 * 1.1  # 0.033 kg/m2 Tedlar covering on 1 sides, with 1.1 coverage factor

    mass_secondary = (W_wr + W_wer + W_wTE) * scaling_factor + W_wsp + W_wc
    mass_covering = W_wc
    mass_skin_panel = W_wsp
    mass_ribs = W_wr
    mass_end_ribs = W_wer

    return mass_secondary, mass_covering, mass_skin_panel, mass_ribs, mass_end_ribs

# todo check skin areal density
skin_areal_density = 0.30  # kg/m^2

mass_wing_secondary, mass_covering, mass_skin_panel, mass_ribs, mass_end_ribs = estimate_mass_wing_secondary(
    span=wing.span(),
    chord=wing.mean_geometric_chord(),
    # n_ribs=n_ribs_wing,
    n_ribs=wing.span() * 3,
    # depron 3mm = 120gsm, 6mm = 200gsm
    skin_density=skin_areal_density,
    n_wing_sections=4,
    t_over_c=wing_airfoil.max_thickness(),
    scaling_factor=1.5  # Suggested by Drela # todo reavaluate this
)

mass_wing = mass_wing_primary + mass_wing_secondary
mass_props['wing_primary'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_wing_primary * structural_mass_margin_multiplier,
    x_cg=wing_x_quarter_chord,
    z_cg=wing.aerodynamic_center()[2],
    radius_of_gyration_x=wing_span / 12,
    radius_of_gyration_y=wing.xsecs[0].xyz_le[0] + wing_root_chord / 2,
    radius_of_gyration_z=wing.xsecs[0].xyz_le[0] + wing_root_chord / 2
)
wing_x_le = wing_x_quarter_chord - wing_root_chord / 4
mass_props['wing_secondary'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_wing_secondary * structural_mass_margin_multiplier,
    x_cg=wing_x_le + wing_root_chord / 2,
    z_cg=wing.aerodynamic_center()[2],
    radius_of_gyration_x=wing_span / 12,
    radius_of_gyration_y=wing.xsecs[0].xyz_le[0] + wing_root_chord / 2,
    radius_of_gyration_z=wing.xsecs[0].xyz_le[0] + wing_root_chord / 2
)

opti.subject_to([
    x_payload_pod < wing_x_le #+ wing_root_chord / 4
])
# Stabilizers
q_ne = opti.variable(
    init_guess=93,
    scale=10,
    category="des"
)  # Never-exceed dynamic pressure [Pa].

def mass_hstab(
        hstab,
        n_ribs_hstab,
):
    mass_hstab_primary = lib_mass_struct.mass_wing_spar(
        span=hstab.span(),
        mass_supported=q_ne * 1.5 * hstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    )

    mass_hstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=hstab.span(),
        chord=hstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_hstab,
        t_over_c=0.08,
        include_spar=False
    )
    mass_hstab = mass_hstab_primary + mass_hstab_secondary  # per hstab
    return mass_hstab


n_ribs_center_hstab = opti.variable(
    init_guess=17,
    scale=10,
    category="des",
    log_transform=True
)

mass_props['center_hstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_hstab(center_hstab, n_ribs_center_hstab) * structural_mass_margin_multiplier,
    x_cg=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
    z_cg=center_hstab.aerodynamic_center()[2],
    radius_of_gyration_x=center_hstab_span / 12,
    radius_of_gyration_y=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
    radius_of_gyration_z=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2
)

n_ribs_outboard_hstab = opti.variable(
    init_guess=7.5,
    scale=1,
    category="des",
    log_transform=True,
)

mass_props['right_hstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_hstab(right_hstab, n_ribs_outboard_hstab) * structural_mass_margin_multiplier,
    x_cg=right_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
    z_cg=right_hstab.aerodynamic_center()[2],
    radius_of_gyration_x=outboard_hstab_span / 12,
    radius_of_gyration_y=right_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
    radius_of_gyration_z=right_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2
)

mass_props['left_hstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_hstab(left_hstab, n_ribs_outboard_hstab) * structural_mass_margin_multiplier,
    x_cg=left_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
    z_cg=left_hstab.aerodynamic_center()[2],
    radius_of_gyration_x=outboard_hstab_span / 12,
    radius_of_gyration_y=left_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
    radius_of_gyration_z=left_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2
)

def mass_vstab(
        vstab,
        n_ribs_vstab,
):
    mass_vstab_primary = lib_mass_struct.mass_wing_spar(
        span=vstab.span(),
        mass_supported=q_ne * 1.5 * vstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    ) * 1.2  # TODO due to asymmetry, a guess
    mass_vstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=vstab.span(),
        chord=vstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_vstab,
        t_over_c=0.08
    )
    # Calculate solar mounting mass.
    if tail_panels is True:
        skin_mass = vstab.area() * skin_areal_density * 2
    else:
        skin_mass = 0
    mass_vstab = skin_mass + mass_vstab_primary + mass_vstab_secondary  # per vstab
    return mass_vstab


n_ribs_vstab = opti.variable(
    init_guess=10,
    scale=1,
    category="des"
)
opti.subject_to(n_ribs_vstab > 0)

mass_props['vstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_vstab(center_vstab, n_ribs_vstab) * structural_mass_margin_multiplier,
    x_cg=center_vstab.xsecs[0].xyz_le[0] + center_vstab_chord / 2,
    z_cg=center_vstab.aerodynamic_center()[2],
    radius_of_gyration_x=center_vstab_span / 12,
    radius_of_gyration_y=center_vstab.xsecs[0].xyz_le[0] + center_vstab_chord / 2,
    radius_of_gyration_z=center_vstab.xsecs[0].xyz_le[0] + center_vstab_chord / 2
)
# Fuselage & Boom
mass_props['center_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=lib_mass_struct.mass_hpa_tail_boom(
        length_tail_boom=center_boom_length,
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=center_hstab.area() + center_vstab.area()
    ) * structural_mass_margin_multiplier * 0.7, # scaled according to demonstrator build mass of fuselage
    x_cg=center_boom_length / 2,
    radius_of_gyration_y=center_boom_length / 3,
    radius_of_gyration_z=center_boom_length / 3,
)

mass_props['left_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=lib_mass_struct.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=right_hstab.area()
    ) * structural_mass_margin_multiplier,
    x_cg=outboard_boom_length / 2,
    radius_of_gyration_y=outboard_boom_length / 3,
    radius_of_gyration_z=outboard_boom_length / 3,
)

mass_props['right_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=lib_mass_struct.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=left_hstab.area()
    ) * structural_mass_margin_multiplier,
    x_cg=outboard_boom_length / 2,
    radius_of_gyration_y=outboard_boom_length / 3,
    radius_of_gyration_z=outboard_boom_length / 3,
)

# The following taken from Daedalus: taken from Daedalus, http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
mass_daedalus = 103.9  # kg, corresponds to 229 lb gross weight. Total mass of the Daedalus aircraft, used as a reference for scaling.
mass_fairings = 2.067 * mass_total / mass_daedalus  # Scale fairing mass to same mass fraction as Daedalus
mass_landing_gear = 0.728 * mass_total / mass_daedalus  # Scale landing gear mass to same mass fraction as Daedalus
mass_strut_forward = payload_pod_forward_strut_length * 0.55 # assumed mass per meter of carbon fiber tube length (about 10 cm diameter)
mass_strut_rear = payload_pod_rear_strut_length * 0.55
mass_payload_pod_shell = payload_pod.area_wetted() * 2 * 0.2 # 0.2 kg/m^2 is the mass of kevlar sheet
mass_payload_pod_structure = 75 * payload_pod.volume() # todo make this better
# areal_density_thermal_mat =   # kg/m^2, assumed mass per unit area of thermal management material
# mass_thermal_management_system = payload_pod.area_wetted() * areal_density_thermal_mat
mass_props['payload_pod'] = asb.MassProperties(
    mass=(
            mass_fairings +
            mass_landing_gear +
            mass_strut_forward +
            mass_strut_rear +
            mass_payload_pod_structure +
            mass_payload_pod_shell
    ),
    x_cg=x_payload_pod + payload_pod_length / 2
) * structural_mass_margin_multiplier

### summation of structural mass
mass_structural = (
        mass_props['wing_primary'] +
        mass_props['wing_secondary'] +
        mass_props['center_hstab'] +
        mass_props['left_hstab'] +
        mass_props['right_hstab'] +
        mass_props['center_boom'] +
        mass_props['left_boom'] +
        mass_props['right_boom'] +
        mass_props['payload_pod']
)

### Avionics
# todo verify against avionics buildup
number_of_actuators = 4
actuator_mass = number_of_actuators * 0.760
processor_mass = 0.140
flight_computer_mass = 0.650
vhf_transciever_mass = 0.453
vhf_antenna_mass = 0.227
radio_transceiver_and_antenna_mass = 0.090
l_band_blade_antenna_mass = 0.055
iridium_satellite_transceiver_mass = 0.185
gps_antenna_mass = 0.176
emergency_locator_transmitter_mass = 0.908
transponder_mass = 0.100
transponder_antenna_mass = 0.109
avionics_margin = 5.6
mass_props['avionics'] = asb.MassProperties(
    mass=(
            actuator_mass +
            processor_mass +
            flight_computer_mass +
            vhf_transciever_mass +
            vhf_antenna_mass +
            radio_transceiver_and_antenna_mass +
            l_band_blade_antenna_mass +
            iridium_satellite_transceiver_mass +
            gps_antenna_mass +
            emergency_locator_transmitter_mass +
            transponder_mass +
            transponder_antenna_mass +
            avionics_margin
    ),
    x_cg=payload_pod_length * 0.75,  # right behind payload # TODO revisit this number
) # todo verify location
avionics_volume = mass_props['avionics'].mass / 1250  # assumed density of electronics
avionics_power = 180  # TODO revisit this number


### Power Systems Mass Accounting
solar_cell_efficiency = 0.30 * 0.93 * 0.93
# 30% efficiency of Sunpower cells at AM0, 7% knockdown for encapsulation, 7% knockdown for wing shape
# as decided 7/17 with Andrew Streett
rho_solar_cells = 0.56 # kg/m^2, areal density of solar cells as determined from 7/17 discussion with Andrew Streett
solar_cost_per_watt = 3 # $/W, cost of Sunpower solar cells
solar_power_ratio = 500 # W/kg

wing_solar_area_fraction = opti.variable(
    init_guess=0.8,
    scale=0.5,
    lower_bound=0,
    upper_bound=max_wing_solar_area_fraction,
    **des
)

vstab_solar_area_fraction = opti.variable(
    init_guess=0.8,
    scale=0.5,
    lower_bound=0,
    upper_bound=max_vstab_solar_area_fraction,
    **des
)

if tail_panels == False:
    opti.subject_to([
        vstab_solar_area_fraction == 0,
    ])

area_solar_wing = wing.area() * wing_solar_area_fraction
area_solar_vstab = center_vstab.area() * vstab_solar_area_fraction

mass_props['solar_panel_wing'] = asb.MassProperties(
    mass=area_solar_wing * rho_solar_cells,
    x_cg=wing_x_le + 0.50 * wing_root_chord
)

mass_props['solar_panel_vstab'] = asb.MassProperties(
    mass=area_solar_vstab * rho_solar_cells,
    x_cg=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
)

mass_solar_cells = mass_props['solar_panel_wing'].mass + mass_props['solar_panel_vstab'].mass
cost_solar_cells = mass_solar_cells * solar_cost_per_watt * solar_power_ratio

### MPPT mass accounting
n_MPPT = 5 # todo reconsider number of mppts
mass_props['MPPT'] = asb.MassProperties(
    mass=n_MPPT * lib_solar.mass_MPPT(power_in_after_panels_max / n_MPPT),
) # todo reconsider MPPT mass based on findings of MPPT study for NASA SIBR
# Model taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

### Battery mass accounting
battery_capacity = opti.variable(
    init_guess=114072085,
    scale=1e8,
    lower_bound=0,
    category="des",
)
battery_capacity_watt_hours = battery_capacity / 3600

battery_pack_mass = lib_prop_elec.mass_battery_pack(
    battery_capacity_Wh=battery_capacity_watt_hours,
    battery_cell_specific_energy_Wh_kg=battery_specific_energy_Wh_kg,
    battery_pack_cell_fraction=battery_pack_cell_percentage
)
battery_cell_mass = battery_pack_mass * battery_pack_cell_percentage
cost_batteries = 4 * battery_capacity_watt_hours
battery_density = 500  # kg/m^3, taken from averaged measurements of commercial cells # todo match to known battery specs
battery_volume = battery_pack_mass / battery_density
battery_voltage = 240 # todo reconsider based on RFI motor findings

battery_pack_specific_energy = battery_specific_energy_Wh_kg * u.hour * battery_pack_cell_percentage  # J/kg #TODO double check this is right
battery_total_energy = battery_pack_mass * battery_pack_specific_energy  # J

battery_cg = (battery_volume / payload_pod_volume) * 0.5 * payload_pod_length + x_payload_pod
mass_props['battery_pack'] = asb.MassProperties(
    mass=battery_pack_mass,
    x_cg=battery_cg,
)

### wiring mass acounting
mass_props['wires'] = asb.MassProperties(
    mass=lib_prop_elec.mass_wires(
        wire_length=wing.span() / 2,
        max_current=power_out_propulsion_max / battery_voltage,
        allowable_voltage_drop=battery_voltage * 0.01,
        material="aluminum"
    ),
    x_cg=wing_x_quarter_chord  # assume most wiring is down spar
)

# propeller mass accounting
propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * atmo(altitude=20000).speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi

area_propulsive = np.pi / 4 * propeller_diameter ** 2 * n_propellers

mass_props['propellers'] = asb.MassProperties(
    mass=n_propellers * lib_prop_prop.mass_hpa_propeller(
        diameter=propeller_diameter,
        max_power=power_out_propulsion_max / n_propellers,
        include_variable_pitch_mechanism=variable_pitch
    ),
    x_cg=wing_x_le - 0.25 * propeller_diameter
)

### motors mass accounting
propeller_max_torque = (power_out_propulsion_max / n_propellers) / propeller_rads_per_sec
motor_kv = propeller_rpm / battery_voltage
mass_motor_raw = lib_prop_elec.mass_motor_electric(
    max_power=power_out_propulsion_max / n_propellers,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * n_propellers
motor_mounting_weight_multiplier = 2.0  # Taken from Raymer guidance.

mass_props['motors'] = asb.MassProperties(
    mass=mass_motor_raw * motor_mounting_weight_multiplier, # scaled according to Raymer guidance
    x_cg=wing_x_le
) # todo verify location

# ESC mass accounting
mass_props['esc'] = asb.MassProperties(
    mass=lib_prop_elec.mass_ESC(
        max_power=power_out_propulsion_max / n_propellers,
    ) * n_propellers,
    x_cg=wing_x_le
) # todo verify location

### summation of power and propulsion system mass
mass_power_systems = (
        mass_props['solar_panel_wing'] +
        mass_props['solar_panel_vstab'] +
        mass_props['MPPT'] +
        mass_props['battery_pack']
)

mass_propulsion = (
        mass_props['propellers'] +
        mass_props['motors'] +
        mass_props['esc']
)


##### Section: Setup Dynamics
#account for winds
def wind_speed_func(alt):
    day_array = np.full(shape=alt.shape[0], fill_value=1) * day_of_year
    latitude_array = np.full(shape=alt.shape[0], fill_value=1) * latitude
    speed_func = lib_winds.wind_speed_world_95(alt, latitude_array, day_array)
    return speed_func

# add trajectory constraints depending on trajectory type
if trajectory == 'straight':
    guess_altitude = 14000
    guess_speed = 20
    air_speed = opti.variable(init_guess=guess_speed, n_vars=n_timesteps, lower_bound=min_speed, scale=10)
    track = vehicle_heading * np.pi / 180
    u_e = air_speed * np.cos(track)
    v_e = air_speed * np.sin(track)
    z_e = opti.variable(
        init_guess=-guess_altitude,
        n_vars=n_timesteps,
        scale=1e4,
        category='ops',
        upper_bound=0,
        lower_bound=-40000,
    )
    altitude = -z_e
    wind_speed = wind_speed_func(altitude)
    if run_with_95th_percentile_wind_condition == False:
        wind_speed = np.zeros(n_timesteps)
    wind_speed_x = np.cosd(wind_direction) * wind_speed
    wind_speed_y = np.sind(wind_direction) * wind_speed
    ground_speed_x = u_e - wind_speed_x
    ground_speed_y = v_e - wind_speed_y
    ground_speed = (ground_speed_x ** 2 + ground_speed_y ** 2) ** 0.5
    opti.subject_to(ground_speed > min_speed)
    vehicle_bearing = np.arctan2d(ground_speed_y, ground_speed_x)
    x_e = opti.variable(init_guess=0, n_vars=n_timesteps, scale=1e4, category='ops')
    y_e = opti.variable(init_guess=0, n_vars=n_timesteps, scale=1e4, category='ops')
    opti.constrain_derivative(
        variable=x_e, with_respect_to=time,
        derivative=ground_speed_x
    )
    opti.constrain_derivative(
        variable=y_e, with_respect_to=time,
        derivative=ground_speed_y
    )
    # distance = (x_e ** 2 + y_e ** 2) ** 0.5
    w_e = opti.variable(
        init_guess=0,
        n_vars=n_timesteps,
        scale=1e-4,
        category='ops',
    )
    # speed = (u_e ** 2 + v_e ** 2) ** 0.5
    gamma = np.arctan2(-w_e, air_speed)
    alpha = opti.variable(
        init_guess=5,
        n_vars=n_timesteps,
        scale=4,
        category='ops'
    )
    distance = opti.variable(init_guess=0, n_vars=n_timesteps, scale=1e5, category='ops')
    opti.subject_to([
        altitude[time_periodic_start_index:] / min_cruise_altitude > 1,
        distance[time_periodic_start_index] / 1e5 == 0,
        distance[time_periodic_end_index] / 1e5 > (distance[time_periodic_start_index] + required_headway_per_day) / 1e5,
    ])
    # vehicle_bearing = vehicle_heading

if trajectory == 'circular':
    guess_altitude = 11500
    guess_speed = 16
    ground_speed = opti.variable(init_guess=guess_speed, n_vars=n_timesteps, lower_bound=min_speed, scale=10, category='ops')
    start_angle = opti.variable(init_guess=-12, scale=1, category='ops')
    distance = opti.variable(init_guess=np.linspace(0, 1406264, n_timesteps), scale=1e5, category='ops')
    opti.constrain_derivative(variable=distance, with_respect_to=time, derivative=ground_speed)
    flight_path_radius = opti.variable(init_guess=57073, lower_bound=0, scale=10000, category='ops')
    circular_trajectory_length = 2 * np.pi * flight_path_radius
    angle_radians = distance / flight_path_radius + start_angle
    track = angle_radians + np.pi / 2
    x_e = flight_path_radius * np.cos(angle_radians)
    y_e = flight_path_radius * np.sin(angle_radians)
    z_e = opti.variable(
        init_guess=-guess_altitude,
        n_vars=n_timesteps,
        scale=1e4,
        category='ops',
        upper_bound=0,
        lower_bound=-40000,
    )
    altitude = -z_e
    wind_speed = wind_speed_func(altitude)
    if run_with_95th_percentile_wind_condition == False:
        wind_speed = np.zeros(n_timesteps)
    wind_speed_x = np.cosd(wind_direction) * wind_speed
    wind_speed_y = np.sind(wind_direction) * wind_speed
    ground_speed_x = ground_speed * np.cos(track)
    ground_speed_y = ground_speed * np.sin(track)
    vehicle_bearing = np.arctan2d(ground_speed_y, ground_speed_x)
    u_e = ground_speed_x + wind_speed_x
    v_e = ground_speed_y + wind_speed_y
    air_speed = (u_e ** 2 + v_e ** 2) ** 0.5
    w_e = opti.variable(
        init_guess=0,
        n_vars=n_timesteps,
        scale=1e-4,
        category='ops',
    )
    gamma = np.arctan2(-w_e, air_speed)
    alpha = opti.variable(
        init_guess=5,
        n_vars=n_timesteps,
        scale=4,
        category='ops'
    )
    opti.subject_to([
        altitude[time_periodic_start_index:] / min_cruise_altitude > 1,
        distance[time_periodic_start_index] == 0,
    ])
    revisit_rate = distance[time_periodic_end_index] / circular_trajectory_length
    revisit_period = 24 / revisit_rate

if trajectory == "racetrack":
    guess_altitude = 12000
    guess_speed = 20
    air_speed = opti.variable(init_guess=guess_speed, n_vars=n_timesteps, lower_bound=min_speed, scale=10,
                              category='ops')
    z_e = opti.variable(
        init_guess=-guess_altitude,
        n_vars=n_timesteps,
        scale=1e4,
        category='ops',
        upper_bound=0,
        lower_bound=-40000,
    )
    altitude = -z_e
    alpha = opti.variable(
        init_guess=5,
        n_vars=n_timesteps,
        scale=4,
        category='ops'
    )

if trajectory == 'lawnmower':
    coverage_length = opti.variable(init_guess=10000, scale=1000, lower_bound=0,
                                    category='des')  # meters, the height of the area the aircraft must sample
    coverage_width = opti.variable(init_guess=10000, scale=1000, lower_bound=0,
                                   category='des')  # meters, the width of the area the aircraft must sample
    guess_altitude = 14000
    guess_speed = 20
    air_speed = opti.variable(init_guess=guess_speed, n_vars=n_timesteps, lower_bound=min_speed, scale=10, category='ops')
    start_angle = 0
    turn_radius_1 = opti.variable(init_guess=1000, lower_bound=0, scale=1000, category='ops')
    turn_radius_2 = opti.variable(init_guess=1000, lower_bound=0, scale=1000, category='ops')
    turn_radius_3 = opti.variable(init_guess=1000, lower_bound=0, scale=1000, category='ops')
    distance = opti.variable(init_guess=np.linspace(0, 10000, n_timesteps), scale=1e5, category='ops')
    single_track_distance = np.mod(distance, (coverage_length * 2 + turn_radius_1 * np.pi + turn_radius_2 * np.pi))
    # single_track_distance = np.mod(distance, (1000))
    # track = 0
    track = np.where(
        single_track_distance > coverage_length,
        start_angle + (single_track_distance - coverage_length) / turn_radius_1,
        start_angle)
    track = np.where(
        single_track_distance > coverage_length + turn_radius_1 * np.pi,
        start_angle + np.pi,
        track)
    track = np.where(
        single_track_distance > coverage_length * 2 + turn_radius_1 * np.pi,
        start_angle + np.pi + (single_track_distance - coverage_length * 2 - turn_radius_1 * np.pi) / turn_radius_2,
        track)
    u_e = air_speed * np.cos(track)
    v_e = air_speed * np.sin(track)
    z_e = opti.variable(
        init_guess=-guess_altitude,
        n_vars=n_timesteps,
        scale=1e4,
        category='ops',
        upper_bound=0,
        lower_bound=-40000,
    )
    altitude = -z_e
    wind_speed = wind_speed_func(altitude)
    if run_with_95th_percentile_wind_condition == False:
        wind_speed = np.zeros(n_timesteps)
    wind_speed_x = np.cosd(wind_direction) * wind_speed
    wind_speed_y = np.sind(wind_direction) * wind_speed
    ground_speed_x = u_e - wind_speed_x
    ground_speed_y = v_e - wind_speed_y
    ground_speed = (ground_speed_x ** 2 + ground_speed_y ** 2) ** 0.5
    opti.constrain_derivative(variable=distance, with_respect_to=time, derivative=ground_speed)
    opti.subject_to(ground_speed > min_speed)
    vehicle_bearing = np.arctan2d(ground_speed_y, ground_speed_x)
    x_e = opti.variable(init_guess=0, n_vars=n_timesteps, scale=1e4, category='ops')
    y_e = opti.variable(init_guess=0, n_vars=n_timesteps, scale=1e4, category='ops')
    opti.constrain_derivative(
        variable=x_e, with_respect_to=time,
        derivative=ground_speed_x
    )
    opti.constrain_derivative(
        variable=y_e, with_respect_to=time,
        derivative=ground_speed_y
    )
    w_e = opti.variable(
        init_guess=0,
        n_vars=n_timesteps,
        scale=1e-4,
        category='ops',
    )

    gamma = np.arctan2(-w_e, air_speed)
    alpha = opti.variable(
        init_guess=5,
        n_vars=n_timesteps,
        scale=4,
        category='ops'
    )
    opti.subject_to([
        altitude[time_periodic_start_index:] / min_cruise_altitude > 1,
        distance[time_periodic_start_index] / 1e5 == 0,
        x_e[0] == 0,
        y_e[0] == 0,
    ])

z_km = altitude / 1e3

if hold_cruise_altitude == True:
    cruise_altitude = opti.variable(init_guess=14000, scale=1e4, category='ops')
    opti.subject_to(altitude[time_periodic_start_index:] == cruise_altitude)

# start on the ground if doing climb optimization
if climb_opt:
    opti.subject_to(altitude[0] / 1e4 == 0)

# region Atmosphere
##### Atmosphere
my_atmosphere = atmo(altitude=altitude)
P = my_atmosphere.pressure()
rho = my_atmosphere.density()
T = my_atmosphere.temperature()
mu = my_atmosphere.dynamic_viscosity()
a = my_atmosphere.speed_of_sound()
mach = air_speed / a
g = 9.81  # gravitational acceleration, m/s^2
q = 1 / 2 * rho * air_speed ** 2  # Solar calculations
opti.subject_to(q_ne / 100 > q * q_ne_over_q_max / 100)


# endregion

##### Section: Aerodynamics
##### Aerodynamics

# Fuselage
def compute_fuse_aerodynamics(fuse: asb.Fuselage):
    fuse.Re = rho / mu * air_speed * fuse.length()
    fuse.CLA = 0
    fuse.CDA = aero_lib.Cf_flat_plate(fuse.Re) * fuse.area_wetted() * 1.2  # wetted area with form factor

    fuse.lift = fuse.CLA * q  # per fuse
    fuse.drag = fuse.CDA * q  # per fuse


compute_fuse_aerodynamics(center_boom)
compute_fuse_aerodynamics(left_boom)
compute_fuse_aerodynamics(right_boom)
compute_fuse_aerodynamics(payload_pod)


# Wing
def compute_wing_aerodynamics(
        surface: asb.Wing,
        incidence_angle: float = 0,
        is_horizontal_surface: bool = True
):
    surface.alpha_eff = incidence_angle + surface.mean_twist_angle()
    if is_horizontal_surface:
        surface.alpha_eff += alpha

    surface.Re = rho / mu * air_speed * surface.mean_geometric_chord()
    surface.airfoil = surface.xsecs[0].airfoil
    try:

        # surface.Cl_inc = surface.airfoil.CL_function(
        #     {'alpha': surface.alpha_eff, 'reynolds': np.log(surface.Re)})  # Incompressible 2D lift coefficient

        airfoil_aero = surface.airfoil.get_aero_from_neuralfoil(
            alpha=surface.alpha_eff, Re=surface.Re, model_size="medium"
        )
        surface.Cl_inc = airfoil_aero["CL"]

        surface.CL = surface.Cl_inc * aero_lib.CL_over_Cl(surface.aspect_ratio(), mach=mach,
                                                      sweep=surface.mean_sweep_angle())  # Compressible 3D lift coefficient
        surface.lift = surface.CL * q * surface.area()

        surface.Cd_profile = np.exp(
            surface.airfoil.CD_function({'alpha': surface.alpha_eff, 'reynolds': np.log(surface.Re)}))
        surface.Cd_profile = airfoil_aero["CD"]
        surface.drag_profile = surface.Cd_profile * q * surface.area()

        surface.oswalds_efficiency = aero_lib.oswalds_efficiency(
            taper_ratio=surface.taper_ratio(),
            aspect_ratio=surface.aspect_ratio(),
            sweep=surface.mean_sweep_angle()
        )
        surface.drag_induced = aero_lib.induced_drag(
            lift=surface.lift,
            span=surface.span(),
            dynamic_pressure=q,
            oswalds_efficiency=surface.oswalds_efficiency
        )

        surface.drag = surface.drag_profile + surface.drag_induced

        surface.Cm_inc = surface.airfoil.CM_function(
            {'alpha': surface.alpha_eff, 'reynolds': np.log(surface.Re)})  # Incompressible 2D moment coefficient
        surface.Cm_inc = airfoil_aero["CM"]
        surface.CM = surface.Cm_inc * aero_lib.CL_over_Cl(surface.aspect_ratio(), mach=mach,
                                                      sweep=surface.mean_sweep_angle())  # Compressible 3D moment coefficient
        surface.moment = surface.CM * q * surface.area() * surface.mean_geometric_chord()
    except TypeError:
        surface.Cl_inc = surface.airfoil.CL_function(surface.alpha_eff, surface.Re, 0)  # Incompressible 2D lift coefficient
        surface.CL = surface.Cl_inc * aero_lib.CL_over_Cl(surface.aspect_ratio(), mach=mach,
                                                      sweep=surface.mean_sweep_angle())  # Compressible 3D lift coefficient
        surface.lift = surface.CL * q * surface.area()

        surface.Cd_profile = surface.airfoil.CD_function(surface.alpha_eff, surface.Re, mach)
        surface.drag_profile = surface.Cd_profile * q * surface.area()

        surface.oswalds_efficiency = aero_lib.oswalds_efficiency(
            taper_ratio=surface.taper_ratio(),
            aspect_ratio=surface.aspect_ratio(),
            sweep=surface.mean_sweep_angle()
        )
        surface.drag_induced = aero_lib.induced_drag(
            lift=surface.lift,
            span=surface.span(),
            dynamic_pressure=q,
            oswalds_efficiency=surface.oswalds_efficiency
        )

        surface.drag = surface.drag_profile + surface.drag_induced

        surface.Cm_inc = surface.airfoil.CM_function(surface.alpha_eff, surface.Re, 0)  # Incompressible 2D moment coefficient
        surface.CM = surface.Cm_inc * aero_lib.CL_over_Cl(surface.aspect_ratio(), mach=mach,
                                                      sweep=surface.mean_sweep_angle())  # Compressible 3D moment coefficient
        surface.moment = surface.CM * q * surface.area() * surface.mean_geometric_chord()


compute_wing_aerodynamics(wing)
compute_wing_aerodynamics(center_hstab, incidence_angle=center_hstab_twist)
compute_wing_aerodynamics(right_hstab, incidence_angle=outboard_hstab_twist)
compute_wing_aerodynamics(left_hstab, incidence_angle=outboard_hstab_twist)
compute_wing_aerodynamics(center_vstab, is_horizontal_surface=False)
compute_wing_aerodynamics(payload_pod_forward_strut)
compute_wing_aerodynamics(payload_pod_rear_strut)

# Increase the wing drag due to tripped flow (8/17/20)
wing_drag_multiplier = opti.parameter(value=1.06)  # TODO review
wing.drag *= wing_drag_multiplier

# Force totals
lift_force = (
        wing.lift +
        center_hstab.lift +
        right_hstab.lift +
        left_hstab.lift +
        center_boom.lift +
        right_boom.lift +
        left_boom.lift +
        payload_pod.lift
)
drag_force = (
        wing.drag +
        center_hstab.drag +
        right_hstab.drag +
        left_hstab.drag +
        center_vstab.drag +
        center_boom.drag +
        right_boom.drag +
        left_boom.drag +
        payload_pod_forward_strut.drag +
        payload_pod_rear_strut.drag +
        payload_pod.drag
)

drag_induced = (
        wing.drag_induced +
        center_hstab.drag_induced +
        right_hstab.drag_induced +
        left_hstab.drag_induced +
        center_vstab.drag_induced
)

drag_parasite = drag_force - drag_induced

moment = (
        -wing.aerodynamic_center()[0] * wing.lift + wing.moment +
        -center_hstab.aerodynamic_center()[0] * center_hstab.lift + center_hstab.moment +
        -right_hstab.aerodynamic_center()[0] * right_hstab.lift + right_hstab.moment +
        -left_hstab.aerodynamic_center()[0] * left_hstab.lift + left_hstab.moment
)

# endregion
#
# # region Stability
# ### Estimate aerodynamic center
x_ac = (
               wing.aerodynamic_center()[0] * wing.area() +
               center_hstab.aerodynamic_center()[0] * center_hstab.area() +
               right_hstab.aerodynamic_center()[0] * right_hstab.area() +
               left_hstab.aerodynamic_center()[0] * left_hstab.area()
       ) / (
               wing.area() +
               center_hstab.area() +
               right_hstab.area() +
               left_hstab.area()
       )
static_margin_fraction = (x_ac - airplane.xyz_ref[0]) / wing.mean_aerodynamic_chord()
opti.subject_to([
    static_margin_fraction == 0.2,  # Stability condition
    # moment / 1e4 == 0  # Trim condition
    # mass_props_TOGW.xyz_cg[0] <=wing_x_quarter_chord,
    # mass_props_TOGW.xyz_cg[0] >= wing_x_le,
])

### Size the tails off of tail volume coefficients
Vv = center_vstab.area() * (
        center_vstab.aerodynamic_center()[0] - wing.aerodynamic_center()[0]
) / (wing.area() * wing.span())
Vh = center_hstab.area() * (
        center_hstab.aerodynamic_center()[0] - wing.aerodynamic_center()[0]
) / (wing.area() * wing.mean_aerodynamic_chord())

# vstab_effectiveness_factor = aero.CL_over_Cl(center_vstab.aspect_ratio()) / aero.CL_over_Cl(wing.aspect_ratio())
# hstab_effectiveness_factor = aero.CL_over_Cl(center_hstab.aspect_ratio()) / aero.CL_over_Cl(wing.aspect_ratio())

opti.subject_to([
    # Vh * hstab_effectiveness_factor > 0.3,
    # Vh * hstab_effectiveness_factor < 0.6,
    # Vh * hstab_effectiveness_factor == 0.45,
    # Vv * vstab_effectiveness_factor > 0.02,
    # Vv * vstab_effectiveness_factor < 0.05,
    # Vv * vstab_effectiveness_factor == 0.035,
    Vh > 0.2,
    # Vh < 0.6,
    # Vh == 0.45,
    Vv > 0.02,
    # Vv < 0.05,
    # Vv == 0.035,
    # center_hstab.aspect_ratio() >= 3,  # TODO review this aspect ratio limit
    # center_vstab.aspect_ratio() == 3.5,  # TODO review this
    # center_vstab.area() < 0.1 * wing.area(),  # checked with Matt on 12/10/21
    # center_vstab.aspect_ratio() > 1.9, # from Jamie, based on ASWing
    # center_vstab.aspect_ratio() < 2.5 # from Jamie, based on ASWing
])

# dyn.add_force(
#     Fx=-np.cosd(dyn.gamma) * drag_force,
#     Fz=np.sind(dyn.gamma) * drag_force,
#     axes="earth"
# )
# dyn.add_force(
#     Fx=np.sind(dyn.gamma) * lift_force,
#     Fz=-np.cosd(dyn.gamma) * lift_force,
#     axes='earth'
# )


##### Section: Payload

c = 299792458  # [m/s] speed of light
k_b = 1.38064852E-23  # [m2 kg s-2 K-1] boltzman constant
radar_width = opti.variable(
    init_guess=0.3,
    scale=0.1,
    lower_bound=0,
    **des
)
radar_length = opti.variable(
    init_guess=0.6,
    scale=0.1,
    lower_bound=0,
    **des
)
look_angle = opti.variable(
    init_guess=45,
    scale=10,
    upper_bound=65,
    lower_bound=25,
    **des
) # limited in lower bound by specular reflection which we get at low look angles and does not enable interferometry
wavelength = opti.variable(
    init_guess=0.03,
    scale=0.01,
    lower_bound=0.015,
    upper_bound=0.3,
    **des
)
bandwidth = opti.variable(
    init_guess=211985277,
    scale=1e7,
    lower_bound=0, # 200 MHz is typical hardware lower limt
    **des
)  # Hz
pulse_rep_freq = opti.variable(
    init_guess=141676,
    scale=10000,
    lower_bound=0,
    **des
) # 1/s (Hz)
power_trans = opti.variable(
    init_guess=862445,
    scale=1e5,
    lower_bound=0,
    upper_bound=1e8,
    **ops
)  # watts

# define key radar parameters
radar_aperture_area = radar_width * radar_length  # meters ** 2
mass_radar_aperture = 5.55 * radar_aperture_area
dist = altitude / np.cosd(look_angle)  # meters
swath_azimuth = wavelength * dist / radar_length  # meters
swath_range = wavelength * dist / (radar_width * np.cosd(look_angle))  # meters
max_length_synth_ap = wavelength * dist / radar_length  # meters
ground_area = swath_range * swath_azimuth * np.pi / 4  # meters ** 2
radius = (swath_azimuth + swath_range) / 4  # meters
ground_imaging_offset = np.tand(look_angle) * altitude  # meters
scattering_cross_sec = 10 ** (scattering_cross_sec_db / 10)
sigma0 = scattering_cross_sec / ground_area
antenna_gain = 4 * np.pi * radar_aperture_area * 0.7 / wavelength ** 2
max_swath_range = opti.variable(init_guess=8500,scale=1e3,lower_bound=0,**ops)
max_swath_azimuth = opti.variable(init_guess=8500,scale=1e3,lower_bound=0,**ops)

# Assumed constants
a_hs = 0.88  # aperture-illumination taper factor associated with the synthetic aperture (value from Ulaby and Long)
F = 4  # receiver noise figure (somewhat randomly chosen value from Ulaby and Long)
a_B = 1  # pulse-taper factor to relate bandwidth and pulse duration

# # constrain SAR resolution to required value
pulse_duration = a_B / bandwidth
range_resolution = c * pulse_duration / (2 * np.sind(look_angle))
azimuth_resolution = radar_length / 2
critical_baseline = wavelength * dist / (2 * range_resolution * (np.cosd(look_angle)) ** 2)

opti.subject_to([
    payload_pod_length >= radar_length,
    payload_pod_diameter >= radar_width,
    max_swath_range >= swath_range,
    max_swath_azimuth >= swath_azimuth,
])
# use SAR specific equations from Ulaby and Long
payload_power = power_trans * pulse_rep_freq * pulse_duration

if trajectory == 'straight':
    # coverage = opti.variable(init_guess=1e11, scale=1e11, lower_bound=0, category='ops')
    max_swath_range = opti.variable(init_guess=8500, scale=1e3, lower_bound=0, category='ops')
    max_swath_azimuth = opti.variable(init_guess=8500, scale=1e3, lower_bound=0, category='ops')
    opti.subject_to([
        max_swath_range > swath_range,
        max_swath_azimuth > swath_azimuth
    ])
    ground_area = max_swath_range * max_swath_azimuth * np.pi / 4  # meters ** 2
    coverage_area = ground_area * distance[time_periodic_end_index]
    # opti.subject_to(coverage_area >= coverage)
    revisit_period = 1 # todo change this


if trajectory == 'circular':
    coverage_radius = opti.variable(init_guess=2500, scale=1000, lower_bound=0,
                                    category='des')  # meters
    opti.subject_to([
        flight_path_radius >= ground_imaging_offset + swath_range,
        coverage_radius <= swath_range,
                ])
    coverage_area = np.pi * coverage_radius ** 2
    payload_power_adjusted = payload_power

if trajectory == "racetrack":
    max_imaging_offset = opti.variable(init_guess=10000, scale=1e3, lower_bound=0, category='ops')
    max_swath_range = opti.variable(init_guess=10000, scale=1e3, lower_bound=0, category='ops')
    swath_overlap = opti.variable(init_guess=0.5, scale=0.1, lower_bound=0, upper_bound=1, category='ops')
    opti.subject_to([
        max_imaging_offset >= ground_imaging_offset,
        max_swath_range > swath_range,
        ])

    guess_altitude = 12000
    guess_speed = 20
    start_angle = 0
    turn_radius = max_imaging_offset + max_swath_range - max_swath_range * swath_overlap
    coverage_length = opti.variable(init_guess=50000, lower_bound=0, scale=1000, category='ops')
    distance = opti.variable(init_guess=np.linspace(0, 10000, n_timesteps), scale=1e5, category='ops')
    track_trajectory_length = 2 * coverage_length + 2 * turn_radius * np.pi
    single_track_distance = np.mod(distance, track_trajectory_length)
    # track = 45 * np.ones(n_timesteps)
    track = np.where(
        single_track_distance > coverage_length,
        start_angle + (single_track_distance - coverage_length) / turn_radius,
        start_angle)
    track = np.where(
        single_track_distance > coverage_length + turn_radius * np.pi,
        start_angle + np.pi,
        track)
    track = np.where(
        single_track_distance > coverage_length * 2 + turn_radius * np.pi,
        start_angle + np.pi + (single_track_distance - coverage_length * 2 - turn_radius * np.pi) / turn_radius,
        track)
    u_e = air_speed * np.cos(track)
    v_e = air_speed * np.sin(track)
    wind_speed = wind_speed_func(altitude)
    if run_with_95th_percentile_wind_condition == False:
        wind_speed = np.zeros(n_timesteps)
    wind_speed_x = np.cosd(wind_direction) * wind_speed
    wind_speed_y = np.sind(wind_direction) * wind_speed
    ground_speed_x = u_e - wind_speed_x
    ground_speed_y = v_e - wind_speed_y
    ground_speed = (ground_speed_x ** 2 + ground_speed_y ** 2) ** 0.5
    opti.constrain_derivative(variable=distance, with_respect_to=time, derivative=ground_speed)
    opti.subject_to(ground_speed > min_speed)
    vehicle_bearing = np.arctan2d(ground_speed_y, ground_speed_x)
    x_e = opti.variable(init_guess=np.linspace(0, 10000, n_timesteps), scale=1e4, category='ops')
    y_e = opti.variable(init_guess=np.linspace(0, 10000, n_timesteps), scale=1e4, category='ops')
    opti.constrain_derivative(
        variable=x_e, with_respect_to=time,
        derivative=ground_speed_x
    )
    opti.constrain_derivative(
        variable=y_e, with_respect_to=time,
        derivative=ground_speed_y
    )
    w_e = opti.variable(
        init_guess=0,
        n_vars=n_timesteps,
        scale=1e-4,
        category='ops',
    )
    gamma = np.arctan2(-w_e, air_speed)
    opti.subject_to([
        altitude[time_periodic_start_index:] / min_cruise_altitude > 1,
        distance[time_periodic_start_index] / 1e5 == 0,
        x_e[0] == 0,
        y_e[0] == 0,
    ])
    revisit_rate = distance[time_periodic_end_index] / track_trajectory_length
    revisit_period = 24 / revisit_rate

    coverage_area = coverage_length * 2 * max_swath_range - (max_swath_range * swath_overlap)
    payload_power_adjusted = np.where(
        track == 0,
        payload_power,
        0)
    payload_power_adjusted = np.where(
        track == np.pi,
        payload_power,
        payload_power_adjusted
    )
y_km = y_e / 1e3
x_km = x_e / 1e3
if trajectory == 'lawnmower':
    max_imaging_offset = opti.variable(init_guess=8500, scale=1e3, lower_bound=0, category='ops')
    max_swath_range = opti.variable(init_guess=8500, scale=1e3, lower_bound=0, category='ops')
    swath_overlap = opti.variable(init_guess=0.5, scale=0.1, lower_bound=0, upper_bound=1, category='ops')
    single_track_coverage = 2 * max_swath_range - (max_swath_range * swath_overlap)
    passes_required = coverage_width / single_track_coverage
    full_coverage_distance = passes_required * (2 * coverage_length + np.pi * turn_radius_1 + np.pi * turn_radius_2) \
                              - np.pi * turn_radius_2 + np.pi * turn_radius_3
    revisit_rate = distance[time_periodic_end_index] / full_coverage_distance
    revisit_period = 24 / revisit_rate
    total_distance = revisit_rate * full_coverage_distance

    track = np.where(
        distance > full_coverage_distance - (np.pi * turn_radius_2),
        start_angle + np.pi + (single_track_distance - coverage_length * 2 - turn_radius_1 * np.pi) / turn_radius_3,
        track)

    opti.subject_to([
        max_imaging_offset >= ground_imaging_offset,
        passes_required >= 1,
        max_imaging_offset <= ground_imaging_offset + 10,
        max_swath_range <= swath_range + 10,
        max_swath_range > max_imaging_offset,
        max_swath_range > swath_range,
        turn_radius_1 == max_imaging_offset + max_swath_range * swath_overlap / 2,
        turn_radius_2 == max_swath_range - max_imaging_offset - 1.5 * max_swath_range * swath_overlap,
        turn_radius_3 == passes_required * turn_radius_1 + (passes_required - 1) * turn_radius_2,
        ])
    coverage_area = coverage_length * coverage_width  # meters ** 2, the area the aircraft must sample
    payload_power_adjusted = np.where(
        track == 0,
        payload_power,
        0)
    payload_power_adjusted = np.where(
        track == np.pi,
        payload_power,
        payload_power_adjusted
    )

snr = payload_power * antenna_gain ** 2 * wavelength ** 3 * a_hs * sigma0 * range_resolution / \
      ((2 * 4 * np.pi) ** 3 * dist ** 3 * k_b * my_atmosphere.temperature() * F * ground_speed * a_B)

snr_db = 10 * np.log(snr)

opti.subject_to([
    pulse_rep_freq >= 2 * ground_speed / radar_length,
    1 / pulse_rep_freq >= pulse_duration
])
# translate to final data product Terms
N_i_range = opti.variable(init_guess=5, scale=1, lower_bound=1, category='des')
N_i_azimuth = opti.variable(init_guess=5, scale=1, lower_bound=1, category='des')
N_i_time = opti.variable(init_guess=5, scale=1, lower_bound=1, category='des')
N_i = N_i_range * N_i_azimuth * N_i_time
# number of pixels in the incoherent averaging window
deviation = 10 # meters
opti.subject_to([
    strain_azimuth_resolution >= N_i_azimuth * azimuth_resolution,
    strain_range_resolution <= strain_azimuth_resolution,
    strain_range_resolution >= N_i_range * range_resolution,
    strain_temporal_resolution >= N_i_time * revisit_period,
    required_strain_temporal_resolution >= strain_temporal_resolution,
])
p_thermal = 1 / (1 + snr ** (-1))
p_position = 1 - (2 * deviation * range_resolution * (np.cosd(look_angle)) ** 2 / (wavelength * dist))
p_time_volume = -0.0021 * revisit_period + 0.95
decorrelation = p_thermal * p_position * p_time_volume
precision = wavelength / (4 * np.pi * N_i * strain_temporal_resolution * strain_range_resolution) * (
        1 - decorrelation ** 2) / decorrelation ** 2
opti.subject_to(required_strain_precision >= (precision * 24 * 365))

### instrument data storage mass requirements
mass_of_data_storage = 0.0053  # kg per TB of data
mass_props['payload'] = asb.MassProperties(
    mass=mass_payload_base +
         mission_length * tb_per_day * mass_of_data_storage +
         mass_radar_aperture
)

# region Propulsion

### Propeller calculations
thrust = opti.variable(
    n_vars=n_timesteps,
    init_guess=100,
    scale=20,
    category="ops",
)
opti.subject_to([
    thrust >= 0
])

# dyn.add_force(
#     Fx=np.cosd(dyn.gamma + dyn.alpha) * thrust,
#     Fz=np.sind(dyn.gamma + dyn.alpha) * thrust,
#     axes="earth"
# )  # Note, this is not a typo: we make the small-angle-approximation on the flight path angle gamma.

if not use_propulsion_fits_from_FL2020_1682_undergrads:
    ### Use older models

    motor_efficiency = 0.955  # Taken from ThinGap estimates

    power_out_propulsion_shaft = lib_prop_prop.propeller_shaft_power_from_thrust(
        thrust_force=thrust,
        area_propulsive=area_propulsive,
        airspeed=speed,
        rho=rho,
        propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
    )

    gearbox_efficiency = 0.986

else:
    ### Use Jamie's model
    from design_opt_utilities.new_models import eff_curve_fit

    opti.subject_to(altitude < 30000)  # Bugs out without this limiter

    propeller_efficiency, motor_efficiency = eff_curve_fit(
        airspeed=air_speed,
        total_thrust=thrust,
        altitude=altitude,
        var_pitch=variable_pitch
    )
    power_out_propulsion_shaft = thrust * air_speed / propeller_efficiency

    gearbox_efficiency = 0.986

power_out_propulsion = power_out_propulsion_shaft / motor_efficiency / gearbox_efficiency

# Motor thermal modeling
heat_motor = power_out_propulsion * (1 - motor_efficiency) / n_propellers
heat_motor_max = 175  # System is designed to reject 175 W
opti.subject_to(heat_motor <= heat_motor_max)

opti.subject_to([
    power_out_propulsion < power_out_propulsion_max,
    power_out_propulsion_max > 0
])

# Account for avionics power
power_out_avionics = 180  # Pulled from Avionics spreadsheet on 5/13/20
# todo update with results from avionics development
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

### Power accounting
power_out = power_out_propulsion + payload_power_adjusted + power_out_avionics

# endregion

##### Section: Power Input (Solar)

MPPT_efficiency = 0.975 # todo revisit efficiency value
# wing_panel_azimuth = 90 # 90 is worst case
wing_panel_azimuth = vehicle_bearing + 90
solar_flux_on_horizontal = lib_solar.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    scattering=True,
)

left_wing_incident_solar_power = 0.5 * area_solar_wing * lib_solar.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    panel_azimuth_angle=wing_panel_azimuth,
    panel_tilt_angle=10,
    scattering=True
)

right_wing_incident_solar_power = 0.5 * area_solar_wing * lib_solar.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    panel_azimuth_angle=wing_panel_azimuth,
    panel_tilt_angle=-10,
    scattering=True
)

wing_incident_solar_power = right_wing_incident_solar_power + left_wing_incident_solar_power

vstab_incident_solar_power_left = area_solar_vstab * lib_solar.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    panel_azimuth_angle=wing_panel_azimuth,
    panel_tilt_angle=90,
    scattering=True
)

vstab_incident_solar_power_right = area_solar_vstab * lib_solar.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    panel_azimuth_angle=-wing_panel_azimuth,
    panel_tilt_angle=90,
    scattering=True
)

vstab_incident_solar_power = vstab_incident_solar_power_right + vstab_incident_solar_power_left

power_in_after_panels_tot = (wing_incident_solar_power * solar_cell_efficiency +
            vstab_incident_solar_power * solar_cell_efficiency)

power_in = (wing_incident_solar_power * solar_cell_efficiency +
            vstab_incident_solar_power * solar_cell_efficiency) \
           * MPPT_efficiency


### Wiring specs
wiring_losses = 0.99
# 12/21/21 Jean marie Bourven

cost_batteries = 4 * battery_capacity_watt_hours  # dollars assuming 355 whr/kg cells
avionics_cost = 80000
materials_cost = mass_total * 0.5 * 220
total_tech_cost = cost_solar_cells + cost_batteries + avionics_cost + materials_cost

opti.subject_to([
    power_in_after_panels_max > power_in_after_panels_tot,
    power_in_after_panels_max > 0
])

# endregion

#### section: Constrain Dynamics
net_accel_x_e = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e-4,
    category="ops"
)
net_accel_y_e = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e-4,
    category="ops"
)
net_accel_z_e = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e-5,
    category="ops"
)

# start by defining speed, then groundspeed vector,then constrain position on circle as a function of time, track angle is then

# opti.constrain_derivative(
#     variable=x_e, with_respect_to=time,
#     derivative=(u_e)
# )
# opti.constrain_derivative(
#     variable=y_e, with_respect_to=time,
#     derivative=(v_e)
# )
opti.constrain_derivative(
    variable=z_e, with_respect_to=time,
    derivative=w_e,
)
# opti.constrain_derivative(
#     variable=distance, with_respect_to=time,
#     derivative=ground_speed
# )
opti.constrain_derivative(
    variable=u_e, with_respect_to=time,
    derivative=net_accel_x_e,
)
# opti.constrain_derivative(
#     variable=v_e, with_respect_to=time,
#     derivative=net_accel_y_e,
# )
opti.constrain_derivative(
    variable=w_e, with_respect_to=time,
    derivative=net_accel_z_e,
)
# opti.subject_to(ground_speed == air_speed - wind_speed)
# dyn.add_force(
#     Fx=-np.cosd(dyn.gamma) * drag_force,
#     Fz=np.sind(dyn.gamma) * drag_force,
#     axes="earth"
# )
# dyn.add_force(
#     Fx=np.sind(dyn.gamma) * lift_force,
#     Fz=-np.cosd(dyn.gamma) * lift_force,
#     axes='earth'
# )
# dyn.add_force(
#     Fx=np.cosd(dyn.gamma + dyn.alpha) * thrust,
#     Fz=np.sind(dyn.gamma + dyn.alpha) * thrust,
#     axes="earth"
# )
gravity_force = g * mass_total
Fx_e = (- np.cosd(gamma) * drag_force \
        + np.sind(gamma) * lift_force) \
        + np.cosd(gamma + alpha) * thrust
Fy_e = 0
Fz_e = np.sind(gamma) * drag_force \
       - np.cosd(gamma) * lift_force \
       + np.sind(gamma + alpha) * thrust \
       + gravity_force


opti.subject_to([
    net_accel_x_e * mass_total / 1e1 == Fx_e / 1e1,
    net_accel_y_e * mass_total / 1e1 == Fy_e / 1e1,
    net_accel_z_e * mass_total / 1e2 == Fz_e / 1e2,
])

# Fx_w, Fy_w, Fz_w = dyn.convert_axes(Fx_e, Fy_e, Fz_e,'earth','wind')
# opti.subject_to([
#     net_accel_x_e * mass_total / 1e1 == Fx_e / 1e1,
#     net_accel_y_e * mass_total / 1e1 == Fy_e / 1e1,
#     net_accel_z_e * mass_total / 1e2 == Fz_e / 1e2
# ])
# opti.subject_to([
#     dyn.speed ** 2 == dyn.u_e ** 2 + dyn.v_e ** 2,
# ])
# opti.subject_to([
#     dyn.Fy_w == 0,
#     dyn.Fz_w == 0,
# ])
# excess_power = Fx_w * speed
# climb_rate = excess_power / 9.81 / mass_total
# opti.constrain_derivative(
#     variable=altitude, with_respect_to=time,
#     derivative=climb_rate,
# )

##### Section: Battery power

battery_charge_state = opti.variable(
    init_guess=0.5,
    n_vars=n_timesteps,
    lower_bound=1-allowable_battery_depth_of_discharge,
    upper_bound=1,
)

net_power = power_in - power_out

opti.constrain_derivative(
    derivative=net_power / battery_total_energy,
    variable=battery_charge_state,
    with_respect_to=time,
)

#### end section

#### summation of total system mass

mass_props_TOGW = sum(mass_props.values())
opti.subject_to([
    mass_total / 100 > mass_props_TOGW.mass / 100,
])

# track remaining volume in payload pod
remaining_volume = (
        payload_pod_volume - (
        payload_volume +
        avionics_volume +
        battery_volume +
        payload_pod_structure_volume
)
)

##### Add periodic constraints
opti.subject_to([
    altitude[time_periodic_end_index] / 1e4 > altitude[time_periodic_start_index] / 1e4,
    air_speed[time_periodic_end_index] / 2e1 > air_speed[time_periodic_start_index] / 2e1,
    battery_charge_state[time_periodic_end_index] > battery_charge_state[time_periodic_start_index],
    gamma[time_periodic_end_index] == gamma[time_periodic_start_index],
    alpha[time_periodic_end_index] == alpha[time_periodic_start_index],
    thrust[time_periodic_end_index] / 1e2 == thrust[time_periodic_start_index] / 1e2,
])


##### Add objective
objective = eval(minimize)

#### Add additional constraints
opti.subject_to([
    # center_hstab_chord <= wing_root_chord,
    # center_hstab_span == outboard_hstab_span,
    # center_hstab_chord == outboard_hstab_chord,
    center_hstab_twist <= 0,  # essentially enforces downforce, prevents hstab from lifting and exploiting config.
    outboard_hstab_twist <= 0, # essentially enforces downforce, prevents hstab from lifting and exploiting config.
    center_hstab_twist >= -15,
    outboard_hstab_twist >= -15,
    remaining_volume >= 0,
    alpha < 12,
    alpha > -8,
    np.diff(alpha) < 2,
    np.diff(alpha) > -2,
    center_boom_length >= outboard_boom_length,
    center_boom_length >= payload_pod_length,
    # outboard_hstab_chord == center_hstab_chord,
    # outboard_hstab_span == center_hstab_span,
    outboard_hstab_chord < wing_root_chord,
    center_vstab.area() <= 0.2 * wing.area(),
])

##### Useful metrics
wing_loading = 9.81 * mass_total / wing.area()
wing_loading_psf = wing_loading / 47.880258888889
empty_wing_loading = 9.81 * mass_structural / wing.area()
empty_wing_loading_psf = empty_wing_loading / 47.880258888889
propeller_efficiency = thrust * air_speed/ power_out_propulsion_shaft
cruise_LD = lift_force / drag_force
avg_cruise_LD = np.mean(cruise_LD)
avg_airspeed = np.mean(air_speed)
max_snr = np.max(snr_db)
max_precision = np.max(precision) * 24 * 365 # 1/yr
cruise_altitude = np.mean(altitude)
sl_atmosphere = atmo(altitude=0)
rho_ratio = np.sqrt(np.mean(my_atmosphere.density()) / sl_atmosphere.density())
avg_ias = avg_airspeed * rho_ratio

##### Add tippers
things_to_slightly_minimize = 0

for tipper_input in [
    wing_span / 50,
    n_propellers / 2,
    propeller_diameter / 3,
    battery_capacity_watt_hours / 50000,
    wing_solar_area_fraction / 0.5,
    (hour[-1] - hour[0]) / 24,
]:
    try:
        things_to_slightly_minimize += tipper_input
    except NameError:
        pass

# Dewiggle
penalty = 0

for penalty_input in [
    thrust / 10,
    Fz_e / 1e-1,
    Fx_e / 5e-1,
    air_speed / 2,
    gamma / 2,
    alpha / 1
]:
    # penalty += np.sum(np.diff(np.diff(penalty_input)) ** 2) / n_timesteps_per_segment ## old version
    penalty += np.mean(integrate_discrete_squared_curvature(penalty_input)) ## peter suggested version
opti.minimize(
    objective
    + penalty
    + 1e-3 * things_to_slightly_minimize
)
# endregion

# Debug set to draw the initial guess to check if things look about right.
if draw_initial_guess_config:
    try:
        opti.solve(max_iter=1)
    except:
        airplane.substitute_solution(opti.debug)
        airplane.draw()

if __name__ == "__main__":
    # wingspan_terms = [1, 0.8, 0.6, 0.4, 0.2, 0]
    # spatial = []
    # wingspan = []
    # coverage = []
    # for wingspan_term in wingspan_terms:
    #                 opti.set_value(wingspan_optimization_scaling_term, wingspan_term)
    #                 coverage_term = 1 - wingspan_term
    #                 opti.set_value(coverage_optimization_scaling_term, coverage_term)
                    try:
                        sol = opti.solve(
                            max_iter=50000,
                            options={
                                "ipopt.max_cpu_time": 60000
                            }
                        )
                        print("Success!")
                        # spatial.append(opti.value(strain_azimuth_resolution))
                        # wingspan.append(opti.value(wing_span))
                        # coverage.append(opti.value(coverage_area))
                    except:
                        sol = opti.debug
                    # Print a warning if the penalty term is unreasonably high
                    penalty_objective_ratio = np.abs(sol.value(penalty / objective))
                    if penalty_objective_ratio > 0.01:
                        print(
                            f"\nWARNING: High penalty term, non-negligible integration error likely! P/O = {penalty_objective_ratio}\n")


                    # # region Postprocessing utilities, console output, etc.
                    def s(x):  # Shorthand for evaluating the value of a quantity x at the optimum
                        return sol.value(x)


                    def output(x: Union[str, List[str]]) -> None:  # Output a scalar variable (give variable name as a string).
                        if isinstance(x, list):
                            for xi in x:
                                output(xi)
                            return
                        print(f"{x}: {sol.value(eval(x)):.3f}")


                    def print_title(s: str) -> None:  # Print a nicely formatted title
                        print(f"\n{'*' * 10} {s.upper()} {'*' * 10}")


                    sl_atmosphere = atmo(altitude=0)
                    rho_ratio = np.sqrt(np.mean(rho) / sl_atmosphere.density())

                    def fmt(x):
                            return f"{s(x):.6g}"

                    print_title("Outputs")
                    for k, v in {
                            "Wing Span": f"{fmt(wing_span)} meters",
                            "Strain Range Resolution": f"{fmt(strain_range_resolution)} meters",
                            "Strain Azimuth Resolution": f"{fmt(strain_azimuth_resolution)} meters",
                            "Strain Temporal Resolution": f"{fmt(strain_temporal_resolution)} hours",
                            "Coverage Area": f"{fmt(coverage_area)} meters^2",
                            "Cruise Altitude": f"{fmt(cruise_altitude / 1000)} kilometers",
                            "Average Airspeed": f"{fmt(avg_airspeed)} m/s",
                            "Wing Root Chord": f"{fmt(wing_root_chord)} meters",
                            "mass_TOGW": f"{fmt(mass_total)} kg",
                            "Average Cruise L/D": fmt(avg_cruise_LD),
                            "CG location": "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
                        }.items():
                            print(f"{k.rjust(25)} = {v}")

                    import csv

                    # Define the data for the "Outputs" section
                    outputs_data = {
                        "Wing Span": f"{fmt(wing_span)} meters",
                        "Strain Range Resolution": f"{fmt(strain_range_resolution)} meters",
                        "Strain Azimuth Resolution": f"{fmt(strain_azimuth_resolution)} meters",
                        "Strain Temporal Resolution": f"{fmt(strain_temporal_resolution)} hours",
                        "Coverage Area": f"{fmt(coverage_area)} meters^2",
                        "Cruise Altitude": f"{fmt(cruise_altitude / 1000)} kilometers",
                        "Average Airspeed": f"{fmt(avg_airspeed)} m/s",
                        "Wing Root Chord": f"{fmt(wing_root_chord)} meters",
                        "mass_TOGW": f"{fmt(mass_total)} kg",
                        "Average Cruise L/D": fmt(avg_cruise_LD),
                        "CG location": "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
                    }

                    fmtpow = lambda x: fmt(x) + " W"

                    print_title("Payload Terms")
                    for k, v in {
                            "payload power": fmtpow(payload_power),
                            "payload mass": f"{fmt(mass_props['payload'].mass)} kg",
                            "aperture length": f"{fmt(radar_length)} meters",
                            "aperture width": f"{fmt(radar_width)} meters",
                            "SAR range resolution": f"{fmt(range_resolution)} meters",
                            "SAR azimuth resolution": f"{fmt(azimuth_resolution)} meters",
                            "SAR revisit period": f"{fmt(revisit_period)} hours",
                            "pixels in the incoherent averaging window": fmt(N_i),
                            "precision": f"{fmt(max_precision)} 1 / yr",
                            "pulse repetition frequency": f"{fmt(pulse_rep_freq)} Hz",
                            "bandwidth": f"{fmt(bandwidth)} Hz",
                            "center wavelength": f"{fmt(wavelength)} meters",
                            "look angle": f"{fmt(look_angle)} degrees",
                            "swath range": f"{fmt(max_swath_range)} meters",
                            "swath azimuth": f"{fmt(max_swath_azimuth)} meters",
                            "SNR": f"{fmt(max_snr)} dB",
                        }.items():
                            print(f"{k.rjust(25)} = {v}")

                    print_title("Powers")
                    for k, v in {
                            "max_power_in": fmtpow(power_in_after_panels_max),
                            "max_power_out": fmtpow(power_out_propulsion_max),
                            "battery_total_energy": fmtpow(battery_total_energy),
                        }.items():
                            print(f"{k.rjust(25)} = {v}")

                    print_title("Mass props")
                    for k, v in mass_props.items():
                            print(f"{k.rjust(25)} = {fmt(v.mass)} kg")

                    # Define a filename for the CSV file with the temporal resolution in the name
                    csv_file = f"outputs/outputs_data.csv"

                    # Write the data to the CSV file
                    with open(csv_file, 'w', newline='') as csvfile:
                        fieldnames = ["Property", "Value"]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                        # Write the header row
                        writer.writeheader()

                        # Write the data rows
                        for k, v in outputs_data.items():
                            writer.writerow({"Property": k, "Value": v})
                        for k, v in mass_props.items():
                            writer.writerow({"Property": k, "Value": fmt(v.mass) + " kg"})
                        for k, v in {
                            "payload power": fmtpow(payload_power),
                            "payload mass": f"{fmt(mass_props['payload'].mass)} kg",
                            "aperture length": f"{fmt(radar_length)} meters",
                            "aperture width": f"{fmt(radar_width)} meters",
                            "SAR range resolution": f"{fmt(range_resolution)} meters",
                            "SAR azimuth resolution": f"{fmt(azimuth_resolution)} meters",
                            "SAR revisit period": f"{fmt(revisit_period)} hours",
                            "pixels in the incoherent averaging window": fmt(N_i),
                            "precision": f"{fmt(max_precision)} 1 / yr",
                            "pulse repetition frequency": f"{fmt(pulse_rep_freq)} Hz",
                            "bandwidth": f"{fmt(bandwidth)} Hz",
                            "center wavelength": f"{fmt(wavelength)} meters",
                            "look angle": f"{fmt(look_angle)} degrees",
                            "swath range": f"{fmt(max_swath_range)} meters",
                            "swath azimuth": f"{fmt(max_swath_azimuth)} meters",
                            "SNR": f"{fmt(max_snr)} dB",
                        }.items():
                            writer.writerow({"Property": k, "Value": v})
                        for k, v in {
                            "max_power_in": fmtpow(power_in_after_panels_max),
                            "max_power_out": fmtpow(power_out_propulsion_max),
                            "battery_total_energy": fmtpow(battery_total_energy),
                        }.items():
                            writer.writerow({"Property": k, "Value": v})

                    print(f"Data from 'Outputs' section saved to {csv_file}")

                    def qp(*args: List[str]):
                        """
                        QuickPlot a variable or set of variables
                        :param args: Variable names, given as strings (e.g. 'x')
                        """
                        n = len(args)
                        if n == 1:
                            fig = px.scatter(y=s(eval(args[0])), title=args[0], labels={'y': args[0]})
                        elif n == 2:
                            fig = px.scatter(
                                x=s(eval(args[0])),
                                y=s(eval(args[1])),
                                title=f"{args[0]} vs. {args[1]}",
                                labels={'x': args[0], 'y': args[1]}
                            )
                        elif n == 3:
                            fig = px.scatter_3d(
                                x=s(eval(args[0])),
                                y=s(eval(args[1])),
                                z=s(eval(args[2])),
                                title=f"{args[0]} vs. {args[1]} vs. {args[2]}",
                                labels={'x': args[0], 'y': args[1], 'z': args[2]},
                                size_max=18
                            )
                        else:
                            raise ValueError("Too many inputs to plot!")
                        fig.data[0].update(mode='markers+lines')
                        fig.show()


                    # endregion

                    ### Draw plots
                    plot_dpi = 200

                    # Find dusk and dawn
                    is_daytime = s(solar_flux_on_horizontal) >= 1  # 1 W/m^2 or greater insolation
                    is_nighttime = np.logical_not(is_daytime)


                    def plot(
                            x_name: str,
                            y_name: str,
                            xlabel: str,
                            ylabel: str,
                            title: str,
                            save_name: str = None,
                            show: bool = True,
                            plot_day_color=(103 / 255, 155 / 255, 240 / 255),
                            plot_night_color=(7 / 255, 36 / 255, 84 / 255),
                    ) -> None:  # Plot a variable x and variable y, highlighting where day and night occur

                        # Make the plot axes
                        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)

                        # Evaluate the data, and plot it. Shade according to whether it's day/night.
                        x = s(eval(x_name))
                        y = s(eval(y_name))
                        plot_average_color = tuple([
                            (d + n) / 2
                            for d, n in zip(plot_day_color, plot_night_color)
                        ])
                        plt.plot(  # Plot a black line through all points
                            x,
                            y,
                            '-',
                            color=plot_average_color,
                        )
                        plt.plot(  # Emphasize daytime points
                            x[is_daytime],
                            y[is_daytime],
                            '.',
                            color=plot_day_color,
                            label="Day"
                        )
                        plt.plot(  # Emphasize nighttime points
                            x[is_nighttime],
                            y[is_nighttime],
                            '.',
                            color=plot_night_color,
                            label="Night"
                        )

                        # Disable offset notation, which makes things hard to read.
                        ax.ticklabel_format(useOffset=False)

                        # Do specific things for certain variable names.
                        if x_name == "hour":
                            ax.xaxis.set_major_locator(
                                ticker.MultipleLocator(base=3)
                            )

                        # Do the usual plot things.
                        plt.xlabel(xlabel)
                        plt.ylabel(ylabel)
                        plt.title(title)
                        plt.legend()
                        plt.tight_layout()
                        if save_name is not None:
                            plt.savefig(save_name)
                        if show:
                            plt.show()

                        return fig, ax


                    if make_plots:
                        plot("hour", "z_km",
                             xlabel="Hours after Solar Noon",
                             ylabel="Altitude [km]",
                             title="Altitude over Simulation",
                             save_name="outputs/altitude.png"
                             )
                        plot("hour", "air_speed",
                             xlabel="Hours after Solar Noon",
                             ylabel="True Airspeed [m/s]",
                             title="True Airspeed over Simulation",
                             save_name="outputs/airspeed.png"
                             )
                        plot("hour", "net_power",
                             xlabel="Hours after Solar Noon",
                             ylabel="Net Power [W] (positive is charging)",
                             title="Net Power to Battery over Simulation",
                             save_name="outputs/net_powerJuly15.png"
                             )
                        plot("hour", "battery_charge_state",
                             xlabel="Hours after Solar Noon",
                             ylabel="State of Charge [%]",
                             title="Battery Charge State over Simulation",
                             save_name="outputs/battery_charge.png"
                             )
                        plot("hour", "distance",
                             xlabel="hours after Solar Noon",
                             ylabel="Downrange Distance [km]",
                             title="Optimal Trajectory over Simulation",
                             save_name="outputs/trajectory.png"
                             )
                        plot("x_km", "y_km",
                             xlabel="Downrange Distance in X [km]",
                             ylabel="Downrange Distance in Y [km]",
                             title="Optimal Trajectory over Simulation",
                             save_name="outputs/trajectory.png"
                             )
                        # plot("hour", "groundspeed",
                        #      xlabel="hours after Solar Noon",
                        #      ylabel="Groundspeed [m/s]",
                        #      title="Groundspeed over Simulation",
                        #      save_name="outputs/trajectory.png"
                        #      )

                        # Draw mass breakdown
                        fig = plt.figure(figsize=(10, 8), dpi=plot_dpi)
                        plt.suptitle("Mass Budget")

                        ax_main = fig.add_axes([0.2, 0.3, 0.6, 0.6], aspect=1)
                        pie_labels = [
                            "Payload",
                            "Structural",
                            "Propulsion",
                            "Power Systems",
                            "Avionics"
                        ]
                        pie_values = [
                            s(mass_props['payload'].mass),
                            s(mass_structural.mass),
                            s(mass_propulsion.mass),
                            s(mass_power_systems.mass),
                            s(mass_props['avionics'].mass),
                        ]
                        colors = plt.cm.Set2(np.arange(5))
                        pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (x * s(mass_total) / 100, x)
                        ax_main.pie(
                            pie_values,
                            labels=pie_labels,
                            autopct=pie_format,
                            pctdistance=0.7,
                            colors=colors,
                            startangle=120
                        )
                        plt.title("Overall Mass")

                        ax_structural = fig.add_axes([0.05, 0.05, 0.3, 0.3], aspect=1)
                        pie_labels = [
                            "Wing Spar",
                            "Wing Secondary Structure",
                            "Stabilizers",
                            "Booms",
                            "Payload Pod"]
                        pie_values = [
                            s(mass_props['wing_primary'].mass),
                            s(mass_props['wing_secondary'].mass),
                            s(
                                mass_props['center_hstab'].mass +
                                mass_props['right_hstab'].mass +
                                mass_props['left_hstab'].mass +
                                mass_props['vstab'].mass
                            ),
                            s(
                                mass_props['center_boom'].mass +
                                mass_props['right_boom'].mass +
                                mass_props['left_boom'].mass
                            ),
                            s(
                                mass_props['payload_pod'].mass
                            )
                        ]
                        colors = plt.cm.Set2(np.arange(5))
                        colors = np.clip(
                            colors[1, :3] + np.expand_dims(
                                np.linspace(-0.1, 0.2, len(pie_labels)),
                                1),
                            0, 1
                        )
                        pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (
                            x * s(mass_structural.mass) / 100, x * s(mass_structural.mass / mass_total))
                        ax_structural.pie(
                            pie_values,
                            labels=pie_labels,
                            autopct=pie_format,
                            pctdistance=0.7,
                            colors=colors,
                            startangle=60,
                        )
                        plt.title("Structural Mass*")

                        ax_power_systems = fig.add_axes([0.65, 0.05, 0.3, 0.3], aspect=1)
                        pie_labels = [
                            "Batt. Pack (Cells)",
                            "Batt. Pack (Non-cell)",
                            "Solar Cells",
                            "Misc. & Wires"
                        ]
                        pie_values = [
                            s(battery_cell_mass),
                            s(mass_props['battery_pack'].mass - battery_cell_mass),
                            s(mass_props['solar_panel_wing'].mass + mass_props['solar_panel_vstab'].mass),
                            s(mass_power_systems.mass - mass_props['battery_pack'].mass - mass_props['solar_panel_vstab'].mass -
                              mass_props['solar_panel_wing'].mass)
                        ]
                        colors = plt.cm.Set2(np.arange(5))
                        colors = np.clip(
                            colors[3, :3] + np.expand_dims(
                                np.linspace(-0.1, 0.2, len(pie_labels)),
                                1),
                            0, 1
                        )[::-1]
                        pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (
                            x * s(mass_power_systems.mass) / 100, x * s(mass_power_systems.mass / mass_total))
                        ax_power_systems.pie(
                            pie_values,
                            labels=pie_labels,
                            autopct=pie_format,
                            pctdistance=0.7,
                            colors=colors,
                            startangle=15,
                        )
                        plt.title("Power Systems Mass*")

                        plt.annotate(
                            text="* percentages referenced to total aircraft mass",
                            xy=(0.01, 0.01),
                            # xytext=(0.03, 0.03),
                            xycoords="figure fraction",
                            # arrowprops={
                            #     "color"     : "k",
                            #     "width"     : 0.25,
                            #     "headwidth" : 4,
                            #     "headlength": 6,
                            # }
                        )
                        plt.annotate(
                            text="""
                                                Total mass: %.1f kg
                                                Wing span: %.2f m
                                                """ % (s(mass_total), s(wing.span())),
                            xy=(0.03, 0.70),
                            # xytext=(0.03, 0.03),
                            xycoords="figure fraction",
                            # arrowprops={
                            #     "color"     : "k",
                            #     "width"     : 0.25,
                            #     "headwidth" : 4,
                            #     "headlength": 6,
                            # }
                        )

                        plt.savefig("outputs/mass_pie_chart.png")
                        plt.show() if make_plots else plt.close(fig)

                    # Write a geometry spreadsheet
                    wing_area = s(wing.area())
                    wing_mac = s(wing.mean_aerodynamic_chord())
                    wing_le_to_hstab_le = s(center_hstab.xsecs[0].xyz_le[0] - wing.xsecs[0].xyz_le[0])
                    wing_le_to_vstab_le = s(center_vstab.xsecs[0].xyz_le[0] - wing.xsecs[0].xyz_le[0])
                    wing_c4_to_hstab_c4 = s(center_hstab.xsecs[0].xyz_le[0] + center_hstab.xsecs[0].chord / 4
                                            - (wing.xsecs[0].xyz_le[0] + wing.xsecs[0].chord / 4))
                    wing_c4_to_vstab_c4 = s(center_vstab.xsecs[0].xyz_le[0] + center_vstab.xsecs[0].chord / 4
                                            - (wing.xsecs[0].xyz_le[0] + wing.xsecs[0].chord / 4))
                    #
                    # hori_tail_vol = s(Vh)
                    # vert_tail_vol = s(Vv)

                    wing_y_taper_break_loc = s(wing_y_taper_break)

                    with open("outputs/geometry.csv", "w+") as f:

                        f.write("Design Variable, Value (all in base SI units or derived units thereof),\n")
                        geometry_vars = [
                            'wing_span',
                            'wing_root_chord',
                            'wing_taper_ratio',
                            'wing_area',
                            'wing_mac',
                            'wing_y_taper_break_loc',
                            '',
                            'center_hstab_span',
                            'center_hstab_chord',
                            'wing_le_to_hstab_le',
                            'wing_c4_to_hstab_c4',
                            '',
                            'outboard_hstab_span',
                            'outboard_hstab_chord',
                            '',
                            'center_vstab_span',
                            'center_vstab_chord',
                            'wing_le_to_vstab_le',
                            'wing_c4_to_vstab_c4',
                            '',
                            'mass_total'
                            '',
                            'wing_solar_area_fraction',
                            'battery_capacity_watt_hours',
                            'n_propellers',
                            'propeller_diameter',
                            '',
                            'center_boom_length',
                            'outboard_boom_length'
                        ]
                        for var_name in geometry_vars:
                            if var_name == '':
                                f.write(",,\n")
                                continue
                            try:
                                value = s(eval(var_name))
                            except:
                                value = eval(var_name)
                            f.write(f"{var_name}, {value},\n")

                    # opti.set_initial_from_sol(sol)


                    # opti.value(net_power)
                    # opti.value(time)
                    # opti.set_value(vehicle_heading, heading)
                    #
                    def draw():  # Draw the geometry of the optimal airplane
                        airplane.substitute_solution(sol)
                        airplane.draw()


                    if make_plots == True:
                        draw()
                        # pass
                    opti.set_initial_from_sol(sol)

    # plt.plot(wingspan, spatial, linestyle='--', marker='o')
    # plt.title('Pareto Front')
    # plt.xlabel('Wingspan [meters]')
    # plt.ylabel('Spatial Resolution [meters]')
    # plt.show()