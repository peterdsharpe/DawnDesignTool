# Imports
import aerosandbox as asb
import aerosandbox.library.aerodynamics as aero
import aerosandbox.library.atmosphere as atmo
from aerosandbox.tools.casadi_tools import *
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.library import winds as lib_winds
from aerosandbox.library.airfoils import naca0008, flat_plate
import plotly.express as px
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import design_opt_utilities as utils
from typing import Union, List

sns.set(font_scale=1)

# region Setup
##### Initialize Optimization
opti = asb.Opti(  # Normal mode - Design Optimization
    cache_filename="cache/optimization_solution.json",
    save_to_cache_on_solve=True
)
# opti = asb.Opti( # Alternate mode - Frozen Design Optimization
#     variable_categories_to_freeze=["des"],
#     cache_filename="cache/optimization_solution.json",
#     load_frozen_variables_from_cache=True,
#     ignore_violated_parametric_constraints=True
# )

minimize = "wing.span() / 50"  # any "eval-able" expression
# minimize = "max_mass_total / 300" # any "eval-able" expression
# minimize = "wing.span() / 50 * 0.9 + max_mass_total / 300 * 0.1"

##### Operating Parameters
climb_opt = False  # are we optimizing for the climb as well?
latitude = opti.parameter(value=49)  # degrees (49 deg is top of CONUS, 26 deg is bottom of CONUS)
day_of_year = opti.parameter(value=244)  # Julian day. June 1 is 153, June 22 is 174, Aug. 31 is 244
min_cruise_altitude = opti.parameter(value=18288)  # meters. 19812 m = 65000 ft, 18288 m = 60000 ft.
required_headway_per_day = 10e3  # meters
allow_trajectory_optimization = True
structural_load_factor = 3  # over static
make_plots = True
mass_payload = opti.parameter(value=30)
wind_speed_func = lambda alt: lib_winds.wind_speed_conus_summer_99(alt, latitude)
battery_specific_energy_Wh_kg = opti.parameter(value=450)
battery_pack_cell_percentage = 0.89  # What percent of the battery pack consists of the module, by weight?
# Accounts for module HW, BMS, pack installation, etc.
# Ed Lovelace (in his presentation) gives 70% as a state-of-the-art fraction.
# Used 75% in the 16.82 CDR.
# According to Kevin Uleck, Odysseus realized an 89% packing factor here.

##### Margins
structural_mass_margin_multiplier = opti.parameter(value=1.25)  # TODO Jamie dropped to 1.215 - why?
energy_generation_margin = opti.parameter(value=1.05)
allowable_battery_depth_of_discharge = opti.parameter(
    value=0.85)  # How much of the battery can you actually use? # Reviewed w/ Annick & Bjarni 4/30/2020

##### Simulation Parameters
n_timesteps_per_segment = 180  # Only relevant if allow_trajectory_optimization is True.
# Quick convergence testing indicates you can get bad analyses below 150 or so...

##### Optimization bounds
min_speed = 1  # Specify a minimum speed - keeps the speed-gamma velocity parameterization from NaNing

##### Time Discretization
if climb_opt:  # roughly 1-day-plus-climb window, starting at ground. Periodicity enforced for last 24 hours.
    assert allow_trajectory_optimization, "You can't do climb optimization without trajectory optimziation!"
    # time_start = -9 * 3600
    time_start = opti.variable(
        init_guess=-18 * 3600,
        scale=3600,
        category="ops",
    )
    time_end = 36 * 3600

    time_periodic_window_start = time_end - 24 * 3600
    time = cas.vertcat(
        cas.linspace(
            time_start,
            time_periodic_window_start,
            n_timesteps_per_segment
        ),
        cas.linspace(
            time_periodic_window_start,
            time_end,
            n_timesteps_per_segment
        )
    )
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

# region Trajectory Optimization Variables
##### Initialize trajectory optimization variables

x = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e5,
    category="ops"
)
x_km = x / 1000
x_mi = x / 1609.34

y = opti.variable(
    n_vars=n_timesteps,
    init_guess=opti.value(min_cruise_altitude),
    scale=1e4,
    category="ops"
)
y_km = y / 1000
y_ft = y / 0.3048

opti.subject_to([
    y[time_periodic_start_index:] / min_cruise_altitude > 1,
    y / 40000 > 0,  # stay above ground
    y / 40000 < 1,  # models break down
])

airspeed = opti.variable(
    n_vars=n_timesteps,
    init_guess=20,
    scale=20,
    category="ops"
)
opti.subject_to([
    airspeed / min_speed > 1
])

flight_path_angle = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=2,
    category="ops"
)
opti.subject_to([
    flight_path_angle / 90 < 1,
    flight_path_angle / 90 > -1,
])

alpha = opti.variable(
    n_vars=n_timesteps,
    init_guess=5,
    scale=4,
    category="ops"
)
opti.subject_to([
    alpha > -8,
    alpha < 12
])

thrust_force = opti.variable(
    n_vars=n_timesteps,
    init_guess=150,
    scale=200,
    category="ops"
)
opti.subject_to([
    thrust_force > 0
])

net_accel_parallel = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e-4,
    category="ops"
)
net_accel_perpendicular = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1e-5,
    category="ops"
)
# endregion

# region Design Optimization Variables
##### Initialize design optimization variables (all units in base SI or derived units)

mass_total = opti.variable(
    init_guess=600,
    scale=600,
    category="ops"
)

max_mass_total = opti.variable(
    init_guess=600,
    scale=600,
    category="des"
)
opti.subject_to(max_mass_total / 600 >= mass_total / 600)

### Initialize geometric variables

# overall layout
boom_location = 0.80  # as a fraction of the half-span
break_location = 0.67  # as a fraction of the half-span

# wing
wing_span = opti.variable(
    init_guess=40,
    scale=60,
    category="des"
)

opti.subject_to([wing_span > 1])

wing_root_chord = opti.variable(
    init_guess=3,
    scale=4,
    category="des"
)
opti.subject_to([wing_root_chord > 0.1])

wing_x_quarter_chord = opti.variable(
    init_guess=0,
    scale=0.01,
    category="des"
)

wing_y_taper_break = break_location * wing_span / 2

wing_taper_ratio = 0.5  # TODO analyze this more

# center hstab
center_hstab_span = opti.variable(
    init_guess=4,
    scale=4,
    category="des"
)
opti.subject_to([
    center_hstab_span > 0.1,
    center_hstab_span < wing_span / 6
])

center_hstab_chord = opti.variable(
    init_guess=3,
    scale=2,
    category="des"
)
opti.subject_to([center_hstab_chord > 0.1])

center_hstab_twist_angle = opti.variable(
    n_vars=n_timesteps,
    init_guess=-3,
    scale=2,
    category="ops"
)

# center hstab
outboard_hstab_span = opti.variable(
    init_guess=4,
    scale=4,
    category="des"
)
opti.subject_to([
    outboard_hstab_span > 2,  # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20
    outboard_hstab_span < wing_span / 6,
])

outboard_hstab_chord = opti.variable(
    init_guess=3,
    scale=2,
    category="des"
)
opti.subject_to([
    outboard_hstab_chord > 0.8,  # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20
])

outboard_hstab_twist_angle = opti.variable(
    n_vars=n_timesteps,
    init_guess=-3,
    scale=2,
    category="ops"
)

# center_vstab
center_vstab_span = opti.variable(
    init_guess=7,
    scale=8,
    category="des"
)
opti.subject_to(center_vstab_span > 0.1)

center_vstab_chord = opti.variable(
    init_guess=2.5,
    scale=2,
    category="des"
)
opti.subject_to([center_vstab_chord > 0.1])

# center_fuselage
center_boom_length = opti.variable(
    init_guess=10,
    scale=2,
    category="des"
)
opti.subject_to([
    center_boom_length - center_vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

# outboard_fuselage
outboard_boom_length = opti.variable(
    init_guess=10,
    scale=2,
    category="des"
)
opti.subject_to([
    outboard_boom_length > wing_root_chord * 3 / 4,
    # outboard_boom_length < 3.5, # TODO review this, driven by Trevor's ASWing findings on turn radius sizing, 8/16/20
])

nose_length = 1.80  # Calculated on 4/15/20 with Trevor and Olek
# https://docs.google.com/spreadsheets/d/1BnTweK-B4Hmmk9piJn8os-LNPiJH-3rJJemYkrKjARA/edit#gid=0

fuse_diameter = 0.24 * 2  # Synced to Jonathan's fuselage CAD as of 8/7/20
boom_diameter = 0.2

# Propeller
propeller_diameter = opti.variable(
    init_guess=5,
    scale=5,
    category="des"
)
opti.subject_to([
    propeller_diameter / 1 > 1,
    propeller_diameter / 10 < 1
])

n_propellers = opti.parameter(value=4)

# import pickle
import dill as pickle

import pathlib

path = str(
    pathlib.Path(__file__).parent.absolute()
)

try:
    with open(path + "/cache/wing_airfoil.cache", "rb") as f:
        wing_airfoil = pickle.load(f)
    with open(path + "/cache/tail_airfoil.cache", "rb") as f:
        tail_airfoil = pickle.load(f)
except (FileNotFoundError, TypeError):
    wing_airfoil = asb.Airfoil(name="HALE_03", coordinates=r"studies/airfoil_optimizer/HALE_03.dat")
    wing_airfoil.populate_sectional_functions_from_xfoil_fits(parallel=False)
    with open(path + "/cache/wing_airfoil.cache", "wb+") as f:
        pickle.dump(wing_airfoil, f)
    tail_airfoil = asb.Airfoil("naca0008")
    tail_airfoil.populate_sectional_functions_from_xfoil_fits(parallel=False)
    with open(path + "/cache/tail_airfoil.cache", "wb+") as f:
        pickle.dump(tail_airfoil, f)

tail_airfoil = naca0008  # TODO remove this and use fits?

wing = asb.Wing(
    name="Main Wing",
    x_le=wing_x_quarter_chord,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=0,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=-wing_root_chord / 4,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=wing_root_chord,
            twist=0,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Break
            x_le=-wing_root_chord / 4,
            y_le=wing_y_taper_break,
            z_le=0,  # wing_span / 2 * cas.pi / 180 * 5,
            chord=wing_root_chord,
            twist=0,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            x_le=-wing_root_chord * wing_taper_ratio / 4,
            y_le=wing_span / 2,
            z_le=0,  # wing_span / 2 * cas.pi / 180 * 5,
            chord=wing_root_chord * wing_taper_ratio,
            twist=0,
            airfoil=wing_airfoil,
        ),
    ]
)

center_hstab = asb.Wing(
    name="Horizontal Stabilizer",
    x_le=center_boom_length - center_vstab_chord * 0.75 - center_hstab_chord,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=0.1,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=center_hstab_chord,
            twist=-3,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0,
            y_le=center_hstab_span / 2,
            z_le=0,
            chord=center_hstab_chord,
            twist=-3,
            airfoil=tail_airfoil,
        ),
    ]
)

right_hstab = asb.Wing(
    name="Horizontal Stabilizer",
    x_le=outboard_boom_length - outboard_hstab_chord * 0.75,  # Coordinates of the wing's leading edge
    y_le=boom_location * wing_span / 2,  # Coordinates of the wing's leading edge
    z_le=0.1,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=outboard_hstab_chord,
            twist=-3,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0,
            y_le=outboard_hstab_span / 2,
            z_le=0,
            chord=outboard_hstab_chord,
            twist=-3,
            airfoil=tail_airfoil,
        ),
    ]
)

left_hstab = copy.deepcopy(right_hstab)
left_hstab.xyz_le[1] *= -1

center_vstab = asb.Wing(
    name="Vertical Stabilizer",
    x_le=center_boom_length - center_vstab_chord * 0.75,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=-center_vstab_span / 2 + center_vstab_span * 0.15,  # Coordinates of the wing's leading edge
    symmetric=False,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=center_vstab_chord,
            twist=0,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0,
            y_le=0,
            z_le=center_vstab_span,
            chord=center_vstab_chord,
            twist=0,
            airfoil=tail_airfoil,
        ),
    ]
)

center_fuse = utils.fuselage(
    boom_length=center_boom_length,
    nose_length=nose_length,
    fuse_diameter=fuse_diameter,
    boom_diameter=boom_diameter,
)

right_fuse = utils.fuselage(
    boom_length=outboard_boom_length,
    nose_length=0.5,  # Review this for fit
    fuse_diameter=boom_diameter,
    boom_diameter=boom_diameter,
)

right_fuse.xyz_le += cas.vertcat(0, boom_location * wing_span / 2, 0)

left_fuse = copy.deepcopy(right_fuse)
left_fuse.xyz_le[1] *= -1

# Assemble the airplane
airplane = asb.Airplane(
    name="Solar1",
    x_ref=0,
    y_ref=0,
    z_ref=0,
    wings=[
        wing,
        center_hstab,
        right_hstab,
        left_hstab,
        center_vstab,
    ],
    fuselages=[
        center_fuse,
        right_fuse,
        left_fuse
    ],
)

# endregion

# region Atmosphere
##### Atmosphere
P = atmo.get_pressure_at_altitude(y)
rho = atmo.get_density_at_altitude(y)
T = atmo.get_temperature_at_altitude(y)
mu = atmo.get_viscosity_from_temperature(T)
a = atmo.get_speed_of_sound_from_temperature(T)
mach = airspeed / a
g = 9.81  # gravitational acceleration, m/s^2
q = 1 / 2 * rho * airspeed ** 2  # Solar calculations
solar_flux_on_horizontal = lib_solar.solar_flux_on_horizontal(
    latitude, day_of_year, time, scattering=True
)


# endregion

# region Aerodynamics
##### Aerodynamics

# Fuselage
def fuse_aero(fuse: asb.Fuselage):
    fuse_Re = rho / mu * airspeed * fuse.length()
    CLA_fuse = 0
    CDA_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted() * 1.2  # wetted area with form factor

    lift_fuse = CLA_fuse * q  # per fuse
    drag_fuse = CDA_fuse * q  # per fuse
    return lift_fuse, drag_fuse


lift_center_fuse, drag_center_fuse = fuse_aero(center_fuse)
lift_left_fuse, drag_left_fuse = fuse_aero(left_fuse)
lift_right_fuse, drag_right_fuse = fuse_aero(right_fuse)


# Wing
def wing_aero(
        wing: asb.Wing,
        incidence_angle: float
):
    alpha_eff = alpha + incidence_angle + wing.mean_twist_angle()

    wing_Re = rho / mu * airspeed * wing.mean_geometric_chord()
    wing_airfoil = wing.xsecs[0].airfoil  # type: asb.Airfoil
    wing_Cl_inc = wing_airfoil.CL_function(alpha_eff, wing_Re, 0,
                                           0)  # Incompressible 2D lift coefficient
    wing_CL = wing_Cl_inc * aero.CL_over_Cl(wing.aspect_ratio(), mach=mach,
                                            sweep=wing.mean_sweep_angle())  # Compressible 3D lift coefficient
    lift_wing = wing_CL * q * wing.area()

    wing_Cd_profile = wing_airfoil.CDp_function(alpha_eff, wing_Re, mach, 0)
    drag_wing_profile = wing_Cd_profile * q * wing.area()

    wing_oswalds_efficiency = aero.oswalds_efficiency(
        taper_ratio=wing.taper_ratio(),
        AR=wing.aspect_ratio(),
        sweep=wing.mean_sweep_angle()
    )
    drag_wing_induced = lift_wing ** 2 / (q * np.pi * wing.span() ** 2 * wing_oswalds_efficiency)

    drag_wing = drag_wing_profile + drag_wing_induced

    wing_Cm_inc = wing_airfoil.Cm_function(alpha_eff, wing_Re, 0,
                                           0)  # Incompressible 2D moment coefficient
    wing_CM = wing_Cm_inc * aero.CL_over_Cl(wing.aspect_ratio(), mach=mach,
                                            sweep=wing.mean_sweep_angle())  # Compressible 3D moment coefficient
    moment_wing = wing_CM * q * wing.area() * wing.mean_geometric_chord()

    return lift_wing, drag_wing, moment_wing


lift_wing, drag_wing, moment_wing = wing_aero(wing, 0)
lift_center_hstab, drag_center_hstab, moment_center_hstab = wing_aero(center_hstab, center_hstab_twist_angle)
lift_right_hstab, drag_right_hstab, moment_right_hstab = wing_aero(right_hstab, outboard_hstab_twist_angle)
lift_left_hstab, drag_left_hstab, moment_left_hstab = wing_aero(left_hstab, outboard_hstab_twist_angle)

# Increase the wing drag due to tripped flow (8/17/20)
wing_drag_multiplier = opti.parameter(value=1.06)  # TODO review
# Was originally 1.25
drag_wing *= wing_drag_multiplier


# center_vstab
def vstab_aero(vstab: asb.Wing):
    vstab_Re = rho / mu * airspeed * vstab.mean_geometric_chord()
    vstab_airfoil = vstab.xsecs[0].airfoil  # type: asb.Airfoil
    vstab_Cd_profile = vstab_airfoil.CDp_function(0, vstab_Re, mach, 0)
    drag_vstab_profile = vstab_Cd_profile * q * vstab.area()
    drag_vstab = drag_vstab_profile  # per vstab
    return drag_vstab


drag_center_vstab = vstab_aero(center_vstab)

# strut drag
strut_y_location = opti.variable(
    init_guess=5,
    scale=5,
    category="des"
)
opti.subject_to([
    strut_y_location > 3,
    strut_y_location <= wing_span / 6  # TODO review why this constraint is here, taken from Jamie DDT 12/30/20
])

strut_span = (strut_y_location ** 2 + (propeller_diameter / 2 + 0.25) ** 2) ** 0.5  # Formula from Jamie
strut_chord = 0.167 * (strut_span * (propeller_diameter / 2 + 0.25)) ** 0.25  # Formula from Jamie
strut_Re = rho / mu * airspeed * strut_chord
strut_airfoil = flat_plate
strut_Cd_profile = flat_plate.CDp_function(0, strut_Re, mach, 0)
drag_strut_profile = strut_Cd_profile * q * strut_chord * strut_span
drag_strut = drag_strut_profile  # per strut

# Force totals
lift_force = (
        lift_wing +
        lift_center_hstab +
        lift_right_hstab +
        lift_left_hstab +
        lift_center_fuse +
        lift_right_fuse +
        lift_left_fuse
)
drag_force = (
        drag_wing +
        drag_center_hstab +
        drag_right_hstab +
        drag_left_hstab +
        drag_center_vstab +
        drag_center_fuse +
        drag_right_fuse +
        drag_left_fuse +
        drag_strut * 2  # 2 struts
)
moment = (
        -wing.approximate_center_of_pressure()[0] * lift_wing + moment_wing +
        -center_hstab.approximate_center_of_pressure()[0] * lift_center_hstab + moment_center_hstab +
        -right_hstab.approximate_center_of_pressure()[0] * lift_right_hstab + moment_right_hstab +
        -left_hstab.approximate_center_of_pressure()[0] * lift_left_hstab + moment_left_hstab
)

# endregion

# region Stability
### Estimate aerodynamic center
x_ac = (
               wing.approximate_center_of_pressure()[0] * wing.area() +
               center_hstab.approximate_center_of_pressure()[0] * center_hstab.area() +
               right_hstab.approximate_center_of_pressure()[0] * right_hstab.area() +
               left_hstab.approximate_center_of_pressure()[0] * left_hstab.area()
       ) / (
               wing.area() +
               center_hstab.area() +
               right_hstab.area() +
               left_hstab.area()
       )
static_margin_fraction = (x_ac - airplane.xyz_ref[0]) / wing.mean_aerodynamic_chord()
opti.subject_to([
    static_margin_fraction == 0.2,  # Stability condition
    moment / 1e4 == 0  # Trim condition
])

### Size the tails off of tail volume coefficients
Vv = center_vstab.area() * (
        center_vstab.approximate_center_of_pressure()[0] - wing.approximate_center_of_pressure()[0]
) / (wing.area() * wing.span())

vstab_effectiveness_factor = aero.CL_over_Cl(center_vstab.aspect_ratio()) / aero.CL_over_Cl(wing.aspect_ratio())

opti.subject_to([
    # Vh * hstab_effectiveness_factor > 0.3,
    # Vh * hstab_effectiveness_factor < 0.6,
    # Vh * hstab_effectiveness_factor == 0.45,
    # Vv * vstab_effectiveness_factor > 0.02,
    # Vv * vstab_effectiveness_factor < 0.05,
    # Vv * vstab_effectiveness_factor == 0.035,
    # Vh > 0.3,
    # Vh < 0.6,
    # Vh == 0.45,
    Vv > 0.02,
    # Vv < 0.05,
    # Vv == 0.035,
    center_vstab.aspect_ratio() == 2.5,  # TODO review this
    # center_vstab.aspect_ratio() > 1.9, # from Jamie, based on ASWing
    # center_vstab.aspect_ratio() < 2.5 # from Jamie, based on ASWing
])

# endregion

# region Propulsion

### Propeller calculations

propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * atmo.get_speed_of_sound_from_temperature(
    atmo.get_temperature_at_altitude(20000)
) / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / cas.pi

area_propulsive = cas.pi / 4 * propeller_diameter ** 2 * n_propellers

use_legacy_propulsion_models = True

if use_legacy_propulsion_models:
    ### Use older models

    motor_efficiency = 0.955  # Taken from ThinGap estimates

    power_out_propulsion_shaft = lib_prop_prop.propeller_shaft_power_from_thrust(
        thrust_force=thrust_force,
        area_propulsive=area_propulsive,
        airspeed=airspeed,
        rho=rho,
        propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
    )

    gearbox_efficiency = 0.986

else:
    ### Use Jamie's model
    from design_opt_utilities.new_models import eff_curve_fit

    opti.subject_to(y < 30000)  # Bugs out without this limiter

    propeller_efficiency, motor_efficiency = eff_curve_fit(
        airspeed=airspeed,
        total_thrust=thrust_force,
        altitude=y,
        var_pitch=False
    )
    power_out_propulsion_shaft = thrust_force * airspeed / propeller_efficiency

    gearbox_efficiency = 0.986

power_out_propulsion = power_out_propulsion_shaft / motor_efficiency / gearbox_efficiency

# Motor thermal modeling
heat_motor = power_out_propulsion * (1 - motor_efficiency) / n_propellers
heat_motor_max = 175  # System is designed to reject 175 W
opti.subject_to(heat_motor <= heat_motor_max)

# Calculate maximum power in/out requirements
power_out_propulsion_max = opti.variable(
    init_guess=5e3,
    scale=5e3,
    category="des",
)
opti.subject_to([
    power_out_propulsion < power_out_propulsion_max,
    power_out_propulsion_max > 0
])

propeller_max_torque = (power_out_propulsion_max / n_propellers) / propeller_rads_per_sec

battery_voltage = 125  # From Olek Peraire >4/2, propulsion slack

motor_kv = propeller_rpm / battery_voltage

mass_motor_raw = lib_prop_elec.mass_motor_electric(
    max_power=power_out_propulsion_max / n_propellers,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * n_propellers

mass_motor_mounted = 2 * mass_motor_raw  # similar to a quote from Raymer, modified to make sensible units, prop weight roughly subtracted

mass_propellers = n_propellers * lib_prop_prop.mass_hpa_propeller(
    diameter=propeller_diameter,
    max_power=power_out_propulsion_max,
    include_variable_pitch_mechanism=True
)
mass_ESC = lib_prop_elec.mass_ESC(max_power=power_out_propulsion_max)

# Total propulsion mass
mass_propulsion = mass_motor_mounted + mass_propellers + mass_ESC

# Account for payload power
power_out_payload = cas.if_else(
    solar_flux_on_horizontal > 1,
    500,
    150
)

# Account for avionics power
power_out_avionics = 180  # Pulled from Avionics spreadsheet on 5/13/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

### Power accounting
power_out = power_out_propulsion + power_out_payload + power_out_avionics

# endregion


# region Solar Power Systems

# Battery design variables
net_power = opti.variable(
    n_vars=n_timesteps,
    init_guess=0,
    scale=1000,
    category="ops"
)

battery_stored_energy_nondim = opti.variable(
    n_vars=n_timesteps,
    init_guess=0.5,
    scale=1,
    category="ops"
)
opti.subject_to([
    battery_stored_energy_nondim > 0,
    battery_stored_energy_nondim < allowable_battery_depth_of_discharge,
])

battery_capacity = opti.variable(
    init_guess=3600 * 60e3,  # Joules, not watt-hours!
    scale=3600 * 60e3,
    category="des",
)
opti.subject_to([
    battery_capacity > 0
])
battery_capacity_watt_hours = battery_capacity / 3600
battery_stored_energy = battery_stored_energy_nondim * battery_capacity
battery_state_of_charge_percentage = 100 * (battery_stored_energy_nondim + (1 - allowable_battery_depth_of_discharge))

### Solar calculations

# solar_cell_efficiency = 0.21 # Sunpower
solar_cell_efficiency = 0.25  # Microlink.
# This figure should take into account all temperature factors,
# spectral losses (different spectrum at altitude), multi-junction effects, etc.
# Should not take into account MPPT losses.
# Kevin Uleck gives this figure as 0.205.
# This paper (https://core.ac.uk/download/pdf/159146935.pdf) gives it as 0.19.
# 3/28/20: Bjarni, MicroLink Devices has flexible triple-junction cells at 31% and 37.75% efficiency.
# 4/5/20: Bjarni, "I'd make the approximation that we can get at least 25% (after accounting for MPPT, mismatch; before thermal effects)."
# 4/13/20: Bjarni "Knock down by 5% since we need to account for things like wing curvature, avionics power, etc."
# 4/17/20: Bjarni, Using SunPower Gen2: 0.223 from from https://us.sunpower.com/sites/default/files/sp-gen2-solar-cell-ds-en-a4-160-506760e.pdf
# 4/21/20: Bjarni, Knock down SunPower Gen2 numbers to 18.6% due to mismatch, gaps, fingers & edges, covering, spectrum losses, temperature corrections.
# 4/29/20: Bjarni, Config slack: SunPower cells can do 21% on a panel level. Area density may change; TBD

# Solar cell weight
# rho_solar_cells = 0.425  # kg/m^2, solar cell area density. Sunpower.
rho_solar_cells = 0.350  # kg/m^2, solar cell area density. Microlink.
# The solar_simple_demo model gives this as 0.27. Burton's model gives this as 0.30.
# This paper (https://core.ac.uk/download/pdf/159146935.pdf) gives it as 0.42.
# This paper (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=4144&context=facpub) effectively gives it as 0.3143.
# According to Bjarni, MicroLink Devices has cells on the order of 250 g/m^2 - but they're prohibitively expensive.
# Bjarni, 4/5/20: "400 g/m^2"
# 4/10/20: 0.35 kg/m^2 taken from avionics spreadsheet: https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0
# 4/17/20: Using SunPower Gen2: 0.425 from https://us.sunpower.com/sites/default/files/sp-gen2-solar-cell-ds-en-a4-160-506760e.pdf

MPPT_efficiency = 1 / 1.04
# Bjarni, 4/17/20 in #powermanagement Slack.

solar_area_fraction = opti.variable(  # TODO log-transform?
    init_guess=0.80,
    scale=0.5,
    category="des"
)
opti.subject_to([
    solar_area_fraction > 0,
    solar_area_fraction < 0.80,  # TODO check
])

area_solar = (
                 wing.area()
             ) * solar_area_fraction

# Energy generation cascade
power_in_from_sun = solar_flux_on_horizontal * area_solar / energy_generation_margin
power_in_after_panels = power_in_from_sun * solar_cell_efficiency
power_in = power_in_after_panels * MPPT_efficiency

mass_solar_cells = rho_solar_cells * area_solar

### Battery calculations

battery_charge_efficiency = 0.985
battery_discharge_efficiency = 0.985
# Taken from Bjarni, 4/17/20 in #powermanagment Slack

mass_battery_pack = lib_prop_elec.mass_battery_pack(
    battery_capacity_Wh=battery_capacity_watt_hours,
    battery_cell_specific_energy_Wh_kg=battery_specific_energy_Wh_kg,
    battery_pack_cell_fraction=battery_pack_cell_percentage
)
mass_battery_cells = mass_battery_pack * battery_pack_cell_percentage

mass_wires = lib_prop_elec.mass_wires(
    wire_length=wing.span() / 2,
    max_current=power_out_propulsion_max / battery_voltage,
    allowable_voltage_drop=battery_voltage * 0.01,
    material="aluminum"
)  # buildup model
# mass_wires = 0.868  # Taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

# Calculate MPPT power requirement
power_in_after_panels_max = opti.variable(
    init_guess=5e3,
    scale=5e3,
    category="des"
)
opti.subject_to([
    power_in_after_panels_max > power_in_after_panels,
    power_in_after_panels_max > 0
])

n_MPPT = 5
mass_MPPT = n_MPPT * lib_solar.mass_MPPT(
    power_in_after_panels_max / n_MPPT)  # Model taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

mass_power_systems_misc = 0.314  # Taken from Avionics spreadsheet on 4/10/20, includes HV-LV convs. and fault isolation mechs
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

# Total system mass
mass_power_systems = mass_solar_cells + mass_battery_pack + mass_wires + mass_MPPT + mass_power_systems_misc

# endregion

# region Weights

### Structural mass

# Wing
n_ribs_wing = opti.variable(
    init_guess=200,
    scale=200,
    category="des",
    log_transform=True,
)

mass_wing_primary = lib_mass_struct.mass_wing_spar(
    span=wing.span() - 2 * strut_y_location,  # effective span, TODO review
    mass_supported=max_mass_total,
    # technically the spar doesn't really have to support its own weight (since it's roughly spanloaded), so this is conservative
    ultimate_load_factor=structural_load_factor,
    n_booms=1
) * 11.382 / 9.222  # scaling factor taken from Daedalus weights to account for real-world effects, non-cap mass, etc.


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
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord
    n_end_ribs = 2 * n_wing_sections - 2
    area = span * chord

    # Rib mass
    W_wr = n_ribs * (chord ** 2 * t_over_c * 5.50e-2 + chord * 1.91e-3) * 1.3
    # x1.3 scales to estimates from structures subteam

    # Half rib mass
    W_whr = (n_ribs - 1) * skin_density * chord * 0.65 * 0.072
    # 40% of cross sectional area, same construction as skin panels

    # End rib mass
    W_wer = n_end_ribs * (chord ** 2 * t_over_c * 6.62e-1 + chord * 6.57e-3)

    # LE sheeting mass
    # W_wLE = 0.456/2 * (span ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # Skin Panel Mass
    W_wsp = area * skin_density * 1.05  # assumed constant thickness from 0.9c around LE to 0.15c

    # TE mass
    W_wTE = span * 2.77e-2

    # Covering
    W_wc = area * 0.076  # 0.033 kg/m2 Tedlar covering on 2 sides, with 1.1 coverage factor

    mass_secondary = (W_wr + W_whr + W_wer + W_wTE) * scaling_factor + W_wsp + W_wc

    return mass_secondary


mass_wing_secondary = estimate_mass_wing_secondary(
    span=wing.span(),
    chord=wing.mean_geometric_chord(),
    n_ribs=n_ribs_wing,
    skin_density=0.220,  # kg/m^2
    n_wing_sections=4,
    t_over_c=0.14,
    scaling_factor=1.5  # Suggested by Drela
)

mass_wing = mass_wing_primary + mass_wing_secondary

# Stabilizers
q_ne = 100  # Never-exceed dynamic pressure [Pa]. TODO verify this against 16.82 CDR.


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
    init_guess=40,
    scale=40,
    category="des",
    log_transform=True
)
mass_center_hstab = mass_hstab(center_hstab, n_ribs_center_hstab)

n_ribs_outboard_hstab = opti.variable(
    init_guess=40,
    scale=30,
    category="des",
    log_transform=True,
)
mass_right_hstab = mass_hstab(right_hstab, n_ribs_outboard_hstab)
mass_left_hstab = mass_hstab(left_hstab, n_ribs_outboard_hstab)


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
    mass_vstab = mass_vstab_primary + mass_vstab_secondary  # per vstab
    return mass_vstab


n_ribs_vstab = opti.variable(
    init_guess=35,
    scale=20,
    category="des"
)
opti.subject_to(n_ribs_vstab > 0)
mass_center_vstab = mass_vstab(center_vstab, n_ribs_vstab)

# Fuselage & Boom
mass_center_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=center_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_ne,
    # mean_tail_surface_area=cas.fmax(center_hstab.area(), center_vstab.area()), # most optimistic
    # mean_tail_surface_area=cas.sqrt(center_hstab.area() ** 2 + center_vstab.area() ** 2),
    mean_tail_surface_area=center_hstab.area() + center_vstab.area(),  # most conservative
)  # / 3 # TODO Jamie divided this by 3 on basis of 11/17/20 structures presentation; check this.
mass_right_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_ne,
    mean_tail_surface_area=right_hstab.area()
)
mass_left_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_ne,
    mean_tail_surface_area=left_hstab.area()
)

# The following taken from Daedalus:  # taken from Daedalus, http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
mass_daedalus = 103.9  # kg, corresponds to 229 lb gross weight. Total mass of the Daedalus aircraft, used as a reference for scaling.
mass_fairings = 2.067 * max_mass_total / mass_daedalus  # Scale fairing mass to same mass fraction as Daedalus
mass_landing_gear = 0.728 * max_mass_total / mass_daedalus  # Scale landing gear mass to same mass fraction as Daedalus
mass_strut = 661 / 2 * (strut_chord / 10) ** 2 * strut_span  # mass per strut, formula from Jamie

mass_center_fuse = mass_center_boom + mass_fairings + mass_landing_gear  # per fuselage
mass_right_fuse = mass_right_boom
mass_left_fuse = mass_left_boom

mass_structural = (
        mass_wing +
        mass_center_hstab +
        mass_right_hstab +
        mass_left_hstab +
        mass_center_vstab +
        mass_center_fuse +
        mass_right_fuse +
        mass_left_fuse +
        mass_strut * 2  # left and right struts
)
mass_structural *= structural_mass_margin_multiplier

### Avionics
mass_avionics = 12.153  # Pulled from Avionics team spreadsheet on 5/13
# Back-calculated from Kevin Uleck's figures in MIT 16.82 presentation: 24.34 kg = 3.7 / 3.8 * 25
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

opti.subject_to([
    mass_total / 250 == (
            mass_payload + mass_structural + mass_propulsion + mass_power_systems + mass_avionics
    ) / 250
])

gravity_force = g * mass_total

# endregion

# region Dynamics

net_force_parallel_calc = (
        thrust_force * cosd(alpha) -
        drag_force -
        gravity_force * sind(flight_path_angle)
)
net_force_perpendicular_calc = (
        thrust_force * sind(alpha) +
        lift_force -
        gravity_force * cosd(flight_path_angle)
)

opti.subject_to([
    net_accel_parallel * mass_total / 1e1 == net_force_parallel_calc / 1e1,
    net_accel_perpendicular * mass_total / 1e2 == net_force_perpendicular_calc / 1e2,
])

speeddot = net_accel_parallel
gammadot = (net_accel_perpendicular / airspeed) * 180 / np.pi

trapz = lambda x: (x[1:] + x[:-1]) / 2

dt = cas.diff(time)
dx = cas.diff(x)
dy = cas.diff(y)
dspeed = cas.diff(airspeed)
dgamma = cas.diff(flight_path_angle)

xdot_trapz = trapz(airspeed * cosd(flight_path_angle))
ydot_trapz = trapz(airspeed * sind(flight_path_angle))
speeddot_trapz = trapz(speeddot)
gammadot_trapz = trapz(gammadot)

##### Winds

wind_speed = wind_speed_func(y)
wind_speed_midpoints = wind_speed_func(trapz(y))

# Total
opti.subject_to([
    dx / 1e4 == (xdot_trapz - wind_speed_midpoints) * dt / 1e4,
    dy / 1e2 == ydot_trapz * dt / 1e2,
    dspeed / 1e-1 == speeddot_trapz * dt / 1e-1,
    dgamma / 1e-2 == gammadot_trapz * dt / 1e-2,
])

# Powertrain-specific
opti.subject_to([
    net_power / 5e3 < (power_in - power_out) / 5e3,
])

# Do the math for battery charging/discharging efficiency
tanh_sigmoid = lambda x: 0.5 + 0.5 * cas.tanh(x)
net_power_to_battery = net_power * (
        1 / battery_discharge_efficiency * (1 - tanh_sigmoid(net_power / 10)) +
        battery_charge_efficiency * tanh_sigmoid(net_power / 10)
)  # tanh blending to avoid optimizer stalling on nonsmooth integrator

# Do the integration
net_power_to_battery_trapz = trapz(net_power_to_battery)

dbattery_stored_energy_nondim = cas.diff(battery_stored_energy_nondim)
opti.subject_to([
    dbattery_stored_energy_nondim / 1e-2 < (net_power_to_battery_trapz / battery_capacity) * dt / 1e-2,
])
# endregion

# region Finalize Optimization Problem

##### Add initial state constraints
opti.subject_to([
    x[0] / 1e5 == 0,  # Start at x-datum of zero
])

##### Add periodic constraints
opti.subject_to([
    x[time_periodic_end_index] / 1e5 > (x[time_periodic_start_index] + required_headway_per_day) / 1e5,
    y[time_periodic_end_index] / 1e4 > y[time_periodic_start_index] / 1e4,
    airspeed[time_periodic_end_index] / 2e1 > airspeed[time_periodic_start_index] / 2e1,
    battery_stored_energy_nondim[time_periodic_end_index] > battery_stored_energy_nondim[time_periodic_start_index],
    flight_path_angle[time_periodic_end_index] == flight_path_angle[time_periodic_start_index],
    alpha[time_periodic_end_index] == alpha[time_periodic_start_index],
    thrust_force[time_periodic_end_index] / 1e2 == thrust_force[time_periodic_start_index] / 1e2,
])

##### Optional constraints
if not allow_trajectory_optimization:
    opti.subject_to([
        flight_path_angle / 100 == 0
    ])
    # Prevent groundspeed loss
    opti.subject_to([
        airspeed / 20 > wind_speed / 20
    ])

###### Climb Optimization Constraints
if climb_opt:
    opti.subject_to(y[0] / 1e4 == 0)

##### Add objective
objective = eval(minimize)

##### Extra constraints
# opti.subject_to([
#     battery_capacity_watt_hours == 68256,  # Fixing to a discrete option from Annick # 4/30/2020
#     propeller_diameter == 2.16,  # Fixing to Dongjoon's design
# ])
opti.subject_to([
    center_hstab_span == outboard_hstab_span,
    center_hstab_chord == outboard_hstab_chord,
    # center_hstab_twist_angle == outboard_hstab_twist_angle,
    # center_boom_length == outboard_boom_length,
    center_hstab_twist_angle <= 0,  # essentially enforces downforce, prevents hstab from lifting and exploiting config.
    outboard_hstab_twist_angle <= 0,
    # essentially enforces downforce, prevents hstab from lifting and exploiting config.
])

##### Useful metrics
wing_loading = 9.81 * max_mass_total / wing.area()
wing_loading_psf = wing_loading / 47.880258888889
empty_wing_loading = 9.81 * mass_structural / wing.area()
empty_wing_loading_psf = empty_wing_loading / 47.880258888889
propeller_efficiency = thrust_force * airspeed / power_out_propulsion_shaft
cruise_LD = lift_force / drag_force

##### Add tippers
things_to_slightly_minimize = (
        wing_span / 80
        + n_propellers / 1
        + propeller_diameter / 2
        + battery_capacity_watt_hours / 30000
        + solar_area_fraction / 0.5
)

# Dewiggle
penalty = 0


def penalty_norm(x):
    return cas.sumsqr(cas.diff(cas.diff(x))) / n_timesteps_per_segment


penalty += penalty_norm(thrust_force / 10)
penalty += penalty_norm(net_accel_parallel / 1e-1)
penalty += penalty_norm(net_accel_perpendicular / 1e-1)
penalty += penalty_norm(airspeed / 2)
penalty += penalty_norm(flight_path_angle / 2)
penalty += penalty_norm(alpha / 1)

opti.minimize(
    objective
    + penalty
    + 1e-6 * things_to_slightly_minimize
)
# endregion

if __name__ == "__main__":
    # Solve
    sol = opti.solve()

    # Print a warning if the penalty term is unreasonably high
    penalty_objective_ratio = np.abs(sol.value(penalty / objective))
    if penalty_objective_ratio > 0.01:
        print(
            f"\nWARNING: High penalty term, non-negligible integration error likely! P/O = {penalty_objective_ratio}\n")


    # # region Postprocessing utilities, console output, etc.
    def s(x: cas.MX) -> np.ndarray:  # Shorthand for evaluating the value of a quantity x at the optimum
        return sol.value(x)


    def output(x: Union[str, List[str]]) -> None:  # Output a scalar variable (give variable name as a string).
        if isinstance(x, list):
            for xi in x:
                output(xi)
            return
        print(f"{x}: {sol.value(eval(x)):.3f}")


    def print_title(s: str) -> None:  # Print a nicely formatted title
        print(f"\n{'*' * 10} {s.upper()} {'*' * 10}")


    print_title("Key Results")
    output([
        "max_mass_total",
        "wing_span",
        "wing_root_chord"
    ])


    def qp(var_name):
        # QuickPlot a variable.
        fig = px.scatter(y=s(eval(var_name)), title=var_name, labels={'y': var_name})
        fig.data[0].update(mode='markers+lines')
        fig.show()


    def qp2(x_name, y_name):
        # QuickPlot two variables.
        fig = px.scatter(
            x=s(eval(x_name)),
            y=s(eval(y_name)),
            title=f"{x_name} vs. {y_name}",
            labels={'x': x_name, 'y': y_name}
        )
        fig.data[0].update(mode='markers+lines')
        fig.show()


    def qp3(x_name, y_name, z_name):
        # QuickPlot three variables.
        fig = px.scatter_3d(
            x=s(eval(x_name)),
            y=s(eval(y_name)),
            z=s(eval(z_name)),
            title=f"{x_name} vs. {y_name} vs. {z_name}",
            labels={'x': x_name, 'y': y_name, 'z': z_name},
            size_max=18
        )
        fig.data[0].update(mode='markers+lines')
        fig.show()


    def draw():  # Draw the geometry of the optimal airplane
        airplane.substitute_solution(sol).draw()


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
        plot("hour", "y_km",
             xlabel="Hours after Solar Noon",
             ylabel="Altitude [km]",
             title="Altitude over Simulation",
             save_name="outputs/altitude.png"
             )
        plot("hour", "airspeed",
             xlabel="Hours after Solar Noon",
             ylabel="True Airspeed [m/s]",
             title="True Airspeed over Simulation",
             save_name="outputs/airspeed.png"
             )
        plot("hour", "net_power",
             xlabel="Hours after Solar Noon",
             ylabel="Net Power [W] (positive is charging)",
             title="Net Power over Simulation",
             save_name="outputs/net_power.png"
             )
        plot("hour", "battery_state_of_charge_percentage",
             xlabel="Hours after Solar Noon",
             ylabel="State of Charge [%]",
             title="Battery Charge State over Simulation",
             save_name="outputs/battery_charge.png"
             )
        plot("x_km", "y_km",
             xlabel="Downrange Distance [km]",
             ylabel="Altitude [km]",
             title="Optimal Trajectory over Simulation",
             save_name="outputs/trajectory.png"
             )

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
            s(mass_payload),
            s(mass_structural),
            s(mass_propulsion),
            s(cas.mmax(mass_power_systems)),
            s(mass_avionics),
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
            "Wing",
            "Stabilizers",
            "Fuses & Booms",
            "Margin"
        ]
        pie_values = [
            s(mass_wing),
            s(
                mass_center_hstab +
                mass_right_hstab +
                mass_left_hstab +
                mass_center_vstab
            ),
            s(
                mass_center_fuse +
                mass_right_fuse +
                mass_left_fuse
            ),
            s(mass_structural - (
                    mass_wing +
                    mass_center_hstab +
                    mass_right_hstab +
                    mass_left_hstab +
                    mass_center_vstab +
                    mass_center_fuse +
                    mass_right_fuse +
                    mass_left_fuse
            )
              ),
        ]
        colors = plt.cm.Set2(np.arange(5))
        colors = np.clip(
            colors[1, :3] + np.expand_dims(
                np.linspace(-0.1, 0.2, len(pie_labels)),
                1),
            0, 1
        )
        pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (
            x * s(mass_structural) / 100, x * s(mass_structural / mass_total))
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
            s(mass_battery_cells),
            s(mass_battery_pack - mass_battery_cells),
            s(mass_solar_cells),
            s(mass_power_systems - mass_battery_pack - mass_solar_cells),
        ]
        colors = plt.cm.Set2(np.arange(5))
        colors = np.clip(
            colors[3, :3] + np.expand_dims(
                np.linspace(-0.1, 0.2, len(pie_labels)),
                1),
            0, 1
        )[::-1]
        pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (
            x * s(mass_power_systems) / 100, x * s(mass_power_systems / mass_total))
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
            s="* percentages referenced to total aircraft mass",
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
            s="""
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

    # Write a mass budget
    with open("outputs/mass_budget.csv", "w+") as f:
        from types import ModuleType

        f.write("Object or Collection of Objects, Mass [kg],\n")
        for var_name in dir():
            if "mass" in var_name and not type(eval(var_name)) == ModuleType and not callable(eval(var_name)):
                f.write("%s, %f,\n" % (var_name, s(eval(var_name))))

    # Write a geometry spreadsheet
    with open("outputs/geometry.csv", "w+") as f:

        f.write("Design Variable, Value (all in base SI units or derived units thereof),\n")
        geometry_vars = [
            'wing_span',
            'wing_root_chord',
            'wing_taper_ratio',
            '',
            'center_hstab_span',
            'center_hstab_chord',
            '',
            'outboard_hstab_span',
            'outboard_hstab_chord',
            '',
            'center_vstab_span',
            'center_vstab_chord',
            '',
            'max_mass_total',
            'mass_total'
            '',
            'solar_area_fraction',
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
