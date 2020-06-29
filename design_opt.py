# Imports
import aerosandbox as asb
import aerosandbox.library.aerodynamics as aero
import aerosandbox.library.atmosphere as atmo
from aerosandbox.tools.casadi_tools import *
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import power_gas as lib_gas
from aerosandbox.library import power_solar as lib_solar
from aerosandbox.library import propulsion_electric as lib_prop_elec
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.library import winds as lib_winds
from aerosandbox.library.airfoils import *
import copy
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import json
import design_opt_utilities as utils

sns.set(font_scale=1)

# region Setup
##### Initialize Optimization
opti = cas.Opti()
des_vars = {}

##### Caching
# # Uncomment these lines to do ANALYSIS and OPTIMIZATION.
file_to_load_from = None
file_to_save_to = "des_vars.json"

# Uncomment these lines to do ANALYSIS on a FROZEN DESIGN.
# file_to_load_from = "des_vars.json"
# file_to_save_to = None

# minimize = "wing.span() / 50"  # any "eval-able" expression
# minimize = "max_mass_total / 300" # any "eval-able" expression
minimize = "wing.span() / 50 * 0.9 + max_mass_total / 300 * 0.1"


# minimize = "cas.sum1(airspeed/20)" # any "eval-able" expression

def des_var(  # design variable
        name,
        initial_guess,
        scale_factor,
        n_variables=1
):
    if file_to_load_from is None:
        var = scale_factor * opti.variable(n_variables)
        opti.set_initial(var, initial_guess)
        des_vars[name] = var
        return var
    else:
        var = scale_factor * opti.variable(n_variables)
        opti.set_initial(var, initial_guess)
        des_vars[name] = var
        with open(file_to_load_from, "r") as f:
            solved_des_vars = json.load(f)
        val = solved_des_vars[name]
        opti.set_initial(var, val)
        opti.subject_to(var == val)
        return var


def ops_var(  # operational variable
        initial_guess,
        scale_factor,
        n_variables=1
):
    var = scale_factor * opti.variable(n_variables)
    opti.set_initial(var, initial_guess)
    return var


##### Operating Parameters
latitude = 49  # degrees (49 deg is top of CONUS, 26 deg is bottom of CONUS)
day_of_year = 244  # Julian day. June 1 is 153, June 22 is 174, Aug. 31 is 244
min_altitude = 19812  # meters. 19812 m = 65000 ft.
# min_altitude = 18288  # meters. 18288 m = 60000 ft.
required_headway_per_day = 10e3  # meters
n_booms = 3
days_to_simulate = opti.parameter()
opti.set_value(days_to_simulate, 1)
enforce_periodicity = True  # Tip: turn this off when looking at gas models or models w/o trajectory opt. enabled.
allow_trajectory_optimization = True
structural_load_factor = 3  # over static
show_plots = True
mass_payload = opti.parameter()
opti.set_value(mass_payload, 30)
wind_speed_func = lambda alt: lib_winds.wind_speed_conus_summer_99(alt, latitude)
battery_specific_energy_Wh_kg = opti.parameter()
opti.set_value(battery_specific_energy_Wh_kg, 450)

##### Margins
structural_mass_margin_multiplier = opti.parameter()
opti.set_value(structural_mass_margin_multiplier, 1.25)
energy_generation_margin = opti.parameter()
opti.set_value(energy_generation_margin, 1.05)
allowable_battery_depth_of_discharge = 0.85  # How much of the battery can you actually use? # Reviewed w/ Annick & Bjarni 4/30/2020

##### Simulation Parameters
n_timesteps = 150  # Only relevant if allow_trajectory_optimization is True.
# Quick convergence testing indicates you can get bad analyses below 150 or so...

##### Optimization bounds
min_speed = 1  # Specify a minimum speed - keeps the speed-gamma velocity parameterization from NaNing

##### Climb Optimization
climb_opt = False  # are we optimizing for the climb as well
seconds_per_day = 86400
if climb_opt:
    simulation_days = 1.5  # must be greater than 1
    opti.set_value(days_to_simulate, simulation_days)
    enforce_periodicity = False  # Leave False
    time_shift = -6.5 * 60 * 60  # 60*((seconds_per_day*2)/(n_timesteps))
    timesteps_of_last_day = int((1 - (1 / simulation_days)) * n_timesteps)
else:
    time_shift = 0

# endregion

# region Trajectory Optimization Variables
##### Initialize trajectory optimization variables

x = ops_var(initial_guess=0, scale_factor=1e5, n_variables=n_timesteps)

y = ops_var(initial_guess=20000, scale_factor=1e4, n_variables=n_timesteps)

if not climb_opt:
    opti.subject_to([
        y / min_altitude > 1,
        y / 40000 < 1,  # models break down
    ])
else:
    opti.subject_to([y[timesteps_of_last_day:-1] > min_altitude, y / 40000 < 1])

airspeed = ops_var(initial_guess=20, scale_factor=20, n_variables=n_timesteps)
opti.subject_to([
    airspeed / min_speed > 1
])

flight_path_angle = ops_var(initial_guess=0, scale_factor=0.1, n_variables=n_timesteps)
opti.subject_to([
    flight_path_angle / 90 < 1,
    flight_path_angle / 90 > -1,
])

alpha = ops_var(initial_guess=5, scale_factor=4, n_variables=n_timesteps)
opti.subject_to([
    alpha > -8,
    alpha < 12
])

thrust_force = ops_var(initial_guess=150, scale_factor=200, n_variables=n_timesteps)
opti.subject_to([
    thrust_force > 0
])

net_accel_parallel = ops_var(initial_guess=0, scale_factor=1e-2, n_variables=n_timesteps)
net_accel_perpendicular = ops_var(initial_guess=0, scale_factor=1e-1, n_variables=n_timesteps)

##### Set up time
time_nondim = cas.linspace(0, 1, n_timesteps)
seconds_per_day = 86400
time = time_nondim * days_to_simulate * seconds_per_day
hour = time / 3600

# endregion

# region Design Optimization Variables
##### Initialize design optimization variables (all units in base SI or derived units)

# log_mass_total = opti.variable()
# opti.set_initial(log_mass_total, cas.log(600))
# mass_total = cas.exp(log_mass_total)

mass_total = des_var(name="mass_total", initial_guess=600, scale_factor=600)

max_mass_total = mass_total

### Initialize geometric variables
# wing
wing_span = des_var(name="wing_span", initial_guess=40, scale_factor=60)
opti.subject_to([wing_span > 1])

wing_root_chord = des_var(name="wing_root_chord", initial_guess=3, scale_factor=4)
opti.subject_to([wing_root_chord > 0.1])

wing_x_quarter_chord = des_var(name="wing_x_quarter_chord", initial_guess=0, scale_factor=0.01)

wing_y_taper_break = 0.57 * wing_span / 2

wing_taper_ratio = 0.5

# center hstab
center_hstab_span = des_var(name="center_hstab_span", initial_guess=12, scale_factor=15)
opti.subject_to([
    center_hstab_span > 0.1,
    center_hstab_span < wing_span / n_booms / 2
])

center_hstab_chord = des_var(name="center_hstab_chord", initial_guess=3, scale_factor=2)
opti.subject_to([center_hstab_chord > 0.1])

center_hstab_twist_angle = ops_var(initial_guess=-7, scale_factor=2, n_variables=n_timesteps)

# center hstab
outboard_hstab_span = des_var(name="outboard_hstab_span", initial_guess=12, scale_factor=15)
opti.subject_to([
    outboard_hstab_span > 0.1,
    outboard_hstab_span < wing_span / n_booms / 2
])

outboard_hstab_chord = des_var(name="outboard_hstab_chord", initial_guess=3, scale_factor=2)
opti.subject_to([outboard_hstab_chord > 0.1])

outboard_hstab_twist_angle = ops_var(initial_guess=-7, scale_factor=2, n_variables=n_timesteps)

# center_vstab
center_vstab_span = des_var(name="center_vstab_span", initial_guess=7, scale_factor=8)
opti.subject_to(center_vstab_span > 0.1)

center_vstab_chord = des_var(name="center_vstab_chord", initial_guess=2.5, scale_factor=2)
opti.subject_to([center_vstab_chord > 0.1])

# center_fuselage
center_boom_length = des_var(name="boom_length", initial_guess=23, scale_factor=2)  # TODO add scale factor
opti.subject_to([
    center_boom_length - center_vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

# outboard_fuselage
outboard_boom_length = des_var(name="boom_length", initial_guess=23, scale_factor=2)  # TODO add scale factor
opti.subject_to([
    outboard_boom_length - center_vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

nose_length = 1.80  # Calculated on 4/15/20 with Trevor and Olek
# https://docs.google.com/spreadsheets/d/1BnTweK-B4Hmmk9piJn8os-LNPiJH-3rJJemYkrKjARA/edit#gid=0

fuse_diameter = 0.6
boom_diameter = 0.2

import dill as pickle

import pathlib
path = str(
    pathlib.Path(__file__).parent.absolute()
)

try:
    with open(path+"/cache/wing_airfoil.cache", "rb") as f:
        wing_airfoil = pickle.load(f)
    with open(path+"/cache/tail_airfoil.cache", "rb") as f:
        tail_airfoil = pickle.load(f)
except FileNotFoundError:
    wing_airfoil = Airfoil(name="HALE_03", coordinates=r"C:\Projects\GitHub\Airfoils\HALE_03.dat")
    wing_airfoil.populate_sectional_functions_from_xfoil_fits(parallel=False)
    with open(path+"/cache/wing_airfoil.cache", "wb+") as f:
        pickle.dump(wing_airfoil, f)
    tail_airfoil = Airfoil("naca0008")
    tail_airfoil.populate_sectional_functions_from_xfoil_fits(parallel=False)
    with open(path+"/cache/tail_airfoil.cache", "wb+") as f:
        pickle.dump(tail_airfoil, f)

tail_airfoil = naca0008  # TODO remove this and use fits?

wing = asb.Wing(
    name="Main Wing",
    x_le=wing_x_quarter_chord,  # Coordinates of the wing's leading edge # TODO make this a free parameter?
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
    y_le=wing_y_taper_break,  # Coordinates of the wing's leading edge
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
    nose_length=nose_length,
    fuse_diameter=boom_diameter,
    boom_diameter=boom_diameter,
)

boom_location = 0.57  # as a fraction of the half-span
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
    latitude, day_of_year, time + time_shift, scattering=True
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

    wing_oswalds_efficiency = 0.95  # TODO make this a function of taper ratio
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


# center_vstab
def vstab_aero(vstab: asb.Wing):
    vstab_Re = rho / mu * airspeed * vstab.mean_geometric_chord()
    vstab_airfoil = vstab.xsecs[0].airfoil  # type: asb.Airfoil
    vstab_Cd_profile = vstab_airfoil.CDp_function(0, vstab_Re, mach, 0)
    drag_vstab_profile = vstab_Cd_profile * q * vstab.area()
    drag_vstab = drag_vstab_profile  # per vstab
    return drag_vstab


drag_center_vstab = vstab_aero(center_vstab)

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
        drag_left_fuse
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
static_margin_fraction = (x_ac - airplane.xyz_ref[0]) / wing.mean_geometric_chord()
opti.subject_to([
    static_margin_fraction == 0.2
])

### Trim
opti.subject_to([
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
])

# endregion

# region Propulsion

### Propeller calculations
propeller_diameter = des_var(name="propeller_diameter", initial_guess=5, scale_factor=5)  # TODO scale factor
opti.subject_to([
    propeller_diameter / 1 > 1,
    propeller_diameter / 10 < 1
])

n_propellers = opti.parameter()
opti.set_value(n_propellers, 6)

propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * atmo.get_speed_of_sound_from_temperature(
    atmo.get_temperature_at_altitude(20000)
) / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / cas.pi

area_propulsive = cas.pi / 4 * propeller_diameter ** 2 * n_propellers
propeller_coefficient_of_performance = 0.90  # a total WAG
motor_efficiency = 0.955  # Taken from ThinGap estimates
# 0.856 / (0.856 + 0.026 + 0.018 + 0.004)  # back-calculated from Odysseus data (94.7%)

power_out_propulsion_shaft = lib_prop_prop.propeller_shaft_power_from_thrust(
    thrust_force=thrust_force,
    area_propulsive=area_propulsive,
    airspeed=airspeed,
    rho=rho,
    propeller_coefficient_of_performance=propeller_coefficient_of_performance
)

power_out_propulsion = power_out_propulsion_shaft / motor_efficiency

power_out_max = des_var(name="power_out_max", initial_guess=5e3, scale_factor=5e3)
opti.subject_to([
    power_out_propulsion < power_out_max,
    power_out_max > 0
])

propeller_max_torque = (power_out_max / n_propellers) / propeller_rads_per_sec

battery_voltage = 225  # From Olek Peraire >4/2, propulsion slack
# battery_voltage = opti.variable()  # From Olek Peraire 4/2, propulsion slack
# opti.set_initial(battery_voltage, 240)
# opti.subject_to([
#     battery_voltage > 0,
#     battery_voltage < 2000
# ])

motor_kv = propeller_rpm / battery_voltage

mass_motor_raw = lib_prop_elec.mass_motor_electric(
    max_power=power_out_max / n_propellers,
    kv_rpm_volt=motor_kv,
    voltage=battery_voltage
) * n_propellers
# mass_motor_raw = lib_prop_elec.mass_motor_electric(max_power=power_out_max / n_propellers) * n_propellers # old model


mass_motor_mounted = 2 * mass_motor_raw  # similar to a quote from Raymer, modified to make sensible units, prop weight roughly subtracted

mass_propellers = n_propellers * lib_prop_prop.mass_hpa_propeller(
    diameter=propeller_diameter,
    max_power=power_out_max,
    include_variable_pitch_mechanism=True
)
mass_ESC = lib_prop_elec.mass_ESC(max_power=power_out_max)

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
net_power = ops_var(initial_guess=0, scale_factor=1000, n_variables=n_timesteps)

battery_stored_energy_nondim = ops_var(initial_guess=0.5, scale_factor=1, n_variables=n_timesteps)
opti.subject_to([
    battery_stored_energy_nondim > 0,
    battery_stored_energy_nondim < allowable_battery_depth_of_discharge,
])

battery_capacity = des_var(name="battery_capacity", initial_guess=3600 * 60e3,
                           scale_factor=3600 * 60e3)  # Joules, not watt-hours!
opti.subject_to([
    battery_capacity > 0
])
battery_capacity_watt_hours = battery_capacity / 3600
battery_stored_energy = battery_stored_energy_nondim * battery_capacity

### Solar calculations

# solar_cell_efficiency = 0.21
solar_cell_efficiency = 0.25
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
# rho_solar_cells = 0.425  # kg/m^2, solar cell area density.
rho_solar_cells = 0.350  # kg/m^2, solar cell area density.
# The solar_simple_demo model gives this as 0.27. Burton's model gives this as 0.30.
# This paper (https://core.ac.uk/download/pdf/159146935.pdf) gives it as 0.42.
# This paper (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=4144&context=facpub) effectively gives it as 0.3143.
# According to Bjarni, MicroLink Devices has cells on the order of 250 g/m^2 - but they're prohibitively expensive.
# Bjarni, 4/5/20: "400 g/m^2"
# 4/10/20: 0.35 kg/m^2 taken from avionics spreadsheet: https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0
# 4/17/20: Using SunPower Gen2: 0.425 from https://us.sunpower.com/sites/default/files/sp-gen2-solar-cell-ds-en-a4-160-506760e.pdf

MPPT_efficiency = 1 / 1.04
# Bjarni, 4/17/20 in #powermanagement Slack.

solar_area_fraction = des_var(name="solar_area_fraction", initial_guess=0.5, scale_factor=1)
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

battery_pack_cell_percentage = 0.89  # What percent of the battery pack consists of the module, by weight?
# Accounts for module HW, BMS, pack installation, etc.
# Ed Lovelace (in his presentation) gives 70% as a state-of-the-art fraction.
# Used 75% in the 16.82 CDR.
# According to Kevin Uleck, Odysseus realized an 89% packing factor here.

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
    max_current=power_out_max / battery_voltage,
    allowable_voltage_drop=battery_voltage * 0.01,
    material="aluminum"
)  # buildup model
# mass_wires = 0.868  # Taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

mass_MPPT = 3*lib_solar.mass_MPPT(5000)  # Model taken from Avionics spreadsheet on 4/10/20
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

mass_power_systems_misc = 0.314  # Taken from Avionics spreadsheet on 4/10/20, includes HV-LV convs. and fault isolation mechs
# https://docs.google.com/spreadsheets/d/1nhz2SAcj4uplEZKqQWHYhApjsZvV9hme9DlaVmPca0w/edit?pli=1#gid=0

# Total system mass
mass_power_systems = mass_solar_cells + mass_battery_pack + mass_wires + mass_MPPT + mass_power_systems_misc

# endregion

# region Weights
# Payload mass
# mass_payload = # defined above

### Structural mass

# Wing
n_ribs_wing = des_var(name="n_ribs_wing", initial_guess=200, scale_factor=200)
opti.subject_to([
    n_ribs_wing > 0,
])

mass_wing_primary = lib_mass_struct.mass_wing_spar(
    span=wing.span(),
    mass_supported=max_mass_total,
    # technically the spar doesn't really have to support its own weight (since it's roughly spanloaded), so this is conservative
    ultimate_load_factor=structural_load_factor,
    n_booms=1
) * 11.382 / 9.222  # scaling factor taken from Daedalus weights
mass_wing_secondary = lib_mass_struct.mass_hpa_wing(
    span=wing.span(),
    chord=wing.mean_geometric_chord(),
    vehicle_mass=max_mass_total,
    n_ribs=n_ribs_wing,
    n_wing_sections=3,
    ultimate_load_factor=structural_load_factor,
    t_over_c=0.14,
    include_spar=False,
) * 1.3  # TODO review this number! Mark says 1.5! 4/30/2020
mass_wing = mass_wing_primary + mass_wing_secondary

# mass_wing = lib_mass_struct.mass_hpa_wing(
#     span=wing.span(),
#     chord=wing.mean_geometric_chord(),
#     vehicle_mass=max_mass_total,
#     n_ribs=n_ribs_wing,
#     n_wing_sections=1,
#     ultimate_load_factor=structural_load_factor,
#     t_over_c=0.12,
#     include_spar=True,
# )

# Stabilizers
q_maneuver = 80  # TODO make this more accurate


def mass_hstab(
        hstab,
        n_ribs_hstab,
):
    mass_hstab_primary = lib_mass_struct.mass_wing_spar(
        span=hstab.span(),
        mass_supported=q_maneuver * 1.5 * hstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    )

    mass_hstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=hstab.span(),
        chord=hstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_maneuver,
        n_ribs=n_ribs_hstab,
        t_over_c=0.08,
        include_spar=False
    )
    mass_hstab = mass_hstab_primary + mass_hstab_secondary  # per hstab
    return mass_hstab


n_ribs_center_hstab = des_var(name="n_ribs_center_hstab", initial_guess=40, scale_factor=30)
opti.subject_to(n_ribs_center_hstab > 0)
mass_center_hstab = mass_hstab(center_hstab, n_ribs_center_hstab)

n_ribs_outboard_hstab = des_var(name="n_ribs_outboard_hstab", initial_guess=40, scale_factor=30)
opti.subject_to(n_ribs_outboard_hstab > 0)
mass_right_hstab = mass_hstab(right_hstab, n_ribs_outboard_hstab)
mass_left_hstab = mass_hstab(left_hstab, n_ribs_outboard_hstab)


def mass_vstab(
        vstab,
        n_ribs_vstab,
):
    mass_vstab_primary = lib_mass_struct.mass_wing_spar(
        span=vstab.span(),
        mass_supported=q_maneuver * 1.5 * vstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    ) * 1.2  # TODO due to asymmetry, a guess
    mass_vstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=vstab.span(),
        chord=vstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_maneuver,
        n_ribs=n_ribs_vstab,
        t_over_c=0.08
    )
    mass_vstab = mass_vstab_primary + mass_vstab_secondary  # per vstab
    return mass_vstab


n_ribs_vstab = des_var(name="n_ribs_vstab", initial_guess=35, scale_factor=20)
opti.subject_to(n_ribs_vstab > 0)
mass_center_vstab = mass_vstab(center_vstab, n_ribs_vstab)

# Fuselage & Boom
mass_center_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=center_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_maneuver,
    # mean_tail_surface_area=cas.fmax(center_hstab.area(), center_vstab.area()), # most optimistic
    # mean_tail_surface_area=cas.sqrt(center_hstab.area() ** 2 + center_vstab.area() ** 2),
    mean_tail_surface_area=center_hstab.area() + center_vstab.area(),  # most conservative
)
mass_right_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_maneuver,
    mean_tail_surface_area=right_hstab.area()
)
mass_left_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
    dynamic_pressure_at_manuever_speed=q_maneuver,
    mean_tail_surface_area=left_hstab.area()
)

# The following taken from Daedalus:  # taken from Daedalus, http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
mass_fairings = 2.067
mass_landing_gear = 0.728

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
        mass_left_fuse
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
opti.subject_to([
    battery_stored_energy_nondim[-1] > battery_stored_energy_nondim[0],
])
# endregion

# region Finalize Optimization Problem
##### Add periodic constraints
if enforce_periodicity:
    opti.subject_to([
        x[-1] / 1e5 > (x[0] + days_to_simulate * required_headway_per_day) / 1e5,
        y[-1] / 1e4 > y[0] / 1e4,
        airspeed[-1] / 2e1 > airspeed[0] / 2e1,
        flight_path_angle[-1] == flight_path_angle[0],
        alpha[-1] == alpha[0],
        thrust_force[-1] / 1e2 == thrust_force[0] / 1e2,
    ])

##### Add initial state constraints
opti.subject_to([  # Air Launch
    x[0] == 0,
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
    opti.subject_to([y[0] == 0])
    opti.subject_to([
        x[-1] / 1e5 > (x[timesteps_of_last_day] + days_to_simulate * required_headway_per_day) / 1e5,
        y[-1] / 1e4 > y[timesteps_of_last_day] / 1e4,
        airspeed[-1] / 2e1 > airspeed[timesteps_of_last_day] / 2e1,
        flight_path_angle[-1] == flight_path_angle[timesteps_of_last_day],
        alpha[-1] == alpha[timesteps_of_last_day],
        thrust_force[-1] / 1e2 == thrust_force[timesteps_of_last_day] / 1e2,
    ])
    opti.subject_to([battery_stored_energy_nondim[0] == allowable_battery_depth_of_discharge])

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
])

##### Useful metrics
wing_loading = 9.81 * max_mass_total / wing.area()
wing_loading_psf = wing_loading / 47.880258888889
empty_wing_loading = 9.81 * mass_structural / wing.area()
empty_wing_loading_psf = empty_wing_loading / 47.880258888889
propeller_efficiency = thrust_force * airspeed / power_out_propulsion_shaft

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
penalty_denominator = n_timesteps
penalty += cas.sum1(cas.diff(thrust_force / 100) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(net_accel_parallel / 1e-1) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(net_accel_perpendicular / 1e-1) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(airspeed / 30) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(flight_path_angle / 10) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(alpha / 5) ** 2) / penalty_denominator

opti.minimize(
    objective
    + penalty
    + 1e-6 * things_to_slightly_minimize
)
# endregion

# region Solve
p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
# s_opts["bound_frac"] = 0.5
# s_opts["bound_push"] = 0.5
# s_opts["slack_bound_frac"] = 0.5
# s_opts["slack_bound_push"] = 0.5
s_opts["mu_strategy"] = "adaptive"
# s_opts["mu_oracle"] = "quality-function"
# s_opts["quality_function_max_section_steps"] = 20
# s_opts["fixed_mu_oracle"] = "quality-function"
# s_opts["alpha_for_y"] = "min"
# s_opts["alpha_for_y"] = "primal-and-full"
# s_opts["watchdog_shortened_iter_trigger"] = 1
# s_opts["expect_infeasible_problem"]="yes"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.001
# s_opts["evaluate_orig_obj_at_resto_trial"] = "yes"
opti.solver('ipopt', p_opts, s_opts)

# endregion

if __name__ == "__main__":
    try:
        sol = opti.solve()
        # If successful, save a cache of the design variables
        solved_des_vars = {k: sol.value(des_vars[k]) for k in des_vars}
        if file_to_save_to is not None:
            with open(file_to_save_to, "w+") as f:
                json.dump(solved_des_vars, f)

    except:
        sol = opti.debug

    if np.abs(sol.value(penalty / objective)) > 0.01:
        print("\nWARNING: high penalty term! P/O = %.3f\n" % sol.value(penalty / objective))

    # save_sol_to_file(sol,
    #                  save_primal=True,
    #                  save_dual=False,
    #                  primal_location="last_solution.sol"
    #                  )

    # Find dusk and dawn
    try:
        si = sol.value(solar_flux_on_horizontal)
        dusk = np.argwhere(si[:round(len(si) / 2)] < 1)[0, 0]
        dawn = np.argwhere(si[round(len(si) / 2):] > 1)[0, 0] + round(len(si) / 2)
    except IndexError:
        print("Could not find dusk and dawn - you likely have a shorter-than-one-day mission.")

    # # region Text Postprocessing & Utilities
    # ##### Text output
    o = lambda x: print(
        "%s: %f" % (x, sol.value(eval(x))))  # A function to Output a scalar variable. Input a variable name as a string
    outs = lambda xs: [o(x) for x in xs] and None  # input a list of variable names as strings
    print_title = lambda s: print("\n********** %s **********" % s.upper())

    print_title("Key Results")
    outs([
        "max_mass_total",
        "wing_span",
        "wing_root_chord"
    ])


    def qp(var_name):
        # QuickPlot a variable.
        fig = px.scatter(y=sol.value(eval(var_name)), title=var_name, labels={'y': var_name})
        fig.data[0].update(mode='markers+lines')
        fig.show()


    def qp2(x_name, y_name):
        # QuickPlot two variables.
        fig = px.scatter(
            x=sol.value(eval(x_name)),
            y=sol.value(eval(y_name)),
            title="%s vs. %s" % (x_name, y_name),
            labels={'x': x_name, 'y': y_name}
        )
        fig.data[0].update(mode='markers+lines')
        fig.show()


    def qp3(x_name, y_name, z_name):
        # QuickPlot two variables.
        fig = px.scatter_3d(
            x=sol.value(eval(x_name)),
            y=sol.value(eval(y_name)),
            z=sol.value(eval(z_name)),
            title="%s vs. %s" % (x_name, y_name),
            labels={'x': x_name, 'y': y_name},
            size_max=18
        )
        fig.data[0].update(mode='markers+lines')
        fig.show()


    s = lambda x: sol.value(x)

    draw = lambda: airplane.substitute_solution(sol).draw()

    # endregion

    # Draw plots
    plot_dpi = 200


    def plot(x, y):
        # plt.plot(s(hour), s(y), ".-")
        plt.plot(s(x)[:dusk], s(y)[:dusk], '.-', color=(103 / 255, 155 / 255, 240 / 255), label="Day")
        plt.plot(s(x)[dawn:], s(y)[dawn:], '.-', color=(103 / 255, 155 / 255, 240 / 255))
        plt.plot(s(x)[dusk - 1:dawn + 1], s(y)[dusk - 1:dawn + 1], '.-', color=(7 / 255, 36 / 255, 84 / 255),
                 label="Night")
        plt.legend()


    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, y / 1000)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("Altitude [km]")
    plt.title("Altitude over a Day (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/altitude.png")
    plt.show() if show_plots else plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, airspeed)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("True Airspeed [m/s]")
    plt.title("True Airspeed over a Day (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/airspeed.png")
    plt.show() if show_plots else plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, net_power)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("Net Power [W] (positive is charging)")
    plt.title("Net Power over a Day (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/net_power.png")
    plt.show() if show_plots else plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, 100 * (battery_stored_energy_nondim + (1 - allowable_battery_depth_of_discharge)))
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("State of Charge [%]")
    plt.title("Battery Charge State over a Day")
    plt.tight_layout()
    plt.savefig("outputs/battery_charge.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(x / 1000, y / 1000)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Downrange Distance [km]")
    plt.ylabel("Altitude [km]")
    plt.title("Optimal Trajectory (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/trajectory.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, power_in)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("Power Generated [W]")
    plt.title("Power Generated over a Day (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/power_in.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=plot_dpi)
    plot(hour, power_out)
    ax.ticklabel_format(useOffset=False)
    plt.xlabel("Hours after Solar Noon")
    plt.ylabel("Power Consumed [W]")
    plt.title("Power Consumed over a Day (Aug. 31)")
    plt.tight_layout()
    plt.savefig("outputs/power_out.png")
    plt.close(fig)

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
    pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (x * s(max_mass_total) / 100, x)
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
    pie_format = lambda x: "%.1f kg\n(%.1f%%)" % (x * s(mass_structural) / 100, x * s(mass_structural / max_mass_total))
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
        x * s(mass_power_systems) / 100, x * s(mass_power_systems / max_mass_total))
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
        """ % (s(max_mass_total), s(wing.span())),
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
    plt.show() if show_plots else plt.close(fig)

    # Write a mass budget
    with open("outputs/mass_budget.csv", "w+") as f:
        from types import ModuleType

        var_names = dir()
        f.write("Object or Collection of Objects, Mass [kg],\n")
        for var_name in var_names:
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
            try:
                f.write("%s, %f,\n" % (var_name, value))
            except:
                f.write("%s, %s,\n" % (var_name, value))
