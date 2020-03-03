# Grab AeroSandbox
import aerosandbox as asb
import casadi as cas
import numpy as np
import plotly.express as px

import atmosphere as atmo
import aerodynamics as aero

# region Setup

##### Operating Parameters
latitude = 49  # degrees (49 deg is top of CONUS, 26 deg is bottom of CONUS)
wind_speed = 29  # m/s (17 m/s is 95% wind speed @ 60000 ft)
day_of_year = 244  # Julian day. June 1 is 153, June 22 is 174, Aug. 31 is 244
min_altitude = 19812 # meters. 19812 m = 65000 ft.
required_headway_per_day = 1e5 # meters
allowable_battery_depth_of_discharge = 0.8 # How much of the battery can you actually use?

##### Simulation Parameters
n_timesteps = 100 # Quick convergence testing indicates you get bad analyses below 200 or so...

##### Optimization bounds
# min_speed = 5  # Specify a minimum speed - keeps the speed-gamma velocity parameterization from NaNing
min_mass = 1  # Specify a minimum mass - keeps the optimization from NaNing.
# max_mach = 1  # Specify a maximum mach number - physics models break down above this.

##### Initialize Optimization
opti = cas.Opti()
# endregion

# region Trajectory Optimization Variables
##### Initialize trajectory optimization variables

y = 1e4 * opti.variable()
opti.set_initial(y,
                 min_altitude
                 )
opti.subject_to([
    y > min_altitude,
    y < 40000,  # models break down
])

airspeed = wind_speed

flight_path_angle = 0

battery_stored_energy_nondim = 1 * opti.variable(n_timesteps)
opti.set_initial(battery_stored_energy_nondim,
                 cas.linspace(0.5, 0.5, n_timesteps)
                 )
opti.subject_to([
    battery_stored_energy_nondim > 0,
    battery_stored_energy_nondim < allowable_battery_depth_of_discharge
])

battery_capacity = 3600 * 40000 * opti.variable()  # Joules, not watt-hours!
opti.set_initial(battery_capacity,
                 3600 * 40000
                 )
opti.subject_to([
    battery_capacity > 0
])
battery_capacity_watt_hours = battery_capacity / 3600
battery_stored_energy = battery_stored_energy_nondim * battery_capacity

alpha = 2 * opti.variable(n_timesteps)
opti.set_initial(alpha,
                 cas.linspace(3, 3, n_timesteps)
                 )
opti.subject_to([
    alpha > -12,
    alpha < 12
])

thrust_force = 1e2 * opti.variable(n_timesteps)
opti.set_initial(thrust_force,
                 1e2
                 )
opti.subject_to([
    thrust_force > 0
])

net_power = 1000 * opti.variable(n_timesteps)
opti.set_initial(net_power,
                 0
                 )

net_accel_parallel = 0

net_accel_perpendicular = 0

##### Set up time
time_nondim = cas.linspace(0, 1, n_timesteps)
seconds_per_day = 86400
time = time_nondim * seconds_per_day

# endregion

# region Design Optimization Variables
##### Initialize design optimization variables (all units in base SI or derived units)
mass_total = 6e2 * opti.variable()
opti.set_initial(mass_total, 3e2)
opti.subject_to([
    mass_total > min_mass
])

# Make the airfoils
e216 = asb.Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (
        aero.Cl_e216(alpha = alpha, Re_c = Re)
    ),
    CDp_function = lambda alpha, Re, mach, deflection,: (
        aero.Cd_profile_e216(alpha = alpha, Re_c = Re)
    )
)
flat_plate = asb.Airfoil(
    CL_function=lambda alpha, Re, mach, deflection,: (
        aero.Cl_flat_plate(alpha = alpha, Re_c = Re)
    ),
    CDp_function = lambda alpha, Re, mach, deflection,: (
        aero.Cf_flat_plate(Re_L = Re) * 2
    )
)

# Initialize any variables
wing_span = 80 #80 * opti.variable()
# opti.set_initial(wing_span, 80)
# opti.subject_to([
#     wing_span > 10,
#     wing_span < 100,
# ])

wing = asb.Wing(
    name="Main Wing",
    x_le=0,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=0,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=-5 / 4,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=5,
            twist=0,  # degrees
            airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=-2.5 / 4,
            y_le=40,# wing_span / 2,
            z_le=4,
            chord=2.5,
            twist=0,
            airfoil=e216,
        ),
    ]
)
hstab = asb.Wing(
    name="Horizontal Stabilitzer",
    x_le=24,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=-1,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=2,
            twist=-3,  # degrees
            airfoil=flat_plate,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0.08,
            y_le=7,
            z_le=0,
            chord=2,
            twist=-3,
            airfoil=flat_plate,
        ),
    ]
)
vstab = asb.Wing(
    name="Vertical Stabilizer",
    x_le=26,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=-4,  # Coordinates of the wing's leading edge
    symmetric=False,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=2,
            twist=0,  # degrees
            airfoil=flat_plate,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0.08,
            y_le=0,
            z_le=8,
            chord=2,
            twist=0,
            airfoil=flat_plate,
        ),
    ]
)
fuse = asb.Fuselage(
    name="Fuselage",
    x_le=0,
    y_le=0,
    z_le=-1,
    xsecs=[
        asb.FuselageXSec(x_c=-5, radius=0),
        asb.FuselageXSec(x_c=-4.5, radius=0.4),
        asb.FuselageXSec(x_c=-4, radius=0.6),
        asb.FuselageXSec(x_c=-3, radius=0.9),
        asb.FuselageXSec(x_c=-2, radius=1),
        asb.FuselageXSec(x_c=-1, radius=1),
        asb.FuselageXSec(x_c=0, radius=1),
        asb.FuselageXSec(x_c=1, radius=1),
        asb.FuselageXSec(x_c=2, radius=1),
        asb.FuselageXSec(x_c=3, radius=1),
        asb.FuselageXSec(x_c=4, radius=1),
        asb.FuselageXSec(x_c=5, radius=0.95),
        asb.FuselageXSec(x_c=6, radius=0.9),
        asb.FuselageXSec(x_c=7, radius=0.8),
        asb.FuselageXSec(x_c=8, radius=0.7),
        asb.FuselageXSec(x_c=9, radius=0.6),
        asb.FuselageXSec(x_c=10, radius=0.5),
        asb.FuselageXSec(x_c=11, radius=0.4),
        asb.FuselageXSec(x_c=12, radius=0.3),
        asb.FuselageXSec(x_c=13, radius=0.25),
        asb.FuselageXSec(x_c=22, radius=0.25),
        asb.FuselageXSec(x_c=24, radius=0.25),
        asb.FuselageXSec(x_c=25, radius=0.25),
        asb.FuselageXSec(x_c=26, radius=0.2),
        asb.FuselageXSec(x_c=27, radius=0.1),
        asb.FuselageXSec(x_c=28, radius=0),
    ]
)

airplane = asb.Airplane(
    name="Solar1",
    x_ref=0,
    y_ref=0,
    z_ref=0,
    wings=[
        wing,
        hstab,
        vstab,
    ],
    fuselages=[
        fuse
    ],
)
# airplane.draw()

# endregion

# region Useful Definitions
##### Useful Definitions
### Constants and shorthand functions
sind = lambda theta: cas.sin(theta * np.pi / 180)
cosd = lambda theta: cas.cos(theta * np.pi / 180)
tand = lambda theta: cas.tan(theta * np.pi / 180)
atan2d = lambda y_val, x_val: cas.atan2(y_val, x_val) * 180 / np.pi
fmax = lambda value1, value2: cas.fmax(value1, value2)
smoothmax = lambda value1, value2, hardness: cas.log(
    cas.exp(hardness * value1) + cas.exp(hardness * value2)) / hardness  # soft maximum
y_eff = y#cas.fmin(cas.fmax(0, y), 40000)  # Fixes any bugs due to being beneath the ground in intermediate steps
P = atmo.get_pressure_at_altitude(y_eff)
rho = atmo.get_density_at_altitude(y_eff)
T = atmo.get_temperature_at_altitude(y_eff)
mu = atmo.get_viscosity_from_temperature(T)
a = atmo.get_speed_of_sound_from_temperature(T)
mach = airspeed / a
g = 9.81  # gravitational acceleration, m/s^2
q = 1 / 2 * rho * airspeed ** 2
# endregion

# region Aerodynamics
##### Aerodynamics
# aerodynamics_type = "buildup" # "buildup", "aerosandbox-point", "aerosandbox-full"
# aerodynamics_type = "aerosandbox-point" # "buildup", "aerosandbox-point", "aerosandbox-full"
aerodynamics_type = "buildup" # "buildup", "aerosandbox-point", "aerosandbox-full"
if aerodynamics_type == "buildup":

    # Fuselage
    fuse_Re = rho / mu * airspeed * fuse.length()
    CLA_fuse = 0
    CDA_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted()

    lift_fuse = CLA_fuse * q
    drag_fuse = CDA_fuse * q

    # wing
    wing_Re = rho / mu * airspeed * wing.mean_geometric_chord()
    wing_airfoil = wing.xsecs[0].airfoil # type: asb.Airfoil
    wing_Cl_inc = wing_airfoil.CL_function(alpha + wing.mean_twist_angle(), wing_Re, 0, 0)  # Incompressible 2D lift_force coefficient
    wing_CL = wing_Cl_inc * aero.CL_over_Cl(wing.aspect_ratio(), mach=mach,
                                            sweep=wing.mean_sweep_angle())  # Compressible 3D lift_force coefficient
    lift_wing = wing_CL * q * wing.area()

    wing_Cd_profile = wing_airfoil.CDp_function(alpha + wing.mean_twist_angle(), wing_Re, 0, 0)
    drag_wing_profile = wing_Cd_profile * q * wing.area()

    wing_oswalds_efficiency = 0.95  # TODO make this a function of taper ratio
    drag_wing_induced = lift_wing ** 2 / (q * np.pi * wing.span() ** 2 * wing_oswalds_efficiency)

    drag_wing = drag_wing_profile + drag_wing_induced

    # hstab
    hstab_Re = rho / mu * airspeed * hstab.mean_geometric_chord()
    hstab_airfoil = hstab.xsecs[0].airfoil # type: asb.Airfoil
    hstab_Cl_inc = hstab_airfoil.CL_function(alpha + hstab.mean_twist_angle(), hstab_Re, 0, 0)  # Incompressible 2D lift_force coefficient
    hstab_CL = hstab_Cl_inc * aero.CL_over_Cl(hstab.aspect_ratio(), mach=mach,
                                            sweep=hstab.mean_sweep_angle())  # Compressible 3D lift_force coefficient
    lift_hstab = hstab_CL * q * hstab.area()

    hstab_Cd_profile = hstab_airfoil.CDp_function(alpha + hstab.mean_twist_angle(), hstab_Re, 0, 0)
    drag_hstab_profile = hstab_Cd_profile * q * hstab.area()

    hstab_oswalds_efficiency = 0.95  # TODO make this a function of taper ratio
    drag_hstab_induced = lift_hstab ** 2 / (q * np.pi * hstab.span() ** 2 * hstab_oswalds_efficiency)

    drag_hstab = drag_hstab_profile + drag_hstab_induced

    # vstab
    vstab_Re = rho / mu * airspeed * vstab.mean_geometric_chord()
    vstab_airfoil = vstab.xsecs[0].airfoil # type: asb.Airfoil
    vstab_Cd_profile = vstab_airfoil.CDp_function(0, vstab_Re, 0, 0)
    drag_vstab_profile = vstab_Cd_profile * q * vstab.area()

    drag_vstab = drag_vstab_profile

    # Totals
    lift_force = lift_fuse + lift_wing + lift_hstab
    drag_force = drag_fuse + drag_wing + drag_hstab + drag_vstab

elif aerodynamics_type == "aerosandbox-point":

    airplane.fuselages = []

    airplane.set_spanwise_paneling_everywhere(8)  # Set the resolution of the analysis
    ap = asb.Casll1(
        airplane = airplane,
        op_point = asb.OperatingPoint(
            density = rho[0],
            viscosity = mu[0],
            velocity = airspeed[0],
            mach = 0,
            alpha = alpha[0],
            beta = 0,
            p = 0,
            q = 0,
            r = 0,
        ),
        opti = opti
    )

    lift_force = -ap.force_total_wind[2]
    drag_force = -ap.force_total_wind[0]

    # Tack on fuselage drag:
    fuse_Re = rho / mu * airspeed * fuse.length()
    drag_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted() * q
    drag_force += drag_fuse

elif aerodynamics_type == "aerosandbox-full":
    lift_force = []
    drag_force = []

    airplane.wings = [wing] # just look at the one wing
    airplane.fuselages = [] # ignore the fuselage

    airplane.set_spanwise_paneling_everywhere(8)  # Set the resolution of the analysis

    aps = [
        asb.Casll1(
            airplane=airplane,
            op_point=asb.OperatingPoint(
                density=rho[i],
                viscosity=mu[i],
                velocity=airspeed[i],
                mach=0,
                alpha=alpha[i],
                beta=0,
                p=0,
                q=0,
                r=0,
            ),
            opti=opti
        )
        for i in range(n_timesteps)
    ]

    lift_force = cas.vertcat(*[-ap.force_total_wind[2] for ap in aps])
    drag_force = cas.vertcat(*[-ap.force_total_wind[0] for ap in aps])

    # Multiply drag force to roughly account for tail
    drag_force *= 1.1

    # Tack on fuselage drag:
    fuse_Re = rho / mu * airspeed * fuse.length()
    drag_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted() * q
    drag_force += drag_fuse

else:
    raise ValueError("Bad value of 'aerodynamics_type'!")

# endregion

# region Power Systems
### Incident solar calculations
# Space effects
# # Source: https://www.itacanet.org/the-sun-as-a-source-of-energy/part-2-solar-energy-reaching-the-earths-surface/#2.1.-The-Solar-Constant
# solar_flux_outside_atmosphere_normal = 1367 * (1 + 0.034 * cas.cos(2 * cas.pi * (day_of_year / 365.25)))  # W/m^2; variation due to orbital eccentricity
# Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-radiation-outside-the-earths-atmosphere
solar_flux_outside_atmosphere_normal = 1366 * (
        1 + 0.033 * cosd(360 * (day_of_year - 2) / 365))  # W/m^2; variation due to orbital eccentricity

# Declination (seasonality)
# Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
declination_angle = -23.45 * cosd(360 / 365 * (day_of_year + 10))  # in degrees

# Solar elevation angle (including seasonality, latitude, and time of day)
# Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
solar_elevation_angle = cas.asin(
    sind(declination_angle) * sind(latitude) +
    cosd(declination_angle) * cosd(latitude) * cosd(time / 86400 * 360)
) * 180 / np.pi  # in degrees
solar_elevation_angle = cas.fmax(solar_elevation_angle, 0)

# Solar cell power flux
incidence_angle_losses_function = lambda theta: cosd(theta)
# To first-order, this is true. In class, Kevin Uleck claimed that you have higher-than-cosine losses at extreme angles,
# since you get reflection losses. However, an experiment by Sharma appears to not reproduce this finding, showing only a
# 0.4-percentage-point drop in cell efficiency from 0 to 60 degrees. So, for now, we'll just say it's a cosine loss.
# Sharma: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6611928/
solar_flux_on_horizontal = (
        solar_flux_outside_atmosphere_normal *
        incidence_angle_losses_function(90 - solar_elevation_angle)
)

# Cell efficiency
# Source: Most pessimistic number in Kevin Uleck's presentation
realizable_solar_cell_efficiency = 0.205  # This figure should take into account all temperature factors, MPPT losses,
# spectral losses (different spectrum at altitude), multi-junction effects, etc.

# Total cell power flux
solar_power_flux = (
        solar_flux_on_horizontal *
        realizable_solar_cell_efficiency
)

solar_area_fraction = opti.variable()
opti.set_initial(solar_area_fraction, 1)
opti.subject_to([
    solar_area_fraction > 0,
    solar_area_fraction < 1,
])

area_solar = wing.area() * solar_area_fraction
power_in = solar_power_flux * area_solar

# Solar cell weight
rho_solar_cells = 0.30  # kg/m^2, solar cell area density. The solar_simple_demo model gives this as 0.27. Burton's model gives this as 0.30.
mass_solar_cells = rho_solar_cells * area_solar

### Battery calculations
battery_specific_energy_Wh_kg = 265  # Wh/kg.
# Burton's solar model uses 350, and public specs from Amprius seem to indicate that's possible.
# Odysseus had cells that were 265 Wh/kg.

battery_specific_energy = battery_specific_energy_Wh_kg * 3600  # J/kg
mass_battery_cells = battery_capacity / battery_specific_energy

battery_module_cell_percentage = 0.75 # What percent of the battery module consists of cells, by weight?
# Accounts for thermal management, case, and electronics.
# Ed Lovelace (in his presentation) gives 75% as a typical fraction.
mass_battery_module = mass_battery_cells / battery_module_cell_percentage

battery_pack_module_percentage = 0.80 # What percent of the battery pack consists of the module, by weight?
# Accounts for module HW, BMS, pack installation, etc.
# Ed Lovelace (in his presentation) gives 80% as a state-of-the-art fraction.
mass_battery_pack = mass_battery_module / battery_pack_module_percentage

mass_wires = 0.015 * (wing.span() / 3) * ((battery_capacity / 86400 * 2) / 3000)  # a guess from 10 AWG aluminum wire

# Total system mass
mass_power_systems = mass_solar_cells + mass_battery_pack + mass_wires

# endregion

# region Propulsion
### Propeller calculations
# propeller_diameter = 3.0
propeller_diameter = opti.variable()
opti.set_initial(propeller_diameter, 3)
opti.subject_to([
    propeller_diameter > 1
])

# n_propellers = 1
n_propellers = opti.variable()
opti.set_initial(n_propellers, 1)
opti.subject_to([
    n_propellers > 1
])

area_propulsive = cas.pi / 4 * propeller_diameter ** 2 * n_propellers
coefficient_of_performance = 0.7  # a total WAG

power_out_propulsion = 0.5 * thrust_force * airspeed * (
        cas.sqrt(
            thrust_force / (area_propulsive * airspeed ** 2 * rho / 2) + 1
        ) + 1
) / coefficient_of_performance

power_out_max = 5e3 * opti.variable()
opti.set_initial(power_out_max, 5e3)
opti.subject_to([
    power_out_propulsion < power_out_max,
    power_out_max > 0
])

mass_motor_raw = power_out_max / 4140.8  # W/kg, taken from Mike Burton's thesis; roughly matches my motor library curve fits and intuition
mass_motor_mounted = 2 * mass_motor_raw  # similar to a quote from Raymer, modified to make sensible units, prop weight roughly subtracted

propeller_n_blades = 2
mass_propellers = n_propellers * 0.495 * (propeller_diameter / 1.25) ** 2 * cas.fmax(1,
                                                                                     power_out_max / 14914) ** 2  # Baselining to a 125cm E-Props Top 80 Propeller for paramotor, with scaling assumptions
# Total propulsion mass
mass_propulsion = mass_motor_mounted + mass_propellers

# Account for payload power
power_out_payload = cas.if_else(
    cas.logic_and(
        time > 4 * 3600, # if it's after dark...
        time < 86400 - 4 * 3600 # or after sunrise...
    ),
    500,
    150
)

### Power accounting
power_out = power_out_propulsion + power_out_payload

# endregion

# region Weights
# Payload mass
mass_payload = 30

# # Structural mass
# def surface_mass(chord, span, type, mean_t_over_c=0.08):
#     chord = cas.fmax(chord, 0)
#     span = cas.fmax(span, 0)
#     mean_t = chord * mean_t_over_c
#     if type == 'balsa-monokote-cf':
#         ### Balsa wood + Monokote + a 1" dia CF tube spar.
#         monokote_mass = 0.061 * chord * span * 2  # 0.2 oz/sqft
#
#         rib_density = 200  # mass density, in kg/m^3
#         rib_spacing = 0.1  # one rib every x meters
#         rib_width = 0.003  # width of an individual rib
#         ribs_mass = (
#                 (mean_t * chord * rib_width) *  # volume of a rib
#                 rib_density *  # density of a rib
#                 (span / rib_spacing)  # number of ribs
#         )
#
#         spar_mass_1_inch = 0.2113 * span * 1.5  # assuming 1.5x 1" CF tube spar
#         spar_mass = spar_mass_1_inch * (
#                 mean_t / 0.0254) ** 2  # Rough GUESS for scaling, FIX THIS before using seriously!
#
#         return (monokote_mass + ribs_mass + spar_mass) * 1.2  # for glue
#     elif type == 'solid-aluminum':
#         ### Solid milled aluminum
#         density = 2700  # mass density, in kg/m^3
#         volume = chord * span * mean_t
#         return density * volume
#     else:
#         raise Exception("Invalid argument for type!")
#
#
# mass_wing = surface_mass(wing.mean_geometric_chord(), wing.span(), type='solid-aluminum',
#                          mean_t_over_c=0.1 * 0.685  # Using the mean/max thickness ratio from a NACA 0010.
#                          )
# mass_hstab = surface_mass(hstab.mean_geometric_chord(), hstab.span(), type='solid-aluminum',
#                           mean_t_over_c=hstab_t_over_c * 0.685  # Using the mean/max thickness ratio from a NACA 0010.
#                           )
# mass_vstab = surface_mass((vstab_root_chord + vstab_tip_chord) / 2, vstab_span, type='solid-aluminum',
#                           mean_t_over_c=vstab_t_over_c * 0.685  # Using the mean/max thickness ratio from a NACA 0010.
#                           )
#
# mass_fuselage_forward = 0.263
# mass_fuselage_aft = 0.352
# mass_fuselage = mass_fuselage_forward + mass_fuselage_aft
#
# mass_aeroshell = 0.010
#
# mass_avionics = 0.075
#
# mass_mechanisms = 0.162
#
# mass_structural = (mass_wing + mass_hstab + mass_vstab  # TODO add vstab mass
#                    + mass_fuselage + mass_aeroshell + mass_avionics + mass_mechanisms)

mass_structural = mass_total * 0.3  # TODO fix this!

opti.subject_to([
    mass_total == mass_payload + mass_structural + mass_propulsion + mass_power_systems,
])

gravity_force = g * mass_total

# endregion

# region Dynamics

opti.subject_to([
    thrust_force == drag_force,
    lift_force == gravity_force
])

opti.subject_to([
    net_power / 5e3 < (power_in - power_out) / 5e3,
])

dt = cas.diff(time)
dbattery_stored_energy_nondim = cas.diff(battery_stored_energy_nondim)

trapz = lambda x: (x[1:] + x[:-1]) / 2

net_power_trapz = trapz(net_power)

# Total
opti.subject_to([
    dbattery_stored_energy_nondim / 1e-3 == (net_power_trapz / battery_capacity) * dt / 1e-3,
])

# endregion

# region Finalize Optimization Problem
##### Add periodic constraints
opti.subject_to([
    battery_stored_energy_nondim[-1] > battery_stored_energy_nondim[0],
])

##### Add initial state constraints

##### Optional constraints
# Match GPKit MTOW
# opti.subject_to([
#     mass_total == 797.4
# ])

# constraints_jacobian = cas.jacobian(opti.g, opti.x)

##### Add objective
objective = mass_total / 1e2

##### Add tippers
things_to_slightly_minimize = 0*(
    wing_span / 80
)

# Dewiggle
penalty = 0
penalty_denominator = n_timesteps
penalty += cas.sum1(cas.diff(net_power / 10000) ** 2) / penalty_denominator

opti.minimize(objective + penalty + 1e-3 * things_to_slightly_minimize)
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
# s_opts["watchdog_shortened_iter_trigger"]=1
# s_opts["expect_infeasible_problem"]="yes"
# s_opts["start_with_resto"] = "yes"
# s_opts["required_infeasibility_reduction"] = 0.1
# s_opts["evaluate_orig_obj_at_resto_trial"] = "yes"
opti.solver('ipopt', p_opts, s_opts)

if __name__ == "__main__":
    try:
        sol = opti.solve()
    except:
        sol = opti.debug



    if aerodynamics_type=="aerosandbox-point":
        import copy
        ap_sol = copy.deepcopy(ap)
        ap_sol.substitute_solution(sol)
    if aerodynamics_type=="aerosandbox-full":
        import copy
        ap_sols = [copy.deepcopy(ap) for ap in aps]
        [ap_sol.substitute_solution(sol) for ap_sol in ap_sols]

    #
    # sol_stats = sol.stats()  # get stats about solve
    #
    # # region Dash Postprocessing
    # ##### Compute processed variables
    # xdot = airspeed * cosd(flight_path_angle)
    # ydot = airspeed * sind(flight_path_angle)
    #
    # import dash_bootstrap_components as dbc
    #
    # app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
    #
    # fig_statevars = sub.make_subplots(rows=3, cols=2)
    # fig_statevars.update_layout(title="Firefly: State Variables", height=1000)
    #
    # fig_statevars.add_trace(go.Scatter(x=sol.value(time), y=sol.value(x), mode='lines+markers', name="X-pos"),
    #                         row=1, col=1)
    # fig_statevars.add_trace(go.Scatter(x=sol.value(time), y=sol.value(y), mode='lines+markers', name="Y-pos"),
    #                         row=1, col=1)
    # fig_statevars.update_yaxes(title_text="Position [m]", row=1, col=1)
    #
    # fig_statevars.add_trace(go.Scatter(x=sol.value(time), y=sol.value(xdot), mode='lines+markers', name="X-vel"),
    #                         row=1, col=2)
    # fig_statevars.add_trace(go.Scatter(x=sol.value(time), y=sol.value(ydot), mode='lines+markers', name="Y-vel"),
    #                         row=1, col=2)
    # fig_statevars.update_yaxes(title_text="Velocity [m/s]", row=1, col=2)
    #
    # fig_statevars.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(mass_propellant), mode='lines+markers', name="Fuel Mass"),
    #     row=2, col=1)
    # fig_statevars.update_yaxes(title_text="Fuel Mass [kg]", row=2, col=1)
    #
    # fig_statevars.add_trace(go.Scatter(x=sol.value(time), y=sol.value(alpha), mode='lines+markers', name="AoA"),
    #                         row=2, col=2)
    # fig_statevars.update_yaxes(title_text="AoA [deg]", row=2, col=2)
    #
    # fig_statevars.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(mass_flow_rate), mode='lines+markers', name="Mass Flow Rate"),
    #     row=3, col=1)
    # fig_statevars.update_yaxes(title_text="Mass Flow Rate [kg/s]", row=3, col=1)
    # fig_statevars.update_xaxes(title_text="Time [s]")
    #
    # fig_trajectory = go.Figure()
    # fig_trajectory.update_layout(title="Firefly: Trajectory<br>Range: %.3f km (%.3f mi)" % (
    #     sol.value(range / 1000), sol.value(range / 1000) * 0.62137119223), height=900)
    # fig_trajectory.add_trace(
    #     go.Scatter(x=sol.value(x[:n_timesteps_boost]), y=sol.value(y[:n_timesteps_boost]), mode="lines+markers",
    #                name="Boost", marker_color='rgb(255, 138, 92)'
    #                ))
    # fig_trajectory.add_trace(
    #     go.Scatter(x=sol.value(x[n_timesteps_boost:]), y=sol.value(y[n_timesteps_boost:]), mode="lines+markers",
    #                name="Glide", marker_color='rgb(82, 200, 255)'))
    # fig_trajectory.update_layout(
    #     # title="Firefly: Trajectory",
    #     xaxis_title="Downrange Distance [m]",
    #     yaxis_title="Altitude [m]",
    #     yaxis=dict(scaleanchor="x", scaleratio=1)
    # )
    #
    # fig_thrust_profile = go.Figure()
    # fig_thrust_profile.update_layout(title="Thrust Profile", height=600)
    # fig_thrust_profile.add_trace(go.Scatter(
    #     x=sol.value(time[:n_timesteps_boost]),
    #     y=sol.value(thrust_force[:n_timesteps_boost]),
    #     mode="lines+markers"
    # ))
    # fig_thrust_profile.update_layout(
    #     xaxis_title="Time [s]",
    #     yaxis_title="Thrust [N]",
    # )
    #
    # fig_speed = sub.make_subplots(rows=2, cols=2)
    # fig_speed.update_layout(title="Speed over Time", height=600)
    # fig_speed.add_trace(go.Scatter(x=sol.value(time), y=sol.value(airspeed), mode="lines+markers", name="Speed"), row=1,
    #                     col=1)
    # fig_speed.update_yaxes(title_text="Speed [m/s]", row=1, col=1)
    # fig_speed.add_trace(go.Scatter(x=sol.value(time), y=sol.value(mach), mode="lines+markers", name="Mach"), row=1,
    #                     col=2)
    # fig_speed.update_yaxes(title_text="Mach Number", row=1, col=2)
    # fig_speed.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(flight_path_angle), mode="lines+markers", name="Flight Path Angle"),
    #     row=2, col=1)
    # fig_speed.update_yaxes(title_text="Flight Path Angle [deg]", row=2, col=1)
    # # fig_speed.add_trace(go.Scatter(x=sol.value(time), y=sol.value(flight_path_angle), mode="lines+markers"), row=2, col=2)
    # # fig_speed.update_yaxes(title_text="Flight Path Angle [deg]", row=2, col=2)
    # fig_speed.update_xaxes(title_text="Time [s]")
    #
    # fig_aero_lift_drag = sub.make_subplots(rows=2, cols=2)
    # fig_aero_lift_drag.update_layout(title="Lift and Drag", height=600)
    # fig_aero_lift_drag.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(lift_force), mode="lines+markers", name="Lift"), row=1,
    #     col=1)
    # fig_aero_lift_drag.update_yaxes(title_text="Lift Force [N]", row=1, col=1)
    # fig_aero_lift_drag.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(drag_force), mode="lines+markers", name="Drag"), row=1,
    #     col=2)
    # fig_aero_lift_drag.update_yaxes(title_text="Drag Force [N]", row=1, col=2)
    # fig_aero_lift_drag.add_trace(
    #     go.Scatter(x=sol.value(time), y=sol.value(lift_force / drag_force), mode="lines+markers", name="L/D"), row=2,
    #     col=1)
    # fig_aero_lift_drag.update_yaxes(title_text="Lift-to-Drag Ratio", row=2, col=1)
    # # fig_aero_lift_drag.add_trace(go.Scatter(x=sol.value(time), y=sol.value(flight_path_angle), mode="lines+markers"), row=2, col=2)
    # # fig_aero_lift_drag.update_yaxes(title_text="Flight Path Angle [deg]", row=2, col=2)
    # fig_aero_lift_drag.update_xaxes(title_text="Time [s]")
    #
    # fig_opt = sub.make_subplots(rows=2, cols=2)
    # fig_opt.update_layout(title="Firefly: Optimization Progress", height=600)
    # iters = np.arange(sol_stats['iter_count']) + 1
    # fig_opt.add_trace(go.Scatter(x=iters, y=sol_stats['iterations']['obj'], mode="lines+markers", name="Objective"),
    #                   row=1,
    #                   col=1)
    # fig_opt.update_yaxes(title_text="Objective Function", row=1, col=1)
    # fig_opt.add_trace(
    #     go.Scatter(x=iters, y=np.log10(sol_stats['iterations']['inf_pr']), mode="lines+markers",
    #                name="Primal Feasibility"),
    #     row=1, col=2)
    # fig_opt.add_trace(
    #     go.Scatter(x=iters, y=np.log10(sol_stats['iterations']['inf_du']), mode="lines+markers",
    #                name="Dual Feasibility"),
    #     row=1, col=2)
    # fig_opt.update_yaxes(title_text="log(feasibility)", row=1, col=2)
    # fig_opt.add_trace(
    #     go.Scatter(x=iters, y=np.log10(sol_stats['iterations']['mu']), mode="lines+markers", name="Objective"), row=2,
    #     col=1)
    # fig_opt.update_yaxes(title_text="log(mu)", row=2, col=1)
    # fig_opt.add_trace(go.Scatter(x=iters, y=sol_stats['iterations']['alpha_pr'], mode="lines+markers",
    #                              name="Primal Step Size Multiplier"), row=2, col=2)
    # fig_opt.add_trace(
    #     go.Scatter(x=iters, y=sol_stats['iterations']['alpha_du'], mode="lines+markers",
    #                name="Dual Step Size Multiplier"),
    #     row=2, col=2)
    # fig_opt.update_yaxes(title_text="Step Size Multiplier (alpha)", row=2, col=2)
    #
    # fig_opt.update_xaxes(title_text="Iteration #")
    #
    # import visualization as vis
    #
    # fig_constraints_jacobian = vis.spy(sol.value(constraints_jacobian))
    # fig_constraints_jacobian.update_xaxes(title_text="Variable index")
    # fig_constraints_jacobian.update_yaxes(title_text="Constraint index")
    # fig_constraints_jacobian.update_layout(title="Constraints Jacobian Sparsity Pattern")
    #
    # app.layout = html.Div([
    #     html.Div(children=[
    #         html.Div(style={
    #             'overflow': 'hidden',
    #             'padding': '10px 13px',
    #             'backgroundColor': '#f1f1f1',
    #         }, children=[
    #             html.Div(style={
    #                 'float': 'left'
    #             }, children=[
    #                 html.H1("Firefly Design and Trajectory Optimization"),
    #                 html.H5("Peter Sharpe"),
    #             ]),
    #             html.Div(style={
    #                 'float': 'right',
    #             }, children=[
    #                 html.Img(src="assets/MIT-logo-red-gray-72x38.svg", alt="MIT Logo", height="30px"),
    #             ])
    #
    #         ])
    #
    #     ]),
    #     dcc.Tabs([
    #         dcc.Tab(label="State Variables", children=[
    #             dcc.Graph(
    #                 figure=fig_statevars
    #             )
    #         ]),
    #         dcc.Tab(label="Trajectory", children=[
    #             dcc.Graph(
    #                 figure=fig_trajectory
    #             ),
    #         ]),
    #         dcc.Tab(label="Speed over Time", children=[
    #             dcc.Graph(
    #                 figure=fig_speed
    #             ),
    #         ]),
    #         dcc.Tab(label="Aerodynamics", children=[
    #             dcc.Graph(
    #                 figure=fig_aero_lift_drag
    #             ),
    #         ]),
    #         dcc.Tab(label="Propulsion", children=[
    #             dcc.Graph(
    #                 figure=fig_thrust_profile
    #             )
    #         ]),
    #         dcc.Tab(label="Optimization", children=[
    #             dcc.Graph(
    #                 figure=fig_opt
    #             ),
    #             dcc.Graph(
    #                 figure=fig_constraints_jacobian
    #             )
    #         ])
    #     ])
    # ])
    # try:  # wrapping this, since a forum post said it may be deprecated at some point.
    #     app.title = "Firefly Design Opt."
    # except:
    #     print("Could not set the page title!")
    # app.run_server(debug=False)
    # # endregion
    #
    # # region Text Postprocessing & Utilities
    # ##### Text output
    out = lambda x: print("%s: %f" % (x, sol.value(eval(x))))  # input a variable name as a string
    outs = lambda xs: [out(x) for x in xs] and None  # input a list of variable names as strings
    print_title = lambda s: print("\n********** %s **********" % s.upper())

    print_title("Objective")
    outs(["mass_total"])

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

    # endregion
