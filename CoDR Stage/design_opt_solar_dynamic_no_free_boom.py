# Grab AeroSandbox
import aerosandbox as asb
import aerosandbox.library.aerodynamics as aero
import aerosandbox.library.atmosphere as atmo
import casadi as cas
import numpy as np
import plotly.express as px
from aerosandbox.casadi_helpers import *
from aerosandbox.library import mass_propulsion as lib_mass_prop
from aerosandbox.library import mass_structural as lib_mass_struct
from aerosandbox.library import propulsion_propeller as lib_prop_prop
from aerosandbox.library import solar
from aerosandbox.library import winds as lib_winds
from aerosandbox.library.airfoils import e216, generic_airfoil

# region Setup

##### Operating Parameters
latitude = 26  # degrees (49 deg is top of CONUS, 26 deg is bottom of CONUS)
day_of_year = 244  # Julian day. June 1 is 153, June 22 is 174, Aug. 31 is 244
min_altitude = 19812  # meters. 19812 m = 65000 ft.
required_headway_per_day = 1e5  # meters
optimistic = False  # Are you optimistic (as opposed to conservative)? Replaces a variety of constants...
allow_altitude_cycling = True
allow_groundtrack_cycling = True

##### Simulation Parameters
n_timesteps = 200  # Quick convergence testing indicates you can get bad analyses below 200 or so...

##### Optimization bounds
min_speed = 1  # Specify a minimum speed - keeps the speed-gamma velocity parameterization from NaNing
min_mass = 30  # Specify a minimum mass - keeps the optimization from NaNing.

##### Initialize Optimization
opti = cas.Opti()
# endregion

# region Trajectory Optimization Variables
##### Initialize trajectory optimization variables

x = 1e5 * opti.variable(n_timesteps)
opti.set_initial(x,
                 0,
                 # cas.linspace(0, required_headway_per_day, n_timesteps)
                 )

y = 2e4 * opti.variable(n_timesteps)
opti.set_initial(y,
                 20000,
                 # 20000 + 2000 * cas.sin(cas.linspace(0, 2 * cas.pi, n_timesteps))
                 )
opti.subject_to([
    y > min_altitude,
    y < 40000,  # models break down
])

airspeed = 2e1 * opti.variable(n_timesteps)
opti.set_initial(airspeed,
                 20
                 )
opti.subject_to([
    airspeed > min_speed
])

flight_path_angle = 1e0 * opti.variable(n_timesteps)
opti.set_initial(flight_path_angle,
                 0
                 )
opti.subject_to([
    flight_path_angle < 90,
    flight_path_angle > -90,
])

battery_stored_energy_nondim = 1 * opti.variable(n_timesteps)
opti.set_initial(battery_stored_energy_nondim,
                 0.5,
                 # 0.5 + 0.5 * cas.sin(cas.linspace(0, 2 * cas.pi, n_timesteps)),
                 )
allowable_battery_depth_of_discharge = 0.9 if optimistic else 0.8  # How much of the battery can you actually use?
opti.subject_to([
    battery_stored_energy_nondim > 0,
    battery_stored_energy_nondim < allowable_battery_depth_of_discharge,
])

battery_capacity = 3600 * 40000 * opti.variable()  # Joules, not watt-hours!
opti.set_initial(battery_capacity,
                 3600 * 10000 if optimistic else 3600 * 15000
                 )
opti.subject_to([
    battery_capacity > 0
])
battery_capacity_watt_hours = battery_capacity / 3600
battery_stored_energy = battery_stored_energy_nondim * battery_capacity

alpha = 4 * opti.variable(n_timesteps)
opti.set_initial(alpha,
                 3
                 )
opti.subject_to([
    alpha > -8,
    alpha < 12
])

thrust_force = 1e2 * opti.variable(n_timesteps)
opti.set_initial(thrust_force,
                 60 if optimistic else 120
                 )
opti.subject_to([
    thrust_force > 0
])

net_power = 1000 * opti.variable(n_timesteps)
opti.set_initial(net_power,
                 0,
                 # 1000 * cas.cos(cas.linspace(0, 2 * cas.pi, n_timesteps))
                 )

net_accel_parallel = 1e-2 * opti.variable(n_timesteps)
opti.set_initial(net_accel_parallel,
                 0
                 )

net_accel_perpendicular = 1e-1 * opti.variable(n_timesteps)
opti.set_initial(net_accel_perpendicular,
                 0
                 )

##### Set up time
time_nondim = cas.linspace(0, 1, n_timesteps)
seconds_per_day = 86400
time = time_nondim * seconds_per_day

# endregion

# region Design Optimization Variables
##### Initialize design optimization variables (all units in base SI or derived units)
mass_total = 3e2 * opti.variable()
opti.set_initial(mass_total,
                 300 if optimistic else 1000
                 )
opti.subject_to([
    mass_total > min_mass
])
mass_total_eff = cas.fmax(mass_total, min_mass)

# Initialize any variables
wing_span = 80 * opti.variable()
opti.set_initial(wing_span,
                 40
                 )
opti.subject_to([
    wing_span > 1,
    # wing_span < 100,  # TODO edit or delete this
])

wing_root_chord = 4 * opti.variable()
opti.set_initial(wing_root_chord,
                 5
                 )
opti.subject_to([
    wing_root_chord > 0.1,
    # wing_root_chord < 10,  # TODO edit or delete this
])

hstab_twist_angle = 2 * opti.variable(n_timesteps)
opti.set_initial(hstab_twist_angle,
                 -2
                 )

wing = asb.Wing(
    name="Main Wing",
    x_le=0.1,  # Coordinates of the wing's leading edge # TODO make this a free parameter?
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
            airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=-wing_root_chord * 0.5 / 4,
            y_le=wing_span / 2,
            z_le=4,
            chord=wing_root_chord * 0.5,
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
            twist=-3,  # degrees # TODO fix
            airfoil=generic_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0,
            y_le=7,
            z_le=0,
            chord=2,
            twist=-3, # TODO fix
            airfoil=generic_airfoil,
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
            airfoil=generic_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0,
            y_le=0,
            z_le=8,
            chord=2,
            twist=0,
            airfoil=generic_airfoil,
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
        asb.FuselageXSec(x_c=17, radius=0.25),
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

# region Atmosphere
##### Atmosphere
P = atmo.get_pressure_at_altitude(y)
rho = atmo.get_density_at_altitude(y)
T = atmo.get_temperature_at_altitude(y)
mu = atmo.get_viscosity_from_temperature(T)
a = atmo.get_speed_of_sound_from_temperature(T)
mach = airspeed / a
g = 9.81  # gravitational acceleration, m/s^2
q = 1 / 2 * rho * airspeed ** 2
# endregion

# region Aerodynamics
##### Aerodynamics
aerodynamics_type = "buildup"  # "buildup", "aerosandbox-point", "aerosandbox-full"

if aerodynamics_type == "buildup":

    # Fuselage
    fuse_Re = rho / mu * airspeed * fuse.length()
    CLA_fuse = 0
    CDA_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted()

    lift_fuse = CLA_fuse * q
    drag_fuse = CDA_fuse * q

    # wing
    wing_Re = rho / mu * airspeed * wing.mean_geometric_chord()
    wing_airfoil = wing.xsecs[0].airfoil  # type: asb.Airfoil
    wing_Cl_inc = wing_airfoil.CL_function(alpha + wing.mean_twist_angle(), wing_Re, 0,
                                           0)  # Incompressible 2D lift_force coefficient
    wing_CL = wing_Cl_inc * aero.CL_over_Cl(wing.aspect_ratio(), mach=mach,
                                            sweep=wing.mean_sweep_angle())  # Compressible 3D lift_force coefficient
    lift_wing = wing_CL * q * wing.area()

    wing_Cd_profile = wing_airfoil.CDp_function(alpha + wing.mean_twist_angle(), wing_Re, mach, 0)
    drag_wing_profile = wing_Cd_profile * q * wing.area()

    wing_oswalds_efficiency = 0.95  # TODO make this a function of taper ratio
    drag_wing_induced = lift_wing ** 2 / (q * np.pi * wing.span() ** 2 * wing_oswalds_efficiency)

    drag_wing = drag_wing_profile + drag_wing_induced

    # hstab
    hstab_Re = rho / mu * airspeed * hstab.mean_geometric_chord()
    hstab_airfoil = hstab.xsecs[0].airfoil  # type: asb.Airfoil
    hstab_Cl_inc = hstab_airfoil.CL_function(alpha + hstab_twist_angle, hstab_Re, 0,
                                             0)  # Incompressible 2D lift_force coefficient
    hstab_CL = hstab_Cl_inc * aero.CL_over_Cl(hstab.aspect_ratio(), mach=mach,
                                              sweep=hstab.mean_sweep_angle())  # Compressible 3D lift_force coefficient
    lift_hstab = hstab_CL * q * hstab.area()

    hstab_Cd_profile = hstab_airfoil.CDp_function(alpha + hstab_twist_angle, hstab_Re, mach, 0)
    drag_hstab_profile = hstab_Cd_profile * q * hstab.area()

    hstab_oswalds_efficiency = 0.95  # TODO make this a function of taper ratio
    drag_hstab_induced = lift_hstab ** 2 / (q * np.pi * hstab.span() ** 2 * hstab_oswalds_efficiency)

    drag_hstab = drag_hstab_profile + drag_hstab_induced

    # vstab
    vstab_Re = rho / mu * airspeed * vstab.mean_geometric_chord()
    vstab_airfoil = vstab.xsecs[0].airfoil  # type: asb.Airfoil
    vstab_Cd_profile = vstab_airfoil.CDp_function(0, vstab_Re, mach, 0)
    drag_vstab_profile = vstab_Cd_profile * q * vstab.area()

    drag_vstab = drag_vstab_profile

    # Force totals
    lift_force = lift_fuse + lift_wing + lift_hstab
    drag_force = drag_fuse + drag_wing + drag_hstab + drag_vstab

    # Moment totals
    m = (
            -wing.approximate_center_of_pressure()[0] * lift_wing
            -hstab.approximate_center_of_pressure()[0] * lift_hstab
    )
    opti.subject_to([
        m==0 # Trim condition
    ])

elif aerodynamics_type == "aerosandbox-point":

    airplane.fuselages = []

    airplane.set_spanwise_paneling_everywhere(8)  # Set the resolution of the analysis
    ap = asb.Casll1(
        airplane=airplane,
        op_point=asb.OperatingPoint(
            density=rho[0],
            viscosity=mu[0],
            velocity=airspeed[0],
            mach=0,
            alpha=alpha[0],
            beta=0,
            p=0,
            q=0,
            r=0,
        ),
        opti=opti
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

    airplane.wings = [wing]  # just look at the one wing
    airplane.fuselages = []  # ignore the fuselage

    airplane.set_spanwise_paneling_everywhere(6)  # Set the resolution of the analysis

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

solar_flux_on_horizontal = solar.solar_flux_on_horizontal(latitude, day_of_year, time, scattering=True)

realizable_solar_cell_efficiency = 0.205 if optimistic else 0.19  # This figure should take into account all temperature factors, MPPT losses,
# spectral losses (different spectrum at altitude), multi-junction effects, etc.
# Kevin Uleck gives this figure as 0.205.
# This paper (https://core.ac.uk/download/pdf/159146935.pdf) gives it as 0.19.

# Total cell power flux
solar_power_flux = (
        solar_flux_on_horizontal *
        realizable_solar_cell_efficiency
)

solar_area_fraction = opti.variable()
opti.set_initial(solar_area_fraction,
                 0.25
                 )
opti.subject_to([
    solar_area_fraction > 0,
    solar_area_fraction < 1,
])

area_solar = wing.area() * solar_area_fraction
power_in = solar_power_flux * area_solar

# Solar cell weight
rho_solar_cells = 0.27 if optimistic else 0.32  # kg/m^2, solar cell area density.
# The solar_simple_demo model gives this as 0.27. Burton's model gives this as 0.30.
# This paper (https://core.ac.uk/download/pdf/159146935.pdf) gives it as 0.42.
# This paper (https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=4144&context=facpub) effectively gives it as 0.3143.
mass_solar_cells = rho_solar_cells * area_solar

### Battery calculations
battery_specific_energy_Wh_kg = 420 if optimistic else 265  # Wh/kg.
# Burton's solar model uses 350, and public specs from Amprius seem to indicate that's possible.
# Odysseus had cells that were 265 Wh/kg.

battery_pack_cell_percentage = 0.70  # What percent of the battery pack consists of the module, by weight?
# Accounts for module HW, BMS, pack installation, etc.
# Ed Lovelace (in his presentation) gives 70% as a state-of-the-art fraction.

mass_battery_pack = lib_mass_prop.mass_battery_pack(
    battery_capacity_Wh=battery_capacity_watt_hours,
    battery_cell_specific_energy_Wh_kg=battery_specific_energy_Wh_kg,
    battery_pack_cell_fraction=battery_pack_cell_percentage
)

mass_wires = 0.015 * (wing.span() / 2) * ((battery_capacity / 86400 * 2) / 3000)  # a guess from 10 AWG aluminum wire

# Total system mass
mass_power_systems = mass_solar_cells + mass_battery_pack + mass_wires

# endregion

# region Propulsion
### Propeller calculations
# propeller_diameter = 3.0
propeller_diameter = opti.variable()
opti.set_initial(propeller_diameter,
                 2.5 if optimistic else 4.5
                 )
opti.subject_to([
    propeller_diameter > 1,
    propeller_diameter < 10
])

# n_propellers = 2
n_propellers = opti.variable()
opti.set_initial(n_propellers,
                 1 if optimistic else 1
                 )
opti.subject_to([
    n_propellers > 1,
    n_propellers < 4
])

area_propulsive = cas.pi / 4 * propeller_diameter ** 2 * n_propellers
propeller_efficiency = 0.85 if optimistic else 0.7  # a total WAG
motor_efficiency = 0.9 if optimistic else 0.85

power_out_propulsion_shaft = lib_prop_prop.propeller_shaft_power_from_thrust(
    thrust_force=thrust_force,
    area_propulsive=area_propulsive,
    airspeed=airspeed,
    rho=rho,
    propeller_efficiency=propeller_efficiency
)

power_out_propulsion = power_out_propulsion_shaft / motor_efficiency

power_out_max = 5e3 * opti.variable()
opti.set_initial(power_out_max, 5e3)
opti.subject_to([
    power_out_propulsion < power_out_max,
    power_out_max > 0
])

mass_motor_raw = lib_mass_prop.mass_motor_electric(max_power=power_out_max)
mass_motor_mounted = 2 * mass_motor_raw  # similar to a quote from Raymer, modified to make sensible units, prop weight roughly subtracted

mass_propellers = n_propellers * lib_mass_prop.mass_hpa_propeller(
    diameter=propeller_diameter,
    max_power=power_out_max,
    include_variable_pitch_mechanism=True
)
mass_ESC = lib_mass_prop.mass_ESC(max_power=power_out_max)

# Total propulsion mass
mass_propulsion = mass_motor_mounted + mass_propellers + mass_ESC

# Account for payload power
power_out_payload = cas.if_else(
    cas.logic_and(
        time > 4 * 3600,  # if it's after dark...
        time < 86400 - 4 * 3600  # or after sunrise...
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

# Structural mass
n_ribs_wing = 100 * opti.variable()
opti.set_initial(n_ribs_wing, 100)
opti.subject_to([
    n_ribs_wing > 0,
])
mass_wing = lib_mass_struct.mass_hpa_wing(
    span=wing.span(),
    chord=wing.mean_geometric_chord(),
    vehicle_mass=mass_total,
    n_ribs=n_ribs_wing,
    n_wing_sections=1,
    ultimate_load_factor=1.75,
    # type="one-wire",
    type="multi-wire",
    t_over_c=0.10
)

q_maneuver = 1 / 2 * atmo.get_density_at_altitude(min_altitude) * 15 ** 2 # TODO make this more accurate

n_ribs_hstab = 30 * opti.variable()
opti.set_initial(n_ribs_hstab, 30)
opti.subject_to([
    n_ribs_hstab > 0
])
mass_hstab = lib_mass_struct.mass_hpa_stabilizer(
    span=hstab.span(),
    chord=hstab.mean_geometric_chord(),
    dynamic_pressure_at_manuever_speed=q_maneuver,
    n_ribs=n_ribs_hstab,
    t_over_c=0.10
)

n_ribs_vstab = 20 * opti.variable()
opti.set_initial(n_ribs_vstab, 20)
opti.subject_to([
    n_ribs_vstab > 0
])
mass_vstab = lib_mass_struct.mass_hpa_stabilizer(
    span=vstab.span(),
    chord=vstab.mean_geometric_chord(),
    dynamic_pressure_at_manuever_speed=q_maneuver,
    n_ribs=n_ribs_vstab,
    t_over_c=0.10
)

mass_tail_boom = lib_mass_struct.mass_hpa_tail_boom(
    length_tail_boom=vstab.xyz_le[0] + vstab.xsecs[0].chord,
    dynamic_pressure_at_manuever_speed=q_maneuver,
    mean_tail_surface_area=hstab.area()
)

mass_structural = mass_wing + mass_hstab + mass_vstab + mass_tail_boom
# mass_structural = mass_total * (0.25 if optimistic else 0.31)

### Avionics
mass_flight_computer = 0.038  # a total guess - Pixhawks are 38 grams?
mass_sensors = 0.120  # GPS receiver, pitot probe, IMU, etc.
mass_communications = 0.75  # a total guess
mass_servos = 6 * 0.100  # a total guess

mass_avionics = mass_flight_computer + mass_sensors + mass_communications + mass_servos

opti.subject_to([
    mass_total == mass_payload + mass_structural + mass_propulsion + mass_power_systems + mass_avionics,
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
    net_accel_parallel / 1e-2 == net_force_parallel_calc / mass_total_eff / 1e-2,
    net_accel_perpendicular / 1e-2 == net_force_perpendicular_calc / mass_total_eff / 1e-2,
    net_power / 5e3 < (power_in - power_out) / 5e3,
])

speeddot = net_accel_parallel
gammadot = (net_accel_perpendicular * 1 / cas.fmax(min_speed, airspeed)) * 180 / np.pi

dt = cas.diff(time)
dx = cas.diff(x)
dy = cas.diff(y)
dspeed = cas.diff(airspeed)
dgamma = cas.diff(flight_path_angle)
dbattery_stored_energy_nondim = cas.diff(battery_stored_energy_nondim)

trapz = lambda x: (x[1:] + x[:-1]) / 2

xdot_trapz = trapz(airspeed * cosd(flight_path_angle))
ydot_trapz = trapz(airspeed * sind(flight_path_angle))
speeddot_trapz = trapz(speeddot)
gammadot_trapz = trapz(gammadot)
net_power_trapz = trapz(net_power)

##### Winds

wind_speed = lib_winds.wind_speed_conus_summer_99(y, latitude)
wind_speed_midpoints = lib_winds.wind_speed_conus_summer_99(trapz(y), latitude)

# Total
opti.subject_to([
    dx / 1e3 == (xdot_trapz - wind_speed_midpoints) * dt / 1e3,
    dy / 1e2 == ydot_trapz * dt / 1e2,
    dspeed / 1e0 == speeddot_trapz * dt / 1e0,
    dgamma / 1e-1 == gammadot_trapz * dt / 1e-1,
    dbattery_stored_energy_nondim / 1e-2 == (net_power_trapz / battery_capacity) * dt / 1e-2,
])
# endregion

# region Finalize Optimization Problem
##### Add periodic constraints
opti.subject_to([
    x[-1] / 1e5 > (x[0] + required_headway_per_day) / 1e5,
    y[-1] / 1e4 > y[0] / 1e4,
    battery_stored_energy_nondim[-1] > battery_stored_energy_nondim[0],
    airspeed[-1] / 2e1 > airspeed[0] / 2e1,
    flight_path_angle[-1] == flight_path_angle[0],
])

##### Add initial state constraints
opti.subject_to([  # Air Launch
    x[0] == 0,
])

##### Optional constraints
# Prevent altitude cycling
if not allow_altitude_cycling:
    y_fixed = min_altitude * opti.variable()
    opti.set_initial(y_fixed, min_altitude)
    opti.subject_to([
        y > y_fixed - 50,
        y < y_fixed + 50
    ])
# Prevent groundspeed loss
if not allow_groundtrack_cycling:
    opti.subject_to([
        # airspeed / 2e1 > wind_speed / 2e1
        x > 0
    ])

# constraints_jacobian = cas.jacobian(opti.g, opti.x)

##### Add objective
objective = mass_total / 3e2

##### Add tippers
things_to_slightly_minimize = (
        wing_span / 80
        - x[-1] / 1e6
        + n_propellers / 1
        + propeller_diameter / 2
        + battery_capacity_watt_hours / 30000
        + solar_area_fraction / 0.5
)

# Dewiggle
penalty = 0
penalty_denominator = n_timesteps
penalty += cas.sum1(cas.diff(net_power / 3000) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(net_accel_parallel / 1) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(net_accel_perpendicular / 1) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(airspeed / 30) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(flight_path_angle / 10) ** 2) / penalty_denominator
penalty += cas.sum1(cas.diff(alpha / 5) ** 2) / penalty_denominator

opti.minimize(objective + penalty + 1e-6 * things_to_slightly_minimize)
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
s_opts["watchdog_shortened_iter_trigger"] = 1
# s_opts["expect_infeasible_problem"]="yes"
s_opts["start_with_resto"] = "yes"
s_opts["required_infeasibility_reduction"] = 0.001
# s_opts["evaluate_orig_obj_at_resto_trial"] = "yes"
opti.solver('ipopt', p_opts, s_opts)

# endregion

if __name__ == "__main__":
    try:
        sol = opti.solve()
    except:
        sol = opti.debug

    if np.abs(sol.value(penalty / objective)) > 0.01:
        print("\nWARNING: high penalty term! P/O = %.3f\n" % sol.value(penalty / objective))

    if aerodynamics_type == "aerosandbox-point":
        import copy

        ap_sol = copy.deepcopy(ap)
        ap_sol.substitute_solution(sol)
    if aerodynamics_type == "aerosandbox-full":
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
    o = lambda x: print(
        "%s: %f" % (x, sol.value(eval(x))))  # A function to Output a scalar variable. Input a variable name as a string
    outs = lambda xs: [o(x) for x in xs] and None  # input a list of variable names as strings
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
