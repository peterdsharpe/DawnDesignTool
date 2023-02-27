import aerosandbox as asb
from aerosandbox.library import winds as lib_winds
import aerosandbox.numpy as np
import pathlib
from aerosandbox.modeling.interpolation import InterpolatedModel
from aerosandbox.library.airfoils import naca0008
from design_opt_utilities.fuselage import make_payload_pod
import aerosandbox.library.mass_structural as mass_lib
from aerosandbox.library import power_solar as solar_lib
from aerosandbox.library import propulsion_electric as elec_lib
from aerosandbox.library import propulsion_propeller as prop_lib
from aerosandbox.atmosphere import Atmosphere as atmo
import aerosandbox.tools.units as u
from typing import Union, List




path = str(
    pathlib.Path(__file__).parent.absolute()
)

##### Section: Initialize Optimization
opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='design'
)
des = dict(category="design")
ops = dict(category="operations")

##### optimization assumptions
minimize = 'wingspan'
make_plots = False

##### Section: Parameters

# Mission Operating Parameters
latitude = opti.parameter(value=-75)  # degrees, the location the sizing occurs
day_of_year = opti.parameter(value=45)  # Julian day, the day of the year the sizing occurs
mission_length = opti.parameter(value=45)  # days, the length of the mission without landing to download data
strat_offset_value = opti.parameter(value=1000)  # meters, margin above the stratosphere height the aircraft is required to stay above
min_cruise_altitude = lib_winds.tropopause_altitude(latitude, day_of_year) + strat_offset_value
climb_opt = False  # are we optimizing for the climb as well?
hold_cruise_altitude = True  # must we hold the cruise altitude (True) or can we altitude cycle (False)?

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
vstab_cells = "sunpower"  # select cells for vtail, options include ascent_solar, sunpower, and microlink
max_wing_solar_area_fraction = opti.parameter(0.8)
max_vstab_solar_area_fraction = opti.parameter(0.8)
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
n_timesteps_per_segment = 180  # number of timesteps in the 25 hour sizing period #todo increase for trajectory stuff

if climb_opt:  # roughly 1-day-plus-climb window, starting at ground. Periodicity enforced for last 24 hours.
    time_start = opti.variable(init_guess=-12 * 3600,
                               scale=3600,
                               upper_bound=0,
                               lower_bound=-24 * 3600
                                           ** ops)
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

# Payload pod
# values roughly to match the demonstrator fuselage and payload pod
boom_diameter = 0.2  # meters
payload_pod_length = 2.0  # meters
payload_pod_diameter = 0.5  # meters
payload_pod_y_offset = 1.5 # meters

payload_pod = make_payload_pod( # TODO ask peter if there's a better way to make this an aero shape
    boom_length=payload_pod_length,
    nose_length=0.5,
    tail_length=1,
    fuse_diameter=payload_pod_diameter,
).translate([0, 0, -payload_pod_y_offset])

payload_pod_volume = payload_pod.volume()

# overall layout wing layout
boom_location = 0.80  # as a fraction of the half-span
taper_break_location = 0.67  # as a fraction of the half-span
field_joint_location = 0.36 # as fraction of the half-span

# Wing
wing_span = opti.variable(
    init_guess=30,
    scale=10,
    lower_bound=0.01,
    **des
)
wing_x_le = opti.variable(
    init_guess=1.2,
    lower_bound=payload_pod_length,
    upper_bound=payload_pod_length * 0.75,
    # freeze=True,
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

wing_y_taper_break = taper_break_location * wing_span / 2

wing_taper_ratio = 0.5  # TODO analyze this more
wing_tip_chord = wing_root_chord * wing_taper_ratio

wing_incidence = opti.variable(
    init_guess=0,
    lower_bound=-15,
    upper_bound=15,
    freeze=True,
    **des
)
wing_x_center = wing_x_le + 0.5 * wing_root_chord

wing = asb.Wing(
    name="Main Wing",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            chord=wing_root_chord,
            twist=wing_incidence,  # degrees
            airfoil=wing_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Break
            xyz_le=np.array([0,
                             wing_y_taper_break,
                             0]),
            chord=wing_root_chord,
            twist=wing_incidence,
            airfoil=wing_airfoil,
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0.25 * (wing_root_chord - wing_tip_chord),
                             wing_span / 2,
                             0]),
            chord=wing_tip_chord,
            twist=wing_incidence,
            airfoil=wing_airfoil,
        ),
    ]
).translate([
    wing_x_le,
    0,
    0
])

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
    lower_bound=wing_root_chord * 3/4,
    **des
)

center_boom = asb.Fuselage(
        name="Center Boom",
        xsecs=[
            asb.FuselageXSec(
                    xyz_c=[0, 0, 0],
                    radius=boom_diameter/2,
                ),
                asb.FuselageXSec(
                    xyz_c=[center_boom_length, 0, 0],
                    radius=boom_diameter/2,
                )
        ]
    )
right_boom = asb.Fuselage(
        name="Right Boom",
        xsecs=[
            asb.FuselageXSec(
                    xyz_c=[0, 0, 0],
                    radius=boom_diameter/2,
                ),
                asb.FuselageXSec(
                    xyz_c=[outboard_boom_length, 0, 0],
                    radius=boom_diameter/2,
                )
        ]
    )
right_boom = right_boom.translate(np.array([
    0,
    boom_offset,
    0]))
left_boom = right_boom.translate(np.array([
    0,
    -2 * boom_offset,
    0]))


# tail section
tail_airfoil = naca0008
# tail_airfoil.generate_polars(
#     cache_filename="cache/naca0008.json",
#     include_compressibility_effects=False,
#     make_symmetric_polars=True
# )

#  vstab
vstab_span = opti.variable(
    init_guess=3,
    scale=2,
    lower_bound=0.1,
    **des
)

vstab_chord = opti.variable(
    init_guess=1,
    scale=1,
    lower_bound=0.1,
    **des
)

vstab_incidence = opti.variable(
    init_guess=0,
    scale=1,
    upper_bound=30,
    lower_bound=-30,
    freeze=True,
    **ops
)

vstab = asb.Wing(
    name="Vertical Stabilizer",
    symmetric=False,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([0, 0, 0]),
            chord=vstab_chord,
            twist=vstab_incidence,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, 0, vstab_span]),
            chord=vstab_chord,
            twist=vstab_incidence,
            airfoil=tail_airfoil,
        ),
    ]
).translate(
    np.array([
        center_boom_length - vstab_chord * 0.75,
        0,
        -vstab_span / 2 + vstab_span * 0.15
]))

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
    **ops
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
).translate(np.array([
    center_boom_length - vstab_chord * 0.75 - center_hstab_chord,
    0,
    0.1]))

opti.subject_to([
    center_boom_length - vstab_chord - center_hstab_chord > wing_x_quarter_chord + wing_root_chord * 3 / 4
])

# tailerons
outboard_hstab_span = opti.variable(
    init_guess=4,
    scale=4,
    lower_bound=2,
    upper_bound=wing_span/6,
    **des
)

outboard_hstab_chord = opti.variable(
    init_guess=0.8,
    scale=0.5,
    lower_bound=0.8,
    upper_bound=wing_root_chord,
    **des
)

outboard_hstab_incidence = opti.variable(
    init_guess=0,
    lower_bound=-15,
    upper_bound=15,
    **ops
)
right_hstab = asb.Wing(
    name="Taileron",
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            xyz_le=np.array([0, 0, 0]),
            chord=outboard_hstab_chord,
            twist=outboard_hstab_incidence,  # degrees
            airfoil=tail_airfoil,  # Airfoils are blended between a given XSec and the next one.
            control_surface_is_symmetric=True,
        ),
        asb.WingXSec(  # Tip
            xyz_le=np.array([0, outboard_hstab_span / 2, 0]),
            chord=outboard_hstab_chord,
            twist=outboard_hstab_incidence,
            airfoil=tail_airfoil,
        ),
    ]
).translate(np.array([
    outboard_boom_length - outboard_hstab_chord * 0.75,
    boom_offset,
    0.1]))
left_hstab = right_hstab.translate([
    0,
    -boom_offset * 2,
    0])

# Assemble the airplane
airplane = asb.Airplane(
    name="Dawn1",
    xyz_ref=np.array([0, 0, 0]),
    wings=[
        wing,
        center_hstab,
        right_hstab,
        left_hstab,
        vstab,
    ],
    fuselages=[
        center_boom,
        right_boom,
        left_boom
    ],
)

# Propeller
propeller_diameter = opti.variable(
    init_guess=1,
    scale=1,
    upper_bound=10,
    lower_bound=0.5,
    **des
)

n_propellers = opti.parameter(value=2) # TODO reconsider 2 or 4

##### Vehicle Overall Specs
mass_total = opti.variable(
    init_guess=110,
    lower_bound=10,
    **des
)

max_power_in = opti.variable(
    init_guess=5000,
    lower_bound=10,
    **des
)

max_power_out_propulsion = opti.variable(
    init_guess=1500,
    lower_bound=10,
    **des
)

##### Section: Internal Geometry and Weights
mass_props = {}

### Wing mass accounting
wing_n_ribs = opti.variable(
    init_guess=200,
    scale=200,
    lower_bound=0,
    log_transform=True,
    **des
)

wing_mass_primary = mass_lib.mass_wing_spar(
    span=wing.span(),
    mass_supported=mass_total,
    # technically the spar doesn't really have to support its own weight (since it's roughly spanloaded), so this is conservative
    ultimate_load_factor=structural_load_factor,
    n_booms=1
) * 11.382 / 9.222  # scaling factor taken from Daedalus weights to account for real-world effects, non-cap mass, etc.

wing_mass_secondary = mass_lib.mass_hpa_wing(
        span=wing_span,
        chord=wing.mean_geometric_chord(),
        vehicle_mass=mass_total,
        n_ribs=wing_n_ribs,
        n_wing_sections=4,
        ultimate_load_factor=structural_load_factor,
        type='cantilevered',
        t_over_c=wing_airfoil.max_thickness(),
        include_spar=False
) * 1.5 # scaling factor suggested by Drela

wing_mass = wing_mass_primary + wing_mass_secondary

wing_y_field_joint_break = field_joint_location * wing_span / 2

mass_props['wing_center'] = asb.mass_properties_from_radius_of_gyration(
    mass=wing_mass * wing_y_field_joint_break * structural_mass_margin_multiplier,
    x_cg=wing_x_le + 0.40 * wing_root_chord,  # quarter-chord,
    radius_of_gyration_x=(wing_y_field_joint_break * wing_span) / 12,
    radius_of_gyration_z=(wing_y_field_joint_break * wing_span) / 12
)
mass_props['wing_tips'] = asb.mass_properties_from_radius_of_gyration(
    mass=wing_mass * (1 - wing_y_field_joint_break) * structural_mass_margin_multiplier,
    x_cg=wing_x_le + 0.40 * wing_root_chord,  # quarter-chord,
    radius_of_gyration_x=(1 + wing_y_field_joint_break) / 2 * (wing_span / 2),
    radius_of_gyration_z=(1 + wing_y_field_joint_break) / 2 * (wing_span / 2),
)

### hstab mass accounting
def mass_hstab(
        hstab,
        n_ribs_hstab,
):
    mass_hstab_primary = mass_lib.mass_wing_spar(
        span=hstab.span(),
        mass_supported=q_ne * 1.5 * hstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    )

    mass_hstab_secondary = mass_lib.mass_hpa_stabilizer(
        span=hstab.span(),
        chord=hstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_hstab,
        t_over_c=0.08,
        include_spar=False
    )
    mass_hstab = mass_hstab_primary + mass_hstab_secondary  # per hstab
    return mass_hstab

q_ne = opti.variable(
    init_guess=160,
    lower_bound=0,
    **des
)

n_ribs_center_hstab = opti.variable(
    init_guess=40,
    scale=40,
    lower_bound=0,
    **des
)

mass_props['center_hstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_hstab(center_hstab, n_ribs_center_hstab) * structural_mass_margin_multiplier,
    x_cg=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
    z_cg=vstab.aerodynamic_center()[2],
    radius_of_gyration_x=center_hstab_span / 12,
    radius_of_gyration_y=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
    radius_of_gyration_z=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2
)

n_ribs_outboard_hstab = opti.variable(
    init_guess=40,
    scale=40,
    lower_bound=0,
    **des
)

mass_props['right_hstab'] = asb.MassProperties(
    mass=mass_hstab(right_hstab, n_ribs_outboard_hstab) * structural_mass_margin_multiplier,
    x_cg=right_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
)

mass_props['left_hstab'] = asb.MassProperties(
    mass=mass_hstab(left_hstab, n_ribs_outboard_hstab) * structural_mass_margin_multiplier,
    x_cg=left_hstab.xsecs[0].xyz_le[0] + outboard_hstab_chord / 2,
)

# vstab mass accounting
def mass_vstab(
        vstab,
        n_ribs_vstab,
):
    mass_vstab_primary = mass_lib.mass_wing_spar(
        span=vstab.span(),
        mass_supported=q_ne * 1.5 * vstab.area() / 9.81,
        ultimate_load_factor=structural_load_factor
    ) * 1.2  # TODO due to asymmetry, a guess
    mass_vstab_secondary = mass_lib.mass_hpa_stabilizer(
        span=vstab.span(),
        chord=vstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_vstab,
        t_over_c=0.08
    )
    mass_vstab = mass_vstab_primary + mass_vstab_secondary  # per vstab
    return mass_vstab

n_ribs_vstab = opti.variable(
    init_guess=40,
    scale=40,
    lower_bound=0,
    **des
)

mass_props['vstab'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_hstab(vstab, n_ribs_vstab) * structural_mass_margin_multiplier,
    x_cg=vstab.xsecs[0].xyz_le[0] + vstab_chord / 2,
    z_cg=vstab.aerodynamic_center()[2],
    radius_of_gyration_x=vstab_span / 12,
    radius_of_gyration_y=vstab.xsecs[0].xyz_le[0] + vstab_chord / 2,
    radius_of_gyration_z=vstab.xsecs[0].xyz_le[0] + vstab_chord / 2
)

### boom mass accounting
mass_props['center_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=center_boom_length-wing_x_quarter_chord,
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=center_hstab.area() + vstab.area()
    ) * structural_mass_margin_multiplier,
    x_cg=center_boom_length / 2,
    radius_of_gyration_y=center_boom_length / 3,
    radius_of_gyration_z=center_boom_length / 3,
)
mass_props['left_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=right_hstab.area()
    ) * structural_mass_margin_multiplier,
    x_cg=outboard_boom_length / 2,
    radius_of_gyration_y=outboard_boom_length / 3,
    radius_of_gyration_z=outboard_boom_length / 3,
)
mass_props['right_boom'] = asb.mass_properties_from_radius_of_gyration(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=left_hstab.area()
    ) * structural_mass_margin_multiplier,
    x_cg=outboard_boom_length / 2,
    radius_of_gyration_y=outboard_boom_length / 3,
    radius_of_gyration_z=outboard_boom_length / 3,
)

### payload pod mass accounting
pod_structure_mass = 20 # kg, corresponds to reduction in pod mass expected for future build
# assumes approximately same size battery system and payload
# taken from Daedalus, http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
mass_daedalus = 103.9  # kg, corresponds to 229 lb gross weight.
# Total mass of the Daedalus aircraft, used as a reference for scaling.
mass_fairings = 2.067 * mass_total / mass_daedalus  # Scale fairing mass to same mass fraction as Daedalus
mass_landing_gear = 0.728 * mass_total / mass_daedalus  # Scale landing gear mass to same mass fraction as Daedalus
mass_strut = 0.5 # mass per strut to the payload pod roughly baselined to dawn demonstrator build

mass_props['payload_pod'] = asb.MassProperties(
    mass=(
        mass_fairings +
        mass_landing_gear +
        mass_strut * 2 +
        pod_structure_mass
    ),
    x_cg=payload_pod_length/2
) * structural_mass_margin_multiplier

### summation of structural mass
structural_mass_props = (
        mass_props['wing_center'] +
        mass_props['wing_tips'] +
        mass_props['center_hstab'] +
        mass_props['left_hstab'] +
        mass_props['right_hstab'] +
        mass_props['center_boom'] +
        mass_props['left_boom'] +
        mass_props['right_boom'] +
        mass_props['payload_pod']
)

### avionics mass accounting
# tons of assumptions made here and taken from other project peter sent over
# roughly +/- 0.5kg the previous ddt avionics mass assumptions
actuator_mass = 6 * 0.575
processor_mass = 0.140
flight_computer_mass = 0.660
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
    x_cg=payload_pod_length * 0.4,  # right behind payload # TODO revisit this number
)
avionics_volume = mass_props['avionics'].mass / 1250  # assumed density of electronics
avionics_power = 180 # TODO revisit this number

# instrument data storage mass requirements
mass_of_data_storage = 0.0053 # kg per TB of data
tb_per_day = 4
mass_props['payload'] = asb.MassProperties(
    mass=mass_payload_base +
           mission_length * tb_per_day * mass_of_data_storage
)
payload_volume = 0.023 * 0.5 # assuming payload mass from gamma remote sensing with 50% margin on volume

### Power Systems Mass Accounting
if vstab_cells == "microlink":
    vstab_solar_cell_efficiency = 0.285 * 0.9  # Microlink
    vstab_rho_solar_cells = 0.255 * 1.1  # kg/m^2, solar cell area density. Microlink.
    vstab_solar_cost_per_watt = 250  # $/W
    vstab_solar_power_ratio = 1100  # W/kg

if vstab_cells == "sunpower":
    vstab_solar_cell_efficiency = 0.243 * 0.9  # Sunpower
    vstab_rho_solar_cells = 0.425 * 1.1 * 1.15  # kg/m^2, solar cell area density. Sunpower.
    vstab_solar_cost_per_watt = 3  # $/W
    vstab_solar_power_ratio = 500  # W/kg

if vstab_cells == "ascent_solar":
    vstab_solar_cell_efficiency = 0.14 * 0.9  # Ascent Solar
    vstab_rho_solar_cells = 0.300 * 1.1  # kg/m^2, solar cell area density. Ascent Solar
    vstab_solar_cost_per_watt = 80  # $/W
    vstab_solar_power_ratio = 300  # W/kg

if wing_cells == "microlink":
    wing_solar_cell_efficiency = 0.285 * 0.9  # Microlink
    wing_rho_solar_cells = 0.255 * 1.1  # kg/m^2, solar cell area density. Microlink.
    wing_solar_cost_per_watt = 250  # $/W
    wing_solar_power_ratio = 1100  # W/kg

if wing_cells == "sunpower":
    wing_solar_cell_efficiency = 0.243 * 0.9  # Sunpower
    wing_rho_solar_cells = 0.425 * 1.1 * 1.15  # kg/m^2, solar cell area density. Sunpower.
    wing_solar_cost_per_watt = 3  # $/W
    wing_solar_power_ratio = 500  # W/kg

if wing_cells == "ascent_solar":
    wing_solar_cell_efficiency = 0.14 * 0.9  # Ascent Solar
    wing_rho_solar_cells = 0.300 * 1.1  # kg/m^2, solar cell area density. Ascent Solar
    wing_solar_cost_per_watt = 80  # $/W
    wing_solar_power_ratio = 300  # W/kg

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

if not tail_panels:
    opti.subject_to([
        vstab_solar_area_fraction == 0,
    ])

area_solar_wing = wing.area() * wing_solar_area_fraction
area_solar_vstab = vstab.area() * vstab_solar_area_fraction

mass_props['solar_panel_wing'] = asb.MassProperties(
    mass=area_solar_wing * wing_rho_solar_cells,
    x_cg=wing_x_le + 0.50 * wing_root_chord
)

mass_props['solar_panel_vstab'] = asb.MassProperties(
    mass=area_solar_vstab * vstab_rho_solar_cells,
    x_cg=center_hstab.xsecs[0].xyz_le[0] + center_hstab_chord / 2,
)

### MPPT mass accounting
mass_props['MPPT'] = asb.MassProperties(
    mass=solar_lib.mass_MPPT(max_power_in)
)

### battery mass accounting
battery_capacity = opti.variable(
    init_guess=5e8,
    scale=5e8,
    lower_bound=0,
    **des
)
battery_capacity_watt_hours = battery_capacity / 3600

battery_pack_mass = elec_lib.mass_battery_pack(
        battery_capacity_Wh=battery_capacity_watt_hours,
        battery_cell_specific_energy_Wh_kg=battery_specific_energy_Wh_kg,
        battery_pack_cell_fraction=battery_pack_cell_percentage
    )

battery_cell_mass = battery_pack_mass * battery_pack_cell_percentage
cost_batteries = 4 * battery_capacity_watt_hours
battery_density = 2000  # kg/m^3, taken from averaged measurements of commercial cells
battery_volume = battery_pack_mass / battery_density

battery_cg = (battery_volume / payload_pod_volume) * 0.5 # TODO figure out if x cg location makes sense
mass_props['battery_pack'] = asb.MassProperties(
    mass=battery_pack_mass,
    x_cg=battery_cg,
)

battery_pack_specific_energy = battery_specific_energy_Wh_kg * u.hour * battery_pack_cell_percentage # J/kg #TODO double check this is right
battery_total_energy = mass_props['battery_pack'].mass * battery_pack_specific_energy  # J

battery_voltage = 125  # From Olek Peraire >4/2, propulsion slack

### wiring mass acounting
mass_props['wires'] = asb.MassProperties(
    mass=elec_lib.mass_wires(
        wire_length=wing.span() / 2,
        max_current=max_power_out_propulsion / battery_voltage,
        allowable_voltage_drop=battery_voltage * 0.01,
        material="aluminum"
    ),
    x_cg=wing_root_chord * 0.25 # assume most wiring is down spar
)

### propellers mass acounting
propeller_tip_mach = 0.36  # From Dongjoon, 4/30/20
propeller_rads_per_sec = propeller_tip_mach * atmo(altitude=20000).speed_of_sound() / (propeller_diameter / 2)
propeller_rpm = propeller_rads_per_sec * 30 / np.pi

area_propulsive = np.pi / 4 * propeller_diameter ** 2 * n_propellers

mass_props['propellers'] = asb.MassProperties(
    mass=n_propellers * prop_lib.mass_hpa_propeller(
        diameter=propeller_diameter,
        max_power=max_power_out_propulsion / n_propellers,
        include_variable_pitch_mechanism=variable_pitch
    ),
    x_cg=wing_x_le - 0.25 * propeller_diameter
)

### motors mass accounting
propeller_max_torque = (max_power_out_propulsion / n_propellers) / propeller_rads_per_sec
motor_kv = propeller_rpm / battery_voltage
motor_mounting_weight_multiplier = 2.0  # Taken from Raymer guidance.

mass_props['motors'] = asb.MassProperties(
    mass=n_propellers * elec_lib.mass_motor_electric(
        max_power=max_power_out_propulsion / n_propellers,
        kv_rpm_volt=motor_kv,
        voltage=battery_voltage,
    ),
    x_cg=wing_x_le - 0.1 * propeller_diameter
) * motor_mounting_weight_multiplier

mass_props['esc'] = asb.MassProperties(
    mass=elec_lib.mass_ESC(
        max_power=max_power_out_propulsion
    ),
    x_cg=wing_x_le - 0.1 * propeller_diameter # co-located with motors
)

### summation of power system mass
power_systems_mass_props = (
        mass_props['solar_panel_wing'] +
        mass_props['solar_panel_vstab'] +
        mass_props['MPPT'] +
        mass_props['battery_pack']
)
propulsion_system_mass_props = (
    mass_props['propellers'] +
    mass_props['motors'] +
    mass_props['esc']
)

#### summation of total system mass
mass_props_TOGW = asb.MassProperties(mass=0)
for k, v in mass_props.items():
    mass_props_TOGW = mass_props_TOGW + v

remaining_volume = (
    payload_pod_volume - (
    payload_volume +
    avionics_volume +
    battery_volume
    )
)

opti.subject_to(mass_total > mass_props_TOGW.mass)

##### Section: Setup Dynamics
guess_altitude = 18000
guess_u_e = 20

dyn = asb.DynamicsPointMass2DCartesian(
    mass_props=mass_props_TOGW,
    x_e=opti.variable(
        init_guess=time * guess_u_e,
        **ops
    ),
    z_e=opti.variable(
        init_guess=-guess_altitude, n_vars=n_timesteps,
        **ops
    ),
    u_e=opti.variable(init_guess=guess_u_e, n_vars=n_timesteps),
    w_e=opti.variable(init_guess=0, n_vars=n_timesteps),
    alpha=opti.variable(
        init_guess=3, n_vars=n_timesteps,
        **ops
    ),
)

dyn.add_gravity_force(g=9.81)

opti.subject_to([
    dyn.x_e[time_periodic_start_index] == 0,
    dyn.altitude[time_periodic_start_index:] / min_cruise_altitude > 1,
    dyn.altitude / guess_altitude > 0,  # stay above ground
    dyn.altitude / 40000 < 1,  # models break down
])

if climb_opt:
    opti.subject_to(dyn.altitude[0] / 1e4 == 0)

if hold_cruise_altitude:

    cruise_altitude = opti.variable(
        init_guess=guess_altitude,
        scale=10000,
        lower_bound=min_cruise_altitude,
        **des
    )

    dyn.altitude[time_periodic_start_index:] / cruise_altitude > 1, # stay at cruise altitude after climb

##### Section: Aerodynamics
aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point,
    xyz_ref=mass_props_TOGW.xyz_cg
).run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=False,
    q=False,
    r=False
)

dyn.add_force(
    *aero['F_w'],
    axes="earth"
)  # Note, this is not a typo: we make the small-angle-approximation on the flight path angle gamma.

##### Section: Payload

c = 299792458  # [m/s] speed of light
k_b = 1.38064852E-23  # [m2 kg s-2 K-1] boltzman constant
# radar_length = opti.variable(
#     init_guess=0.1,
#     scale=1,
#     category='des',
#     lower_bound=0.1,
#     upper_bound=1,
# ) # meters
bandwidth = opti.variable(
    init_guess=1e8,
    scale=1e6,
    lower_bound=0,
    **des
)  # Hz
pulse_rep_freq = opti.variable(
    init_guess=353308,
    scale=10000,
    lower_bound=0,
    **des
)
power_trans = opti.variable(
    init_guess=1e6,
    scale=1e5,
    lower_bound=0,
    upper_bound=1e8,
    **ops
) # watts
groundspeed = opti.variable( # TODO relate to airspeed using windspeed
    init_guess=10,
    scale=5,
    lower_bound=0, # TODO revisit this as lower bound
    **ops
)

# define key radar parameters
radar_area = radar_width * radar_length # meters ** 2
dist = dyn.altitude / np.cosd(look_angle) # meters
swath_azimuth = center_wavelength * dist / radar_length # meters
swath_range = center_wavelength * dist / (radar_width * np.cosd(look_angle)) # meters
max_length_synth_ap = center_wavelength * dist / radar_length # meters
ground_area = swath_range * swath_azimuth * np.pi / 4 # meters ** 2
radius = (swath_azimuth + swath_range) / 4 # meters
ground_imaging_offset = np.tand(look_angle) * dyn.altitude # meters
scattering_cross_sec = 10 ** (scattering_cross_sec_db / 10)
sigma0 = scattering_cross_sec / ground_area
antenna_gain = 4 * np.pi * radar_area * 0.7 / center_wavelength ** 2

# Assumed constants
a_hs = 0.88 # aperture-illumination taper factor associated with the synthetic aperture (value from Ulaby and Long)
F = 4 # receiver noise figure (somewhat randomly chosen value from Ulaby and Long)
a_B = 1 # pulse-taper factor to relate bandwidth and pulse duration

# # constrain SAR resolution to required value
pulse_duration = a_B / bandwidth
range_resolution = c * pulse_duration / (2 * np.sind(look_angle))
azimuth_resolution = radar_length / 2
opti.subject_to([
    range_resolution <= required_resolution,
    azimuth_resolution <= required_resolution,
])

# use SAR specific equations from Ulaby and Long
payload_power = power_trans * pulse_rep_freq * pulse_duration # TODO make a function of location in trajectory

snr = payload_power * antenna_gain ** 2 * center_wavelength ** 3 * a_hs * sigma0 * range_resolution / \
      ((2 * 4 * np.pi) ** 3 * dist ** 3 * k_b * dyn.op_point.atmosphere.temperature() * F * groundspeed * a_B)

snr_db = 10 * np.log(snr)

opti.subject_to([
    required_snr <= snr_db,
    pulse_rep_freq >= 2 * groundspeed / radar_length,
    pulse_rep_freq <= c / (2 * swath_azimuth),
])

##### Section: Propulsion and Power Output

thrust = opti.variable(
    init_guess=600 * 9.81 / 30,
    n_vars=n_timesteps,
    lower_bound=0,
    **ops
)

dyn.add_force(
    Fx=thrust,
    axes="earth"
)  # Note, this is not a typo: we make the small-angle-approximation on the flight path angle gamma.

if not use_propulsion_fits_from_FL2020_1682_undergrads:
    ### Use older models

    motor_efficiency = 0.955  # Taken from ThinGap estimates

    power_out_propulsion_shaft = prop_lib.propeller_shaft_power_from_thrust(
        thrust_force=thrust,
        area_propulsive=area_propulsive,
        airspeed=dyn.op_point.velocity,
        rho=dyn.op_point.atmosphere.density(),
        propeller_coefficient_of_performance=0.90  # calibrated to QProp output with Dongjoon
    )

    gearbox_efficiency = 0.986

else:
    ### Use Jamie's model
    from design_opt_utilities.new_models import eff_curve_fit

    opti.subject_to(dyn.altitude < 30000)  # Bugs out without this limiter

    propeller_efficiency, motor_efficiency = eff_curve_fit(
        airspeed=dyn.op_point.velocity,
        total_thrust=thrust,
        altitude=dyn.altitude,
        var_pitch=variable_pitch
    )
    power_out_propulsion_shaft = thrust * dyn.op_point.velocity / propeller_efficiency

    gearbox_efficiency = 0.986

power_out_propulsion = power_out_propulsion_shaft / motor_efficiency / gearbox_efficiency

opti.subject_to(power_out_propulsion < max_power_out_propulsion)

power_out = power_out_propulsion + payload_power + avionics_power

##### Section: Power Input (Solar)

MPPT_efficiency = 1 / 1.04

left_wing_incident_solar_power = 0.5 * area_solar_wing * solar_lib.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    altitude=dyn.altitude,
    panel_azimuth_angle=dyn.beta + 90,  # TODO check beta is applied correctly
    panel_tilt_angle=10 + dyn.gamma,  # TODO check gamma is applied correctly
    scattering=True
)

right_wing_incident_solar_power = 0.5 * area_solar_wing * solar_lib.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    altitude=dyn.altitude,
    panel_azimuth_angle=dyn.beta - 90,  # TODO check beta is applied correctly
    panel_tilt_angle=-10 + dyn.gamma,  # TODO check gamma is applied correctly
    scattering=True
)

wing_incident_solar_power = right_wing_incident_solar_power + left_wing_incident_solar_power

vstab_incident_solar_power_left = area_solar_vstab * solar_lib.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    altitude=dyn.altitude,
    panel_azimuth_angle=dyn.beta + 90,  # TODO check beta is applied correctly
    panel_tilt_angle=90,  # TODO check gamma is applied correctly
    scattering=True
)

vstab_incident_solar_power_right = area_solar_vstab * solar_lib.solar_flux(
    latitude=latitude,
    day_of_year=day_of_year,
    time=time,
    altitude=dyn.altitude,
    panel_azimuth_angle=dyn.beta - 90,  # TODO check beta is applied correctly
    panel_tilt_angle=90,  # TODO check gamma is applied correctly
    scattering=True
)

vstab_incident_solar_power = vstab_incident_solar_power_right + vstab_incident_solar_power_left

power_in = (wing_incident_solar_power * wing_solar_cell_efficiency +
            vstab_incident_solar_power * vstab_solar_cell_efficiency) \
           * MPPT_efficiency

opti.subject_to(
    power_in / 5e3 < max_power_in / 5e3
)

##### Section: Battery power

battery_charge_state = opti.variable(
    init_guess=0.5,
    n_vars=n_timesteps,
    lower_bound=allowable_battery_depth_of_discharge,
    upper_bound=1,
)

net_power = power_in - power_out

opti.constrain_derivative(
    derivative=net_power,
    variable=battery_charge_state * battery_total_energy,
    with_respect_to=time,
)

##### Section: Constrain Dynamics
opti.constrain_derivative(
    variable=dyn.x_e, with_respect_to=time,
    derivative=dyn.u_e
)
opti.constrain_derivative(
    variable=dyn.z_e, with_respect_to=time,
    derivative=dyn.w_e,
)

opti.subject_to(dyn.Fz_e / 1e3 == 0)  # L == W
excess_power = dyn.u_e * dyn.Fx_e
climb_rate = excess_power / (dyn.mass_props.mass * 9.81)
opti.subject_to((excess_power + dyn.w_e * dyn.mass_props.mass * 9.81) / 5e3 == 0)
opti.subject_to(q_ne > dyn.op_point.dynamic_pressure())

##### Section: Perodicity Constraints
opti.subject_to([
    dyn.x_e[time_periodic_end_index] / 1e5 > dyn.x_e[time_periodic_start_index] / 1e5 + required_headway_per_day,
    dyn.altitude[time_periodic_end_index] / 1e4 > dyn.altitude[time_periodic_start_index] / 1e4,
    dyn.u_e[time_periodic_end_index] / 1e1 > dyn.u_e[time_periodic_start_index] / 1e1,
    battery_charge_state[time_periodic_end_index] > battery_charge_state[time_periodic_end_index],
    dyn.gamma[time_periodic_end_index] == dyn.gamma[time_periodic_start_index],
    dyn.alpha[time_periodic_end_index] == dyn.alpha[time_periodic_start_index],
    thrust[time_periodic_end_index] == thrust[time_periodic_start_index]
])

#### Section: Add imposed constraints
opti.subject_to([
    center_hstab_span == outboard_hstab_span,
    center_hstab_chord == outboard_hstab_chord,
    # center_hstab_twist_angle == outboard_hstab_twist_angle,
    center_boom_length >= outboard_boom_length,
    center_hstab_incidence <= 0,  # essentially enforces downforce, prevents hstab from lifting and exploiting config.
    outboard_hstab_incidence <= 0,
    # essentially enforces downforce, prevents hstab from lifting and exploiting config.
    # TODO double check below constraints to make sure they are correct for this application
    center_hstab_span < wing_span / 2,
    aero["CL"] > 0,
    np.mean(aero["Cm"]) == 0,
    aero["Cma"] < -0.5,
    aero["Cnb"] > 0.05,
    remaining_volume / u.inch ** 3 > 0,
    dyn.alpha < 12,
    dyn.alpha > -12,
    # np.diff(np.degrees(dyn.gamma)) < 5,
    # np.diff(np.degrees(dyn.gamma)) > -5,
    np.diff(dyn.alpha) < 2,
    np.diff(dyn.alpha) > -2,
])

##### Section: Useful metrics
wing_loading = 9.81 * mass_total / wing.area()
wing_loading_psf = wing_loading / 47.880258888889
empty_wing_loading = 9.81 * structural_mass_props.mass / wing.area()
empty_wing_loading_psf = empty_wing_loading / 47.880258888889
propeller_efficiency = thrust * dyn.u_e / power_out_propulsion_shaft
cruise_LD = aero['L'] / aero['D']
avg_cruise_LD = np.mean(cruise_LD)
avg_CL = np.mean(wing.CL)
avg_airspeed = np.mean(dyn.u_e)
sl_atmosphere = atmo(altitude=0)
rho_ratio = np.sqrt(np.mean(dyn.op_point.atmosphere.density()) / sl_atmosphere.density())
avg_ias = avg_airspeed * rho_ratio

##### Add objective
objective = eval(minimize)

##### Section: Add tippers
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
    groundspeed,
    dyn.gamma / 2,
    dyn.alpha / 1,
]:
    penalty += np.sum(np.diff(np.diff(penalty_input)) ** 2) / n_timesteps_per_segment

opti.minimize(
    objective
    + penalty
    + 1e-3 * things_to_slightly_minimize
)

if __name__ == "__main__":
    # Solve
    try:
        sol = opti.solve(
            max_iter=10000,
            options={
                "ipopt.max_cpu_time": 3000
            }
        )
    except RuntimeError as e:
        print(e)
        sol = opti.debug

    airplane.substitute_solution(sol)
    dyn.substitute_solution(sol)
    mass_props_TOGW.substitute_solution(sol)

    ### Macros
    s = lambda x: sol.value(x)

    print("\nVolume Accounting\n" + "-" * 50)
    volumes = {
        k: s(eval(k))
        for k in list(vars().keys())
        if ("volume" in k and
            k != "payload_pod_volume" and
            k != "volumes" and
            "volumetric" not in k
            )
    }

    for k, v in sorted(volumes.items(), key=lambda item: -item[1]):
        print(" | ".join([
            f"{k:25}",
            f"{format(v, '8.3g').rjust(9)} m^3",
            f"{v / s(payload_pod_volume) * 100:.1f}%"
        ]))

    print("\nMass Accounting\n" + "-" * 50)
    for v in mass_props.values():
        v.substitute_solution(sol)

    aero = {
        k: s(v)
        for k, v in aero.items() if not isinstance(v, list)
    }

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p
    from aerosandbox.tools.string_formatting import eng_string

    ##### Section: Printout
    print_title = lambda s: print(s.upper().join(["*" * 20] * 2))

    def fmt(x):
        return f"{s(x):.6g}"


    print_title("Outputs")
    for k, v in {
        "Wing Span": f"{fmt(wing_span)} meters",
        "Wing Root Chord": f"{fmt(wing_root_chord)} meters",
        "mass_TOGW": f"{fmt(mass_props_TOGW.mass)} kg",
        "Propeller Diameter": f"{fmt(propeller_diameter)} m ({fmt(propeller_diameter)} meters)",
        "Average Cruise L/D": fmt(avg_cruise_LD),
        "CG location"           : "(" + ", ".join([fmt(xyz) for xyz in mass_props_TOGW.xyz_cg]) + ") m",
    }.items():
        print(f"{k.rjust(25)} = {v}")

    fmtpow = lambda x: fmt(x) + " W"

    print_title("Powers")
    for k, v in {
        "max_power_in"        : fmtpow(max_power_in),
        "max_power_out": fmtpow(max_power_out_propulsion),
        "battery_total_energy": fmtpow(battery_total_energy),
        "payload_power": fmtpow(payload_power),
    }.items():
        print(f"{k.rjust(25)} = {v}")

    print_title("Mass props")
    for k, v in mass_props.items():
        print(f"{k.rjust(25)} = {v.mass:.3f} kg ({v.mass / u.oz:.2f} oz)")

    if make_plots:
        ##### Section: Geometry
        airplane.draw_three_view(show=False)
        p.show_plot(tight_layout=False, savefig="figures/three_view.png")

        ##### Section: Mass Budget
        # fig, ax = plt.subplots(figsize=(12, 5), subplot_kw=dict(aspect="equal"), dpi=300)
        #
        # name_remaps = {
        #     **{
        #         k: k.replace("_", " ").title()
        #         for k in mass_props.keys()
        #     },
        #     "apu"                         : "APU",
        #     "wing"                        : "Wing Structure",
        #     "hstab"                       : "H-Stab Structure",
        #     "vstab"                       : "V-Stab Structure",
        #     "fuselage"                    : "Fuselage Structure",
        #     "payload_proportional_weights": "FAs, Food, Galleys, Lavatories,\nLuggage Hold, Doors, Lighting,\nAir Cond., Entertainment"
        # }
        #
        # p.pie(
        #     values=[
        #         v.mass
        #         for v in mass_props.values()
        #     ],
        #     names=[
        #         n if n not in name_remaps.keys() else name_remaps[n]
        #         for n in mass_props.keys()
        #     ],
        #     center_text=f"$\\bf{{Mass\\ Budget}}$\nTOGW: {s(mass_props_TOGW.mass):.3f} kg",
        #     label_format=lambda name, value, percentage: f"{name}, {value:.3f} kg, {percentage:.1f}%",
        #     startangle=110,
        #     arm_length=30,
        #     arm_radius=20,
        #     y_max_labels=1.1
        # )
        # p.show_plot(savefig="figures/mass_budget.png")