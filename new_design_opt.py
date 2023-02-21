import aerosandbox as asb
from aerosandbox.library import winds as lib_winds
import aerosandbox.numpy as np
import pathlib
from aerosandbox.modeling.interpolation import InterpolatedModel
from design_opt_utilities.fuselage import make_payload_pod
import aerosandbox.library.mass_structural as mass_lib


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
payload_pod_length = 2.0  # meters
payload_pod_diameter = 0.5  # meters
payload_pod_y_offset = 1.5 # meters

payload_pod = make_payload_pod( # TODO ask peter if there's a better way to make this an aero shape
    boom_length=payload_pod_length,
    nose_length=0.5,
    tail_length=1,
    fuse_diameter=payload_pod_diameter,
).translate(np.array[0, 0, -payload_pod_y_offset])

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
right_boom = right_boom.translate(np.array([0, boom_offset, 0]))
left_boom = right_boom.translate(np.array([0, -2 * boom_offset, 0]))


# tail section
tail_airfoil = asb.Airfoil("naca0008")
tail_airfoil.generate_polars(
    cache_filename="cache/naca0008.json",
    include_compressibility_effects=False,
    make_symmetric_polars=True
)

#  vstab
vstab_span = opti.variable(
    init_guess=3,
    scale=2,
    lower_bound=0.1
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
    **tra
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
    np.array([center_boom_length - vstab_chord * 0.75, 0, -vstab_span / 2 + vstab_span * 0.15]))

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
    **tra
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
).translate(np.array([center_boom_length - vstab_chord * 0.75 - center_hstab_chord, 0, 0.1]))

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
    n_vars=n_timesteps,
    init_guess=-3,
    scale=1,
    upper_bound=30,
    lower_bound=-30,
    freeze=True,
    **tra
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
).translate(np.array([outboard_boom_length - outboard_hstab_chord * 0.75, boom_offset, 0.1]))
left_hstab = right_hstab.translate([0, -boom_offset * 2, 0])

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

n_propellers = opti.parameter(value=4) # TODO reconsider 2 or 4

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

wing_mass_props = asb.MassProperties(
    mass=wing_mass,
    x_cg=0.40 * wing_root_chord
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

center_hstab_mass_props = asb.MassProperties(
    mass=mass_hstab(center_hstab, n_ribs_center_hstab)
)

n_ribs_outboard_hstab = opti.variable(
    init_guess=40,
    scale=40,
    lower_bound=0,
    **des
)

right_hstab_mass_props = asb.MassProperties(
    mass=mass_hstab(right_hstab, n_ribs_outboard_hstab)
)

left_hstab_mass_props = asb.MassProperties(
    mass=mass_hstab(left_hstab, n_ribs_outboard_hstab)
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

vstab_mass_props = asb.MassProperties(
    mass=mass_hstab(vstab, n_ribs_vstab)
)

### boom mass accounting
center_boom_mass_props = asb.MassProperties(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=center_boom_length-wing_x_quarter_chord,
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=center_hstab.area() + vstab.area()
    )
)
left_boom_mass_props = asb.MassProperties(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=right_hstab.area()
    )
)
right_boom_mass_props = asb.MassProperties(
    mass=mass_lib.mass_hpa_tail_boom(
        length_tail_boom=outboard_boom_length - wing_x_quarter_chord,  # support up to the quarter-chord
        dynamic_pressure_at_manuever_speed=q_ne,
        mean_tail_surface_area=left_hstab.area()
    )
)

