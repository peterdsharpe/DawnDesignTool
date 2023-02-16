import aerosandbox as asb
from aerosandbox.library import winds as lib_winds

##### Section: Initialize Optimization
opti = asb.Opti(
    freeze_style='float',
    # variable_categories_to_freeze='design'
)
des = dict(category="design")
tra = dict(category="trajectory")

##### optimization assumptions
make_plots = False

##### Mission Operating Parameters
latitude = opti.parameter(value=-75)  # degrees, the location the sizing occurs
day_of_year = opti.parameter(vlaue=45)  # Julian day, the day of the year the sizing occurs
mission_length = opti.parameter(value=45)  # days, the length of the mission without landing to download data
strat_offset_value = opti.parameter(value=1000)  # meters, margin above the stratosphere height the aircraft is required to stay above
min_cruise_altitude = lib_winds.tropopause_altitude(latitude, day_of_year) + strat_offset_value
climb_opt = False  # are we optimizing for the climb as well?
hold_cruise_altitude = True # must we hold the cruise altitude (True) or can we altitude cycle (False)?

##### Trajectory Parameters
sample_area_height = opti.parameter(value=150000)  # meters, the height of the area the aircraft must sample
sample_area_width = opti.parameter(value=100000)  # meters, the width of the area the aircraft must sample
required_headway_per_day = opti.parameter(value=0)  # meters, the minimum distance the aircraft must cover in the sizing day
trajectory = 1 # value to determine the particular trajectory

##### Aircraft Parameters
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

