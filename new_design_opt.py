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
required_revisit_rate = opti.parameter(value=0) # How many times must the aircraft fully cover the sample area in the sizing day?
swath_overlap = opti.parameter(value=0.1) # What fraction of the adjacent swaths must overlap? Typically ranges from 0.1 to 0.5

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

##### Instrument Parameters
required_resolution = opti.parameter(value=2)  # meters from conversation with Brent on 2/18/22
required_snr = opti.parameter(value=6)  # 6 dB min and 20 dB ideally from conversation w Brent on 2/18/22
center_wavelength = opti.parameter(value=0.024)  # meters given from Brent based on the properties of the ice sampled by the radar
scattering_cross_sec_db = opti.parameter(value=-10)  # meters ** 2 ranges from -20 to 0 db according to Charles in 4/19/22 email
radar_length = opti.parameter(value=1)  # meters, given from existing Gamma Remote Sensing instrument
radar_width = opti.parameter(value=0.3)  # meters, given from existing Gamma Remote Sensing instrument
look_angle = opti.parameter(value=45)  # degrees
