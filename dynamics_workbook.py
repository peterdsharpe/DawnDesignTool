import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools import units as u
import matplotlib.pyplot as plt
import casadi as cas
from aerosandbox.library import winds as lib_winds


from cessna152 import airplane

### Initialize the problem
opti = asb.Opti()

# define the area to cover
sample_area_x = 2000  # meters, the height of the area the aircraft must sample
sample_area_y = 2000  # meters, the width of the area the aircraft must sample
sample_area_radius = 2000

# define the location and day of year
latitude = -73  # degrees, the location the sizing occurs
day_of_year = 60  # Julian day, the day of the year the sizing occurs
wind_direction = 0

time = np.linspace(
    0,
    500,
    100
)
N = np.length(time)
num_laps = opti.variable(
    init_guess=1,
    lower_bound=0)

### Create a dynamics instance
mass_total = 1151.8 * u.lbm
dyn = asb.DynamicsPointMass2DCartesian(
    mass_props=asb.MassProperties(mass=mass_total),
    x_e=opti.variable(
        init_guess=np.linspace(0, sample_area_radius, N),
    ),
    z_e=opti.variable(init_guess=-10000,
                      n_vars=N,),
    u_e=opti.variable(init_guess=50,
                        n_vars=N,
                        lower_bound=-100,
                        upper_bound=100
                      ),
    w_e=opti.variable(init_guess=0,
                        n_vars=N,
                        lower_bound=-100,
                        upper_bound=100
                        ),
    alpha=opti.variable(init_guess=5,
                        n_vars=N,
                        lower_bound=-12,
                        upper_bound=12
                        ),
)

### add wind
def wind_speed_func(alt):
    day_array = np.full(shape=alt.shape[0], fill_value=1) * day_of_year
    latitude_array = np.full(shape=alt.shape[0], fill_value=1) * latitude
    speed_func = lib_winds.wind_speed_world_95(alt, latitude_array, day_array)
    return speed_func

wind_speed = wind_speed_func(-dyn.z_e)


### Constrain the tracks for each lap
distance = opti.variable(
    init_guess=np.linspace(0, 100000, N),
    lower_bound=0,
)
groundspeed = dyn.u_e - wind_speed
opti.constrain_derivative(
    variable=distance, with_respect_to=time,
    derivative=groundspeed
)
# turn_1_length = opti.variable(init_guess=2000, scale=100, lower_bound=0)
# turn_2_length = opti.variable(init_guess=2000, scale=100, lower_bound=0)
# single_track_length = sample_area_x * 2 + turn_1_length + turn_2_length
circular_trajectory_length = 2 * np.pi * sample_area_radius
place_on_track = np.mod(distance, circular_trajectory_length)

angular_displacement = place_on_track / circular_trajectory_length * 360
arc_length = np.radians(angular_displacement) * sample_area_radius
vehicle_bearing = 360 - angular_displacement

# opti.subject_to([
#     dyn.x_e == groundspeed * np.cosd(vehicle_bearing),
#     dyn.y_e == groundspeed * np.sind(vehicle_bearing),
# ])

opti.subject_to(num_laps <= distance[-1] / circular_trajectory_length)

groundspeed_x = groundspeed * np.cosd(vehicle_bearing)
groundspeed_y = groundspeed * np.sind(vehicle_bearing)
windspeed_x = wind_speed * np.cosd(wind_direction)
windspeed_y = wind_speed * np.sind(wind_direction)
airspeed_x = groundspeed_x - windspeed_x
airspeed_y = groundspeed_y - windspeed_y
vehicle_heading = np.arctan2d(airspeed_y, airspeed_x)

# # Constrain the initial state
opti.subject_to([
    dyn.x_e[0] == 0,
    # dyn.speed[0] == 67 * u.knot,
    # # dyn.track[0] == 0,
    # dyn.bank[0] == 0,
    distance[0] == 0,
    dyn.z_e[0] == -10000,
])


# # Constrain the final state

# Add some constraints on rate of change of inputs (alpha and bank angle)
pitch_rate = np.diff(dyn.alpha) / np.diff(time)  # deg/sec
opti.subject_to([
    pitch_rate > -5,
    pitch_rate < 5,
])

### Add in forces
dyn.add_gravity_force(g=9.81)
thrust = opti.variable(
    init_guess=1000,
    lower_bound=0,
    n_vars=N
)

aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=dyn.op_point,
    xyz_ref=airplane.xyz_ref,
    include_wave_drag=False,
).run_with_stability_derivatives(
    alpha=True,
    beta=True,
    p=True,
    q=True,
    r=True
)

dyn.add_force(
    Fx=np.cosd(dyn.alpha) * thrust,
    Fz=np.sind(dyn.alpha) * thrust,
    axes="earth"
)
dyn.add_force(
    *aero["F_w"],
    axes="wind"
)


### Constrain the altitude to be above ground at all times
opti.subject_to(
    dyn.altitude / 1000 > 0
)

### Finalize the problem
net_accel_x = opti.variable(
    n_vars=N,
    init_guess=0,
    scale=1e-4,
)
net_accel_z = opti.variable(
    n_vars=N,
    init_guess=0,
    scale=1e-5,
)
opti.constrain_derivative(
    variable=dyn.x_e, with_respect_to=time,
    derivative=(dyn.u_e-wind_speed)
)
opti.constrain_derivative(
    variable=dyn.z_e, with_respect_to=time,
    derivative=dyn.w_e,
)
opti.constrain_derivative(
    variable=dyn.u_e, with_respect_to=time,
    derivative=net_accel_x,
)
opti.constrain_derivative(
    variable=dyn.w_e, with_respect_to=time,
    derivative=net_accel_z,
)
opti.subject_to([
    net_accel_x * mass_total / 1e1 == dyn.Fx_e / 1e1,
    net_accel_z * mass_total / 1e2 == dyn.Fz_e / 1e2,
])
opti.minimize(-num_laps)  #

### Solve it
try:
    sol = opti.solve(
        max_iter=50000,
        options={
            "ipopt.max_cpu_time": 10000
        }
    )
    opti.set_initial_from_sol(sol)
    airplane = sol(airplane)
    dyn = sol(dyn)

except RuntimeError as e:
    print(e)
    sol = asb.OptiSol(opti=opti, cas_optisol=opti.debug)
    dyn = sol(dyn)


# def plot_trajectory(x, y):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(x, y, label='Trajectory')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.legend()
#     plt.show()
#
# # Call the function to plot the trajectory
# plot_trajectory(dyn.x_e, dyn.y_e)

def plot_traj_circ(angular_displacement, radius):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(np.radians(angular_displacement), radius, label='Trajectory')
    ax.set_xlabel('Theta')
    ax.set_ylabel('R')
    ax.legend()
    plt.show()

plot_traj_circ(sol(angular_displacement), sol(sample_area_radius) * np.ones(N))
