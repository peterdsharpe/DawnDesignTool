import aerosandbox.numpy as np


def solar_flux_outside_atmosphere_normal(day_of_year):
    """
    Normal solar flux at the top of the atmosphere (variation due to orbital eccentricity)
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: Solar flux [W/m^2]
    """
    # Space effects
    # # Source: https://www.itacanet.org/the-sun-as-a-source-of-energy/part-2-solar-energy-reaching-the-earths-surface/#2.1.-The-Solar-Constant
    # solar_flux_outside_atmosphere_normal = 1367 * (1 + 0.034 * cas.cos(2 * cas.pi * (day_of_year / 365.25)))  # W/m^2; variation due to orbital eccentricity
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/solar-radiation-outside-the-earths-atmosphere
    return 1366 * (
            1 + 0.033 * np.cosd(360 * (day_of_year - 2) / 365))  # W/m^2; variation due to orbital eccentricity


def declination_angle(day_of_year):
    """
    Declination angle, in degrees, as a func. of day of year. (Seasonality)
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: Declination angle [deg]
    """
    # Declination (seasonality)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/declination-angle
    return -23.45 * np.cosd(360 / 365 * (day_of_year + 10))  # in degrees


def solar_elevation_angle(latitude, day_of_year, time):
    """
    Elevation angle of the sun [degrees] for a local observer.

    :param latitude: Latitude [degrees]

    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)

    :param time: Time after local solar noon [seconds]

    :return: Solar elevation angle [degrees] (angle between horizon and sun). Returns negative values if the sun is
    below the horizon.
    """

    # Solar elevation angle (including seasonality, latitude, and time of day)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/elevation-angle
    declination = declination_angle(day_of_year)

    solar_elevation_angle = np.arcsind(
        np.sind(declination) * np.sind(latitude) +
        np.cosd(declination) * np.cosd(latitude) * np.cosd(time / 86400 * 360)
    )  # in degrees
    return solar_elevation_angle


def solar_azimuth_angle(latitude, day_of_year, time):
    """
    Zenith angle of the sun [degrees] for a local observer.
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time after local solar noon [seconds]
    :return: Solar azimuth angle [degrees] (the compass direction from which the sunlight is coming).
    """

    # Solar azimuth angle (including seasonality, latitude, and time of day)
    # Source: https://www.pveducation.org/pvcdrom/properties-of-sunlight/azimuth-angle
    declination = declination_angle(day_of_year)
    sdec = np.sind(declination)
    cdec = np.cosd(declination)
    slat = np.sind(latitude)
    clat = np.cosd(latitude)
    cos_time = np.cosd(time / 86400 * 360)

    elevation = solar_elevation_angle(latitude, day_of_year, time)
    cele = np.cosd(elevation)

    solar_azimuth_angle = np.where(
        np.mod(time, 86400) > 43200,
        np.arccosd(
            (sdec * clat - cdec * slat * cos_time) / cele
        ),
        360 - np.arccosd(
            (sdec * clat - cdec * slat * cos_time) / cele
        )
    )

    return solar_azimuth_angle


def incidence_angle_function(
        latitude,
        day_of_year,
        time,
        scattering=True,
        panel_orientation="horizontal"
):
    """
    What is the fraction of insolation that a horizontal surface will receive as a function of sun position in the sky?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    """
    # Old description:
    # To first-order, this is true. In class, Kevin Uleck claimed that you have higher-than-cosine losses at extreme angles,
    # since you get reflection losses. However, an experiment by Sharma appears to not reproduce this finding, showing only a
    # 0.4-percentage-point drop in cell efficiency from 0 to 60 degrees. So, for now, we'll just say it's a cosine loss.
    # Sharma: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6611928/

    elevation_angle = solar_elevation_angle(latitude, day_of_year, time)
    sun_above_horizon_multiplier = np.where(elevation_angle > 0, 1, 0)

    if panel_orientation == "horizontal":
        theta = 90 - elevation_angle  # Angle between panel normal and the sun, in degrees
    elif panel_orientation == "vertical":
        theta = elevation_angle
    else:
        raise ValueError("Bad value of `panel_orientation`!")

    cosine_factor = np.cosd(theta)

    illumination_factor = cosine_factor

    if scattering:
        illumination_factor = illumination_factor * scattering_factor(elevation_angle)

    illumination_factor = illumination_factor * sun_above_horizon_multiplier
    return illumination_factor


def incidence_angle_function_vert(latitude, day_of_year, time, scattering=True):
    """
    What is the fraction of insolation that a horizontal surface will receive as a function of sun position in the sky?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    """
    # Old description:
    # To first-order, this is true. In class, Kevin Uleck claimed that you have higher-than-cosine losses at extreme angles,
    # since you get reflection losses. However, an experiment by Sharma appears to not reproduce this finding, showing only a
    # 0.4-percentage-point drop in cell efficiency from 0 to 60 degrees. So, for now, we'll just say it's a cosine loss.
    # Sharma: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6611928/

    elevation_angle = solar_elevation_angle(latitude, day_of_year, time)
    theta = 90 - elevation_angle - 90  # Angle between panel normal and the sun, in degrees

    cosine_factor = np.cosd(theta)

    if not scattering:
        return cosine_factor
    else:
        return cosine_factor * scattering_factor_vert(elevation_angle)

def incidence_angle_function_new(latitude, day_of_year, time, panel_heading, panel_angle, scattering=True):
    """
    This website will be useful for accounting for direction of the vertical surface
    https://www.pveducation.org/pvcdrom/properties-of-sunlight/arbitrary-orientation-and-tilt
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param panel_heading: the directionality of the solar panel (0 degrees if pointing North and 180 if South)
    :param panel_angle: the degrees of horizontal the array is mounted (0 if hoirzontal and 90 if vertical)
    :param scattering: Boolean: include scattering effects at very low angles?
    """
    elevation_angle = solar_elevation_angle(latitude, day_of_year, time)
    azimuth_angle = solar_azimuth_angle(latitude, day_of_year, time)
    cosine_factor = np.cosd(elevation_angle) * np.sind(panel_angle) * np.cosd(panel_heading - azimuth_angle) + np.sind(elevation_angle) * np.cosd(panel_angle)

    if not scattering:
        incidence = cosine_factor
    else:
        incidence = cosine_factor * scattering_factor(elevation_angle)
    incidence = np.fmax(incidence, 0)
    return incidence


def scattering_factor(elevation_angle):
    """
    Calculates a scattering factor (a factor that gives losses due to atmospheric scattering at low elevation angles).
    Source: AeroSandbox/studies/SolarPanelScattering
    :param elevation_angle: Angle between the horizon and the sun [degrees]
    :return: Fraction of the light that is not lost to scattering.
    """
    elevation_angle = np.clip(elevation_angle, 0, 90)
    theta = 90 - elevation_angle  # Angle between panel normal and the sun, in degrees

    # # Model 1
    # c = (
    #     0.27891510500505767300438719757949,
    #     -0.015994330894744987481281839336589,
    #     -19.707332432605799255043166340329,
    #     -0.66260979582573353852126274432521
    # )
    # scattering_factor = c[0] + c[3] * theta_rad + cas.exp(
    #     c[1] * (
    #             cas.tan(theta_rad) + c[2] * theta_rad
    #     )
    # )

    # Model 2
    c = (
        -0.04636,
        -0.3171
    )
    scattering_factor = np.exp(
        c[0] * (
                np.tand(theta * 0.999) + c[1] * np.radians(theta)
        )
    )

    # # Model 3
    # p1 = -21.74
    # p2 = 282.6
    # p3 = -1538
    # p4 = 1786
    # q1 = -923.2
    # q2 = 1456
    # x = theta_rad
    # scattering_factor = ((p1*x**3 + p2*x**2 + p3*x + p4) /
    #            (x**2 + q1*x + q2))

    # Keep this:
    # scattering_factor = cas.fmin(cas.fmax(scattering_factor, 0), 1)
    return scattering_factor

def scattering_factor_vert(elevation_angle):
    """
    Calculates a scattering factor (a factor that gives losses due to atmospheric scattering at low elevation angles).
    Source: AeroSandbox/studies/SolarPanelScattering
    :param elevation_angle: Angle between the horizon and the sun [degrees]
    :return: Fraction of the light that is not lost to scattering.
    """
    elevation_angle = np.clip(elevation_angle, 0, 90)
    theta = 90 - elevation_angle - 90 # Angle between panel normal and the sun, in degrees

    # # Model 1
    # c = (
    #     0.27891510500505767300438719757949,
    #     -0.015994330894744987481281839336589,
    #     -19.707332432605799255043166340329,
    #     -0.66260979582573353852126274432521
    # )
    # scattering_factor = c[0] + c[3] * theta_rad + cas.exp(
    #     c[1] * (
    #             cas.tan(theta_rad) + c[2] * theta_rad
    #     )
    # )

    # Model 2
    c = (
        -0.04636,
        -0.3171
    )
    scattering_factor = np.exp(
        c[0] * (
                np.tand(theta * 0.999) + c[1] * np.radians(theta)
        )
    )

    # # Model 3
    # p1 = -21.74
    # p2 = 282.6
    # p3 = -1538
    # p4 = 1786
    # q1 = -923.2
    # q2 = 1456
    # x = theta_rad
    # scattering_factor = ((p1*x**3 + p2*x**2 + p3*x + p4) /
    #            (x**2 + q1*x + q2))

    # Keep this:
    # scattering_factor = cas.fmin(cas.fmax(scattering_factor, 0), 1)
    return scattering_factor


def solar_flux_on_horizontal(
        latitude: float,
        day_of_year: float,
        time: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a horizontal surface for some given conditions?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    return (
            solar_flux_outside_atmosphere_normal(day_of_year) *
            incidence_angle_function(latitude, day_of_year, time, scattering)
    )

def solar_flux_new(
        latitude: float,
        day_of_year: float,
        time: float,
        panel_heading: float,
        panel_angle: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a horizontal surface for some given conditions?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    return (
            solar_flux_outside_atmosphere_normal(day_of_year) *
            incidence_angle_function_new(latitude, day_of_year, time, panel_heading, panel_angle)
    )

def solar_flux_on_vertical(
        latitude: float,
        day_of_year: float,
        time: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a vertical surface for some given conditions?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    S_horz = solar_flux_outside_atmosphere_normal(day_of_year) * (incidence_angle_function(latitude, day_of_year, time, scattering))
    S_vert =  S_horz * np.sind(solar_elevation_angle(latitude, day_of_year, time) + 90) / np.fmax(np.sind(solar_elevation_angle(latitude, day_of_year, time)), 0.0000001)
    return S_vert

def solar_flux_on_angle(
        angle: float,
        latitude: float,
        day_of_year: float,
        time: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a surface at a given angle for some given conditions?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    S_horz = solar_flux_outside_atmosphere_normal(day_of_year) * (incidence_angle_function(latitude, day_of_year, time, scattering))
    S_angle =  S_horz * np.sind(solar_elevation_angle(latitude, day_of_year, time) + 90 - angle) / np.fmax(np.sind(solar_elevation_angle(latitude, day_of_year, time)), 0.0000001)
    return S_angle

def solar_flux_circular_flight_path(
        latitude: float,
        day_of_year: float,
        time: float,
        panel_angle: float,
        panel_heading: float,
        scattering: bool = True
) -> float:
    """
    What is the solar flux on a surface at a given angle for a circular flight path given the radius?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param panel_angle: the degrees of horizontal the array is mounted (0 if hoirzontal and 90 if vertical)
    :param panel_heading: the directionality of the solar panel (0 degrees if pointing North and 180 if South)
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """

    solar_flux_on_panel = solar_flux_outside_atmosphere_normal(day_of_year) * incidence_angle_function_new(
        latitude, day_of_year, time, panel_heading, panel_angle)
    return solar_flux_on_panel


def peak_sun_hours_per_day_on_horizontal(latitude, day_of_year, scattering=True):
    """
    How many hours of equivalent peak sun do you get per day?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :param time: Time since (local) solar noon [seconds]
    :param scattering: Boolean: include scattering effects at very low angles?
    :return:
    """
    times = np.linspace(0, 86400, 1000)
    dt = np.diff(times)
    normalized_fluxes = (
        # solar_flux_outside_atmosphere_normal(day_of_year) *
        incidence_angle_function(latitude, day_of_year, times, scattering)
    )
    sun_hours = np.sum(
        (normalized_fluxes[1:] + normalized_fluxes[:-1]) / 2 * dt
    ) / 3600

    return sun_hours
def length_day(latitude, day_of_year):
    """
    How many hours of with incoming solar energy  per day?
    :param latitude: Latitude [degrees]
    :param day_of_year: Julian day (1 == Jan. 1, 365 == Dec. 31)
    :return: hours of no sun
    """
    times = np.linspace(0, 86400, 100)
    dt = np.diff(times)
    sun_time = 0
    for time in times:
        solar_flux = solar_flux_on_horizontal(
            latitude, day_of_year, time, scattering=True
        )
        if solar_flux > 1:
            sun_time = sun_time + (872.72727273 / 60 / 60)
    return sun_time

def mass_MPPT(
        power: float
) -> float:
    """
    Gives the estimated mass of a Maximum Power Point Tracking (MPPT) unit for solar energy
    collection. Based on regressions at AeroSandbox/studies/SolarMPPTMasses.

    Args:
        power: Power of MPPT [watts]

    Returns:
        Estimated MPPT mass [kg]
    """
    constant = 0.066343
    exponent = 0.515140
    return constant * power ** exponent


if __name__ == "__main__":

    # circular flight path function test

    import matplotlib.pyplot as plt
    ## fun integration problem
    airspeed = 30 # m/s
    radius = 50000 # m
    latitude = 60
    day_of_year = 174
    angle = 90
    day = 60 * 60 * 24
    # wind_speed = 25
    # wind_direction = 180
    # solar_flux = []
    # ground_dist_list = [0]
    # x_vector = []
    # time_step = times[2] - times[1]
    # groundspeed = 0
    # for i, time in enumerate(times):
    #     x = airspeed * time
    #     x_vector.append(x)
    #     vehicle_direction = x / (np.pi / 180) / radius + 90
    #     heading_x = airspeed * np.sind(vehicle_direction) - wind_speed * np.sind(wind_direction)
    #     heading_y = airspeed * np.cosd(vehicle_direction) - wind_speed * np.cosd(wind_direction)
    #     groundspeed = np.sqrt(heading_x ** 2 + heading_y ** 2)
    #     ground_dist = ground_dist_list[i-1] + groundspeed * time_step
    #     ground_dist_list.append(ground_dist)
    #     vehicle_heading = np.arctan2d(heading_y, heading_x)
    #     panel_heading = vehicle_heading - 90
    #     solar_flux.append(solar_flux_circular_flight_path(latitude, day_of_year, time, -angle, panel_heading) + solar_flux_circular_flight_path(latitude, day_of_year, time, angle, panel_heading) )

    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # plt.plot(times / 3600, solar_flux)
    # plt.grid(True)
    # plt.title("Solar Flux on Vertical as  Aircraft Completes a Circular Flight Path")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel(r"Solar Flux [W/m$^2$]")
    # plt.tight_layout()
    # # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    # plt.show()
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # plt.plot(times / 3600, ground_dist_list[:-1])
    # plt.grid(True)
    # plt.title("Ground Distance Covered as Aircraft Completes a Circular Flight Path")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel("Ground Distance [m]")
    # plt.tight_layout()
    # # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    # plt.show()

    #circular flight path test w optimization
    import aerosandbox as asb
    opti = asb.Opti()
    radius = 50000 # m
    latitude = 60
    day_of_year = 174
    angle = 90
    wind_speed = 20
    wind_direction = 90

    n_timesteps_per_segment = 300
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

    # initialize variables
    airspeed = opti.variable(
        n_vars=n_timesteps,
        init_guess=25,
        scale=10,
    )
    x = opti.variable(
        n_vars=n_timesteps,
        init_guess=0,
        scale=1e5,
    )
    heading_x = opti.variable(
        n_vars=n_timesteps,
        init_guess=90,
        scale=20,
    )
    heading_y = opti.variable(
        n_vars=n_timesteps,
        init_guess=180,
        scale=20,
    )
    vehicle_heading = opti.variable(
        n_vars=n_timesteps,
        init_guess=90,
        scale=20,
    )
    vehicle_direction = opti.variable(
        n_vars=n_timesteps,
        init_guess=80,
        scale=20,
    )
    groundspeed = opti.variable(
        n_vars=n_timesteps,
        init_guess=3,
        scale=1,
    )

    # constraints
    trapz = lambda x: (x[1:] + x[:-1]) / 2
    dt = np.diff(time)
    dx = np.diff(x)
    opti.subject_to([
        airspeed >= 0,
        dx / 1e4 == trapz(groundspeed) * dt / 1e4,
        x[-1] >= 10000,
        vehicle_direction == x / (np.pi / 180) / radius + 90 , # direction the vehicle must fly on to remain in the circular trajectory
        heading_x == airspeed * np.sind(vehicle_direction) - wind_speed * np.sind(wind_direction),  # x component of heading vector
        heading_y == airspeed * np.cosd(vehicle_direction) - wind_speed * np.cosd(wind_direction) , # y component of heading vector
        vehicle_heading == np.arctan2d(heading_y, heading_x), # actual directionality of the vehicle as modified by the wind speed and direction
        groundspeed ** 2 == heading_x ** 2 + heading_y ** 2, # speed of aircraft as measured from observer on the ground
        groundspeed >= 2,
        airspeed <= 40,
        airspeed >= 10,
        x > 0,
        x[0] == 0,
    ])

    panel_heading = vehicle_heading - 90  # actual directionality of the solar panel

    solar_flux_on_vertical_left = solar_flux_circular_flight_path(
        latitude, day_of_year, time, 90, panel_heading, scattering=True,
    )
    solar_flux_on_vertical_right = solar_flux_circular_flight_path(
        latitude, day_of_year, time, -90, panel_heading, scattering=True,
    )
    solar_flux = solar_flux_on_vertical_left + solar_flux_on_vertical_right
    solar_flux_total = np.sum(solar_flux_on_vertical_left + solar_flux_on_vertical_right)
    opti.minimize(-solar_flux_total)
    sol = opti.solve(
        max_iter=10000,
    )

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(opti.value(time / 3600), opti.value(solar_flux))
    plt.grid(True)
    plt.title("Solar Flux on Vertical as  Aircraft Completes a Circular Flight Path")
    plt.xlabel("Time after Solar Noon [hours]")
    plt.ylabel(r"Solar Flux [W/m$^2$]")
    plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(opti.value(time / 3600), opti.value(x))
    plt.grid(True)
    plt.title("Ground Distance Covered as Aircraft Completes a Circular Flight Path")
    plt.xlabel("Time after Solar Noon [hours]")
    plt.ylabel("Ground Distance [m]")
    plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    plt.plot(opti.value(time / 3600), opti.value(groundspeed))
    plt.grid(True)
    plt.title("Groundspeed as Aircraft Completes a Circular Flight Path")
    plt.xlabel("Time after Solar Noon [hours]")
    plt.ylabel("Groundspeed [m/s]")
    plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    plt.show()


    # Run some checks
    import plotly.graph_objects as go

    latitudes = np.linspace(0, 80, 200)
    day_of_years = np.arange(0, 365) + 1
    times = np.linspace(0, 86400, 400)
    #
    headings = np.linspace(0, 180, 400)
    #
    # Times, Latitudes = np.meshgrid(times, latitudes, indexing="ij")
    # fluxes = np.array(solar_flux_on_horizontal(Latitudes, 244, Times))
    # fig = go.Figure(
    #     data=[
    #         go.Surface(
    #             x=Times / 3600,
    #             y=Latitudes,
    #             z=fluxes,
    #         )
    #     ],
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="Time after Solar Noon [hours]"),
    #         yaxis=dict(title="Latitude [deg]"),
    #         zaxis=dict(title="Solar Flux [W/m^2]"),
    #         camera=dict(
    #             eye=dict(x=-1, y=-1, z=1)
    #         )
    #     ),
    #     title="Solar Flux on Horizontal",
    # )
    # fig.show()
    ## test new function for off horizontal solar arrays
    # latitude = 60
    # angle = 90
    # Times, Headings = np.meshgrid(times, headings, indexing="ij")
    # fluxes = np.array(solar_flux_new(latitude, 174, times, headings, angle))
    # fig = go.Figure(
    #     data=[
    #         go.Surface(
    #             x=Times / 3600,
    #             y=Headings,
    #             z=fluxes,
    #         )
    #     ],
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="Time after Solar Noon [hours]"),
    #         yaxis=dict(title="Aircraft Heading [deg]"),
    #         zaxis=dict(title="Solar Flux [W/m^2]"),
    #         camera=dict(
    #             eye=dict(x=-1, y=-1, z=1)
    #         )
    #     ),
    #     title="Solar Flux on Vertical",
    # )
    # fig.show()
    # for heading in range(np.linspace(0, 180, 10)):
    #     fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    #     # lats_to_plot = [26, 49]
    #     lats_to_plot = np.linspace(0, 90, 7)
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(lats_to_plot)))[::-1]
    #     [
    #         plt.plot(
    #             times / 3600,
    #             solar_flux_on_new(lats_to_plot[i], 173, times),
    #             label="%iN Latitude" % lats_to_plot[i],
    #             color=colors[i],
    #             linewidth=3
    #         ) for i in range(len(lats_to_plot))
    #     ]
    #     plt.grid(True)
    #     plt.legend()
    #     plt.title("Solar Flux on a Vertical Surface (Summer Solstice)")
    #     plt.xlabel("Time after Solar Noon [hours]")
    #     plt.ylabel(r"Solar Flux [W/m$^2$]")
    #     plt.tight_layout()
    #     # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_vertical")
    #     plt.show()
    #
    # fig = go.Figure(
    #     data=[
    #         go.Contour(
    #             z=fluxes.T,
    #             x=times/3600,
    #             y=latitudes,
    #             colorbar=dict(
    #                 title="Solar Flux [W/m^2]"
    #             ),
    #             colorscale="Viridis",
    #         )
    #     ]
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="Hours after Solar Noon [hours]"),
    #         yaxis=dict(title="Latitude [deg]"),
    #     ),
    #     title="Solar Flux on Horizontal",
    #     xaxis_title="Time after Solar Noon [hours]",
    #     yaxis_title="Latitude [deg]",
    # )
    # fig.show()

    import matplotlib.pyplot as plt
    import seaborn as sns
    #
    # sns.set(font_scale=1)
    # latitudes = np.linspace(0, 80, 200)
    # day_of_years = np.arange(0, 365) + 1
    # times = np.linspace(0, 86400, 400)
    # latitude = 60
    # angle = 90
    # headings = np.linspace(0, 180, 400)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # headings_to_plot = np.linspace(0, 180, 5)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(headings_to_plot)))[::-1]
    # [
    #     plt.plot(
    #         times / 3600,
    #         solar_flux_new(latitude,173, times, headings_to_plot[i], angle=0),
    #         label="%iN Headings" % headings_to_plot[i],
    #         color=colors[i],
    #         linewidth=3
    #     ) for i in range(len(headings_to_plot))
    # ]
    # plt.grid(True)
    # plt.legend()
    # plt.title("Solar Flux on a Horizontal Surface (Summer Solstice)")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel(r"Solar Flux [W/m$^2$]")
    # plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    # plt.show()
    #
    #
    sns.set(font_scale=1)

    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # lats_to_plot = [26, 49]
    # lats_to_plot = np.linspace(0, 90, 7)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(lats_to_plot)))[::-1]
    # [
    #     plt.plot(
    #         times / 3600,
    #         solar_flux_on_horizontal(lats_to_plot[i], 173, times),
    #         label="%iN Latitude" % lats_to_plot[i],
    #         color=colors[i],
    #         linewidth=3
    #     ) for i in range(len(lats_to_plot))
    # ]
    # plt.grid(True)
    # plt.legend()
    # plt.title("Solar Flux on a Horizontal Surface (Summer Solstice)")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel(r"Solar Flux [W/m$^2$]")
    # plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_horizontal")
    # plt.show()
    # # #
    # sns.set(font_scale=1)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # lats_to_plot = [26, 49]
    # lats_to_plot = np.linspace(0, 90, 7)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(lats_to_plot)))[::-1]
    # [
    #     plt.plot(
    #         times / 3600,
    #         solar_flux_on_vertical(lats_to_plot[i], 173, times),
    #         label="%iN Latitude" % lats_to_plot[i],
    #         color=colors[i],
    #         linewidth=3
    #     ) for i in range(len(lats_to_plot))
    # ]
    # plt.grid(True)
    # plt.legend()
    # plt.title("Solar Flux on a Vertical Surface (Summer Solstice)")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel(r"Solar Flux [W/m$^2$]")
    # plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_vertical")
    # plt.show()
    #
    # sns.set(font_scale=1)
    #
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # lats_to_plot = [26, 49]
    # lats_to_plot = np.linspace(0, 90, 7)
    # colors = plt.cm.rainbow(np.linspace(0, 1, len(lats_to_plot)))[::-1]
    # [
    #     plt.plot(
    #         times / 3600,
    #         solar_flux_on_angle(10, lats_to_plot[i], 173, times),
    #         label="%iN Latitude" % lats_to_plot[i],
    #         color=colors[i],
    #         linewidth=3
    #     ) for i in range(len(lats_to_plot))
    # ]
    # plt.grid(True)
    # plt.legend()
    # plt.title("Solar Flux on a Angle 10 Degrees off Vertical Surface (Summer Solstice)")
    # plt.xlabel("Time after Solar Noon [hours]")
    # plt.ylabel(r"Solar Flux [W/m$^2$]")
    # plt.tight_layout()
    # plt.savefig("/Users/annickdewald/Desktop/Thesis/Photos/solar_angle")
    # plt.show()
    #
    # # Check scattering factor
    # elevations = np.linspace(-10,90,800)
    # scatter_factors = scattering_factor(elevations)
    #
    # import matplotlib.pyplot as plt
    # import matplotlib.style as style
    # import seaborn as sns
    # sns.set(font_scale=1)
    # fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=200)
    # plt.plot(elevations, scatter_factors,'.-')
    # plt.xlabel(r"Elevation Angle [deg]")
    # plt.ylabel(r"Scattering Factor")
    # plt.title(r"Scattering Factor")
    # plt.tight_layout()
    # # plt.legend()
    # # plt.savefig("C:/Users/User/Downloads/temp.svg")
    # plt.show()

