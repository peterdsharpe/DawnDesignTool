import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np

style.use("ggplot")


def mass_motor_electric(
        max_power,
        kv_rpm_volt=1000,  # This is in rpm/volt, not rads/sec/volt!
        voltage=20,
        method="astroflight"
):
    """
    Estimates the mass of a brushless DC electric motor.
    Curve fit to scraped Hobbyking BLDC motor data as of 2/24/2020.
    Estimated range of validity: 50 < max_power < 10000
    :param max_power: maximum power [W]
    :param kv: Voltage constant of the motor, measured in rpm/volt, not rads/sec/volt! [rpm/volt]
    :param voltage: Operating voltage of the motor [V]
    :param method: method to use. "burton", "hobbyking", or "astroflight" (increasing level of detail).
    Burton source: https://dspace.mit.edu/handle/1721.1/112414
    Hobbyking source: C:\Projects\GitHub\MotorScraper, https://github.com/austinstover/MotorScraper
    Astroflight source: Gates, et. al., "Combined Trajectory, Propulsion, and Battery Mass Optimization for Solar-Regen..."
        https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=3932&context=facpub
        Validity claimed from 1.5 kW to 15 kW, kv from 32 to 1355.
    :return: estimated motor mass [kg]
    """
    if method == "burton":
        return max_power / 4128  # Less sophisticated model. 95% CI (3992, 4263), R^2 = 0.866
    elif method == "hobbyking":
        return 10 ** (0.8205 * np.log10(max_power) - 3.155)  # More sophisticated model
    elif method == "astroflight":
        max_current = max_power / voltage
        return 2.464 * max_current / kv_rpm_volt + 0.368  # Even more sophisticated model


def mass_gearbox(
        power,
        rpm_in,
        rpm_out,
):
    """
    Estimates the mass of a gearbox.

    Based on data from NASA/TM-2009-215680, available here:
        https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20090042817.pdf

    R^2 = 0.92 to the data.

    To quote this document:
        "The correlation was developed based on actual weight
        data from over fifty rotorcrafts, tiltrotors, and turboprop
        aircraft."

    Data fits in the NASA document were thrown out and refitted to extrapolate more sensibly; see:
        C:\Projects\GitHub\AeroSandbox\studies\GearboxMassFits
    :param power: Shaft power through the gearbox [W]
    :param rpm_in: RPM of the input to the gearbox [rpm]
    :param rpm_out: RPM of the output of the gearbox [rpm]
    :return: Estimated mass of the gearbox [kg]
    """
    power_hp = power / 745.7

    beta = (power_hp / rpm_out) ** 0.75 * (rpm_in / rpm_out) ** 0.15
    # Beta is a parametric value that tends to collapse the masses of gearboxes onto a line.
    # Data fit is considered tightly valid for gearboxes with 1 < beta < 100. Sensible extrapolations are made beyond that.

    p1 = 1.0445171124733774
    p2 = 2.0083615496306910

    mass_lb = 10 ** (p1 * np.log10(beta) + p2)

    mass = mass_lb / 2.20462262185

    return mass


voltage = 240  # V
kv_des = 4
rpm_des = kv_des * voltage  # rpm
power = 1200  # W
torque_des = power / (rpm_des * np.pi / 30)  # Nm

plt.plot(1, mass_motor_electric(power, kv_des, voltage), ".", color="blue", zorder=5, label="No Gearbox: Total")

gear_ratios = np.logspace(0, 2, 50)
motor_masses = np.array([
    mass_motor_electric(power, kv_des * gear_ratio, voltage)
    for gear_ratio in gear_ratios
])
gearbox_masses = np.array([
    2 * mass_gearbox(power, rpm_des * gear_ratio, rpm_des)
    # the 2 is to roughly calibrate it to data that I found through quick google searches - again, super preliminary haha
    for gear_ratio in gear_ratios
])
system_masses = motor_masses + gearbox_masses
plt.semilogx(gear_ratios, system_masses, "-", color="green", label="With Gearbox: Total")
plt.plot(gear_ratios, motor_masses, ":", color="green", label="With Gearbox: Motor")
plt.plot(gear_ratios, gearbox_masses, "--", color="green", label="With Gearbox: Gearbox")
plt.grid(True, which="both")
plt.xlabel(r"Gear Ratio")
plt.ylabel(r"Total Motor System Mass (per motor) [kg]")
plt.ylim(0, 4)
plt.title("Is Gearing Worth It?\n(Brief study, needs more validation)")
plt.tight_layout()
plt.legend()
plt.show()
