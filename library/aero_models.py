"""
Aero models to be used with the solar model.
"""
from typing import Union
import aerosandbox as asb
from aerosandbox.library import aerodynamics
from aerosandbox.atmosphere import Atmosphere
import numpy as np


def dynamic_pressure(airspeed, density):
    """
    Calculate the dynamic pressure from airspeed and density. Requires
    consistent units and will return consistent units.

    Args:
        airspeed: Airspeed in consistent units.
        density: Density in consistent units.

    Returns:
        Dynamic pressure in consistent units with inputs.
    """
    return 1 / 2 * density * airspeed**2


def mach_number(airspeed, speed_of_sound):
    """
    Calculate the mach number from airspeed and speed of sound.

    Args:
        airspeed: Airspeed in consistent units.
        speed_of_sound: Speed of sound in consistent units.

    Returns:
        Mach number
    """
    return airspeed / speed_of_sound


def compute_fuse_aerodynamics(
    fuse: asb.Fuselage,
    atmosphere: Atmosphere,
    airspeed: Union[asb.cas.MX, float],
) -> None:
    """
    Modify fuselage object in place with appropriate drag.

    Args:
        fuse: Fuselage object to be updated
        atmosphere: Atmosphere model to use
        airspeed: Airspeed in m/s, either as a plain float or casadi variable

    Returns:
        None, object modified in place.
    """
    q = dynamic_pressure(airspeed, atmosphere.density())
    fuse.Re = (atmosphere.density() / atmosphere.dynamic_viscosity() * airspeed
               * fuse.length())
    fuse.CLA = 0
    # wetted area with form factor
    fuse.CDA = (aerodynamics.Cf_flat_plate(fuse.Re) * fuse.area_wetted() * 1.2)

    fuse.lift = fuse.CLA * q  # per fuse
    fuse.drag = fuse.CDA * q  # per fuse


def compute_wing_aerodynamics(
    surface: asb.Wing,
    atmosphere: Atmosphere,
    airspeed: Union[asb.cas.MX, float],
    alpha,
    incidence_angle: float = 0,
    is_horizontal_surface: bool = True,
) -> None:
    """
    Modify the surface object in place with appropriate drag.

    Args:
        surface: Surface object to be updated
        atmosphere: Atmosphere model to use
        airspeed: Airspeed in m/s, either as plain float or casadi variable
        alpha: angle of attack in degrees
        incidence_angle: Incidence angle of surface in degrees
        is_horizontal_surface: Flag for indicating a horizontal surface
                               vs vertical

    Returns:
        Nothing, surface modified in place.
    """
    surface.alpha_eff = incidence_angle + surface.mean_twist_angle()
    if is_horizontal_surface:
        surface.alpha_eff += alpha

    q = dynamic_pressure(airspeed, atmosphere.density())
    mach = mach_number(airspeed, atmosphere.speed_of_sound())

    surface.Re = (atmosphere.density() / atmosphere.dynamic_viscosity()
                  * airspeed * surface.mean_geometric_chord())
    surface.airfoil = surface.xsecs[0].airfoil

    try:
        surface.Cl_inc = surface.airfoil.CL_function({
            "alpha": surface.alpha_eff,
            "reynolds": np.log(surface.Re)
        })  # Incompressible 2D lift coefficient
        surface.CL = surface.Cl_inc * aerodynamics.CL_over_Cl(
            surface.aspect_ratio(), mach=mach, sweep=surface.mean_sweep_angle(
            ))  # Compressible 3D lift coefficient
        surface.lift = surface.CL * q * surface.area()

        surface.Cd_profile = np.exp(
            surface.airfoil.CD_function({
                "alpha": surface.alpha_eff,
                "reynolds": np.log(surface.Re)
            }))
        surface.drag_profile = surface.Cd_profile * q * surface.area()

        surface.oswalds_efficiency = aerodynamics.oswalds_efficiency(
            taper_ratio=surface.taper_ratio(),
            aspect_ratio=surface.aspect_ratio(),
            sweep=surface.mean_sweep_angle(),
        )
        surface.drag_induced = aerodynamics.induced_drag(
            lift=surface.lift,
            span=surface.span(),
            dynamic_pressure=q,
            oswalds_efficiency=surface.oswalds_efficiency,
        )

        surface.drag = surface.drag_profile + surface.drag_induced

        surface.Cm_inc = surface.airfoil.CM_function({
            "alpha": surface.alpha_eff,
            "reynolds": np.log(surface.Re)
        })  # Incompressible 2D moment coefficient
        surface.CM = surface.Cm_inc * aerodynamics.CL_over_Cl(
            surface.aspect_ratio(), mach=mach, sweep=surface.mean_sweep_angle(
            ))  # Compressible 3D moment coefficient
        surface.moment = (surface.CM * q * surface.area()
                          * surface.mean_geometric_chord())
    except TypeError:
        surface.Cl_inc = surface.airfoil.CL_function(
            surface.alpha_eff, surface.Re, 0,
            0)  # Incompressible 2D lift coefficient
        surface.CL = surface.Cl_inc * aerodynamics.CL_over_Cl(
            surface.aspect_ratio(), mach=mach, sweep=surface.mean_sweep_angle(
            ))  # Compressible 3D lift coefficient
        surface.lift = surface.CL * q * surface.area()

        surface.Cd_profile = surface.airfoil.CD_function(
            surface.alpha_eff, surface.Re, mach, 0)
        surface.drag_profile = surface.Cd_profile * q * surface.area()

        surface.oswalds_efficiency = aerodynamics.oswalds_efficiency(
            taper_ratio=surface.taper_ratio(),
            aspect_ratio=surface.aspect_ratio(),
            sweep=surface.mean_sweep_angle(),
        )
        surface.drag_induced = aerodynamics.induced_drag(
            lift=surface.lift,
            span=surface.span(),
            dynamic_pressure=q,
            oswalds_efficiency=surface.oswalds_efficiency,
        )

        surface.drag = surface.drag_profile + surface.drag_induced

        surface.Cm_inc = surface.airfoil.CM_function(
            surface.alpha_eff, surface.Re, 0,
            0)  # Incompressible 2D moment coefficient
        surface.CM = surface.Cm_inc * aerodynamics.CL_over_Cl(
            surface.aspect_ratio(), mach=mach, sweep=surface.mean_sweep_angle(
            ))  # Compressible 3D moment coefficient
        surface.moment = (surface.CM * q * surface.area()
                          * surface.mean_geometric_chord())
