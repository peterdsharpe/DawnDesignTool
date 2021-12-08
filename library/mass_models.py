"""
Mass models.

Set of functions to return masses of wings and tails, largely based off of Juan
Cruz Daedalus empirical functions.
"""
from aerosandbox.library import mass_structural as lib_mass_struct

gravity = 9.81

def mass_vstab(
    vstab,
    n_ribs_vstab,
    structural_load_factor,
    q_ne,
):
    # TODO due to asymmetry, a guess
    mass_knockup = 1.2

    thickness_ratio = 0.08
    cl_max = 1.5

    mass_vstab_primary = (lib_mass_struct.mass_wing_spar(
        span=vstab.span(),
        mass_supported=q_ne * cl_max * vstab.area() / gravity,
        ultimate_load_factor=structural_load_factor,
    ) * mass_knockup)  
    mass_vstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=vstab.span(),
        chord=vstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_vstab,
        t_over_c=thickness_ratio,
    )
    mass_vstab_total = mass_vstab_primary + mass_vstab_secondary  # per vstab
    return mass_vstab_total


def mass_hstab(
    hstab,
    n_ribs_hstab,
    structural_load_factor,
    q_ne,
):
    thickness_ratio = 0.08
    cl_max = 1.5

    mass_hstab_primary = lib_mass_struct.mass_wing_spar(
        span=hstab.span(),
        mass_supported=q_ne * cl_max * hstab.area() / gravity,
        ultimate_load_factor=structural_load_factor,
    )

    mass_hstab_secondary = lib_mass_struct.mass_hpa_stabilizer(
        span=hstab.span(),
        chord=hstab.mean_geometric_chord(),
        dynamic_pressure_at_manuever_speed=q_ne,
        n_ribs=n_ribs_hstab,
        t_over_c=thickness_ratio,
        include_spar=False,
    )
    mass_hstab_total = mass_hstab_primary + mass_hstab_secondary  # per hstab
    return mass_hstab_total


def estimate_mass_wing_secondary(
    span,
    chord,
    # You should optimize on ribs, there's a trade
    # between rib weight and LE sheeting weight!
    n_ribs,
    skin_density,
    # defaults to a single-section wing
    # (be careful: can you disassemble/transport this?)
    n_wing_sections=1,
    t_over_c=0.128,  # default from DAE11
    # Should we include the mass of the spar?
    # Useful if you want to do your own primary structure calculations.
    # Scale-up factor for masses that haven't yet totally been pinned down
    # experimentally
    scaling_factor=1.0,
):
    """
    Finds the mass of the wing structure of a human powered aircraft (HPA),
    following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    :param span: wing span [m]
    :param chord: wing mean chord [m]
    :param vehicle_mass: aircraft gross weight [kg]
    :param n_ribs: number of ribs in the wing
    :param n_wing_sections: number of wing sections or panels (for disassembly?)
    :param ultimate_load_factor: ultimate load factor [unitless]
    :param type: Type of bracing: "cantilevered", "one-wire", "multi-wire"
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :param include_spar: Should we include the mass of the spar?
                         Useful if you want to do your own primary
                         structure calculations. [boolean]
    :return: Wing structure mass [kg]
    """
    ### Secondary structure
    n_end_ribs = 2 * n_wing_sections - 2
    area = span * chord

    # Rib mass
    weight_ribs = n_ribs * (chord**2 * t_over_c * 5.50e-2
                            + chord * 1.91e-3) * 1.3
    # x1.3 scales to estimates from structures subteam

    # Half rib mass
    weight_half_ribs = (n_ribs - 1) * skin_density * chord * 0.65 * 0.072
    # 40% of cross sectional area, same construction as skin panels

    # End rib mass
    weight_end_ribs = n_end_ribs * (chord**2 * t_over_c * 6.62e-1
                                    + chord * 6.57e-3)

    # LE sheeting mass
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord
    weight_leading_edge_sheeting = 0.456 / 2 * (
        span**2 * ratio_of_rib_spacing_to_chord**(4 / 3) / span)

    # Skin Panel Mass
    weight_skin_panel = (
        area * skin_density * 1.05
    )  # assumed constant thickness from 0.9c around LE to 0.15c

    # TE mass
    weight_trailing_edge = span * 2.77e-2

    # Covering
    weight_covering = (
        area * 0.076
    )  # 0.033 kg/m2 Tedlar covering on 2 sides, with 1.1 coverage factor

    mass_secondary = (weight_ribs + weight_half_ribs + weight_end_ribs
                      + weight_trailing_edge) * scaling_factor + (
                          weight_skin_panel + weight_covering
                          + weight_leading_edge_sheeting)

    return mass_secondary
