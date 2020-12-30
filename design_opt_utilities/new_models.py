import casadi as cas


def mass_hpa_wing(
        span,
        chord,
        vehicle_mass,
        n_ribs,  # You should optimize on this, there's a trade between rib weight and LE sheeting weight!
        skin_density,
        n_wing_sections=1,  # defaults to a single-section wing (be careful: can you disassemble/transport this?)
        ultimate_load_factor=1.75,  # default taken from Daedalus design
        type="cantilevered",  # "cantilevered", "one-wire", "multi-wire"
        t_over_c=0.128,  # default from DAE11
        include_spar=True,
        # Should we include the mass of the spar? Useful if you want to do your own primary structure calculations.
):
    """
    Finds the mass of the wing structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    :param span: wing span [m]
    :param chord: wing mean chord [m]
    :param vehicle_mass: aircraft gross weight [kg]
    :param n_ribs: number of ribs in the wing
    :param n_wing_sections: number of wing sections or panels (for disassembly?)
    :param ultimate_load_factor: ultimate load factor [unitless]
    :param type: Type of bracing: "cantilevered", "one-wire", "multi-wire"
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :param include_spar: Should we include the mass of the spar? Useful if you want to do your own primary structure calculations. [boolean]
    :return: Wing structure mass [kg]
    """
    ### Primary structure
    if include_spar:
        if type == "cantilevered":
            mass_primary_spar = (
                    (span * 1.17e-1 + span ** 2 * 1.10e-2) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        elif type == "one-wire":
            mass_primary_spar = (
                    (span * 3.10e-2 + span ** 2 * 7.56e-3) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        elif type == "multi-wire":
            mass_primary_spar = (
                    (span * 1.35e-1 + span ** 2 * 1.68e-3) *
                    (1 + (ultimate_load_factor * vehicle_mass / 100 - 2) / 4)
            )
        else:
            raise ValueError("Bad input for 'type'!")

        mass_primary = mass_primary_spar * (
                11382.3 / 9222.2)  # accounts for rear spar, struts, fittings, kevlar x-bracing, and wing-fuselage mounts
    else:
        mass_primary = 0

    ### Secondary structure
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord
    n_end_ribs = 2 * n_wing_sections - 2
    area = span * chord

    # Rib mass
    W_wr = n_ribs * (chord ** 2 * t_over_c * 5.50e-2 + chord * 1.91e-3) * 1.3
    # x1.3 scales to estimates from structures subteam

    # Half rib mass
    W_whr = (n_ribs - 1) * skin_density * chord * 0.65 * 0.072
    # 40% of cross sectional area, same construction as skin panels

    # End rib mass
    W_wer = n_end_ribs * (chord ** 2 * t_over_c * 6.62e-1 + chord * 6.57e-3)

    # LE sheeting mass
    # W_wLE = 0.456/2 * (span ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # Skin Panel Mass
    W_wsp = area * skin_density * 1.05 / 1.3  # assumed constant thickness from 0.9c around LE to 0.15c
    # 1.3 removes the correction factor included later - prevents double counting

    # TE mass
    W_wTE = span * 2.77e-2

    # Covering
    W_wc = area * 0.076 / 1.3  # 0.033 kg/m2 Tedlar covering on 2 sides, with 1.1 coverage factor

    mass_secondary = W_wr + W_wer + W_wsp + W_wTE + W_wc + W_whr

    return mass_primary + mass_secondary


def mass_wing_spar(
        span,
        mass_supported,
        ultimate_load_factor=1.75,  # default taken from Daedalus design
        n_booms=1,
        strut_loc=5,
):
    """
    Finds the mass of the spar for a wing on a single- or multi-boom lightweight aircraft. Model originally designed for solar aircraft.
    Data was fit to the range 3 < wing_span < 120 [m] and 5 < supported_mass < 3000 [kg], but validity should extend somewhat beyond that.
    Extremely accurate fits within this range; R^2 > 0.995 for all fits.
    Source: AeroSandbox\studies\MultiBoomSparMass_v2                                
    Assumptions:
        * Elliptical lift distribution
        * Constraint that local wing dihedral/anhedral angle must not exceed 10 degrees anywhere in the ultimate load case.
        * If multi-boom, assumes roughly static-aerostructurally-optimal placement of the outer booms and equal boom weights.
    :param span: Wing span [m]
    :param mass_supported: Total mass of all fuselages + tails
    :param ultimate_load_factor: Design load factor. Default taken from Daedalus design.
    :param n_booms: Number of booms on the design. Can be 1, 2, or 3. Assumes optimal placement of the outer booms.
    :return:
        
    """
    if n_booms == 1:
        c = 20.7100792220283090
        span_exp = 1.6155586404697364
        mass_exp = 0.3779456295164249
    elif n_booms == 2:
        c = 12.3247625359796285
        span_exp = 1.5670343007798109
        mass_exp = 0.4342199756794465
    elif n_booms == 3:
        c = 10.0864141678007844
        span_exp = 1.5614086940653213
        mass_exp = 0.4377206254456823
    else:
        raise ValueError("Bad value of n_booms!")

    mass_eff = mass_supported * ultimate_load_factor
    span_eff = span - 2 * strut_loc

    spar_mass = c * (span_eff / 40) ** span_exp * (mass_eff / 300) ** mass_exp

    return spar_mass


def mass_hpa_stabilizer(
        span,
        chord,
        dynamic_pressure_at_manuever_speed,
        n_ribs,  # You should optimize on this, there's a trade between rib weight and LE sheeting weight!
        t_over_c=0.128,  # default from DAE11
        include_spar=True,
        # Should we include the mass of the spar? Useful if you want to do your own primary structure calculations.
):
    """
    Finds the mass of a stabilizer structure of a human powered aircraft (HPA), following Juan Cruz's correlations in
    http://journals.sfu.ca/ts/index.php/ts/article/viewFile/760/718
    Note: apply this once to BOTH the rudder and elevator!!!
    :param span: stabilizer span [m]
    :param chord: stabilizer mean chord [m]
    :param dynamic_pressure_at_manuever_speed: dynamic pressure at maneuvering speed [Pa]
    :param n_ribs: number of ribs in the wing
    :param t_over_c: wing airfoil thickness-to-chord ratio
    :param include_spar: Should we include the mass of the spar? Useful if you want to do your own primary structure calculations. [boolean]
    :return: Stabilizer structure mass [kg]
    """
    ### Primary structure
    area = span * chord
    q = dynamic_pressure_at_manuever_speed
    if include_spar:
        W_tss = (
                (span * 4.15e-2 + span ** 2 * 3.91e-3) *
                (1 + ((q * area) / 78.5 - 1) / 2)
        )

        mass_primary = W_tss
    else:
        mass_primary = 0

    ### Secondary structure
    ratio_of_rib_spacing_to_chord = (span / n_ribs) / chord

    # Rib mass
    W_tsr = n_ribs * (chord ** 2 * t_over_c * 1.16e-1 + chord * 4.01e-3)

    # Leading edge sheeting
    W_tsLE = 0.174 * (area ** 2 * ratio_of_rib_spacing_to_chord ** (4 / 3) / span)

    # Covering
    W_tsc = area * 1.93e-2
    # W_tsc = area * 0.076

    mass_secondary = W_tsr + W_tsLE + W_tsc

    ### Totaling
    correction_factor = ((537.8 / (537.8 - 23.7 - 15.1)) * (623.3 / (623.3 - 63.2 - 8.1))) ** 0.5
    # geometric mean of Daedalus elevator and rudder corrections from misc. weight

    return (mass_primary + mass_secondary) * correction_factor


def eff_curve_fit(airspeed, total_thrust, altitude, var_pitch):
    """
    Curve fit to yield propeller and motor efficiency estimates.

    Based on curve fit to many data points generated by Matlab script.

    Given by Jamie Abel (jmabel@mit.edu) to Peter Sharpe (pds@mit.edu) on 12/15/2020.
    Presumably based on propulsion analysis by Julia Gaubatz.

    :param airspeed: Airspeed [m/s]
    :param total_thrust: Total thrust force for the aircraft [N]
    :param altitude: Altitude [m]
    :param var_pitch: Variable pitch
    :return:
    """

    # logistic fit on data generated by Matlab script
    # currently fixed pitch, stock motors, geared drive

    airspeed_norm = airspeed / 28
    thrust_norm = total_thrust / 140
    altitude_norm = altitude / 18288

    x = {'airspeed'         : airspeed_norm,
         'thrust'           : thrust_norm,
         'altitude'         : altitude_norm,
         'airspeed2'        : airspeed_norm ** 2,
         'thrust2'          : thrust_norm ** 2,
         'altitude2'        : altitude_norm ** 2,
         'airspeed_thrust'  : airspeed_norm * thrust_norm,
         'thrust_altitude'  : thrust_norm * altitude_norm,
         'altitude_airspeed': altitude_norm * airspeed_norm
         }
    if var_pitch:
        p_prop = {'c0': 1.575218678109346,
                  'c1': 1.2870422616757415,
                  'c2': -0.055743218813845787,
                  'c3': -0.26095248557114475,
                  'c4': -1.434100367832706,
                  'c5': 0.004215266351764378,
                  'c6': -0.9352579377409926,
                  'c7': 0.46767421485606164,
                  'c8': -0.474587756160942,
                  'c9': 1.7378050393774664}

        p_motor = {'c0': 1.4847520195589505,
                   'c1': 1.3178449851433132,
                   'c2': 0.9550224442521187,
                   'c3': 0.36550134597090395,
                   'c4': -0.5913485808906566,
                   'c5': -0.20106568389993254,
                   'c6': -0.7889569190668057,
                   'c7': -0.1351433888968255,
                   'c8': -0.19182814050736208,
                   'c9': 0.4890002450542662}

    else:
        p_prop = {'c0': 1.1737048834956951,
                  'c1': 2.951217734244729,
                  'c2': -0.3831881923070517,
                  'c3': -0.9099965850192641,
                  'c4': -2.5653630486804717,
                  'c5': 0.038232751115058985,
                  'c6': -1.0619892648665925,
                  'c7': 0.7296445420142255,
                  'c8': -0.7052582749278459,
                  'c9': 2.6010856594186325}

        p_motor = {'c0': 1.7337119601446356,
                   'c1': 0.39718800409519256,
                   'c2': 1.002057633578034,
                   'c3': 0.12632735211740692,
                   'c4': -0.11856769124210348,
                   'c5': -0.2682997995501091,
                   'c6': -0.2548726025978303,
                   'c7': -0.02272953412184562,
                   'c8': 0.09844251106050433,
                   'c9': -0.16565544086978443}

    prop_eff = (1 / (1 + cas.exp(
        -(p_prop['c0'] +
          p_prop['c1'] * x['airspeed'] +
          p_prop['c2'] * x['thrust'] +
          p_prop['c3'] * x['altitude'] +
          p_prop['c4'] * x['airspeed2'] +
          p_prop['c5'] * x['thrust2'] +
          p_prop['c6'] * x['altitude2'] +
          p_prop['c7'] * x['airspeed_thrust'] +
          p_prop['c8'] * x['thrust_altitude'] +
          p_prop['c9'] * x['altitude_airspeed']
          ))))

    motor_eff = (1 / (1 + cas.exp(
        -(p_motor['c0'] +
          p_motor['c1'] * x['airspeed'] +
          p_motor['c2'] * x['thrust'] +
          p_motor['c3'] * x['altitude'] +
          p_motor['c4'] * x['airspeed2'] +
          p_motor['c5'] * x['thrust2'] +
          p_motor['c6'] * x['altitude2'] +
          p_motor['c7'] * x['airspeed_thrust'] +
          p_motor['c8'] * x['thrust_altitude'] +
          p_motor['c9'] * x['altitude_airspeed']
          ))))

    return prop_eff, motor_eff
