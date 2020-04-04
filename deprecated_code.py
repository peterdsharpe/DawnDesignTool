"""
Snippets of old code that might be useful later.
"""


# elif aerodynamics_type == "aerosandbox-point":
#
#     airplane.fuselages = []
#
#     airplane.set_spanwise_paneling_everywhere(8)  # Set the resolution of the analysis
#     ap = asb.Casll1(
#         airplane=airplane,
#         op_point=asb.OperatingPoint(
#             density=rho[0],
#             viscosity=mu[0],
#             velocity=airspeed[0],
#             mach=0,
#             alpha=alpha[0],
#             beta=0,
#             p=0,
#             q=0,
#             r=0,
#         ),
#         opti=opti
#     )
#
#     lift_force = -ap.force_total_wind[2]
#     drag_force = -ap.force_total_wind[0]
#
#     # Tack on fuselage drag:
#     fuse_Re = rho / mu * airspeed * fuse.length()
#     drag_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted() * q
#     drag_force += drag_fuse
#
# elif aerodynamics_type == "aerosandbox-full":
#     lift_force = []
#     drag_force = []
#
#     airplane.wings = [wing]  # just look at the one wing
#     airplane.fuselages = []  # ignore the fuselage
#
#     airplane.set_spanwise_paneling_everywhere(6)  # Set the resolution of the analysis
#
#     aps = [
#         asb.Casll1(
#             airplane=airplane,
#             op_point=asb.OperatingPoint(
#                 density=rho[i],
#                 viscosity=mu[i],
#                 velocity=airspeed[i],
#                 mach=0,
#                 alpha=alpha[i],
#                 beta=0,
#                 p=0,
#                 q=0,
#                 r=0,
#             ),
#             opti=opti
#         )
#         for i in range(n_timesteps)
#     ]
#
#     lift_force = cas.vertcat(*[-ap.force_total_wind[2] for ap in aps])
#     drag_force = cas.vertcat(*[-ap.force_total_wind[0] for ap in aps])
#
#     # Multiply drag force to roughly account for tail
#     drag_force *= 1.1
#
#     # Tack on fuselage drag:
#     fuse_Re = rho / mu * airspeed * fuse.length()
#     drag_fuse = aero.Cf_flat_plate(fuse_Re) * fuse.area_wetted() * q
#     drag_force += drag_fuse



# if aerodynamics_type == "aerosandbox-point":
#     import copy
#
#     ap_sol = copy.deepcopy(ap)
#     ap_sol.substitute_solution(sol)
# if aerodynamics_type == "aerosandbox-full":
#     import copy
#
#     ap_sols = [copy.deepcopy(ap) for ap in aps]
#     ap_sols = [ap_sol.substitute_solution(sol) for ap_sol in ap_sols]
