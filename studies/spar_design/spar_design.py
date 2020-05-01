from aerosandbox.structures.beams import TubeBeam1
import casadi as cas

opti = cas.Opti()
beam = TubeBeam1(
    opti=opti,
    length=49 / 2,
    points_per_point_load=50,
    diameter_guess=100,
    thickness = 0.70e-3,
    bending=True,
    torsion=False
)
lift_force = 9.81 * 385
load_location = opti.variable()
opti.set_initial(load_location, beam.length * 0.50)
opti.subject_to([
    load_location == beam.length * 0.50,
])
beam.add_point_load(load_location, -lift_force / 3)
beam.add_elliptical_load(force=lift_force / 2)
beam.setup()

# Constraints (in addition to stress)
opti.subject_to([
    # beam.u[-1] < 2,  # tip deflection. Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    # beam.u[-1] > -2  # tip deflection. Source: http://web.mit.edu/drela/Public/web/hpa/hpa_structure.pdf
    beam.du * 180 / cas.pi < 10, # dihedral constraint
    beam.du * 180 / cas.pi > -10, # anhedral constraint
    cas.diff(beam.nominal_diameter) < 0, # manufacturability
])

# # Zero-curvature constraint (restrict to conical tube spars only)
# opti.subject_to([
#     cas.diff(cas.diff(beam.nominal_diameter)) == 0
# ])

opti.minimize(beam.mass)

p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
# s_opts["mu_strategy"] = "adaptive"
opti.solver('ipopt', p_opts, s_opts)

sol = opti.solve()

beam.substitute_solution(sol)

print("Beam mass: %f kg" % beam.mass)
print("Spar mass: %f kg" % (2 * beam.mass))
beam.draw_bending()
