# Grab AeroSandbox
import aerosandbox as asb
import casadi as cas

import aerosandbox.library.aerodynamics as aero
from aerosandbox.library.airfoils import e216, flat_plate

##### Initialize Optimization
opti = cas.Opti()
# endregion

wing = asb.Wing(
    name="Main Wing",
    x_le=0,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=0.25,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=-5 / 4,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=5,
            twist=0,  # degrees
            airfoil=e216,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=-2.5 / 4,
            y_le=40,  # wing_span / 2,
            z_le=4,
            chord=2.5,
            twist=0,
            airfoil=e216,
        ),
    ]
)
hstab = asb.Wing(
    name="Horizontal Stabilitzer",
    x_le=24,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=-0.5,  # Coordinates of the wing's leading edge
    symmetric=True,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=2,
            twist=-3,  # degrees
            airfoil=flat_plate,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0.08,
            y_le=7,
            z_le=0,
            chord=2,
            twist=-3,
            airfoil=flat_plate,
        ),
    ]
)
vstab = asb.Wing(
    name="Vertical Stabilizer",
    x_le=26,  # Coordinates of the wing's leading edge
    y_le=0,  # Coordinates of the wing's leading edge
    z_le=-4,  # Coordinates of the wing's leading edge
    symmetric=False,
    xsecs=[  # The wing's cross ("X") sections
        asb.WingXSec(  # Root
            x_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            y_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            z_le=0,  # Coordinates of the XSec's leading edge, relative to the wing's leading edge.
            chord=2,
            twist=0,  # degrees
            airfoil=flat_plate,  # Airfoils are blended between a given XSec and the next one.
            control_surface_type='symmetric',
            # Flap # Control surfaces are applied between a given XSec and the next one.
            control_surface_deflection=0,  # degrees
        ),
        asb.WingXSec(  # Tip
            x_le=0.08,
            y_le=0,
            z_le=8,
            chord=2,
            twist=0,
            airfoil=flat_plate,
        ),
    ]
)
fuse = asb.Fuselage(
    name="Fuselage",
    x_le=0,
    y_le=0,
    z_le=-1,
    symmetric=False,
    xsecs=[
        asb.FuselageXSec(x_c=-5, radius=0),
        asb.FuselageXSec(x_c=-4.5, radius=0.4),
        asb.FuselageXSec(x_c=-4, radius=0.6),
        asb.FuselageXSec(x_c=-3, radius=0.9),
        asb.FuselageXSec(x_c=-2, radius=1),
        asb.FuselageXSec(x_c=-1, radius=1),
        asb.FuselageXSec(x_c=0, radius=1),
        asb.FuselageXSec(x_c=1, radius=1),
        asb.FuselageXSec(x_c=2, radius=1),
        asb.FuselageXSec(x_c=3, radius=1),
        asb.FuselageXSec(x_c=4, radius=1),
        asb.FuselageXSec(x_c=5, radius=0.95),
        asb.FuselageXSec(x_c=6, radius=0.9),
        asb.FuselageXSec(x_c=7, radius=0.8),
        asb.FuselageXSec(x_c=8, radius=0.7),
        asb.FuselageXSec(x_c=9, radius=0.6),
        asb.FuselageXSec(x_c=10, radius=0.5),
        asb.FuselageXSec(x_c=11, radius=0.4),
        asb.FuselageXSec(x_c=12, radius=0.3),
        asb.FuselageXSec(x_c=13, radius=0.25),
        asb.FuselageXSec(x_c=22, radius=0.25),
        asb.FuselageXSec(x_c=24, radius=0.25),
        asb.FuselageXSec(x_c=25, radius=0.25),
        asb.FuselageXSec(x_c=26, radius=0.2),
        asb.FuselageXSec(x_c=27, radius=0.1),
        asb.FuselageXSec(x_c=28, radius=0),
    ]
)

airplane = asb.Airplane(
    name="Solar1",
    x_ref=0,
    y_ref=0,
    z_ref=0,
    wings=[
        wing,
        hstab,
        vstab,
    ],
    fuselages=[
        fuse
    ],
)
airplane.set_spanwise_paneling_everywhere(20)  # Set the resolution of the analysis
ap = asb.Casll1(
    airplane=airplane,
    op_point=asb.OperatingPoint(
        density=1.225,
        viscosity=1.81e-5,
        velocity=22,
        mach=0,
        alpha=0,
        beta=0,
        p=0,
        q=0,
        r=0,
    ),
    opti=opti,
) # type: asb.Casll1

# region Solve
p_opts = {}
s_opts = {}
s_opts["max_iter"] = 1e6  # If you need to interrupt, just use ctrl+c
s_opts["mu_strategy"] = "adaptive"
s_opts["start_with_resto"] = "yes"
s_opts["required_infeasibility_reduction"] = 0.001
opti.solver('ipopt', p_opts, s_opts)

if __name__ == "__main__":
    try:
        sol = opti.solve()
    except:
        sol = opti.debug

    import copy

    ap_sol = copy.deepcopy(ap)
    ap_sol.substitute_solution(sol)

    ap_sol.draw()
