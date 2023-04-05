import aerosandbox as asb
import aerosandbox.numpy as np
import asb.modeling.splines as splines

def make_fuselage(
        boom_length,
        nose_length,
        fuse_diameter,
        boom_diameter,
        fuse_resolution=10,

) -> asb.Fuselage:
    ### Build the fuselage geometry
    blend = lambda x: (1 - np.cos(np.pi * x)) / 2
    fuse_x_c = []
    fuse_z_c = []
    fuse_radius = []
    # Nose geometry
    fuse_nose_theta = np.linspace(0, np.pi / 2, fuse_resolution)
    fuse_x_c.extend([
        - nose_length * np.cos(theta) for theta in fuse_nose_theta
    ])
    fuse_z_c.extend([-fuse_diameter / 2] * fuse_resolution)
    fuse_radius.extend([
        fuse_diameter / 2 * np.sin(theta) for theta in fuse_nose_theta
    ])
    # Taper
    fuse_taper_x_nondim = np.linspace(0, 1, fuse_resolution)
    fuse_x_c.extend([
        0.0 * boom_length + (0.6 - 0.0) * boom_length * x_nd for x_nd in fuse_taper_x_nondim
    ])
    fuse_z_c.extend([
        -fuse_diameter / 2 * blend(1 - x_nd) - boom_diameter / 2 * blend(x_nd) for x_nd in fuse_taper_x_nondim
    ])
    fuse_radius.extend([
        fuse_diameter / 2 * blend(1 - x_nd) + boom_diameter / 2 * blend(x_nd) for x_nd in fuse_taper_x_nondim
    ])
    # Tail
    # fuse_tail_x_nondim = np.linspace(0, 1, fuse_resolution)[1:]
    # fuse_x_c.extend([
    #     0.9 * boom_length + (1 - 0.9) * boom_length * x_nd for x_nd in fuse_taper_x_nondim
    # ])
    # fuse_z_c.extend([
    #     -boom_diameter / 2 * blend(1 - x_nd) for x_nd in fuse_taper_x_nondim
    # ])
    # fuse_radius.extend([
    #     boom_diameter / 2 * blend(1 - x_nd) for x_nd in fuse_taper_x_nondim
    # ])
    fuse_straight_resolution = 4
    fuse_x_c.extend([
        0.6 * boom_length + (1 - 0.6) * boom_length * x_nd for x_nd in np.linspace(0, 1, fuse_straight_resolution)[1:]
    ])
    fuse_z_c.extend(
        [-boom_diameter / 2] * (fuse_straight_resolution - 1)
    )
    fuse_radius.extend(
        [boom_diameter / 2] * (fuse_straight_resolution - 1)
    )

    fuse = asb.Fuselage(
        name="Fuselage",
        # xyz_le = np.array([0, 0, 0]),
        xsecs=[
            asb.FuselageXSec(
                # TODO have Peter check this is the correct change
                xyz_c = [fuse_x_c[i], 0, fuse_z_c[i]],
                radius=fuse_radius[i]
            ) for i in range(len(fuse_x_c))
        ]
    )

    return fuse

def make_payload_pod(
    total_length,
    nose_length,
    tail_length,
    fuse_diameter,
    fuse_resolution = 10,

) -> asb.Fuselage:
    ### Build the fuselage geometry
    blend = lambda x: (1 - np.cos(np.pi * x)) / 2
    fuse_x_c = []
    fuse_z_c = []
    fuse_radius = []
    # Nose geometry
    fuse_nose_theta = np.linspace(0, np.pi / 2, fuse_resolution)
    fuse_x_c.extend([
        - nose_length * np.cos(theta) for theta in fuse_nose_theta
    ])
    fuse_z_c.extend([-fuse_diameter / 2] * fuse_resolution)
    fuse_radius.extend([
        fuse_diameter / 2 * np.sin(theta) for theta in fuse_nose_theta
    ])

    # center section
    fuse_x_c.extend([
        total_length
    ])
    fuse_z_c.extend([
        (-fuse_diameter / 2)
    ])
    fuse_radius.extend([
        fuse_diameter / 2
    ])
    # Tail geometry
    fuse_tail_theta = np.flip(np.linspace(0, np.pi / 2, fuse_resolution))
    fuse_x_c.extend([
        total_length + tail_length * np.cos(theta) for theta in fuse_tail_theta
    ])
    fuse_z_c.extend([-fuse_diameter / 2] * fuse_resolution)
    fuse_radius.extend([
        fuse_diameter / 2 * np.sin(theta) for theta in fuse_tail_theta
    ])

    fuse = asb.Fuselage(
    name = "payload pod",
    xsecs = [
        asb.FuselageXSec(
            # TODO have Peter check this is the correct change
            xyz_c=[fuse_x_c[i], 0, fuse_z_c[i]],
            radius=fuse_radius[i]
        ) for i in range(len(fuse_x_c))
    ]
    )

    return fuse

#Below is a function that takes in the diameter and length of the payload pod and returns a fuselage object with
# an aerodynamic shape defined from the bezier spline function in aerosandbox
def aero_payload_pod(
    body_length, #length of the payload pod
    diameter, #maximum diameter of the payload pod
    nose_length, #length of the nose of the payload pod
    tail_length, #length of the tail of the payload pod
    resolution = 10,
) -> asb.Fuselage:
    fuse_x_c = []
    fuse_z_c = []
    fuse_radius = []
    # define a bezier spline for the nose of the payload pod
    x_n, y_n = splines.quadratic_bezier_patch_from_tangents(
        t=np.linspace(0, 1, resolution),
        x_a=0,
        x_b=nose_length,
        y_a=0,
        y_b=diameter/2,
        dydx_a=4,
        dydx_b=0,
    )
    z_n = [-diameter/2] * resolution
    # define a bezier spline for the tail of the payload pod
    x_t, y_t = splines.quadratic_bezier_patch_from_tangents(
        t=np.linspace(0, 1, resolution),
        x_a=body_length,
        x_b=body_length+tail_length,
        y_a=diameter/2,
        y_b=0,
        dydx_a=0,
        dydx_b=-0.3,
    )
    z_t = [-diameter/2] * resolution
    fuse_x_c.extend(x_n, x_t)
    fuse_z_c.extend(z_n, z_t)
    fuse_radius.extend(y_n, y_t)
    fuse = asb.Fuselage(
    name = "payload pod",
    xsecs = [
        asb.FuselageXSec(
            xyz_c=[fuse_x_c[i], 0, fuse_z_c[i]],
            radius=fuse_radius[i]
        ) for i in range(len(fuse_x_c))
    ]
    )



