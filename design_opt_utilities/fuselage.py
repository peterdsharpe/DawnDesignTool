import aerosandbox as asb
import aerosandbox.numpy as np


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
