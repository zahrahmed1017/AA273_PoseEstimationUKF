"""
spacecraft.py

Physical constants and geometry for the Mango (servicer) and Tango (target)
spacecraft in the SHIRT dataset.

All quaternions use [qw, qx, qy, qz] convention.

Frame definitions:
    spri  = Mango principal frame  (= pri for Mango)
    scam  = Mango camera frame
    tpri  = Tango principal frame  (= pri for Tango)
    tbdy  = Tango body frame

Source: ukfspn_cpp/src/ukf/spacecraft.hh
"""

import numpy as np


class Mango:
    """
    Servicer spacecraft (camera platform).

    q_pri2cam: rotation from Mango principal frame to Mango camera frame.
               Identity here -- camera is aligned with the principal frame.
    r_pri2cam_pri: position of camera origin w.r.t. Mango CoM, in principal frame [m].
    """

    # Principal moments of inertia [kg·m^2], diagonal of inertia tensor in principal frame
    I_pri = np.array([16.704476666666668, 19.44025, 18.278726666666664])

    # Position of camera origin w.r.t. Mango CoM, expressed in principal frame [m]
    r_pri2cam_pri = np.array([0.1, 0.161683937823834, 0.41])

    # Rotation from Mango principal frame → Mango camera frame [qw, qx, qy, qz]
    # Identity: camera frame is aligned with the principal frame
    q_pri2cam = np.array([1., 0., 0., 0.])

    # Position of Mango body origin w.r.t. CoM, in principal frame [m]
    r_pri2bdy_pri = np.array([0., 0.011683937823834, 0.])

    # Rotation from Mango principal frame → Mango body frame [qw, qx, qy, qz]
    q_pri2bdy = np.array([1., 0., 0., 0.])


class Tango:
    """
    Target spacecraft.

    q_pri2bdy: rotation from Tango principal frame to Tango body frame.
               Non-trivial -- there is a misalignment between principal and body frames.
    r_pri2bdy_pri: position of Tango body origin w.r.t. Tango CoM, in principal frame [m].
    """

    # Principal moments of inertia [kg·m^2]
    I_pri = np.array([2.685703125, 3.458072723298021, 3.106463735035312])

    # Position of Tango body origin w.r.t. Tango CoM, expressed in principal frame [m]
    r_pri2bdy_pri = np.array([0., -0.041845452916084, 0.156390637826117])

    # Rotation from Tango principal frame → Tango body frame [qw, qx, qy, qz]
    q_pri2bdy = np.array([-0.187867580617481, 0.982194365771324, 0., 0.])

    # 3D keypoint locations in Tango body frame [m], shape [11 x 3]
    # These are the 11 points the SPN is trained to detect.
    # Source: ukfspn_cpp/src/ukf/spacecraft.hh  (Tango::keypoints)
    keypoints = np.array([
        [-0.3700, -0.3850,  0.3215],
        [-0.3700,  0.3850,  0.3215],
        [ 0.3700,  0.3850,  0.3215],
        [ 0.3700, -0.3850,  0.3215],
        [-0.3700, -0.2640,  0.0000],
        [-0.3700,  0.3040,  0.0000],
        [ 0.3700,  0.3040,  0.0000],
        [ 0.3700, -0.2640,  0.0000],
        [-0.5427,  0.4877,  0.2535],
        [ 0.5427,  0.4877,  0.2591],
        [ 0.3050, -0.5790,  0.2515],
    ], dtype=np.float64)   # [11 x 3]
