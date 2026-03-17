"""
attitude.py

Quaternion math, USQUE MRP/GRP attitude parameterisation, and RK4 attitude
propagation for the UKF.

All quaternions use [qw, qx, qy, qz] convention throughout.

Rotation matrix convention:
    quat_to_dcm(q_AB) @ v_B = v_A
    i.e. the DCM maps vectors FROM frame B INTO frame A.
    To map A → B use:  quat_to_dcm(q_AB).T  (or equivalently quat_to_dcm(q_BA))

    This is the PASSIVE (coordinate transform) convention and matches
    both ukfspn_cpp (dcm_from) and cnnukf (quat2dcm in postprocess.py).

USQUE attitude parameterisation (a=1, f=4):
    MRP p is related to error quaternion dq = [dqw, dqv] by:
        dqw  = (16 - |p|^2) / (16 + |p|^2)
        dqvec = (1 + dqw) / 4 * p
    Reference: Crassidis & Markley 2003, Terzakis et al. JMIV 2018.

Sources:
    ukfspn_cpp/src/ukf/navigation.cc
    cnnukf/src/kalmanFilter/core/UnscentedKalmanFilter.m
    cnnukf/src/kalmanFilter/core/UKF_SPNv2.m
"""

import numpy as np


# ---------------------------------------------------------------------------
# Basic quaternion operations  [qw, qx, qy, qz]
# ---------------------------------------------------------------------------

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product of two quaternions.

    Both inputs and output are [qw, qx, qy, qz].
    Composition convention:  quat_multiply(q_AB, q_BC) = q_AC
    i.e. q1 is applied first, then q2.
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])
    return q / np.linalg.norm(q)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate (= inverse for unit quaternions).

    [qw, qx, qy, qz] → [qw, -qx, -qy, -qz]
    If q = q_AB, then quat_conjugate(q) = q_BA.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalise quaternion to unit length."""
    return q / np.linalg.norm(q)


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [qw, qx, qy, qz] to direction cosine matrix.

    Convention:  quat_to_dcm(q_AB) @ v_B = v_A   (maps frame B → frame A)

    To get a matrix that maps A → B:
        quat_to_dcm(q_AB).T   or   quat_to_dcm(quat_conjugate(q_AB))

    Note: this is R^T of the standard active rotation matrix, matching
    postprocess.py::quat2dcm and ukfspn_cpp::dcm_from.
    """
    q  = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [2*qw**2 - 1 + 2*qx**2,  2*qx*qy + 2*qw*qz,  2*qx*qz - 2*qw*qy],
        [2*qx*qy - 2*qw*qz,  2*qw**2 - 1 + 2*qy**2,  2*qy*qz + 2*qw*qx],
        [2*qx*qz + 2*qw*qy,  2*qy*qz - 2*qw*qx,  2*qw**2 - 1 + 2*qz**2],
    ])


# ---------------------------------------------------------------------------
# USQUE: MRP ↔ error quaternion  (a=1, f=4)
# ---------------------------------------------------------------------------

def error_quaternion_from_mrp(mrp: np.ndarray) -> np.ndarray:
    """
    Convert MRP (GRP with a=1, f=4) to error quaternion [qw, qx, qy, qz].

    This is the USQUE "sigma point to quaternion" step:
        dqw  = (16 - |p|^2) / (16 + |p|^2)
        dqvec = (1 + dqw) / 4 * p

    Reference: UnscentedKalmanFilter.m::grpToErrorQuat() with a=1, f=4
               navigation.cc -- "factor of 4 in our MRP"
    """
    p2  = float(np.dot(mrp, mrp))
    dqw = (16.0 - p2) / (16.0 + p2)
    dqv = (1.0 + dqw) / 4.0 * mrp
    dq  = np.array([dqw, dqv[0], dqv[1], dqv[2]])
    return dq / np.linalg.norm(dq)


def mrp_from_error_quaternion(dq: np.ndarray) -> np.ndarray:
    """
    Convert error quaternion [qw, qx, qy, qz] to MRP (GRP with a=1, f=4).

    Chooses the shadow MRP (shorter representation) if |dqw| < 0:
        mrp = 4 / (1 + dqw) * dqvec

    Reference: UnscentedKalmanFilter.m::errorQuatToGRP() with a=1, f=4
               navigation.cc::mrp_from_error_quaternion()
    """
    if dq[0] < 0:
        dq = -dq
    qw  = dq[0]
    qv  = dq[1:]
    return (4.0 / (1.0 + qw)) * qv


# ---------------------------------------------------------------------------
# Attitude dynamics
# ---------------------------------------------------------------------------

def _attitude_dynamics(
    q:                  np.ndarray,   # [4] [qw,qx,qy,qz]  q_spri2tpri
    w_tpri2spri_tpri:   np.ndarray,   # [3] RAV [rad/s]
    w_eci2spri_spri:    np.ndarray,   # [3] Mango ang. vel. in spri frame [rad/s]
    m_spri:             np.ndarray,   # [3] external torque on Mango in spri [N·m]
    I_s:                np.ndarray,   # [3] Mango principal MOI [kg·m^2]
    I_t:                np.ndarray,   # [3] Tango principal MOI [kg·m^2]
) -> tuple:
    """
    Continuous-time attitude dynamics.  Returns (qdot, wdot).

    State:
        q_spri2tpri       : quaternion from servicer to target principal frame
        w_tpri2spri_tpri  : angular velocity of tpri w.r.t. spri, in tpri frame

    Equations:
        qdot = 0.5 * Omega(w_spri2tpri_tpri) * q

        wdot = R_spri2tpri * wdot_spri - wdot_tpri
               - cross(w_eci2tpri_tpri, w_tpri2spri_tpri)

        where:
            w_spri2tpri_tpri = -w_tpri2spri_tpri
            R_spri2tpri      = quat_to_dcm(q_spri2tpri).T  (maps spri → tpri)
            w_eci2tpri_tpri  = R_spri2tpri @ w_eci2spri_spri + w_spri2tpri_tpri
            wdot_spri        = I_s^{-1} * (m_spri - cross(w_eci2spri, I_s * w_eci2spri))
            wdot_tpri        = I_t^{-1} * (      - cross(w_eci2tpri,  I_t * w_eci2tpri))

    Reference: navigation.cc::propagate() ODE lambda
               UKF_SPNv2.m::attitudeDynamics()
    """
    q = quat_normalize(q)
    qw, qx, qy, qz = q

    # Angular velocity of spri w.r.t. tpri, in tpri frame (note sign flip)
    w_spri2tpri_tpri = -w_tpri2spri_tpri

    wx, wy, wz = w_spri2tpri_tpri

    # Quaternion kinematics: qdot = 0.5 * Omega * q
    # Omega uses w_spri2tpri_tpri (the "body rate" of the relative rotation)
    Omega = np.array([
        [  0, -wx, -wy, -wz],
        [ wx,   0,  wz, -wy],
        [ wy, -wz,   0,  wx],
        [ wz,  wy, -wx,   0],
    ])
    qdot = 0.5 * Omega @ q

    # Rotation matrix: maps spri → tpri
    # quat_to_dcm(q_spri2tpri) maps tpri → spri (passive convention)
    # so .T maps spri → tpri
    R_spri2tpri = quat_to_dcm(q).T

    # Absolute angular velocity of tpri in tpri frame
    w_eci2tpri_tpri = R_spri2tpri @ w_eci2spri_spri + w_spri2tpri_tpri

    # Angular acceleration of Mango (Euler rigid-body, in spri frame)
    wdot_spri = (1.0 / I_s) * (m_spri - np.cross(w_eci2spri_spri, I_s * w_eci2spri_spri))

    # Angular acceleration of Tango (no external torque, in tpri frame)
    wdot_tpri = (1.0 / I_t) * (       - np.cross(w_eci2tpri_tpri, I_t * w_eci2tpri_tpri))

    # Relative angular acceleration
    wdot = (R_spri2tpri @ wdot_spri
            - wdot_tpri
            - np.cross(w_eci2tpri_tpri, w_tpri2spri_tpri))

    return qdot, wdot


def attitude_dynamics_rk4(
    q:                np.ndarray,   # [4] [qw,qx,qy,qz]  q_spri2tpri
    w_tpri2spri_tpri: np.ndarray,   # [3] RAV [rad/s]
    w_eci2spri_spri:  np.ndarray,   # [3] Mango ang. vel. in spri [rad/s]
    m_spri:           np.ndarray,   # [3] torque on Mango in spri [N·m]
    I_s:              np.ndarray,   # [3] Mango principal MOI
    I_t:              np.ndarray,   # [3] Tango principal MOI
    dt:               float,
) -> tuple:
    """
    Integrate attitude dynamics over one timestep using RK4.

    Returns:
        q_new: [4] [qw,qx,qy,qz]  normalised propagated quaternion
        w_new: [3] propagated w_tpri2spri_tpri [rad/s]

    w_eci2spri_spri and m_spri are treated as constant over the interval
    (they come from the Mango absolute state at the start of the step).

    Reference: UKF_SPNv2.m::dynamicsUpdate() RK4 branch
               navigation.cc::propagate() via rk4_single()
    """
    def deriv(q_i, w_i):
        return _attitude_dynamics(q_i, w_i, w_eci2spri_spri, m_spri, I_s, I_t)

    # RK4 stages
    kq1, kw1 = deriv(q,                   w_tpri2spri_tpri)
    kq2, kw2 = deriv(q + kq1 * dt / 2,   w_tpri2spri_tpri + kw1 * dt / 2)
    kq3, kw3 = deriv(q + kq2 * dt / 2,   w_tpri2spri_tpri + kw2 * dt / 2)
    kq4, kw4 = deriv(q + kq3 * dt,        w_tpri2spri_tpri + kw3 * dt)

    q_new = q + (kq1 + 2*kq2 + 2*kq3 + kq4) * dt / 6
    w_new = w_tpri2spri_tpri + (kw1 + 2*kw2 + 2*kw3 + kw4) * dt / 6

    return quat_normalize(q_new), w_new
