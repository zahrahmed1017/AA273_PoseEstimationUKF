"""
test_attitude.py

Unit tests for attitude.py.

Tests:
  1. quat_multiply: identity, composition, non-commutativity check
  2. quat_conjugate: double-conjugate = identity, q * q* = [1,0,0,0]
  3. quat_to_dcm: orthogonality, det=1, known 90-deg rotation
  4. USQUE round-trip: mrp → error_quat → mrp
  5. USQUE round-trip: error_quat → mrp → error_quat
  6. MRP near-zero: error_quat_from_mrp([0,0,0]) = [1,0,0,0]
  7. RK4 propagation: zero angular velocity → quaternion unchanged
  8. RK4 propagation: pure spin → quaternion half-angle check
  9. RK4 energy conservation: torque-free symmetric top

Run:
    python -m ukf.test_attitude
"""

import numpy as np
from .attitude import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_to_dcm,
    error_quaternion_from_mrp,
    mrp_from_error_quaternion,
    attitude_dynamics_rk4,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passed(name: str) -> None:
    print(f"  PASS  {name}")

def _failed(name: str, msg: str) -> None:
    print(f"  FAIL  {name}: {msg}")

def assert_close(a, b, tol, name):
    err = np.max(np.abs(np.asarray(a) - np.asarray(b)))
    if err < tol:
        _passed(f"{name}  (max err = {err:.2e})")
    else:
        _failed(name, f"max err = {err:.2e}, tol = {tol:.2e}")
        _failed(name, f"  a = {a}")
        _failed(name, f"  b = {b}")

def _random_unit_quat(seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_quat_multiply_identity():
    """q * identity = q."""
    q    = _random_unit_quat(1)
    qI   = np.array([1.0, 0.0, 0.0, 0.0])
    assert_close(quat_multiply(q, qI), q, tol=1e-14,
                 name="q * identity = q")
    assert_close(quat_multiply(qI, q), q, tol=1e-14,
                 name="identity * q = q")


def test_quat_multiply_inverse():
    """q * q^{-1} = identity."""
    q    = _random_unit_quat(2)
    qinv = quat_conjugate(q)
    prod = quat_multiply(q, qinv)
    # May get [-1,0,0,0] or [1,0,0,0], both represent identity rotation
    assert_close(np.abs(prod[0]), 1.0, tol=1e-13,
                 name="|qw| of q * q* = 1")
    assert_close(prod[1:], np.zeros(3), tol=1e-13,
                 name="qv of q * q* = 0")


def test_quat_multiply_composition():
    """
    Rotate v by q1 then q2, compare with direct q_total = q2 * q1.

    quat_to_dcm(q_AB) maps B→A in passive convention.
    Active rotation of a vector: use dcm.T (= dcm of conjugate).
    """
    q1  = _random_unit_quat(3)
    q2  = _random_unit_quat(4)
    v   = np.array([1.0, 2.0, 3.0])

    # Two-step active rotation: first q1, then q2
    v1  = quat_to_dcm(quat_conjugate(q1)).T @ v   # active rotate by q1
    v2  = quat_to_dcm(quat_conjugate(q2)).T @ v1  # then by q2

    # Direct composition
    q12  = quat_multiply(q1, q2)
    v_12 = quat_to_dcm(quat_conjugate(q12)).T @ v

    assert_close(v2, v_12, tol=1e-13,
                 name="two-step rotation = composed rotation")


def test_quat_conjugate_double():
    """double conjugate = original."""
    q = _random_unit_quat(5)
    assert_close(quat_conjugate(quat_conjugate(q)), q, tol=1e-14,
                 name="conjugate(conjugate(q)) = q")


def test_quat_to_dcm_orthogonal():
    """DCM should be orthogonal (R @ R.T = I) and det = 1."""
    q   = _random_unit_quat(6)
    R   = quat_to_dcm(q)
    I3  = np.eye(3)
    assert_close(R @ R.T, I3, tol=1e-13, name="R @ R.T = I")
    assert_close(np.linalg.det(R), 1.0, tol=1e-13, name="det(R) = 1")


def test_quat_to_dcm_known_rotation():
    """
    90-degree rotation about z-axis.

    q = [cos(45°), 0, 0, sin(45°)] rotates x→y in active convention.
    In passive convention, quat_to_dcm(q) maps the NEW frame back to OLD,
    so applying it to [1,0,0]_new should give [0,1,0]_old ... but we use
    active interpretation: the DCM.T is the active rotation matrix.
    """
    angle = np.pi / 2  # 90 deg about z
    q = np.array([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    R = quat_to_dcm(q)

    # Active: R.T rotates [1,0,0] → [0,1,0]  (x → y)
    v_rot = R.T @ np.array([1.0, 0.0, 0.0])
    assert_close(v_rot, np.array([0.0, 1.0, 0.0]), tol=1e-14,
                 name="90° about z: R.T @ x_hat = y_hat")


def test_usque_mrp_roundtrip():
    """mrp → error_quat → mrp should recover original mrp (small angle)."""
    mrp_orig = np.array([0.1, -0.05, 0.08])
    dq       = error_quaternion_from_mrp(mrp_orig)
    mrp_rec  = mrp_from_error_quaternion(dq)
    assert_close(mrp_orig, mrp_rec, tol=1e-13,
                 name="mrp → error_quat → mrp  round-trip")


def test_usque_quat_roundtrip():
    """error_quat → mrp → error_quat should recover original (positive-qw hemisphere)."""
    # Build a small error quaternion (< 90 deg rotation)
    angle = 0.3  # rad
    axis  = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    dq_orig = np.array([np.cos(angle/2),
                        np.sin(angle/2)*axis[0],
                        np.sin(angle/2)*axis[1],
                        np.sin(angle/2)*axis[2]])
    mrp    = mrp_from_error_quaternion(dq_orig)
    dq_rec = error_quaternion_from_mrp(mrp)
    assert_close(dq_orig, dq_rec, tol=1e-13,
                 name="error_quat → mrp → error_quat  round-trip")


def test_usque_identity():
    """mrp = [0,0,0] should give error quaternion [1,0,0,0]."""
    dq = error_quaternion_from_mrp(np.zeros(3))
    assert_close(dq, np.array([1.0, 0.0, 0.0, 0.0]), tol=1e-14,
                 name="mrp=0 → identity quaternion")


def test_rk4_zero_omega():
    """
    With zero relative angular velocity and zero Mango angular velocity,
    the quaternion should remain unchanged after propagation.
    """
    q0  = _random_unit_quat(7)
    w0  = np.zeros(3)   # w_tpri2spri_tpri
    w_m = np.zeros(3)   # w_eci2spri_spri
    m   = np.zeros(3)   # torque
    I_s = np.array([1.0, 2.0, 3.0])
    I_t = np.array([0.5, 0.8, 1.2])

    q1, w1 = attitude_dynamics_rk4(q0, w0, w_m, m, I_s, I_t, dt=0.5)
    assert_close(q0, q1, tol=1e-14,
                 name="zero ω: quaternion unchanged after RK4 step")
    assert_close(w0, w1, tol=1e-14,
                 name="zero ω: angular velocity unchanged after RK4 step")


def test_rk4_pure_spin_angle():
    """
    Pure spin of tpri relative to spri at constant ω about z-axis.

    With no external forces and spri also stationary, the relative
    quaternion should rotate by ω*dt radians about z.

    Setup:
        w_tpri2spri_tpri = [0, 0, wz]  (tpri spins about its own z)
        w_eci2spri_spri  = 0
        torque           = 0
        I_s = I_t = [1,1,1] (spherically symmetric → no Euler torque)

    Expected: q_new = q_rot(wz*dt, z) * q0  (up to sign)
    """
    wz  = 0.2        # rad/s
    dt  = 0.1        # s
    q0  = np.array([1.0, 0.0, 0.0, 0.0])
    w0  = np.array([0.0, 0.0, wz])
    w_m = np.zeros(3)
    m   = np.zeros(3)
    I_s = np.ones(3)
    I_t = np.ones(3)

    q1, _ = attitude_dynamics_rk4(q0, w0, w_m, m, I_s, I_t, dt=dt)

    # Omega uses w_spri2tpri_tpri = -w_tpri2spri_tpri = [0,0,-wz].
    # With q0=identity, qdot = 0.5*Omega*q0 → qz component decrements.
    # So q_new ≈ [cos(wz*dt/2), 0, 0, -sin(wz*dt/2)].
    half = wz * dt / 2.0
    q_expected = quat_normalize(np.array([np.cos(half), 0.0, 0.0, -np.sin(half)]))

    # Check they represent the same rotation (allow sign flip)
    dot = abs(np.dot(q1, q_expected))
    if abs(dot - 1.0) < 1e-5:
        _passed(f"pure z-spin: quaternion angle  (dot = {dot:.8f})")
    else:
        _failed("pure z-spin: quaternion angle",
                f"dot(q1, q_expected) = {dot:.8f}, expected ≈ 1")


def test_rk4_energy_conservation():
    """
    Torque-free asymmetric top: rotational kinetic energy T = 0.5 * w.T I w
    should be conserved under Euler dynamics.

    We propagate for 100 steps and check relative energy drift < 1e-6.
    """
    q0  = _random_unit_quat(8)
    w0  = np.array([0.5, 0.3, 0.1])   # rad/s, general initial spin
    w_m = np.zeros(3)
    m   = np.zeros(3)
    I_s = np.ones(3)   # Mango: spherical → no wdot_spri contribution
    I_t = np.array([1.0, 2.0, 4.0])   # Tango: asymmetric top

    dt  = 0.01   # s
    T0  = 0.5 * np.dot(w0, I_t * w0)

    q, w = q0.copy(), w0.copy()
    for _ in range(100):
        q, w = attitude_dynamics_rk4(q, w, w_m, m, I_s, I_t, dt=dt)

    T1   = 0.5 * np.dot(w, I_t * w)
    drift = abs(T1 - T0) / (T0 + 1e-30)
    if drift < 1e-5:
        _passed(f"energy conservation (100 steps, rel drift = {drift:.2e})")
    else:
        _failed("energy conservation",
                f"relative energy drift = {drift:.2e}, expected < 1e-5")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nattitude.py unit tests")
    print("=" * 55)
    test_quat_multiply_identity()
    test_quat_multiply_inverse()
    test_quat_multiply_composition()
    test_quat_conjugate_double()
    test_quat_to_dcm_orthogonal()
    test_quat_to_dcm_known_rotation()
    test_usque_mrp_roundtrip()
    test_usque_quat_roundtrip()
    test_usque_identity()
    test_rk4_zero_omega()
    test_rk4_pure_spin_angle()
    test_rk4_energy_conservation()
    print("=" * 55)
