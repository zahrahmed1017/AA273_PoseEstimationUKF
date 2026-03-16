"""
test_orbit.py

Unit tests for orbit.py.

Tests:
  1. Keplerian round-trip: cart → kep → cart
  2. Equinoctial round-trip: kep → eq → kep
  3. ROE round-trip: kep → roe → eq → kep (deputy recovers original)
  4. ROE = 0 when chief == deputy

Run:
    python -m ukf.test_orbit
"""

import numpy as np
from .orbit import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    keplerian_to_equinoctial,
    equinoctial_to_keplerian,
    roe_ns_to_equinoctial,
    kep_to_roe_ns,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _passed(name: str) -> None:
    print(f"  PASS  {name}")

def _failed(name: str, msg: str) -> None:
    print(f"  FAIL  {name}: {msg}")

def assert_close(a, b, tol, name):
    err = np.max(np.abs(a - b))
    if err < tol:
        _passed(f"{name}  (max err = {err:.2e})")
    else:
        _failed(name, f"max err = {err:.2e}, tol = {tol:.2e}")
        _failed(name, f"  a = {a}")
        _failed(name, f"  b = {b}")


# ---------------------------------------------------------------------------
# Test data: ISS-like circular orbit
# ---------------------------------------------------------------------------

# Chief: circular orbit at ~400 km altitude, 51.6 deg inclination
MU   = 3.986004418e14
a_c  = 6778e3         # [m]  ~400 km altitude
e_c  = 0.001          # near-circular
i_c  = np.radians(51.6)
O_c  = np.radians(30.0)
w_c  = np.radians(45.0)
M_c  = np.radians(120.0)
KP_CHIEF = np.array([a_c, e_c, i_c, O_c, w_c, M_c])

# Deputy: small separation (~100m along-track)
a_d  = a_c + 5.0       # 5 m SMA difference
e_d  = e_c + 1e-5
i_d  = i_c + 1e-6
O_d  = O_c
w_d  = w_c
M_d  = M_c + 1e-5
KP_DEPUTY = np.array([a_d, e_d, i_d, O_d, w_d, M_d])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cartesian_roundtrip():
    """cart → kep → cart should recover original r, v."""
    rv      = keplerian_to_cartesian(KP_CHIEF)
    kp_back = cartesian_to_keplerian(rv[:3], rv[3:])
    rv_back = keplerian_to_cartesian(kp_back)
    assert_close(rv, rv_back, tol=1e-3, name="cart → kep → cart  (1 mm / 0.001 m/s)")


def test_keplerian_equinoctial_roundtrip():
    """kep → eq → kep should recover original elements."""
    eq      = keplerian_to_equinoctial(KP_CHIEF)
    kp_back = equinoctial_to_keplerian(eq)
    assert_close(KP_CHIEF, kp_back, tol=1e-10, name="kep → eq → kep")


def test_roe_zero_for_identical_orbits():
    """ROE should be zero when chief == deputy."""
    roe = kep_to_roe_ns(KP_CHIEF, KP_CHIEF)
    assert_close(roe, np.zeros(6), tol=1e-8, name="ROE = 0 when chief == deputy")


def test_roe_roundtrip():
    """
    kep_chief + kep_deputy → ROE → deputy equinoctial → kep_deputy
    should recover the original deputy Keplerian elements.
    """
    roe    = kep_to_roe_ns(KP_CHIEF, KP_DEPUTY)
    eq_c   = keplerian_to_equinoctial(KP_CHIEF)
    eq_d   = roe_ns_to_equinoctial(eq_c, roe)
    kp_rec = equinoctial_to_keplerian(eq_d)
    assert_close(KP_DEPUTY, kp_rec, tol=1e-6, name="kep → ROE → eq → kep  (deputy round-trip)")


def test_roe_cartesian_roundtrip():
    """
    kep_deputy → kep_to_roe_ns → roe_ns_to_equinoctial → equinoctial_to_keplerian
    → keplerian_to_cartesian  should give the same ECI position as direct
    keplerian_to_cartesian(kep_deputy).
    """
    rv_direct = keplerian_to_cartesian(KP_DEPUTY)

    roe   = kep_to_roe_ns(KP_CHIEF, KP_DEPUTY)
    eq_c  = keplerian_to_equinoctial(KP_CHIEF)
    eq_d  = roe_ns_to_equinoctial(eq_c, roe)
    kp_d  = equinoctial_to_keplerian(eq_d)
    rv_roe = keplerian_to_cartesian(kp_d)

    assert_close(rv_direct[:3], rv_roe[:3], tol=1e-3,
                 name="ROE path vs direct: position  (1 mm)")
    assert_close(rv_direct[3:], rv_roe[3:], tol=1e-6,
                 name="ROE path vs direct: velocity  (1 um/s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\norbit.py unit tests")
    print("=" * 50)
    test_cartesian_roundtrip()
    test_keplerian_equinoctial_roundtrip()
    test_roe_zero_for_identical_orbits()
    test_roe_roundtrip()
    test_roe_cartesian_roundtrip()
    print("=" * 50)
