"""
orbit.py

Orbital mechanics functions for the UKF measurement model and initialization.

All units are SI (metres, seconds, radians) unless noted otherwise.

Keplerian elements convention:
    [a, e, i, O, w, M]
    a = semi-major axis          [m]
    e = eccentricity             [-]
    i = inclination              [rad]
    O = RAAN                     [rad]
    w = argument of perigee      [rad]
    M = mean anomaly             [rad]

Equinoctial elements convention (used internally, [p, f, g, h, k, L]):
    p = semi-latus rectum  = a(1 - e^2)    [m]
    f = e * cos(w + O)                      [-]
    g = e * sin(w + O)                      [-]
    h = tan(i/2) * cos(O)                   [-]
    k = tan(i/2) * sin(O)                   [-]
    L = O + w + M  (mean longitude)         [rad]

NS-ROE convention (scaled, [da, dL, dex, dey, dix, diy]):
    da  = a_d - a_c                         [m]
    dL  = a_c * (L_d - L_c)                [m]
    dex = a_c * (f_d - f_c)                [m]
    dey = a_c * (g_d - g_c)                [m]
    dix = a_c * (h_d - h_c)               [m]
    diy = a_c * (k_d - k_c)               [m]

Source: ukfspn_cpp/slab/orbit-shared.cc + kepler.cc
"""

import numpy as np

MU_EARTH = 3.986004418e14   # [m^3/s^2]


# ---------------------------------------------------------------------------
# Kepler equation solver
# ---------------------------------------------------------------------------

def _M2E(M: float, e: float, tol: float = 1e-12) -> float:
    """
    Solve Kepler's equation  E - e*sin(E) = M  via Newton-Raphson.

    Initial guess: nearest odd multiple of pi to M (avoids slow convergence
    near M=0 or M=2pi for low eccentricities).

    Source: ukfspn_cpp/slab/kepler.cc::M2E()
    """
    # Initial guess: floor(M / 2pi) * 2pi + pi
    E = np.floor(M / (2 * np.pi)) * (2 * np.pi) + np.pi
    while True:
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E -= dE
        if abs(dE) < tol:
            break
    return E


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Keplerian ↔ Cartesian ECI
# ---------------------------------------------------------------------------

def cartesian_to_keplerian(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Convert ECI Cartesian state to Keplerian orbital elements.

    r: [3] position [m]
    v: [3] velocity [m/s]

    Returns: [a, e, i, O, w, M]

    Source: ukfspn_cpp/slab/orbit-shared.cc::cartesian_to_keplerian()
    """
    mu  = MU_EARTH
    h   = np.cross(r, v)                        # specific angular momentum
    hh  = h / np.linalg.norm(h)                 # unit vector

    i   = np.arctan2(np.sqrt(hh[0]**2 + hh[1]**2), hh[2])
    O   = np.arctan2(hh[0], -hh[1])

    r_n = np.linalg.norm(r)
    v_n = np.linalg.norm(v)
    p   = np.dot(h, h) / mu
    a   = 1.0 / (2.0 / r_n - v_n**2 / mu)

    n   = np.sqrt(mu / a**3)                    # mean motion
    e   = np.sqrt(max(1.0 - p / a, 0.0))        # eccentricity

    # Eccentric anomaly
    E   = np.arctan2(np.dot(r, v) / (a**2 * n), 1.0 - r_n / a)
    M   = _wrap_to_pi(E - e * np.sin(E))

    # Argument of latitude u = w + nu
    u   = np.arctan2(r[2], -r[0] * hh[1] + r[1] * hh[0])
    nu  = np.arctan2(np.sqrt(max(1.0 - e**2, 0.0)) * np.sin(E), np.cos(E) - e)
    w   = _wrap_to_pi(u - nu)

    return np.array([a, e, i, O, w, M])


def keplerian_to_cartesian(oe: np.ndarray) -> np.ndarray:
    """
    Convert Keplerian orbital elements to ECI Cartesian state.

    oe: [a, e, i, O, w, M]

    Returns: [r(3), v(3)]  -- position [m] and velocity [m/s]

    Source: ukfspn_cpp/slab/orbit-shared.cc::keplerian_to_cartesian()
    """
    a, e, i, O, w, M = oe
    mu = MU_EARTH

    E   = _M2E(M, e)
    si, ci = np.sin(i), np.cos(i)
    sO, cO = np.sin(O), np.cos(O)
    sw, cw = np.sin(w), np.cos(w)
    sE, cE = np.sin(E), np.cos(E)

    # Perifocal basis vectors in ECI
    P = np.array([ cw*cO - sw*ci*sO,  cw*sO + sw*ci*cO,  sw*si])
    Q = np.array([-sw*cO - cw*ci*sO, -sw*sO + cw*ci*cO,  cw*si])

    r   = a * (cE - e) * P + a * np.sqrt(max(1.0 - e**2, 0.0)) * sE * Q
    r_n = np.linalg.norm(r)
    v   = (np.sqrt(mu * a) / r_n) * (-sE * P + np.sqrt(max(1.0 - e**2, 0.0)) * cE * Q)

    return np.concatenate([r, v])


# ---------------------------------------------------------------------------
# Keplerian ↔ Equinoctial
# ---------------------------------------------------------------------------

def keplerian_to_equinoctial(oe: np.ndarray) -> np.ndarray:
    """
    Convert Keplerian elements to equinoctial elements [p, f, g, h, k, L].

    Source: ukfspn_cpp/slab/orbit-shared.cc::keplerian_to_equinoctial()
    """
    a, e, i, O, w, M = oe
    p = a * (1.0 - e**2)
    f = e * np.cos(w + O)
    g = e * np.sin(w + O)
    h = np.tan(0.5 * i) * np.cos(O)
    k = np.tan(0.5 * i) * np.sin(O)
    L = _wrap_to_pi(O + w + M)
    return np.array([p, f, g, h, k, L])


def equinoctial_to_keplerian(eq: np.ndarray) -> np.ndarray:
    """
    Convert equinoctial elements [p, f, g, h, k, L] to Keplerian elements.

    Source: ukfspn_cpp/slab/orbit-shared.cc::equinoctial_to_keplerian()
    """
    p, f, g, h, k, L = eq
    fg = f**2 + g**2
    hk = h**2 + k**2
    a  = p / (1.0 - fg)
    e  = np.sqrt(fg)
    i  = np.arctan2(2.0 * np.sqrt(hk), 1.0 - hk)
    O  = np.arctan2(k, h)
    w  = np.arctan2(g * h - f * k, f * h + g * k)
    M  = _wrap_to_pi(L - O - w)
    return np.array([a, e, i, O, w, M])


# ---------------------------------------------------------------------------
# NS-ROE conversions
# ---------------------------------------------------------------------------

def roe_ns_to_equinoctial(chief_eq: np.ndarray, roe: np.ndarray) -> np.ndarray:
    """
    Convert chief equinoctial elements + NS-ROE to deputy equinoctial elements.

    chief_eq: [p, f, g, h, k, L]  chief equinoctial (p = semi-latus rectum)
    roe:      [da, dL, dex, dey, dix, diy]  scaled NS-ROE [m]

    Returns: [p, f, g, h, k, L]  deputy equinoctial

    Source: ukfspn_cpp/slab/orbit-shared.cc::roe_ns_to_equinoctial()
    """
    p_c, f_c, g_c, h_c, k_c, L_c = chief_eq
    da, dL, dex, dey, dix, diy    = roe

    # Chief semi-major axis from semi-latus rectum
    a_c = p_c / (1.0 - f_c**2 - g_c**2)

    a_d = a_c + da
    f_d = f_c + dex / a_c
    g_d = g_c + dey / a_c
    p_d = a_d * (1.0 - f_d**2 - g_d**2)
    h_d = h_c + dix / a_c
    k_d = k_c + diy / a_c
    L_d = L_c + dL  / a_c

    return np.array([p_d, f_d, g_d, h_d, k_d, L_d])


def kep_to_roe_ns(chief_kep: np.ndarray, deputy_kep: np.ndarray) -> np.ndarray:
    """
    Compute scaled NS-ROE from chief and deputy Keplerian elements.

    Returns: [da, dL, dex, dey, dix, diy]  [m]

    Source: ukfspn_cpp/slab/orbit-shared.cc::kep_to_roe_ns()
    """
    a_c = chief_kep[0]

    eq_c = keplerian_to_equinoctial(chief_kep)
    eq_d = keplerian_to_equinoctial(deputy_kep)

    _, f_c, g_c, h_c, k_c, L_c = eq_c
    _, f_d, g_d, h_d, k_d, L_d = eq_d

    da  = deputy_kep[0] - a_c
    dL  = a_c * (L_d - L_c)
    dex = a_c * (f_d - f_c)
    dey = a_c * (g_d - g_c)
    dix = a_c * (h_d - h_c)
    diy = a_c * (k_d - k_c)

    return np.array([da, dL, dex, dey, dix, diy])
