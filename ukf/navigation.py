"""
navigation.py

Core UKF implementation for spacecraft relative pose estimation.

State vector (12D):
    x[0:6]  = NS-ROE  [da, dL, dex, dey, dix, diy]    (meters, scaled by chief SMA)
    x[6:9]  = MRP     [p1, p2, p3]                    (GRP attitude error, a=1 f=4)
    x[9:12] = RAV     [w1, w2, w3]                    (rad/s, w_tpri2spri in tpri frame)

Quaternion (stored separately, not part of x):
    q = [qw, qx, qy, qz]  -- q_spri2tpri (servicer principal → target principal)
    ALL quaternions throughout this file use [qw, qx, qy, qz] convention.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Sub-module imports -- these will be implemented in their respective files.
# See attitude.py and orbit.py for function signatures and docstrings.
from .spacecraft import Mango, Tango
from .attitude import (
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    quat_to_dcm,
    error_quaternion_from_mrp,
    mrp_from_error_quaternion,
    attitude_dynamics_rk4,
)
from .orbit import (
    keplerian_to_equinoctial,
    roe_ns_to_equinoctial,
    equinoctial_to_keplerian,
    keplerian_to_cartesian,
    cartesian_to_keplerian,
    kep_to_roe_ns,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU_EARTH  = 3.986004418e14   # [m^3/s^2]
N_STATES  = 12
N_KP      = 11               # number of Tango keypoints
N_MEAS    = 22               # 11 keypoints x 2 (x, y pixel coords)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class UKFParams:
    """
    UKF tuning parameters.

    """

    alpha: float = 0.1
    beta:  float = 2.0
    kappa: float = 0.0
    dt:    float = 5.0   # timestep [s] -- set from metadata at runtime

    # Initial state covariance (1-sigma values, will be squared to get P diagonal)
    sigma_roe_da: float = 0.5    # [m]   semi-major axis difference
    sigma_roe_dl: float = 1.0    # [m]   mean longitude difference
    sigma_roe_de: float = 0.5     # [m]   eccentricity vector components
    sigma_roe_di: float = 0.01     # [m]   inclination vector components
    sigma_mrp:    float = np.deg2rad(10)     # [-]   MRP (attitude error)
    sigma_rav:    float = np.deg2rad(0.01)    # [rad/s] relative angular velocity

    # Fixed process noise diagonal values
    q_roe: float = 1e-7   # applied to all 6 ROE states
    q_att: float = 1e-7   # applied to all 6 attitude+RAV states

    # Mahalanobis outlier rejection threshold (chi-squared, 2 DOF)
    # 16.0 corresponds to ~4-sigma for a 2D measurement
    mahalanobis_threshold: float = 9.21

    @property
    def lambda_(self) -> float:
        return self.alpha**2 * (N_STATES + self.kappa) - N_STATES

    @property
    def n_sigmas(self) -> int:
        return 2 * N_STATES + 1


@dataclass
class MangoAbsState:
    """
    Mango (servicer) absolute state at one timestep.
    Read from the meta CSV: rv_eci2com_eci(6), oe_osc_kep(6), oe_osc_eq(6),
                            q_eci2pri(4), w_pri(3), m_pri(3)

    Quaternion convention: [qw, qx, qy, qz]
    """
    r_eci2com_eci: np.ndarray   # [3]  ECI position of Mango CoM [m]
    v_eci2com_eci: np.ndarray   # [3]  ECI velocity of Mango CoM [m/s]
    oe_osc_kep:    np.ndarray   # [6]  Keplerian elements [a, e, i, O, w, M]  (a in metres)
    oe_osc_eq:     np.ndarray   # [6]  Equinoctial elements [a, f, g, h, k, L] (note: a, not p)
    q_eci2pri:     np.ndarray   # [4]  [qw, qx, qy, qz]  ECI → Mango principal frame
    w_pri:         np.ndarray   # [3]  Angular velocity of spri w.r.t. ECI, in spri frame [rad/s]
    m_pri:         np.ndarray   # [3]  External torque on Mango in spri frame [N·m]

    @staticmethod
    def from_csv_row(row: np.ndarray) -> "MangoAbsState":
        """
        Parse one row of the meta CSV.
        Column order: rv_eci2com_eci(6), oe_osc_kep(6), oe_osc_eq(6),
                      q_eci2pri(4), w_pri(3), m_pri(3)
        """
        return MangoAbsState(
            r_eci2com_eci = row[0:3],
            v_eci2com_eci = row[3:6],
            oe_osc_kep    = row[6:12],
            oe_osc_eq     = row[12:18],
            q_eci2pri     = row[18:22],   # [qw, qx, qy, qz]
            w_pri         = row[22:25],
            m_pri         = row[25:28],
        )


@dataclass
class FilterRecord:
    """Output record for one UKF timestep."""

    # Filter state
    x:      np.ndarray = field(default_factory=lambda: np.zeros(N_STATES))
    P_diag: np.ndarray = field(default_factory=lambda: np.zeros(N_STATES))
    q:      np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))
    # q = [qw, qx, qy, qz]  q_spri2tpri

    # Residuals
    prefit_residual:  np.ndarray = field(default_factory=lambda: np.zeros(N_MEAS))
    postfit_residual: np.ndarray = field(default_factory=lambda: np.zeros(N_MEAS))

    # Predicted keypoints from post-update state [11 x 2] in pixels
    predicted_keypoints: np.ndarray = field(default_factory=lambda: np.zeros((N_KP, 2)))

    # Camera-to-target-body pose derived from filter state
    pose_q: np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))
    # pose_q = [qw, qx, qy, qz]  q_scam2tbdy
    pose_t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # pose_t = r_scam2tbdy_scam [m]

    # Errors vs ground truth (only populated when gt is available)
    rotation_error_deg:  float = 0.0
    translation_error_m: float = 0.0

    # Measurement bookkeeping
    n_keypoints_used: int  = 0
    reject_all:       bool = False


# ---------------------------------------------------------------------------
# UKF class
# ---------------------------------------------------------------------------

class UKF:
    """
    Unscented Kalman Filter for spacecraft relative pose estimation.

    State:  x = [NS-ROE (6), MRP (3), RAV (3)]
    Quat:   q = q_spri2tpri  [qw, qx, qy, qz]

    Measurement model:  project Tango 3D keypoints to 2D image plane
                        via ROE → relative Cartesian → camera pose → projection.

    Attitude parameterisation:  USQUE (Generalised Rodrigues Parameters)
        a = 1, f = 4  (same as cnnukf and ukfspn_cpp)
    """

    def __init__(
        self,
        params:        UKFParams,
        camera_matrix: np.ndarray,   # [3x3]
        dist_coeffs:   np.ndarray,   # [5]
        keypoints_3d:  np.ndarray,   # [11x3]  Tango 3D keypoints in body frame [m]
    ):
        self.p             = params
        self.camera_matrix = camera_matrix.astype(np.float64)
        self.dist_coeffs   = dist_coeffs.astype(np.float64)
        self.keypoints_3d  = keypoints_3d.astype(np.float64)  # [11 x 3]

        # State slice indices
        self.i_roe = slice(0, 6)
        self.i_mrp = slice(6, 9)
        self.i_rav = slice(9, 12)

        # UKF weights
        n   = N_STATES
        lam = self.p.lambda_
        self.w_mean    = np.full(self.p.n_sigmas, 0.5 / (n + lam))
        self.w_mean[0] = lam / (n + lam)
        self.w_cov     = self.w_mean.copy()
        self.w_cov[0] += (1.0 - self.p.alpha**2 + self.p.beta)

        # Filter state (set properly in initialize())
        self.x = np.zeros(N_STATES)
        self.P = np.eye(N_STATES)
        self.q = np.array([1., 0., 0., 0.])  # [qw, qx, qy, qz]

        # Fixed process noise
        self.Q = np.zeros((N_STATES, N_STATES))
        self.Q[self.i_roe, self.i_roe] = np.eye(6) * self.p.q_roe
        self.Q[6:12,  6:12]            = np.eye(6) * self.p.q_att

        # Pre-allocated sigma point arrays (avoids repeated allocation in the loop)
        self.sigma_x = np.zeros((N_STATES, self.p.n_sigmas))  # [12 x 25]
        self.sigma_q = np.zeros((4,        self.p.n_sigmas))  # [4  x 25]
        self.sigma_z = np.zeros((N_MEAS,   self.p.n_sigmas))  # [22 x 25]

    def initialize(self, meas, m_abs: MangoAbsState) -> None:
        """
        Initialise filter state from first SPN measurement.

        Uses EPnP on the accepted keypoints to get an initial camera→target pose,
        then converts that pose to the filter state representation.

        meas: SPNMeasurement from spn/model.py (fields: peaks [11x2], covs [11x2x2],
              maxvals [11], reject [11 bool])
        m_abs: MangoAbsState for the first frame
        """
        from spn.model import solve_pose

        # PnP → initial camera-to-target-body pose
        pose = solve_pose(meas, self.keypoints_3d, self.camera_matrix, self.dist_coeffs)
        if pose is None:
            raise RuntimeError("PnP failed on initialization frame -- too few keypoints.")

        # pose.q = [qw, qx, qy, qz]  q_scam2tbdy
        # pose.t = r_scam2tbdy_scam  [m]
        q_scam2tbdy      = pose.q.ravel()
        r_scam2tbdy_scam = pose.t.ravel()

        # Convert camera pose → filter state representation
        q_spri2tpri, r_scom2tcom_spri = _cam_pose_to_filter_state(
            q_scam2tbdy, r_scam2tbdy_scam
        )

        # Reference quaternion
        self.q = quat_normalize(q_spri2tpri)   # [qw, qx, qy, qz]

        # ROE: need Cartesian relative state → NS-ROE
        # Assume target not tumbling at t=0 → v ≈ cross(w_mango, r_rel)
        v_scom2tcom_spri = np.cross(m_abs.w_pri, r_scom2tcom_spri)
        roe = _cartesian_to_nsroe(
            r_scom2tcom_spri, v_scom2tcom_spri, m_abs
        )

        # Initial RAV: assume target angular velocity ≈ Mango's (no relative tumbling)
        # w_tpri2spri_tpri = R_spri2tpri * w_mango
        # quat_to_dcm(q_spri2tpri) maps tpri→spri (passive convention)
        # so .T maps spri→tpri
        R_spri2tpri = quat_to_dcm(q_spri2tpri).T
        w_tpri2spri_tpri = R_spri2tpri @ m_abs.w_pri

        # Assemble state
        self.x                = np.zeros(N_STATES)
        self.x[self.i_roe]    = roe
        self.x[self.i_mrp]    = np.zeros(3)   # MRP = 0 at reference quaternion
        self.x[self.i_rav]    = w_tpri2spri_tpri

        # Initial covariance
        sigmas = np.array([
            self.p.sigma_roe_da, self.p.sigma_roe_dl,
            self.p.sigma_roe_de, self.p.sigma_roe_de,
            self.p.sigma_roe_di, self.p.sigma_roe_di,
            self.p.sigma_mrp,    self.p.sigma_mrp,    self.p.sigma_mrp,
            self.p.sigma_rav,    self.p.sigma_rav,    self.p.sigma_rav,
        ])
        self.P = np.diag(sigmas**2)

    def step(
        self,
        meas,
        m_abs:  MangoAbsState,
        gt_q:   Optional[np.ndarray] = None,   # [qw,qx,qy,qz] ground-truth q_scam2tbdy
        gt_t:   Optional[np.ndarray] = None,   # [3] ground-truth r_scam2tbdy_scam [m]
    ) -> FilterRecord:
        """
        Run one full UKF step:
            1. Sigma point generation  (unscented transform)
            2. Sigma point propagation (dynamics)
            3. A priori mean & covariance
            4. Predicted measurements  (project sigma points to image plane)
            5. Innovation covariance   (+ measurement noise R from heatmap covs)
            6. Innovation              (actual - predicted measurement)
            7. Outlier rejection + Kalman update

        meas:  SPNMeasurement for this frame
        m_abs: MangoAbsState for this frame
        gt_q, gt_t: optional ground-truth pose for error logging

        Returns a FilterRecord with state, residuals, predicted keypoints, and errors.
        """
        # --- 1. Sigma points ---
        self._unscented_transform()

        # --- 2. Propagate ---
        self._propagate(m_abs)

        # --- 3. A priori mean & covariance ---
        self._sigmas_quat_to_mrp()
        self._inverse_ut_state()

        # --- 4. Re-generate sigma points from a priori mean/cov ---
        # Fresh sigma points are needed so the cross-covariance T = E[dx dz^T]
        # uses symmetric deviations centered on the a priori mean
        self._unscented_transform()

        # --- 5. Predicted measurements ---
        self._predict_measurements(m_abs)

        # --- 6 & 7. Measurement mean, covariance, innovation ---
        z_mean, S = self._innovation_covariance(meas)
        innovation = self._compute_innovation(meas, z_mean)

        # Store pre-fit residual before Kalman update
        prefit = innovation.copy()

        # --- 8. Outlier rejection + Kalman update ---
        reject_all = self._post_fit_update(meas, S, innovation, z_mean)

        # --- Build output record ---
        return self._make_record(meas, m_abs, prefit, reject_all, gt_q, gt_t)

    # -----------------------------------------------------------------------
    # UKF internals
    # -----------------------------------------------------------------------

    def _unscented_transform(self) -> None:
        """
        Generate 2N+1 sigma points from current (x, P, q).

        For the attitude component, MRP sigma point values are converted to
        full quaternion sigma points via USQUE:
            q_i = q_ref * error_quaternion_from_mrp(x_i[i_mrp])

        Reference: navigation.cc::unscented_transform()
        """
        n   = N_STATES
        lam = self.p.lambda_
        L   = np.linalg.cholesky((n + lam) * self.P)   # lower-triangular, [12x12]

        # Centre sigma point (index 0)
        self.sigma_x[:, 0] = self.x
        self.sigma_q[:, 0] = self.q   # [qw, qx, qy, qz]

        for i in range(n):
            # Positive perturbation (index i+1)
            x_pos               = self.x + L[:, i]
            dq_pos              = error_quaternion_from_mrp(x_pos[self.i_mrp])
            self.sigma_x[:, i + 1]     = x_pos
            self.sigma_q[:, i + 1]     = quat_multiply(self.q, dq_pos)

            # Negative perturbation (index i+N+1)
            x_neg               = self.x - L[:, i]
            dq_neg              = error_quaternion_from_mrp(x_neg[self.i_mrp])
            self.sigma_x[:, i + n + 1] = x_neg
            self.sigma_q[:, i + n + 1] = quat_multiply(self.q, dq_neg)

    def _propagate(self, m_abs: MangoAbsState) -> None:
        """
        Propagate all sigma points through the dynamics model.

        ROE:      linear STM (two-body Keplerian)
                  STM[1,0] = -1.5 * n * dt   (only dL drifts under Keplerian motion)
        Attitude: RK4 integration of rigid-body Euler equations
        MRP:      reset to zero after propagation (reference quaternion is updated)

        Reference: navigation.cc::propagate()
        """
        a  = m_abs.oe_osc_kep[0]                      # chief semi-major axis [m]
        n  = np.sqrt(MU_EARTH / a**3)                  # mean motion [rad/s]
        STM    = np.eye(6)
        STM[1, 0] = -1.5 * n * self.p.dt

        for i in range(self.p.n_sigmas):
            state = self.sigma_x[:, i].copy()
            q_i   = self.sigma_q[:, i].copy()          # [qw, qx, qy, qz]

            # ROE propagation
            state[self.i_roe] = STM @ state[self.i_roe]

            # Attitude + RAV propagation via RK4
            # attitude_dynamics_rk4 is implemented in attitude.py
            q_new, w_new = attitude_dynamics_rk4(
                q_i,                     # [qw,qx,qy,qz] q_spri2tpri
                state[self.i_rav],       # [3] w_tpri2spri_tpri [rad/s]
                m_abs.w_pri,             # [3] w_eci2spri_spri  [rad/s]
                m_abs.m_pri,             # [3] torque on Mango in spri [N·m]
                Mango.I_pri,             # [3] Mango principal MOI
                Tango.I_pri,             # [3] Tango principal MOI
                self.p.dt,
            )

            # MRP is reset to zero; reference quaternion will be updated in
            # _sigmas_quat_to_mrp() via the new quaternion sigma points
            state[self.i_mrp] = np.zeros(3)
            state[self.i_rav] = w_new

            self.sigma_x[:, i] = state
            self.sigma_q[:, i] = q_new                 # [qw, qx, qy, qz]

    def _sigmas_quat_to_mrp(self) -> None:
        """
        After propagation, convert quaternion sigma points back to MRP
        perturbations relative to the new reference quaternion (sigma_q[:,0]).

        This is the USQUE "back-projection" step: each propagated quaternion
        q_i is expressed as an error quaternion dq_i = q_ref^{-1} * q_i,
        then mapped to an MRP.

        Reference: navigation.cc::sigmas_quat_to_mrp()
        """
        q_ref_inv = quat_conjugate(self.sigma_q[:, 0])   # [qw, qx, qy, qz]

        # Centre point: error is zero by definition
        self.sigma_x[self.i_mrp, 0] = np.zeros(3)

        for i in range(1, self.p.n_sigmas):
            dq  = quat_multiply(q_ref_inv, self.sigma_q[:, i])   # [qw,qx,qy,qz]
            mrp = mrp_from_error_quaternion(dq)
            self.sigma_x[self.i_mrp, i] = mrp

    def _inverse_ut_state(self) -> None:
        """
        Compute a priori state mean and covariance from propagated sigma points,
        then add process noise Q.

        The reference quaternion after propagation is the propagated centre sigma
        point quaternion (sigma_q[:,0]).

        Reference: navigation.cc::inverse_unscented_transform_state()
        """
        # Mean state (weighted sum of sigma points)
        self.x = self.sigma_x @ self.w_mean                 # [12]

        # Absorb mean MRP into reference quaternion, then reset MRP to zero.
        # This maintains the invariant x[i_mrp] == 0 after every predict step,
        # making the state easier to inspect and debug.
        dq     = error_quaternion_from_mrp(self.x[self.i_mrp])
        self.q = quat_normalize(quat_multiply(self.sigma_q[:, 0], dq))
        self.x[self.i_mrp] = np.zeros(3)

        # State error matrix [12 x 25]: deviation of each sigma point from mean
        dx = self.sigma_x - self.x[:, None]

        # Covariance = weighted outer products + process noise
        self.P = (self.w_cov * dx) @ dx.T + self.Q          # [12 x 12]
        self.P = 0.5 * (self.P + self.P.T)                  # enforce symmetry

    def _predict_measurements(self, m_abs: MangoAbsState) -> None:
        """
        Project each sigma point through the measurement model to get predicted
        2D keypoint locations in the image plane.

        Measurement model:
            ROE + q_spri2tpri  →  camera pose (q_scam2tbdy, r_scam2tbdy_scam)
                               →  cv2.projectPoints  →  [x0,y0, x1,y1, ..., x10,y10]

        Reference: navigation.cc::measurement_update() inner loop
        """
        for i in range(self.p.n_sigmas):
            roe = self.sigma_x[self.i_roe, i]
            q_i = self.sigma_q[:, i]                    # [qw, qx, qy, qz]

            q_scam2tbdy, r_scam2tbdy_scam = self._roe_to_camera_pose(roe, q_i, m_abs)

            kpts = _project_keypoints(
                q_scam2tbdy, r_scam2tbdy_scam,
                self.keypoints_3d, self.camera_matrix, self.dist_coeffs,
            )  # [22] flat: [x0, y0, x1, y1, ...]

            self.sigma_z[:, i] = kpts

    def _innovation_covariance(self, meas) -> tuple:
        """
        Compute predicted measurement mean and innovation covariance S = Pzz + R.

        R is assembled per-keypoint from heatmap covariances (adaptive measurement
        noise): meas.covs[k] is the [2x2] covariance for keypoint k from the SPN.
        Rejected keypoints contribute zero to R (their S diagonal is already
        inflated by outlier rejection in _post_fit_update).

        Reference: navigation.cc::inverse_unscented_transform_meas()
        """
        # Predicted measurement mean
        z_mean = self.sigma_z @ self.w_mean              # [22]

        # Innovation covariance (from sigma point spread)
        dz  = self.sigma_z - z_mean[:, None]             # [22 x 25]
        Pzz = (self.w_cov * dz) @ dz.T                  # [22 x 22]

        # Measurement noise R -- per-keypoint heatmap covariance + noise floor.
        # The floor accounts for systematic SPN bias that heatmap spread alone
        # does not capture (peaks can be tight but far from ground truth).
        R = np.zeros((N_MEAS, N_MEAS))
        floor = self.p.r_floor * np.eye(2)
        for k in range(N_KP):
            if not meas.reject[k]:
                R[2*k:2*k+2, 2*k:2*k+2] = meas.covs[k] + floor  # [2x2]

        S = Pzz + R
        S = 0.5 * (S + S.T)

        return z_mean, S

    def _compute_innovation(self, meas, z_mean: np.ndarray) -> np.ndarray:
        """
        Innovation = actual SPN measurement - predicted measurement mean.

        For rejected keypoints the innovation will be zeroed out in
        _post_fit_update, so we can safely compute it here for all keypoints.
        """
        innovation = np.zeros(N_MEAS)
        for k in range(N_KP):
            innovation[2*k]   = meas.peaks[k, 0] - z_mean[2*k]
            innovation[2*k+1] = meas.peaks[k, 1] - z_mean[2*k+1]
        return innovation

    def _post_fit_update(self, meas, S: np.ndarray, innovation: np.ndarray, z_mean: np.ndarray) -> bool:
        """
        Mahalanobis outlier rejection followed by Kalman gain update.

        For each keypoint:
          - If meas.reject[k] is True: inflate S diagonal, zero innovation.
          - Otherwise: compute 2D Mahalanobis distance using the 2x2 block of S.
            If distance > threshold: inflate S diagonal, zero innovation.

        If more than 7 of the 11 keypoints are rejected, skip the entire update.

        Reference: navigation.cc::mahalanobis_outlier_rejection() +
                   navigation.cc::post_fit_update()
        """
        n_rejects = 0
        for k in range(N_KP):
            if meas.reject[k]:
                S[2*k, 2*k]     = 1e15
                S[2*k+1, 2*k+1] = 1e15
                innovation[2*k]   = 0.0
                innovation[2*k+1] = 0.0
                n_rejects += 1
            else:
                dx   = innovation[2*k]
                dy   = innovation[2*k+1]
                Sxx  = S[2*k,   2*k]
                Sxy  = S[2*k,   2*k+1]
                Syy  = S[2*k+1, 2*k+1]
                denom = Sxx * Syy - Sxy**2
                if denom > 0.0:
                    maha_sq = (Syy*dx**2 - 2*Sxy*dx*dy + Sxx*dy**2) / denom
                    if maha_sq > self.p.mahalanobis_threshold:
                        S[2*k, 2*k]     = 1e15
                        S[2*k+1, 2*k+1] = 1e15
                        innovation[2*k]   = 0.0
                        innovation[2*k+1] = 0.0
                        n_rejects += 1

        reject_all = (n_rejects > 7)

        if not reject_all:
            self._kalman_update(S, innovation, z_mean)

        return reject_all

    def _kalman_update(self, S: np.ndarray, innovation: np.ndarray, z_mean: np.ndarray) -> None:
        """
        Kalman gain computation and state/covariance update.

        T = sum_i w_i * (x_i - x_mean)(z_i - z_mean)^T   [12 x 22]

        State update (avoids forming K = T S^{-1} explicitly):
            x += T * S^{-1} * innovation
            q  = q * error_quaternion_from_mrp(x[i_mrp])
            x[i_mrp] = 0  (MRP reset)

        Covariance update:
            P -= T * S^{-1} * T^T
        """
        dx = self.sigma_x - self.x[:, None]          # [12 x 25]
        dz = self.sigma_z - z_mean[:, None]          # [22 x 25]

        # Cross-covariance T [12 x 22]
        T = (self.w_cov * dx) @ dz.T

        # State correction: 
        self.x += T @ np.linalg.solve(S, innovation)

        # Quaternion correction: absorb MRP correction into reference quaternion
        dq     = error_quaternion_from_mrp(self.x[self.i_mrp])
        self.q = quat_normalize(quat_multiply(self.q, dq))
        self.x[self.i_mrp] = np.zeros(3)              # reset MRP to zero

        # Covariance update:
        self.P -= T @ np.linalg.solve(S, T.T)
        self.P  = 0.5 * (self.P + self.P.T)

    # -----------------------------------------------------------------------
    # Geometry: ROE + attitude -> camera pose
    # -----------------------------------------------------------------------

    def _roe_to_camera_pose(
        self,
        roe:          np.ndarray,   # [6] NS-ROE
        q_spri2tpri:  np.ndarray,   # [4] [qw, qx, qy, qz]
        m_abs:        MangoAbsState,
    ) -> tuple:
        """
        Convert NS-ROE + attitude to camera-to-target-body pose.

        Steps:
          1. ROE → deputy (Tango) Keplerian elements
          2. Tango Keplerian → ECI Cartesian
          3. Relative ECI position → principal frame (spri)
          4. Attitude conversion:  q_spri2tpri → q_scam2tbdy
          5. Position conversion:  r_scom2tcom_spri → r_scam2tbdy_scam

        All quaternions [qw, qx, qy, qz].
        Rotation matrix convention:  quat_to_dcm(q_AB) maps frame-B vectors to
        frame-A  (i.e.  quat_to_dcm(q_AB) @ v_B = v_A ).

        Reference: navigation.cc::spri2tpri_to_scam2tbdy()
        """
        # --- Step 1-2: ROE → Tango ECI Cartesian ---
        # Chief (Mango) equinoctial elements in [p, f, g, h, k, L] form
        eq_mango = keplerian_to_equinoctial(m_abs.oe_osc_kep)
        # Deputy (Tango) equinoctial via NS-ROE
        eq_tango = roe_ns_to_equinoctial(eq_mango, roe)
        # Tango ECI state [r(3), v(3)]
        kep_tango    = equinoctial_to_keplerian(eq_tango)
        rv_tango_eci = keplerian_to_cartesian(kep_tango)    # [6]

        # --- Step 3: Relative position ECI → spri ---
        # quat_to_dcm(q_pri2eci) @ v_eci = v_pri  (maps ECI → spri)
        # q_pri2eci = conjugate of q_eci2pri
        q_pri2eci  = quat_conjugate(m_abs.q_eci2pri)        # [qw, qx, qy, qz]
        R_eci2spri = quat_to_dcm(q_pri2eci)                 # maps ECI → spri
        r_scom2tcom_spri = R_eci2spri @ (rv_tango_eci[:3] - m_abs.r_eci2com_eci)

        # --- Step 4: Attitude ---
        # q_scam2tbdy = q_cam2pri * q_spri2tpri * q_pri2bdy
        # where q_cam2pri = conj(q_pri2cam)  [all [qw,qx,qy,qz]]
        q_cam2pri   = quat_conjugate(Mango.q_pri2cam)
        q_scam2tbdy = quat_multiply(
            quat_multiply(q_cam2pri, q_spri2tpri),
            Tango.q_pri2bdy,
        )

        # --- Step 5: Position ---
        # R_spri2scam = quat_to_dcm(q_cam2pri)  maps spri → scam
        # (quat_to_dcm(q_AB) @ v_B = v_A, so q_AB=q_cam2pri maps pri → cam ✓)
        R_spri2scam      = quat_to_dcm(q_cam2pri)
        r_scam2tcom_scam = R_spri2scam @ (r_scom2tcom_spri - Mango.r_pri2cam_pri)

        # R_tpri2scam = quat_to_dcm(q_scam2tpri)  maps tpri → scam
        q_bdy2tpri   = quat_conjugate(Tango.q_pri2bdy)
        q_scam2tpri  = quat_multiply(q_scam2tbdy, q_bdy2tpri)
        R_tpri2scam  = quat_to_dcm(q_scam2tpri)

        r_scam2tbdy_scam = r_scam2tcom_scam + R_tpri2scam @ Tango.r_pri2bdy_pri

        return q_scam2tbdy, r_scam2tbdy_scam

    # -----------------------------------------------------------------------
    # Record builder
    # -----------------------------------------------------------------------

    def _make_record(
        self,
        meas,
        m_abs:     MangoAbsState,
        prefit:    np.ndarray,
        reject_all: bool,
        gt_q:      Optional[np.ndarray],
        gt_t:      Optional[np.ndarray],
    ) -> FilterRecord:
        """
        Assemble a FilterRecord from the current (post-update) filter state.
        Also computes post-fit residuals and optionally pose errors vs ground truth.
        """
        rec            = FilterRecord()
        rec.x          = self.x.copy()
        rec.P_diag     = np.diag(self.P).copy()
        rec.q          = self.q.copy()       # [qw, qx, qy, qz]
        rec.prefit_residual = prefit
        rec.reject_all      = reject_all
        rec.n_keypoints_used = int((~meas.reject).sum())

        # Camera pose from post-update state
        q_scam2tbdy, r_scam2tbdy_scam = self._roe_to_camera_pose(
            self.x[self.i_roe], self.q, m_abs
        )
        rec.pose_q = q_scam2tbdy         # [qw, qx, qy, qz]
        rec.pose_t = r_scam2tbdy_scam    # [m]

        # Predicted keypoints from post-update pose
        kpts = _project_keypoints(
            q_scam2tbdy, r_scam2tbdy_scam,
            self.keypoints_3d, self.camera_matrix, self.dist_coeffs,
        )
        rec.predicted_keypoints = kpts.reshape(N_KP, 2)

        # Post-fit residuals: actual measurement - post-update prediction
        postfit = np.zeros(N_MEAS)
        for k in range(N_KP):
            postfit[2*k]   = meas.peaks[k, 0] - kpts[2*k]
            postfit[2*k+1] = meas.peaks[k, 1] - kpts[2*k+1]
        rec.postfit_residual = postfit

        # Errors vs ground truth (optional)
        if gt_q is not None and gt_t is not None:
            rec.rotation_error_deg  = _rotation_error_deg(q_scam2tbdy, gt_q)
            rec.translation_error_m = float(np.linalg.norm(r_scam2tbdy_scam - gt_t))

        return rec


# ---------------------------------------------------------------------------
# Module-level geometry helpers
# ---------------------------------------------------------------------------

def _cam_pose_to_filter_state(
    q_scam2tbdy:      np.ndarray,   # [4] [qw, qx, qy, qz]
    r_scam2tbdy_scam: np.ndarray,   # [3] [m]
) -> tuple:
    """
    Convert camera-to-target-body pose to filter state representation.
    Inverse of UKF._roe_to_camera_pose (attitude and position parts only).

    Returns:
        q_spri2tpri:      [4] [qw, qx, qy, qz]
        r_scom2tcom_spri: [3] [m]

    Reference: UnscentedKalmanFilter.m::setInitialStateViaCNN()
    """
    # --- Attitude ---
    # q_spri2tpri = q_pri2cam * q_scam2tbdy * q_bdy2tpri
    q_bdy2tpri  = quat_conjugate(Tango.q_pri2bdy)
    q_spri2tpri = quat_multiply(
        quat_multiply(Mango.q_pri2cam, q_scam2tbdy),
        q_bdy2tpri,
    )

    # --- Position ---
    # r_scam2tbdy_scam = r_scam2tcom_scam + R_tpri2scam * r_tpri2tbdy_tpri
    # → r_scam2tcom_scam = r_scam2tbdy_scam - R_tpri2scam * r_pri2bdy_pri
    q_bdy2tpri  = quat_conjugate(Tango.q_pri2bdy)
    q_scam2tpri = quat_multiply(q_scam2tbdy, q_bdy2tpri)
    R_tpri2scam = quat_to_dcm(q_scam2tpri)     # maps tpri → scam
    r_scam2tcom_scam = r_scam2tbdy_scam - R_tpri2scam @ Tango.r_pri2bdy_pri

    # r_scam2tcom_scam → r_scom2tcom_spri
    # quat_to_dcm(q_pri2cam) maps cam → pri  (q_AB maps B → A)
    R_cam2spri       = quat_to_dcm(Mango.q_pri2cam)
    r_scom2tcom_spri = R_cam2spri @ r_scam2tcom_scam + Mango.r_pri2cam_pri

    return q_spri2tpri, r_scom2tcom_spri


def _cartesian_to_nsroe(
    r_scom2tcom_spri: np.ndarray,   # [3] relative position in spri [m]
    v_scom2tcom_spri: np.ndarray,   # [3] relative velocity in spri [m/s]
    m_abs:            MangoAbsState,
) -> np.ndarray:
    """
    Convert relative Cartesian state (in spri frame) to NS-ROE for filter init.

    Steps:
      1. Rotate relative state from spri to ECI
      2. Add Mango ECI state to get Tango ECI state
      3. Both → Keplerian → NS-ROE via kep_to_roe_ns

    Reference: navigation.cc::initialize() + UnscentedKalmanFilter.m::setInitialStateViaCNN()
    """
    # quat_to_dcm(q_eci2pri) maps pri → eci  (q_AB maps B → A, so q_eci2pri maps pri → eci ✓)
    R_spri2eci  = quat_to_dcm(m_abs.q_eci2pri)
    r_tango_eci = m_abs.r_eci2com_eci + R_spri2eci @ r_scom2tcom_spri
    v_tango_eci = m_abs.v_eci2com_eci + R_spri2eci @ v_scom2tcom_spri

    kep_mango = cartesian_to_keplerian(m_abs.r_eci2com_eci, m_abs.v_eci2com_eci)
    kep_tango = cartesian_to_keplerian(r_tango_eci, v_tango_eci)

    return kep_to_roe_ns(kep_mango, kep_tango)


def _project_keypoints(
    q_scam2tbdy:      np.ndarray,   # [4] [qw, qx, qy, qz]
    r_scam2tbdy_scam: np.ndarray,   # [3] [m]
    keypoints_3d:     np.ndarray,   # [11 x 3]
    camera_matrix:    np.ndarray,   # [3 x 3]
    dist_coeffs:      np.ndarray,   # [5]
) -> np.ndarray:
    """
    Project 3D Tango keypoints (body frame) to 2D image plane using cv2.

    cv2.projectPoints computes X_cam = R * X_body + t, so R must map body → cam.
    quat_to_dcm(q_scam2tbdy) produces the same matrix as slab-spn's quat2dcm(q),
    which maps cam → body. The transpose gives the body → cam rotation needed here.
    (slab-spn reference: postprocess.py::project_keypoints uses quat2dcm(q).T)

    Returns: [22] flat array [x0, y0, x1, y1, ..., x10, y10] in pixels.
    """
    import cv2

    # quat_to_dcm(q_scam2tbdy) maps cam → body; transpose gives body → cam ✓
    R_tbdy2scam = quat_to_dcm(q_scam2tbdy).T
    rvec, _     = cv2.Rodrigues(R_tbdy2scam)
    tvec        = r_scam2tbdy_scam.reshape(3, 1)

    pts2d, _ = cv2.projectPoints(
        keypoints_3d.astype(np.float64),
        rvec, tvec,
        camera_matrix, dist_coeffs,
    )
    return pts2d.reshape(-1)   # [22]


def _rotation_error_deg(q_est: np.ndarray, q_gt: np.ndarray) -> float:
    """
    Angular difference between two [qw, qx, qy, qz] quaternions, in degrees.
    """
    from scipy.spatial.transform import Rotation
    R_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]]).as_matrix()
    R_gt  = Rotation.from_quat([q_gt[1],  q_gt[2],  q_gt[3],  q_gt[0]]).as_matrix()
    trace = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))
