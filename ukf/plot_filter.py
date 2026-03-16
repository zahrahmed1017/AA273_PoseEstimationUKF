"""
plot_filter.py

Navigation filter plots for the UKF pose estimation results.

Produces two figures:
  Figure 1 — Translation error in camera frame (X, Y, Z) with ±3σ bounds
  Figure 2 — Attitude error as Yaw / Pitch / Roll (ZYX) with ±3σ bounds

Additionally prints per-axis mean absolute error statistics.

The ±3σ position bounds are computed by numerically propagating the ROE
covariance (from P_diag[0:6]) through the ROE → camera-position mapping
(Jacobian computed via finite differences on _roe_to_camera_pose).

The ±3σ attitude bounds use the MRP covariance (P_diag[6:9]).
For small errors the MRP components are approximately equal to the rotation
vector components, so sigma_mrp[i] ≈ sigma_angle[i]  (in radians).
This is an approximation — the MRP axes are in tpri body frame, not aligned
with camera-frame Euler angle axes — but it gives a reasonable consistency
check for the filter.

Euler angle convention matches spn_output_errors.m:
    R_error = R_pred.T @ R_gt
    yaw, pitch, roll = scipy ZYX decomposition of R_error
where R = quat_to_dcm(q)  (passive: maps body → camera).

Usage:
    python -m ukf.plot_filter [--results path/to/results.npz]
                               [--meta-csv path/to/roe1_meta.csv]
                               [--camera-json path/to/camera.json]
                               [--keypoints-mat path/to/tangoPoints.mat]
                               [--save-dir path/to/output/]
                               [--dt 1.0]
"""

import os
import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

# Use non-interactive backend if no display is available
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS      = os.path.join(_HERE, '..', 'results', 'run_filter_out.npz')
DEFAULT_META_CSV     = os.path.join(_HERE, '..', 'data',    'roe1_meta.csv')
DEFAULT_DATASET_ROOT = '/Users/Zahra1/Documents/Stanford/Research/datasets/shirtv1'
DEFAULT_CAMERA_JSON  = os.path.join(DEFAULT_DATASET_ROOT, 'camera.json')
DEFAULT_KP_MAT       = os.path.join(DEFAULT_DATASET_ROOT, '..', 'models',
                                    'tango', 'tangoPoints.mat')
DEFAULT_SAVE_DIR     = ''   # empty = show interactively, don't save

# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------

BLUE  = '#1f77b4'
GRAY  = '#aaaaaa'
SIGMA_ALPHA = 0.25    # fill transparency for ±3σ band

plt.rcParams.update({
    'font.size':       13,
    'axes.grid':       True,
    'grid.alpha':      0.4,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'figure.dpi':      120,
})


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Passive DCM from [qw, qx, qy, qz]: quat_to_dcm(q_AB) @ v_B = v_A."""
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q
    return np.array([
        [2*qw**2 - 1 + 2*qx**2,  2*qx*qy + 2*qw*qz,  2*qx*qz - 2*qw*qy],
        [2*qx*qy - 2*qw*qz,  2*qw**2 - 1 + 2*qy**2,  2*qy*qz + 2*qw*qx],
        [2*qx*qz + 2*qw*qy,  2*qy*qz - 2*qw*qx,  2*qw**2 - 1 + 2*qz**2],
    ])


def _euler_zyx_error_deg(q_pred: np.ndarray, q_gt: np.ndarray):
    """
    Compute attitude error as ZYX Euler angles (yaw, pitch, roll) in degrees.

    Matches MATLAB:
        R1 = quat2dcm(quat_gt);   R2 = quat2dcm(quat_pred);
        eul_ZYX = rotm2eul(R2' * R1);

    Both quaternions are [qw, qx, qy, qz] representing q_scam2tbdy.
    quat_to_dcm maps tbdy → scam.  R.T maps scam → tbdy.
    R_pred.T @ R_gt = (scam→tbdy_pred) @ (tbdy→scam_gt) = error rotation.
    """
    R_pred = _quat_to_dcm(q_pred)   # maps tbdy → scam
    R_gt   = _quat_to_dcm(q_gt)     # maps tbdy → scam
    R_err  = R_pred.T @ R_gt        # error in tbdy_pred frame
    yaw, pitch, roll = Rotation.from_matrix(R_err).as_euler('ZYX', degrees=True)
    return yaw, pitch, roll


# ---------------------------------------------------------------------------
# Numerical Jacobian: ROE → camera-frame translation
# ---------------------------------------------------------------------------

def _position_sigma3(ukf, roe: np.ndarray, q: np.ndarray,
                     P_diag_roe: np.ndarray, m_abs) -> np.ndarray:
    """
    Propagate ROE diagonal covariance to a 3σ bound on camera-frame translation.

    Computes  J = ∂t/∂roe  via central finite differences,
    then  P_t = J * diag(P_diag_roe) * J.T,
    returning  3 * sqrt(diag(P_t)).

    Perturbation size: 1% of the 1σ value for each element, clamped to [1mm, 1m].
    """
    _, t0 = ukf._roe_to_camera_pose(roe, q, m_abs)
    J = np.zeros((3, 6))
    for i in range(6):
        h = float(np.clip(0.01 * np.sqrt(P_diag_roe[i]), 1e-3, 1.0))
        roe_p = roe.copy(); roe_p[i] += h
        roe_m = roe.copy(); roe_m[i] -= h
        _, t_p = ukf._roe_to_camera_pose(roe_p, q, m_abs)
        _, t_m = ukf._roe_to_camera_pose(roe_m, q, m_abs)
        J[:, i] = (t_p - t_m) / (2.0 * h)
    P_t = J @ np.diag(P_diag_roe) @ J.T
    return 3.0 * np.sqrt(np.maximum(np.diag(P_t), 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Plot UKF navigation filter results')
    parser.add_argument('--results',      default=DEFAULT_RESULTS)
    parser.add_argument('--meta-csv',     default=DEFAULT_META_CSV)
    parser.add_argument('--camera-json',  default=DEFAULT_CAMERA_JSON)
    parser.add_argument('--keypoints-mat', default=DEFAULT_KP_MAT)
    parser.add_argument('--save-dir',     default=DEFAULT_SAVE_DIR,
                        help='Directory to save figures (empty = show interactively)')
    parser.add_argument('--dt',           type=float, default=1.0)
    args = parser.parse_args()

    # --- Load results --------------------------------------------------------
    print(f'Loading results from {args.results}')
    d = np.load(args.results)
    pose_q   = d['pose_q']       # (N, 4)  q_scam2tbdy estimate
    pose_t   = d['pose_t']       # (N, 3)  r_scam2tbdy_scam estimate [m]
    gt_q     = d['gt_q']         # (N, 4)  ground-truth q_scam2tbdy
    gt_t     = d['gt_t']         # (N, 3)  ground-truth r_scam2tbdy_scam [m]
    P_diag   = d['P_diag']       # (N, 12) filter state covariance diagonal
    filter_q = d['filter_q']     # (N, 4)  internal q_spri2tpri
    filter_roe = d['filter_roe'] # (N, 6)  NS-ROE filter state
    meta_kep = d['meta_kep']     # (N, 6)  chief Keplerian [a, e, i, O, w, M]

    N = len(pose_q)
    frames = np.arange(1, N + 1)
    print(f'  {N} frames loaded')

    # --- Build UKF (needed for _roe_to_camera_pose) --------------------------
    from .navigation import UKF, UKFParams, MangoAbsState

    with open(args.camera_json) as f:
        cam = json.load(f)
    K_mat = np.array(cam['cameraMatrix'], dtype=np.float64)
    dist  = np.array(cam.get('distCoeffs') or np.zeros(5), dtype=np.float64)

    kpts_3d = np.array(loadmat(args.keypoints_mat)['tango3Dpoints'],
                       dtype=np.float64).T     # [11 x 3]

    ukf = UKF(UKFParams(dt=args.dt), K_mat, dist, kpts_3d)

    # --- Load meta CSV for MangoAbsState -------------------------------------
    meta_rows = []
    with open(args.meta_csv) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(',')]
            if not parts or not parts[0]:
                continue
            meta_rows.append(MangoAbsState.from_csv_row(
                np.array(parts[:28], dtype=np.float64)
            ))

    # --- Compute per-frame errors --------------------------------------------
    trans_err    = pose_t - gt_t                    # (N, 3)  [m]

    yaw_err   = np.zeros(N)
    pitch_err = np.zeros(N)
    roll_err  = np.zeros(N)
    for i in range(N):
        yaw_err[i], pitch_err[i], roll_err[i] = _euler_zyx_error_deg(
            pose_q[i], gt_q[i]
        )

    # --- Attitude ±3σ from MRP covariance (P_diag[6:9]) ---------------------
    # MRP ≈ rotation vector for small errors (rad), then convert to degrees.
    # The three components correspond to the three attitude DOF; we use them
    # as an approximate per-axis bound for each Euler angle subplot.
    att_sigma3_deg = 3.0 * np.sqrt(np.maximum(P_diag[:, 6:9], 0.0)) * (180.0 / np.pi)
    # (N, 3) -- approximate ±3σ for yaw, pitch, roll respectively

    # --- Position ±3σ via numerical Jacobian ---------------------------------
    print('Computing camera-frame position covariance (numerical Jacobian)...')
    pos_sigma3 = np.zeros((N, 3))   # (N, 3)  3σ [m]
    for i in range(N):
        m_abs = meta_rows[min(i, len(meta_rows) - 1)]
        pos_sigma3[i] = _position_sigma3(
            ukf, filter_roe[i], filter_q[i], P_diag[i, :6], m_abs
        )
    print('  done.')

    # --- Time axis (in orbits) -----------------------------------------------
    # Use chief SMA from meta to compute orbital period
    a_c = meta_kep[0, 0]                              # chief SMA [m]
    MU  = 3.986004418e14
    T   = 2.0 * np.pi * np.sqrt(a_c**3 / MU)          # orbital period [s]
    time_s      = (frames - 1) * args.dt               # [s]
    time_orbits = time_s / T                           # [orbits]

    # --- Statistics ----------------------------------------------------------
    print('\nMean absolute errors (all frames):')
    print(f'  Translation X:  {np.nanmean(np.abs(trans_err[:, 0])):.4f} m')
    print(f'  Translation Y:  {np.nanmean(np.abs(trans_err[:, 1])):.4f} m')
    print(f'  Translation Z:  {np.nanmean(np.abs(trans_err[:, 2])):.4f} m')
    print(f'  Yaw:            {np.nanmean(np.abs(yaw_err)):.2f} deg')
    print(f'  Pitch:          {np.nanmean(np.abs(pitch_err)):.2f} deg')
    print(f'  Roll:           {np.nanmean(np.abs(roll_err)):.2f} deg')

    # --- Figure 1: Translation errors ----------------------------------------
    fig1, axes1 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig1.suptitle('Translation Error: Camera Frame', fontsize=15, fontweight='bold')

    trans_labels = ['X  [m]', 'Y  [m]', 'Z  [m]']
    for k, (ax, lbl) in enumerate(zip(axes1, trans_labels)):
        err_k    = trans_err[:, k]
        sigma3_k = pos_sigma3[:, k]
        mu       = np.nanmean(np.abs(err_k))

        # ±3σ band
        ax.fill_between(time_orbits, -sigma3_k, sigma3_k,
                        color=GRAY, alpha=SIGMA_ALPHA, label=r'$\pm 3\sigma$')
        # Error trace
        ax.plot(time_orbits, err_k, color=BLUE, linewidth=1.5,
                label=f'error  (|μ| = {mu:.4f} m)')
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')

        ax.set_ylabel(lbl)
        ax.legend(loc='upper right', fontsize=10)
        if k == 0:
            ax.set_title('Filter estimate − ground truth', fontsize=11)

    axes1[-1].set_xlabel('Time [orbits]')
    fig1.tight_layout()

    # --- Figure 2: Attitude errors (Yaw / Pitch / Roll) ----------------------
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig2.suptitle('Attitude Error: ZYX Euler Angles', fontsize=15, fontweight='bold')

    att_errors = [yaw_err, pitch_err, roll_err]
    att_labels = [
        'Yaw  [deg]',
        'Pitch  [deg]',
        'Roll  [deg]',
    ]

    for k, (ax, err_k, lbl) in enumerate(zip(axes2, att_errors, att_labels)):
        sigma3_k = att_sigma3_deg[:, k]
        mu       = np.nanmean(np.abs(err_k))

        ax.fill_between(time_orbits, -sigma3_k, sigma3_k,
                        color=GRAY, alpha=SIGMA_ALPHA,
                        label=r'$\pm 3\sigma$ (MRP, approx.)')
        ax.plot(time_orbits, err_k, color=BLUE, linewidth=1.5,
                label=f'error  (|μ| = {mu:.2f}°)')
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')

        ax.set_ylabel(lbl)
        ax.legend(loc='upper right', fontsize=10)
        if k == 0:
            ax.set_title(r'$R_{\rm pred}^T R_{\rm gt}$ decomposed ZYX', fontsize=11)

    axes2[-1].set_xlabel('Time [orbits]')
    fig2.tight_layout()

    # --- Figure 3: Total errors + ROE states ----------------------------------
    fig3, axes3 = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig3.suptitle('Total Pose Errors', fontsize=15, fontweight='bold')

    tot_trans = np.linalg.norm(trans_err, axis=1)
    tot_rot   = d['rot_errors']

    axes3[0].plot(time_orbits, tot_rot,   color=BLUE, linewidth=1.5)
    axes3[0].set_ylabel('Rotation error [deg]')
    axes3[0].set_title(
        f'Rotation  |μ| = {np.nanmean(tot_rot):.2f}°  '
        f'median = {np.nanmedian(tot_rot):.2f}°', fontsize=11
    )

    axes3[1].plot(time_orbits, tot_trans, color='tomato', linewidth=1.5)
    axes3[1].set_ylabel('Translation error [m]')
    axes3[1].set_xlabel('Time [orbits]')
    axes3[1].set_title(
        f'Translation  |μ| = {np.nanmean(tot_trans):.4f} m  '
        f'median = {np.nanmedian(tot_trans):.4f} m', fontsize=11
    )

    fig3.tight_layout()

    # --- Figure 4: ROE filter states with ±3σ ---------------------------------
    fig4, axes4 = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig4.suptitle('NS-ROE Filter States ± 3σ', fontsize=15, fontweight='bold')

    roe_labels = [
        r'$a\,\delta a$  [m]',
        r'$a\,\delta\lambda$  [m]',
        r'$a\,\delta e_x$  [m]',
        r'$a\,\delta e_y$  [m]',
        r'$a\,\delta i_x$  [m]',
        r'$a\,\delta i_y$  [m]',
    ]

    for k, (lbl) in enumerate(roe_labels):
        row, col = divmod(k, 2)
        ax       = axes4[row, col]
        roe_k    = filter_roe[:, k]
        sigma3_k = 3.0 * np.sqrt(np.maximum(P_diag[:, k], 0.0))

        ax.fill_between(time_orbits, roe_k - sigma3_k, roe_k + sigma3_k,
                        color=GRAY, alpha=SIGMA_ALPHA, label=r'$\pm 3\sigma$')
        ax.plot(time_orbits, roe_k, color=BLUE, linewidth=1.5, label='estimate')
        ax.axhline(0, color='k', linewidth=0.6, linestyle='--')
        ax.set_ylabel(lbl)
        ax.legend(loc='upper right', fontsize=9)

    for col in range(2):
        axes4[-1, col].set_xlabel('Time [orbits]')

    fig4.tight_layout()

    # --- Save or show --------------------------------------------------------
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        fig1.savefig(os.path.join(args.save_dir, 'translation_errors.png'),
                     bbox_inches='tight')
        fig2.savefig(os.path.join(args.save_dir, 'attitude_errors.png'),
                     bbox_inches='tight')
        fig3.savefig(os.path.join(args.save_dir, 'total_errors.png'),
                     bbox_inches='tight')
        fig4.savefig(os.path.join(args.save_dir, 'roe_states.png'),
                     bbox_inches='tight')
        print(f'\nFigures saved to {args.save_dir}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
