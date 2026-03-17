"""
run_filter.py

End-to-end UKF runner for the SHIRT roe1 dataset.

For each frame:
  1. Load image and denormalize GT bounding box
  2. Run SPN inference → SPNMeasurements (peaks, covs, reject)
  3. Feed into UKF (initialize on first frame, step thereafter)
  4. Record FilterRecord (state, covariance, pose, errors, residuals)

Usage:
    python -m ukf.run_filter [--dataset-root PATH] [--traj roe1] [--max-frames N]
                             [--dt 1.0] [--save results.npz]

All paths can also be set by editing the DEFAULT_* constants below.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
from spn.model import solve_pose
from .navigation import _rotation_error_deg


# ---------------------------------------------------------------------------
# Default paths  (edit these for your machine if you don't want to use flags)
# ---------------------------------------------------------------------------

DEFAULT_SLAB_SPN_ROOT = '/Users/Zahra1/Documents/Stanford/Classes/AA273/FinalProject/slab-spn'
DEFAULT_CFG_YAML      = os.path.join(DEFAULT_SLAB_SPN_ROOT,
                                     'experiments', 'cfg_shirt_baseline.yaml')
DEFAULT_CHECKPOINT    = os.path.join(DEFAULT_SLAB_SPN_ROOT, 'output',
                                     'efficientnet_b0.ra_in1k',
                                     'baseline_incAug_20251022',
                                     'model_best.pth.tar')

DEFAULT_DATASET_ROOT  = '/Users/Zahra1/Documents/Stanford/Research/datasets/shirtv1'
DEFAULT_TRAJ          = 'roe1'
DEFAULT_DOMAIN        = 'synthetic'
DEFAULT_METADATA_JSON = os.path.join(DEFAULT_DATASET_ROOT, DEFAULT_TRAJ, 'metadata.json')

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LABEL_CSV     = os.path.join(_HERE, '..', 'data', 'roe1.csv')
DEFAULT_META_CSV      = os.path.join(_HERE, '..', 'data', 'roe1_meta.csv')
DEFAULT_KEYPOINTS_MAT = os.path.join(DEFAULT_DATASET_ROOT, '..', 'models',
                                     'tango', 'tangoPoints.mat')
DEFAULT_CAMERA_JSON   = os.path.join(DEFAULT_DATASET_ROOT, 'camera.json')
DEFAULT_SAVE          = os.path.join(_HERE, '..', 'results', 'run_filter_out.mat')

IMAGE_SIZE = (1920, 1200)   # (W, H) -- VBS camera, used to denormalize bbox


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def load_label_csv(path: str):
    """
    Load roe1.csv into parallel arrays.

    Columns (SHIRT / SPEED+ format):
        [0]     filename
        [1:5]   bbox_norm  [xmin, ymin, xmax, ymax]  normalised to [0,1]
        [5:9]   q_gt       [qw, qx, qy, qz]  ground-truth q_scam2tbdy
        [9:12]  t_gt       [tx, ty, tz]  ground-truth r_scam2tbdy_scam [m]
        [12:14] (unused)
        [14:36] kpts_norm  11 keypoints as [x0_n, y0_n, x1_n, y1_n, ...]  normalised to [0,1]
    """
    filenames  = []
    bboxes     = []
    gt_qs      = []
    gt_ts      = []
    gt_kpts    = []   # (N, 11, 2) in normalised image coordinates

    with open(path) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(',')]
            if not parts or not parts[0]:
                continue
            filenames.append(parts[0])
            bboxes.append(np.array(parts[1:5],  dtype=np.float64))
            gt_qs.append(np.array(parts[5:9],   dtype=np.float64))   # [qw, qx, qy, qz]
            gt_ts.append(np.array(parts[9:12],  dtype=np.float64))
            kpts = np.array(parts[14:36], dtype=np.float64).reshape(11, 2)  # (11, 2) normalised
            gt_kpts.append(kpts)

    return filenames, np.array(bboxes), np.array(gt_qs), np.array(gt_ts), np.array(gt_kpts)


def load_meta_csv(path: str):
    """
    Load roe1_meta.csv into a list of MangoAbsState objects.

    Columns (28 total):
        [0:6]   rv_eci2com_eci  [rx, ry, rz, vx, vy, vz]  ECI position+velocity [m, m/s]
        [6:12]  oe_osc_kep      [a, e, i, O, w, M]  Keplerian [m, rad]
        [12:18] oe_osc_eq       [a, f, g, h, k, L]  Equinoctial (NOTE: a, not p)
        [18:22] q_eci2pri       [qw, qx, qy, qz]
        [22:25] w_pri           [rad/s]
        [25:28] m_pri           [N·m]
    """
    from .navigation import MangoAbsState

    rows = []
    with open(path) as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(',')]
            if not parts or not parts[0]:
                continue
            row = np.array(parts[:28], dtype=np.float64)
            rows.append(MangoAbsState.from_csv_row(row))
    return rows


# ---------------------------------------------------------------------------
# Camera / keypoints loaders
# ---------------------------------------------------------------------------

def load_camera_intrinsics(camera_json: str):
    with open(camera_json) as f:
        cam = json.load(f)
    cam = {k: np.array(v, dtype=np.float64) for k, v in cam.items()}
    if 'distCoeffs' not in cam or cam['distCoeffs'] is None:
        cam['distCoeffs'] = np.zeros(5, dtype=np.float64)
    return cam


def load_tango_keypoints(mat_path: str) -> np.ndarray:
    """Load 3D Tango keypoints from tangoPoints.mat.  Returns [11 x 3] float64."""
    pts = loadmat(mat_path)['tango3Dpoints']   # [3 x 11]
    return np.array(pts, dtype=np.float64).T   # [11 x 3]


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _plot_keypoints(
    image:       np.ndarray,   # (H, W) uint8 grayscale
    kpts_raw:    np.ndarray,   # (11, 2) GT keypoints from CSV, already in pixel coords
    kpts_solved: np.ndarray,   # (11, 2) GT pose re-projected to image
    meas,                      # SPNMeasurements  (.peaks (11,2), .reject (11,) bool)
    frame_idx:   int,
    block:       bool = False,
) -> None:
    """
    Plot three keypoint sets on the raw image:

      Yellow circles  — Ground truth raw keypoints (direct from label CSV)
      Cyan   circles  — Ground truth solved keypoints (GT pose re-projected)
      Green  circles  — SPN accepted detections
      Red    x marks  — SPN rejected detections
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)

    # GT raw keypoints (from label CSV)
    ax.scatter(kpts_raw[:, 0], kpts_raw[:, 1],
               s=100, facecolors='none', edgecolors='yellow', linewidths=2,
               label='GT raw (CSV)', zorder=3)

    # GT solved keypoints (re-projected from GT pose)
    ax.scatter(kpts_solved[:, 0], kpts_solved[:, 1],
               s=100, facecolors='none', edgecolors='cyan', linewidths=2,
               label='GT solved (re-projected)', zorder=3)

    # SPN detections
    accepted = ~meas.reject
    if accepted.any():
        ax.scatter(meas.peaks[accepted, 0], meas.peaks[accepted, 1],
                   s=70, facecolors='none', edgecolors='limegreen', linewidths=2,
                   label=f'SPN accepted ({accepted.sum()})', zorder=4)
    if meas.reject.any():
        ax.scatter(meas.peaks[meas.reject, 0], meas.peaks[meas.reject, 1],
                   s=70, marker='x', c='red', linewidths=2,
                   label=f'SPN rejected ({meas.reject.sum()})', zorder=4)

    # Keypoint index labels on GT raw points
    for k, pt in enumerate(kpts_raw):
        ax.text(pt[0] + 6, pt[1], str(k), color='yellow', fontsize=7)

    ax.set_title(f'Frame {frame_idx} — keypoint comparison', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show(block=block)
    plt.pause(0.001)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run UKF on SHIRT roe1 dataset')
    parser.add_argument('--slab-spn-root', default=DEFAULT_SLAB_SPN_ROOT)
    parser.add_argument('--cfg-yaml',      default=DEFAULT_CFG_YAML)
    parser.add_argument('--checkpoint',    default=DEFAULT_CHECKPOINT)
    parser.add_argument('--dataset-root',  default=DEFAULT_DATASET_ROOT)
    parser.add_argument('--traj',          default=DEFAULT_TRAJ)
    parser.add_argument('--domain',        default=DEFAULT_DOMAIN)
    parser.add_argument('--label-csv',     default=DEFAULT_LABEL_CSV)
    parser.add_argument('--meta-csv',      default=DEFAULT_META_CSV)
    parser.add_argument('--keypoints-mat', default=DEFAULT_KEYPOINTS_MAT)
    parser.add_argument('--camera-json',   default=DEFAULT_CAMERA_JSON)
    parser.add_argument('--max-frames',    type=int, default=0,
                        help='Process at most N frames (0 = all)')
    parser.add_argument('--dt',            type=float, default=5.0,
                        help='Filter timestep [s]')
    parser.add_argument('--metadata-json', default=DEFAULT_METADATA_JSON,
                        help='Path to metadata.json (provides GT ROE, quaternion, RAV)')
    parser.add_argument('--save',          default=DEFAULT_SAVE,
                        help='Path to save results .mat (empty = skip)')
    parser.add_argument('--use-gt-measurements', action='store_true',
                        help='Replace SPN keypoints with GT projected keypoints + small '
                             'noise (R=1 px^2 per axis). Disables outlier rejection. '
                             'Use to isolate dynamics/filter bugs from SPN noise.')
    parser.add_argument('--visualize', action='store_true',
                        help='Show per-frame keypoint comparison plot.')
    parser.add_argument('--viz-frames', type=int, default=0,
                        help='Only visualize the first N frames (0 = all).')
    args = parser.parse_args()

    # --- Load ground truth data -----------------------------------------------------------
    print('Loading data...')
    filenames, bboxes, gt_qs, gt_ts, gt_kpts_norm = load_label_csv(args.label_csv)
    meta_rows = load_meta_csv(args.meta_csv)

    # Load metadata.json (GT filter-state quantities)
    with open(args.metadata_json) as _f:
        _meta = json.load(_f)
    _trel = _meta['tRelState']
    gt_roe_full   = np.array(_trel['roe_osc_ns'],        dtype=np.float64)   # [Nsim x 6]
    gt_q_spri_full = np.array(_trel['q_spri2tpri'],      dtype=np.float64)   # [Nsim x 4]
    gt_rav_full   = np.array(_trel['w_tpri2spri_tpri'],  dtype=np.float64)   # [Nsim x 3]

    n_frames = len(filenames)
    if args.max_frames > 0:
        n_frames = min(n_frames, args.max_frames)
    print(f'  {n_frames} frames to process')

    # Verify row counts match
    if len(meta_rows) < n_frames:
        raise RuntimeError(
            f'label CSV has {len(filenames)} rows but meta CSV has only '
            f'{len(meta_rows)} rows')

    camera   = load_camera_intrinsics(args.camera_json)
    K_mat    = camera['cameraMatrix'].astype(np.float64)
    dist     = camera['distCoeffs'].astype(np.float64)
    kpts_3d  = load_tango_keypoints(args.keypoints_mat)

    # --- Load SPN model ------------------------------------------------------
    print('Loading SPN model...')
    from .navigation import UKF, UKFParams
    from spn.model import SPNWrapper

    spn = SPNWrapper(
        slab_spn_root=args.slab_spn_root,
        cfg_yaml=args.cfg_yaml,
        checkpoint=args.checkpoint,
    )
    print(f'  num_keypoints = {spn.num_keypoints}')
    print(f'  input size    = {spn.input_w} x {spn.input_h}')

    # --- Build UKF -----------------------------------------------------------
    params = UKFParams(dt=args.dt)
    if args.use_gt_measurements:
        # With noiseless GT measurements, disable outlier rejection (always accept)
        # and use a small R floor consistent with the injected 1px noise.
        params.mahalanobis_threshold = 1e9   # effectively never reject
        params.r_floor               = 0.0   # no extra floor; R = 1 px^2 from covs
        print('  [GT mode] using ground-truth projected keypoints as measurements')
        print('  [GT mode] outlier rejection disabled, R = 1 px^2/axis')
    ukf = UKF(params, K_mat, dist, kpts_3d)

    # --- Output Arrays -------------------------------------------------
    rot_errors   = []
    trans_errors = []
    n_kp_used    = []
    reject_flags = []

    # For saving: store per-frame arrays
    all_pose_q    = np.zeros((n_frames, 4))
    all_pose_t    = np.zeros((n_frames, 3))
    all_gt_q      = gt_qs[:n_frames]
    all_gt_t      = gt_ts[:n_frames]
    all_P_diag    = np.zeros((n_frames, 12))
    all_prefit    = np.zeros((n_frames, 22))
    all_postfit   = np.zeros((n_frames, 22))
    all_kpts_pred = np.zeros((n_frames, 11, 2))
    # Internal filter state arrays
    all_filter_q   = np.zeros((n_frames, 4))   # internal q_spri2tpri
    all_filter_roe = np.zeros((n_frames, 6))   # NS-ROE filter state
    all_filter_rav = np.zeros((n_frames, 3))   # RAV x[9:12] [rad/s]
    all_meta_kep   = np.zeros((n_frames, 6))   # chief Keplerian

    # Ground-truth from metadata.json (frame index aligned directly)
    all_gt_roe    = gt_roe_full[:n_frames].copy()    # [n_frames x 6]  NS-ROE [m]
    all_gt_q_spri = gt_q_spri_full[:n_frames].copy() # [n_frames x 4]  q_spri2tpri
    all_gt_rav    = gt_rav_full[:n_frames].copy()    # [n_frames x 3]  w_tpri2spri_tpri [rad/s]

    # Import geometry helpers used for init
    from .navigation import _cam_pose_to_filter_state, _cartesian_to_nsroe

    # --- Main loop -----------------------------------------------------------
    print(f'\n{"Frame":>6}  {"det":>5}'
          f'  {"spn_rot[°]":>10}  {"spn_t[m]":>9}'
          f'  {"ukf_rot[°]":>10}  {"ukf_t[m]":>9}'
          f'  {"prefit[px]":>10}')
    print('-' * 80)

    initialized = False

    for i in range(n_frames):
        filename = filenames[i]
        bbox_norm = bboxes[i]
        m_abs     = meta_rows[i]
        gt_q      = gt_qs[i]      # [qw, qx, qy, qz]  q_scam2tbdy
        gt_t      = gt_ts[i]      # r_scam2tbdy_scam [m]

        # Denormalize bbox to pixel coordinates
        bbox = np.array([
            bbox_norm[0] * IMAGE_SIZE[0],
            bbox_norm[1] * IMAGE_SIZE[1],
            bbox_norm[2] * IMAGE_SIZE[0],
            bbox_norm[3] * IMAGE_SIZE[1],
        ])

        # Load image
        img_path = os.path.join(
            args.dataset_root, args.traj, args.domain, 'images', filename
        )
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f'{i+1:>6}  WARNING: image not found at {img_path}, skipping')
            continue

        # SPN inference
        meas = spn.run_inference(image, bbox)

        # --- GT measurement override (--use-gt-measurements) ----------------
        # Replace SPN keypoints with GT-projected keypoints + tiny noise.
        # Sets R = 1 px^2/axis for all keypoints (effectively noiseless).
        if args.use_gt_measurements:
            from .navigation import _project_keypoints
            gt_kpts = _project_keypoints(gt_q, gt_t, kpts_3d, K_mat, dist).reshape(11, 2)
            # Add tiny Gaussian noise so the filter is not fed exact values
            gt_kpts += np.random.randn(*gt_kpts.shape) * 1.0  # 1 px std
            from spn.model import SPNMeasurements
            meas = SPNMeasurements(
                peaks   = gt_kpts,
                covs    = np.stack([np.eye(2, dtype=np.float64)] * 11),   # 1 px^2/axis
                maxvals = np.ones(11, dtype=np.float64),
                reject  = np.zeros(11, dtype=bool),
            )
        # -------------------------------------------------------------------

        n_det = int((~meas.reject).sum())

        # --- Keypoint visualization ------------------------------------------
        if args.visualize and (args.viz_frames == 0 or i < args.viz_frames):
            from .navigation import _project_keypoints
            # GT raw keypoints: denormalise from [0,1] to image pixels
            kpts_raw_px = gt_kpts_norm[i] * np.array([IMAGE_SIZE[0], IMAGE_SIZE[1]])
            # GT solved: re-project GT pose onto image
            kpts_solved_px = _project_keypoints(
                gt_q, gt_t, kpts_3d, K_mat, dist
            ).reshape(11, 2)
            _plot_keypoints(image, kpts_raw_px, kpts_solved_px, meas, i + 1,
                            block=(i == n_frames - 1))
        # --------------------------------------------------------------------

        # --- Raw SPN pose (EPnP on accepted keypoints, no filter) -----------
        raw_pose = solve_pose(meas, kpts_3d, K_mat, dist)
        if raw_pose is not None:
            spn_rot_err   = _rotation_error_deg(raw_pose.q, gt_q)
            spn_trans_err = float(np.linalg.norm(raw_pose.t - gt_t))
        else:
            spn_rot_err   = float('nan')
            spn_trans_err = float('nan')

        if not initialized:
            # Frame 0: initialize filter from PnP
            ukf.initialize(meas, m_abs)
            initialized = True

            # --- Debug: store init state so we can test measurement model
            #     at frame 2 WITHOUT any propagation (see check below).
            _debug_init_roe  = ukf.x[:6].copy()
            _debug_init_q    = ukf.q.copy()
            _debug_init_mabs = m_abs

            # Record initialisation pose
            q_out, t_out = ukf._roe_to_camera_pose(ukf.x[:6], ukf.q, m_abs)
            rot_err   = _rotation_error_deg(q_out, gt_q)
            trans_err = float(np.linalg.norm(t_out - gt_t))

            # Save init-frame result
            all_pose_q[i]    = q_out
            all_pose_t[i]    = t_out
            all_P_diag[i]    = np.diag(ukf.P)
            all_kpts_pred[i] = np.zeros((11, 2))   # no prediction on init

            all_filter_rav[i] = ukf.x[9:12]

            print(f'{i+1:>6}  {n_det:>5}'
                  f'  {spn_rot_err:>10.2f}  {spn_trans_err:>9.4f}'
                  f'  {rot_err:>10.2f}  {trans_err:>9.4f}'
                  f'  {"---":>10}  [INIT]')
        else:
            # Subsequent frames: full UKF step
            rec = ukf.step(meas, m_abs, gt_q=gt_q, gt_t=gt_t)

            rot_err   = rec.rotation_error_deg
            trans_err = rec.translation_error_m

            # Save record
            all_pose_q[i]     = rec.pose_q
            all_pose_t[i]     = rec.pose_t
            all_P_diag[i]     = rec.P_diag
            all_prefit[i]     = rec.prefit_residual
            all_postfit[i]    = rec.postfit_residual
            all_kpts_pred[i]  = rec.predicted_keypoints

            all_filter_rav[i]  = ukf.x[9:12]

            reject_flags.append(rec.reject_all)
            n_kp_used.append(rec.n_keypoints_used)

            # Mean absolute prefit residual in pixels (over accepted keypoints only)
            prefit_px = rec.prefit_residual.reshape(11, 2)
            accepted  = ~meas.reject
            mean_prefit = float(np.mean(np.linalg.norm(prefit_px[accepted], axis=1))) \
                          if accepted.any() else float('nan')

            suffix = ' [REJ]' if rec.reject_all else ''
            print(f'{i+1:>6}  {n_det:>5}'
                  f'  {spn_rot_err:>10.2f}  {spn_trans_err:>9.4f}'
                  f'  {rot_err:>10.2f}  {trans_err:>9.4f}'
                  f'  {mean_prefit:>10.1f}{suffix}')

        rot_errors.append(rot_err)
        trans_errors.append(trans_err)

        # Save common per-frame data for plotting
        all_filter_q[i]   = ukf.q.copy()
        all_filter_roe[i] = ukf.x[:6].copy()
        all_filter_rav[i] = ukf.x[9:12].copy()
        all_meta_kep[i]   = m_abs.oe_osc_kep.copy()

    # --- Summary -------------------------------------------------------------
    if rot_errors:
        print('\n' + '=' * 52)
        print(f'Summary over {len(rot_errors)} frames:')
        print(f'  Rotation    error:  mean={np.mean(rot_errors):.2f}°  '
              f'median={np.median(rot_errors):.2f}°  '
              f'max={np.max(rot_errors):.2f}°')
        print(f'  Translation error:  mean={np.mean(trans_errors):.4f}m  '
              f'median={np.median(trans_errors):.4f}m  '
              f'max={np.max(trans_errors):.4f}m')
        if reject_flags:
            n_rej = sum(reject_flags)
            print(f'  Rejected (all kp): {n_rej}/{len(reject_flags)} frames')

    # --- Save results --------------------------------------------------------
    if args.save:
        from scipy.io import savemat
        save_path = args.save
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Pad reject_flags / n_kp_used (step-frames only) to full length
        _reject_arr = np.zeros(n_frames, dtype=np.float64)
        _nkp_arr    = np.zeros(n_frames, dtype=np.float64)
        _reject_arr[1:1 + len(reject_flags)] = reject_flags
        _nkp_arr[1:1 + len(n_kp_used)]      = n_kp_used

        savemat(save_path, {
            # Time / metadata
            'time_s'      : (np.arange(n_frames) * args.dt).reshape(-1, 1),
            'dt'          : float(args.dt),
            'sma_m'       : float(meta_rows[0].oe_osc_kep[0]),
            # NS-ROE  [N x 6]  metres  (filter estimate vs GT)
            'filter_roe'  : all_filter_roe[:n_frames],
            'gt_roe'      : all_gt_roe[:n_frames],
            'P_roe_diag'  : all_P_diag[:n_frames, 0:6],
            # Attitude: q_spri2tpri  [N x 4]  (filter estimate vs GT)
            'filter_q'    : all_filter_q[:n_frames],
            'gt_q_spri'   : all_gt_q_spri[:n_frames],
            'P_mrp_diag'  : all_P_diag[:n_frames, 6:9],
            # Angular velocity  [N x 3]  rad/s  (filter estimate vs GT)
            'filter_rav'  : all_filter_rav[:n_frames],
            'gt_rav'      : all_gt_rav[:n_frames],
            'P_rav_diag'  : all_P_diag[:n_frames, 9:12],
            # Camera-frame pose  [N x 4/3]  (for position error plot)
            'pose_q'      : all_pose_q[:n_frames],
            'pose_t'      : all_pose_t[:n_frames],
            'gt_q'        : all_gt_q[:n_frames],
            'gt_t'        : all_gt_t[:n_frames],
            # Residuals  [N x 22]  pixels
            'prefit'      : all_prefit[:n_frames],
            'postfit'     : all_postfit[:n_frames],
            # Flags / bookkeeping
            'reject_all'  : _reject_arr.reshape(-1, 1),
            'n_kp_used'   : _nkp_arr.reshape(-1, 1),
        })
        print(f'\nResults saved to: {save_path}')


if __name__ == '__main__':
    main()
