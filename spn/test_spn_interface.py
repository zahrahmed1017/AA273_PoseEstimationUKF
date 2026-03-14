"""
test_spn_interface.py

Verifies SPNWrapper end-to-end:
  1. Loads model and runs inference on SHIRT images
  2. Checks keypoint peaks and per-keypoint covariances
  3. Estimates pose via EPnP and compares against ground-truth labels
"""

import os
import json
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.io import loadmat

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spn.model import SPNWrapper, solve_pose

# ---------------------------------------------------------------------------
# Paths (TO DO: update this to read in a cfg file with a parser or something. For now just hardcoding necessary paths here)
# ---------------------------------------------------------------------------

SLAB_SPN_ROOT = '/Users/Zahra1/Documents/Stanford/Classes/AA273/FinalProject/slab-spn'
CFG_YAML      = os.path.join(SLAB_SPN_ROOT, 'experiments', 'cfg_shirt_baseline.yaml')
CHECKPOINT    = os.path.join(SLAB_SPN_ROOT, 'output',
                             'efficientnet_b0.ra_in1k',
                             'baseline_incAug_20251022',
                             'model_best.pth.tar')

DATASET_ROOT  = '/Users/Zahra1/Documents/Stanford/Research/datasets/shirtv1'
TRAJ          = 'roe1'
DOMAIN        = 'synthetic'   # or 'lightbox'
IMAGE_EXT     = 'jpg'

LABEL_CSV     = os.path.join(os.path.dirname(__file__), '..', 'data', 'roe1.csv')
KEYPOINTS_MAT = os.path.join(DATASET_ROOT, '..', 'models', 'tango', 'tangoPoints.mat')
CAMERA_JSON   = os.path.join(DATASET_ROOT, 'camera.json')

IMAGE_SIZE    = (1920, 1200)   # (W, H)
NUM_FRAMES    = 10             # how many frames to test

# ---------------------------------------------------------------------------
# Copied from slab-spn/core/utils/utils.py
# ---------------------------------------------------------------------------

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)
    cam = {k: np.array(v, dtype=np.float32) for k, v in cam.items()}
    if 'distCoeffs' not in cam.keys():
        cam['distCoeffs'] = None
    cam['horizontalFOV'] = 2 * np.arctan2(0.5 * cam['ppx'] * cam['Nu'], cam['fx'])
    return cam


def load_tango_3d_keypoints(mat_dir):
    vertices  = loadmat(mat_dir)['tango3Dpoints']  # [3 x 11]
    corners3D = np.transpose(np.array(vertices, dtype=np.float32))  # [11 x 3]
    return corners3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rotation_error_deg(q_est, q_gt):
    """Angular difference between two [qw, qx, qy, qz] quaternions, in degrees."""
    R_est = Rotation.from_quat([q_est[1], q_est[2], q_est[3], q_est[0]]).as_matrix()
    R_gt  = Rotation.from_quat([q_gt[1],  q_gt[2],  q_gt[3],  q_gt[0]]).as_matrix()
    trace = np.clip((np.trace(R_est.T @ R_gt) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(trace))


def translation_error_m(t_est, t_gt):
    return float(np.linalg.norm(t_est - t_gt))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Load 3D keypoints (11, 3) [m]
    keypoints_3d = load_tango_3d_keypoints(KEYPOINTS_MAT).astype(np.float64)

    # --- Load camera intrinsics
    camera = load_camera_intrinsics(CAMERA_JSON)
    camera_matrix = camera['cameraMatrix'].astype(np.float64)
    dist_coeffs   = camera['distCoeffs'].astype(np.float64) if camera['distCoeffs'] is not None else np.zeros(5)

    # --- Load label CSV
    # Columns: filename, xmin, ymin, xmax, ymax, qw, qx, qy, qz, tx, ty, tz, [keypoints...]
    csv = pd.read_csv(LABEL_CSV, header=None)

    # --- Build SPN wrapper
    print("Loading SPN model...")
    spn = SPNWrapper(
        slab_spn_root=SLAB_SPN_ROOT,
        cfg_yaml=CFG_YAML,
        checkpoint=CHECKPOINT,
    )
    print(f"  num_keypoints = {spn.num_keypoints}")
    print(f"  input size    = {spn.input_w} x {spn.input_h}")

    # --- Run over first NUM_FRAMES frames
    rot_errors   = []
    trans_errors = []

    for i in range(min(NUM_FRAMES, len(csv))):
        row       = csv.iloc[i]
        filename  = str(row[0]).strip()
        bbox_norm = row[1:5].to_numpy(dtype=np.float64)   # normalized [0,1]
        q_gt      = row[5:9].to_numpy(dtype=np.float64)   # [qw, qx, qy, qz]
        t_gt      = row[9:12].to_numpy(dtype=np.float64)  # [tx, ty, tz] metres

        # Denormalize bbox to pixels
        bbox = np.array([
            bbox_norm[0] * IMAGE_SIZE[0],
            bbox_norm[1] * IMAGE_SIZE[1],
            bbox_norm[2] * IMAGE_SIZE[0],
            bbox_norm[3] * IMAGE_SIZE[1],
        ])

        # Load image
        imgname = filename if '.' in filename else f"{filename}.{IMAGE_EXT}"
        imgpath = os.path.join(DATASET_ROOT, TRAJ, DOMAIN, 'images', imgname)
        image   = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[{i}] WARNING: image not found at {imgpath}, skipping")
            continue

        # --- SPN inference
        meas = spn.run_inference(image, bbox)

        num_detected = int((~meas.reject).sum())
        print(f"\n[{i}] {filename}  detected={num_detected}/{spn.num_keypoints}")

        # Keypoint check
        print("  peaks (x, y) [px]:")
        for k in range(spn.num_keypoints):
            status = "OK " if not meas.reject[k] else "REJ"
            print(f"    kp{k:02d} [{status}]  "
                  f"x={meas.peaks[k,0]:7.2f}  y={meas.peaks[k,1]:7.2f}  "
                  f"maxval={meas.maxvals[k]:.4f}  "
                  f"cov_trace={np.trace(meas.covs[k]):.2f}")

        # --- Pose estimation
        pose = solve_pose(meas, keypoints_3d, camera_matrix, dist_coeffs)
        if pose is None:
            print("  solve_pose: FAILED (too few keypoints)")
            continue

        r_err = rotation_error_deg(pose.q, q_gt)
        t_err = translation_error_m(pose.t, t_gt)
        rot_errors.append(r_err)
        trans_errors.append(t_err)

        print(f"  pose:  q={np.round(pose.q, 4)}  t={np.round(pose.t, 3)}")
        print(f"  gt:    q={np.round(q_gt,  4)}   t={np.round(t_gt,  3)}")
        print(f"  error: rot={r_err:.2f} deg   trans={t_err:.4f} m")

    # --- Summary
    if rot_errors:
        print(f"\n=== Summary over {len(rot_errors)} frames ===")
        print(f"  Rotation    error: mean={np.mean(rot_errors):.2f} deg  "
              f"median={np.median(rot_errors):.2f} deg  "
              f"max={np.max(rot_errors):.2f} deg")
        print(f"  Translation error: mean={np.mean(trans_errors):.4f} m  "
              f"median={np.median(trans_errors):.4f} m  "
              f"max={np.max(trans_errors):.4f} m")


if __name__ == '__main__':
    main()
