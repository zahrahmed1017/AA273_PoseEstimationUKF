"""
spn/model.py

Inference wrapper around slab-spn's SPN. Handles preprocessing,
forward pass, peak extraction, UDP sub-pixel correction, and
per-keypoint heatmap covariance computation for use as UKF measurements.

Porting notes vs. ukfspn_cpp:
  - Preprocessing  : spn-torch.cc  (to_tensor, ImageNet normalization)
  - Peak extraction : postprocess.cc (process_hmap, argmax)
  - UDP correction  : postprocess.cc (udp_correction)
  - Covariance      : postprocess.cc (process_hmap weighted 2nd moments)
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

# ImageNet normalization constants (matching spn-torch.cc)
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Heatmap post-processing constants (matching ukf-config.hh / postprocess.cc)
_HEATMAP_THRESHOLD = 0.003  # values < threshold * peak_val are zeroed
_UDP_KERNEL        = 11     # Gaussian blur kernel size for UDP correction


# ---------------------------------------------------------------------------

@dataclass
class SPNMeasurements:
    """Output of a single SPN inference call.

    Attributes:
        peaks:   (K, 2) float64  keypoint (x, y) locations in original image pixels.
        covs:    (K, 2, 2) float64  per-keypoint covariance from heatmap spread (px^2).
                           Used directly as the block-diagonal R matrix in the UKF.
        maxvals: (K,) float64  peak heatmap values (confidence proxy).
        reject:  (K,) bool  True if keypoint peak value is <= 0 (not detected).
    """
    peaks:   np.ndarray  # (K, 2)
    covs:    np.ndarray  # (K, 2, 2)
    maxvals: np.ndarray  # (K,)
    reject:  np.ndarray  # (K,) bool


# ---------------------------------------------------------------------------

@dataclass
class SPNPose:
    """Pose estimate from PnP on SPN keypoint detections.

    Attributes:
        q: (4,) float64 quaternion [qw, qx, qy, qz] (scalar-first).
           Represents the rotation from camera frame to target body frame.
        t: (3,) float64 translation vector in camera frame (metres).
           Vector from camera origin to target body origin, expressed in camera.
        num_keypoints: number of non-rejected keypoints used in PnP.
    """
    q:             np.ndarray  # (4,)
    t:             np.ndarray  # (3,)
    num_keypoints: int


def solve_pose(
    meas:          SPNMeasurements,
    keypoints_3d:  np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs:   Optional[np.ndarray] = None,
    min_keypoints: int = 4,
) -> Optional[SPNPose]:
    """Estimate pose from SPN keypoint detections via EPnP.

    Mirrors MyCv::pnp in ukfspn_cpp/src/utils/postprocess.cc and
    slab-spn/core/utils/postprocess.py:pnp(). Intended for standalone
    testing of the CNN pipeline against ground-truth labels before UKF
    integration.

    Args:
        meas:          SPNMeasurements from SPNWrapper.run_inference().
        keypoints_3d:  (K, 3) float64 array of 3D keypoint locations in
                       the target body frame (metres). Must match the K
                       keypoints in meas.
        camera_matrix: (3, 3) float64 camera intrinsic matrix.
        dist_coeffs:   (5,) float64 distortion coefficients [k1,k2,p1,p2,k3].
                       Pass None or zeros for no distortion.
        min_keypoints: Minimum number of accepted keypoints required to
                       attempt PnP. Returns None if not met. Default 4
                       (minimum for EPnP).

    Returns:
        SPNPose with q (4,) and t (3,), or None if too few keypoints.
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    # Select non-rejected keypoints
    accepted = np.where(~meas.reject)[0]
    if len(accepted) < min_keypoints:
        return None

    pts_3d = keypoints_3d[accepted].astype(np.float64)       # (N, 3)
    pts_2d = meas.peaks[accepted].astype(np.float64)         # (N, 2)

    pts_3d = np.ascontiguousarray(pts_3d).reshape(-1, 1, 3)
    pts_2d = np.ascontiguousarray(pts_2d).reshape(-1, 1, 2)

    _, rvec, tvec = cv2.solvePnP(
        pts_3d, pts_2d,
        camera_matrix.astype(np.float64),
        dist_coeffs.astype(np.float64),
        flags=cv2.SOLVEPNP_EPNP,
    )

    R, _ = cv2.Rodrigues(rvec)
    # scipy returns [qx, qy, qz, qw]; convert to scalar-first [qw, qx, qy, qz]
    from scipy.spatial.transform import Rotation
    q_xyzw = Rotation.from_matrix(R).as_quat()
    q = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    t = tvec.ravel().astype(np.float64)

    return SPNPose(q=q, t=t, num_keypoints=len(accepted))


# ---------------------------------------------------------------------------

class SPNWrapper:
    """Inference wrapper around slab-spn's SPNv3.

    Loads the model once and exposes a single `run_inference` method that
    accepts a raw image + bounding box and returns SPNMeasurements.

    Args:
        slab_spn_root: Path to the slab-spn repository root (must contain core/).
        cfg_yaml:      Path to a slab-spn YAML experiment config file.
        checkpoint:    Path to a trained .pth.tar checkpoint.
                       Handles both model_best format (bare state_dict) and
                       checkpoint format (dict with 'state_dict' key).
        device:        torch.device to run inference on. Defaults to CPU.
        use_udp:       Apply UDP sub-pixel correction (default True, matches C++).
    """

    def __init__(
        self,
        slab_spn_root: str,
        cfg_yaml: str,
        checkpoint: str,
        device: Optional[torch.device] = None,
        use_udp: bool = True,
    ):
        self.device  = device or torch.device('cpu')
        self.use_udp = use_udp

        # Add slab-spn/core to sys.path so we can import its modules directly,
        # matching the pattern used by slab-spn's tools/_init_paths.py
        core_path = os.path.join(slab_spn_root, 'core')
        if core_path not in sys.path:
            sys.path.insert(0, core_path)

        from config import cfg, update_config
        from nets   import build_spnv3

        # Load YAML config using the same pattern as slab-spn's test.py
        class _Args:
            cfg  = cfg_yaml
            opts = []

        update_config(cfg, _Args())

        self.input_w       = cfg.DATASET.INPUT_SIZE[0]   # CNN input width
        self.input_h       = cfg.DATASET.INPUT_SIZE[1]   # CNN input height
        self.num_keypoints = cfg.DATASET.NUM_KEYPOINTS

        # Build model and load weights
        self.model = build_spnv3(cfg, train=False)

        ckpt = torch.load(checkpoint, map_location='cpu')
        # model_best.pth.tar saves the state_dict directly;
        # checkpoint.pth.tar wraps it under a 'state_dict' key
        state_dict = ckpt.get('state_dict', ckpt)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    # -----------------------------------------------------------------------
    # Interface to use with the UKF

    def run_inference(self, image: np.ndarray, bbox: np.ndarray) -> SPNMeasurements:
        """Run SPN inference on a single image.

        Args:
            image: (H, W) uint8 grayscale or (H, W, 3) uint8 image.
            bbox:  [xmin, ymin, xmax, ymax] bounding box in original image pixels.

        Returns:
            SPNMeasurements with peaks (K,2), covs (K,2,2), maxvals (K,), reject (K,).
        """
        tensor   = self._preprocess(image, bbox)   # (1, 3, H_in, W_in) on device
        heatmaps = self._forward(tensor)            # (K, H_hm, W_hm) numpy on CPU
        return self._postprocess(heatmaps, bbox)

    # -----------------------------------------------------------------------
    # Internal helpers

    def _preprocess(self, image: np.ndarray, bbox: np.ndarray) -> torch.Tensor:
        """Crop to bbox, resize to CNN input size, apply ImageNet normalization."""
        xmin, ymin, xmax, ymax = [int(round(v)) for v in bbox]

        # Crop
        crop = image[ymin:ymax, xmin:xmax]

        # Replicate grayscale to 3 channels (matching spn-torch.cc: tensor.repeat({3,1,1}))
        if crop.ndim == 2:
            crop = np.stack([crop, crop, crop], axis=-1)  # (H, W, 3)

        # Resize to CNN input size [W_in, H_in]
        crop = cv2.resize(crop, (self.input_w, self.input_h),
                          interpolation=cv2.INTER_LINEAR)

        # uint8 -> float32 [0, 1] -> subtract ImageNet mean / divide std
        x = crop.astype(np.float32) / 255.0   # (H_in, W_in, 3)
        x = (x - _MEAN) / _STD

        # HWC -> CHW, add batch dimension
        tensor = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0)
        return tensor.to(self.device)

    def _forward(self, tensor: torch.Tensor) -> np.ndarray:
        """Run model forward pass; return heatmaps as (K, H_hm, W_hm) numpy array."""
        with torch.no_grad():
            outputs = self.model(tensor)  # list of head outputs
        # outputs[0] is the heatmap head: (1, K, H_hm, W_hm)
        return outputs[0].squeeze(0).cpu().numpy()  # (K, H_hm, W_hm)

    def _postprocess(self, heatmaps: np.ndarray, bbox: np.ndarray) -> SPNMeasurements:
        """Extract peaks, apply UDP, compute covariances, map to image coords."""
        K, H_hm, W_hm = heatmaps.shape
        xmin, ymin, xmax, ymax = bbox

        # Scale factors: heatmap pixels -> original image pixels
        Wf = (xmax - xmin) / W_hm
        Hf = (ymax - ymin) / H_hm

        peaks   = np.zeros((K, 2),    dtype=np.float64)
        covs    = np.zeros((K, 2, 2), dtype=np.float64)
        maxvals = np.zeros(K,         dtype=np.float64)
        reject  = np.ones(K,          dtype=bool)

        for k in range(K):
            hmap   = heatmaps[k]
            maxval = float(hmap.max())
            maxvals[k] = maxval

            if maxval <= 0.0:
                continue

            # Peak location in heatmap pixel coordinates (argmax)
            idx = int(hmap.argmax())
            px  = float(idx % W_hm)
            py  = float(idx // W_hm)

            # UDP sub-pixel correction (matching postprocess.cc: udp_correction)
            if self.use_udp:
                px, py = _udp_correction(hmap, px, py)

            # Map to original image coordinates
            peaks[k, 0] = px * Wf + xmin
            peaks[k, 1] = py * Hf + ymin
            reject[k]   = False

            # Per-keypoint heatmap covariance for UKF R matrix
            # (matching postprocess.cc: process_hmap weighted 2nd moments)
            covs[k] = _compute_heatmap_covariance(hmap, px, py, Wf, Hf)

        return SPNMeasurements(peaks=peaks, covs=covs, maxvals=maxvals, reject=reject)


# ---------------------------------------------------------------------------
# Heatmap post-processing utilities (module-level for testability)

def _compute_heatmap_covariance(
    hmap: np.ndarray,
    px: float,
    py: float,
    Wf: float,
    Hf: float,
    threshold_frac: float = _HEATMAP_THRESHOLD,
) -> np.ndarray:
    """Compute the 2x2 covariance of the heatmap probability distribution around a peak.

    Ported from process_hmap in ukfspn_cpp/src/utils/postprocess.cc.
    The covariance is in original image pixel^2 units and feeds directly into
    the UKF measurement noise matrix R as a 2x2 block for each keypoint.

    Args:
        hmap:           (H, W) float32 heatmap.
        px, py:         Peak location in heatmap pixel coordinates.
        Wf, Hf:         Scale factors: heatmap pixels -> image pixels.
        threshold_frac: Values below (threshold_frac * peak_val) are zeroed
                        before computing the distribution (99.7% tail cutoff).

    Returns:
        cov: (2, 2) covariance matrix in image pixel^2.
    """
    maxval = hmap.max()
    h = hmap.copy().astype(np.float64)
    h[h < maxval * threshold_frac] = 0.0

    total = h.sum()
    if total < 1e-10:
        return np.zeros((2, 2), dtype=np.float64)
    h /= total  # normalize to a probability distribution

    H, W = h.shape
    # x-offsets (column direction) and y-offsets (row direction) in image pixels
    dx = (np.arange(W, dtype=np.float64) - px) * Wf  # (W,)
    dy = (np.arange(H, dtype=np.float64) - py) * Hf  # (H,)

    cxx = np.sum(h * dx[np.newaxis, :] ** 2)
    cxy = np.sum(h * dx[np.newaxis, :] * dy[:, np.newaxis])
    cyy = np.sum(h * dy[:, np.newaxis] ** 2)

    return np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)


def _udp_correction(hmap: np.ndarray, px: float, py: float) -> Tuple[float, float]:
    """Apply UDP sub-pixel correction to a peak location.

    Ported from udp_correction in ukfspn_cpp/src/utils/postprocess.cc.
    Fits a local 2nd-order Taylor expansion on the log of the Gaussian-blurred
    heatmap and solves for the sub-pixel peak shift analytically.

    Args:
        hmap: (H, W) float32 heatmap.
        px, py: Peak location in heatmap pixel coordinates (integer argmax values).

    Returns:
        (px_refined, py_refined) sub-pixel corrected peak.
    """
    H, W = hmap.shape

    # Gaussian blur + log (matching postprocess.cc)
    h = cv2.GaussianBlur(hmap, (_UDP_KERNEL, _UDP_KERNEL), 0).astype(np.float64)
    np.clip(h, 0.001, 50.0, out=h)
    np.log(h, out=h)

    # Pad by 1 on all sides with edge replication
    h_pad = np.pad(h, 1, mode='edge')

    ix = int(px)
    iy = int(py)

    # Flat index into padded array; row stride = W + 2
    stride = W + 2
    i = (iy + 1) * stride + (ix + 1)
    f = h_pad.ravel()

    # 3x3 neighborhood values (matching index arithmetic in postprocess.cc)
    i_      = f[i]
    ix1     = f[i + 1]           # right
    iy1     = f[i + stride]      # below
    ix1y1   = f[i + stride + 1]  # right + below
    ix1_y1_ = f[i - stride - 1]  # left  + above
    ix1_    = f[i - 1]           # left
    iy1_    = f[i - stride]      # above

    # First derivatives
    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)

    # Second derivatives (Hessian)
    dxx = ix1 - 2.0 * i_ + ix1_
    dyy = iy1 - 2.0 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)

    # Solve 2x2 system for sub-pixel shift
    det   = dxx * dyy - dxy * dxy + 1e-10
    diffx = (dyy * dx - dxy * dy) / det
    diffy = (-dxy * dx + dxx * dy) / det

    return px - diffx, py - diffy
