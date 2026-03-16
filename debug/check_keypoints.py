"""
check_keypoints.py

Quick sanity check: compare the hardcoded Tango keypoints in spacecraft.py
against the tangoPoints.mat file used by the SPN/PnP pipeline.

Run from AA273_PoseEstimationCNN/:
    python -m ukf.check_keypoints
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

from .spacecraft import Tango

KEYPOINTS_MAT = (
    "/Users/Zahra1/Documents/Stanford/Research/datasets/shirtv1"
    "/../models/tango/tangoPoints.mat"
)

# Edges connecting keypoints for a wireframe (index pairs)
# Based on the 4 top corners, 4 mid corners, and 3 antenna/solar panel points
EDGES = [
    # Top face rectangle (kp 0-3)
    (0, 1), (1, 2), (2, 3), (3, 0),
    # Mid face rectangle (kp 4-7)
    (4, 5), (5, 6), (6, 7), (7, 4),
    # Vertical edges connecting top to mid
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def load_mat_keypoints(mat_path: str) -> np.ndarray:
    """Load tango3Dpoints from .mat file, returns [11 x 3]."""
    data = loadmat(mat_path)
    pts  = data["tango3Dpoints"]          # [3 x 11]
    return pts.T.astype(np.float64)       # [11 x 3]


def compare_and_plot(kp_mat: np.ndarray, kp_code: np.ndarray) -> None:
    """Print numerical comparison and show 3D plot."""

    print("=" * 60)
    print(f"{'kp':>4}  {'mat x':>10} {'mat y':>10} {'mat z':>10}  "
          f"{'code x':>10} {'code y':>10} {'code z':>10}  {'max diff':>10}")
    print("-" * 60)

    max_diffs = []
    for i in range(len(kp_mat)):
        diff = np.max(np.abs(kp_mat[i] - kp_code[i]))
        max_diffs.append(diff)
        flag = "  <<<" if diff > 1e-4 else ""
        print(f"{i:>4}  "
              f"{kp_mat[i,0]:>10.4f} {kp_mat[i,1]:>10.4f} {kp_mat[i,2]:>10.4f}  "
              f"{kp_code[i,0]:>10.4f} {kp_code[i,1]:>10.4f} {kp_code[i,2]:>10.4f}  "
              f"{diff:>10.6f}{flag}")

    print("=" * 60)
    print(f"Max difference across all keypoints: {max(max_diffs):.2e}")
    if max(max_diffs) < 1e-4:
        print("PASS -- keypoints match.")
    else:
        print("MISMATCH -- check flagged rows above.")

    # --- 3D plot ---
    fig = plt.figure(figsize=(12, 5))

    for col_idx, (kp, label, color) in enumerate([
        (kp_mat,  "tangoPoints.mat",  "tab:blue"),
        (kp_code, "spacecraft.py",    "tab:orange"),
    ]):
        ax = fig.add_subplot(1, 2, col_idx + 1, projection="3d")
        ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2],
                   c=color, s=60, zorder=5, label=label)

        # Label each keypoint
        for i, (x, y, z) in enumerate(kp):
            ax.text(x, y, z, f" {i}", fontsize=8, color=color)

        # Draw wireframe edges
        for i, j in EDGES:
            ax.plot(
                [kp[i, 0], kp[j, 0]],
                [kp[i, 1], kp[j, 1]],
                [kp[i, 2], kp[j, 2]],
                color="gray", linewidth=0.8,
            )

        ax.set_title(label)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_box_aspect([1, 1, 1])

    plt.suptitle("Tango 3D Keypoints Comparison", fontsize=13)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    kp_mat  = load_mat_keypoints(KEYPOINTS_MAT)
    kp_code = Tango.keypoints

    print(f"\ntangoPoints.mat shape: {kp_mat.shape}")
    print(f"spacecraft.py shape:   {kp_code.shape}\n")

    if kp_mat.shape != kp_code.shape:
        print(f"SHAPE MISMATCH: {kp_mat.shape} vs {kp_code.shape}")
    else:
        compare_and_plot(kp_mat, kp_code)
