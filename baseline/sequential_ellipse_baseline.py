from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import image_center_physical
from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import resample_image_with_transform
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"


@dataclass
class StepResult:
    step_name: str
    angle_deg: float
    status: str
    n_points: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline sequential 3-angle estimation from bone mask with slab-based "
            "2D ellipse-axis approximation."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to NIfTI file or directory with .nii/.nii.gz files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for rotated volumes, angle CSV, and QC images.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing in mm for preprocessing (default: 1.0).",
    )
    parser.add_argument(
        "--bone-threshold-hu",
        type=float,
        default=300.0,
        help="HU threshold to keep bone structures (default: 300).",
    )
    parser.add_argument(
        "--axial-offset-mm",
        type=float,
        default=30.0,
        help="Offset from top-most bone slice for axial step (default: 30 mm).",
    )
    parser.add_argument(
        "--top-slab-mm",
        type=float,
        default=50.0,
        help="Top slab thickness used for coronal and sagittal steps (default: 50 mm).",
    )
    parser.add_argument(
        "--mid-band-mm",
        type=float,
        default=10.0,
        help="Midline slab thickness for coronal/sagittal projections (default: 10 mm).",
    )
    parser.add_argument(
        "--save-qc",
        action="store_true",
        help="Save QC figures with fitted major axis for each step.",
    )
    return parser.parse_args()


def collect_nifti_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    nii = list(input_path.rglob("*.nii"))
    nii_gz = list(input_path.rglob("*.nii.gz"))
    return sorted({*nii, *nii_gz})


def study_id_from_path(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    if path.name.endswith(".nii"):
        return path.name[: -len(".nii")]
    return path.stem


def preprocess_ct(path: Path, spacing: float) -> sitk.Image:
    ct = read_nifti(path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    return resample_ct_to_isotropic(
        ct,
        target_spacing=(spacing, spacing, spacing),
    )


def make_bone_mask(ct_img: sitk.Image, threshold_hu: float) -> np.ndarray:
    hu = sitk_image_to_numpy(ct_img)  # [z, y, x]
    return (hu >= threshold_hu).astype(np.uint8)


def normalize_angle_to_half_turn(angle_deg: float) -> float:
    # map to [-90, 90)
    return ((angle_deg + 90.0) % 180.0) - 90.0


def fit_major_axis_angle_deg(binary_2d: np.ndarray) -> tuple[float | None, np.ndarray]:
    """
    Approximate ellipse major-axis orientation using PCA of non-zero points.
    Returns angle in degrees relative to +x axis in image coordinates.
    """
    ys, xs = np.nonzero(binary_2d > 0)
    if ys.size < 10:
        return None, np.empty((0, 2), dtype=np.float64)

    pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])  # [N,2]
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    angle_rad = math.atan2(float(major_vec[1]), float(major_vec[0]))
    angle_deg = math.degrees(angle_rad)
    return angle_deg, pts


def rotate_ct_single_axis(ct_img: sitk.Image, angle_deg: float, axis: str) -> sitk.Image:
    tx = sitk.Euler3DTransform()
    tx.SetCenter(image_center_physical(ct_img))
    ax = ay = az = 0.0
    angle_rad = math.radians(angle_deg)
    if axis == "x":
        ax = angle_rad
    elif axis == "y":
        ay = angle_rad
    elif axis == "z":
        az = angle_rad
    else:
        raise ValueError(f"Unknown axis: {axis}")
    tx.SetRotation(ax, ay, az)

    return resample_image_with_transform(
        image=ct_img,
        transform=tx,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        expand_to_fit=True,
    )


def estimate_axial_angle(
    bone_zyx: np.ndarray,
    spacing: float,
    offset_mm: float,
) -> tuple[StepResult, np.ndarray]:
    z_nonzero = np.where(bone_zyx.any(axis=(1, 2)))[0]
    if z_nonzero.size == 0:
        return StepResult("axial", 0.0, "empty_bone_mask", 0), np.zeros((1, 1), dtype=np.uint8)

    z_top = int(z_nonzero.min())
    z_idx = min(
        int(z_top + round(offset_mm / spacing)),
        bone_zyx.shape[0] - 1,
    )
    img = bone_zyx[z_idx, :, :]

    raw_angle, pts = fit_major_axis_angle_deg(img)
    if raw_angle is None:
        return StepResult("axial", 0.0, "not_enough_points", int(pts.shape[0])), img

    # Correct towards horizontal major axis.
    angle_deg = normalize_angle_to_half_turn(-raw_angle)
    return StepResult("axial", angle_deg, "ok", int(pts.shape[0])), img


def estimate_coronal_angle(
    bone_zyx: np.ndarray,
    spacing: float,
    top_slab_mm: float,
    mid_band_mm: float,
) -> tuple[StepResult, np.ndarray]:
    z_nonzero = np.where(bone_zyx.any(axis=(1, 2)))[0]
    if z_nonzero.size == 0:
        return StepResult("coronal", 0.0, "empty_bone_mask", 0), np.zeros((1, 1), dtype=np.uint8)

    z_top = int(z_nonzero.min())
    slab = max(1, int(round(top_slab_mm / spacing)))
    z1 = min(z_top + slab, bone_zyx.shape[0])

    y_mid = bone_zyx.shape[1] // 2
    band = max(1, int(round(mid_band_mm / spacing / 2.0)))
    y0 = max(0, y_mid - band)
    y1 = min(bone_zyx.shape[1], y_mid + band + 1)

    # [z, y, x] -> collapse y band => [z, x]
    img = bone_zyx[z_top:z1, y0:y1, :].max(axis=1)

    raw_angle, pts = fit_major_axis_angle_deg(img)
    if raw_angle is None:
        return StepResult("coronal", 0.0, "not_enough_points", int(pts.shape[0])), img

    angle_deg = normalize_angle_to_half_turn(-raw_angle)
    return StepResult("coronal", angle_deg, "ok", int(pts.shape[0])), img


def estimate_sagittal_angle(
    bone_zyx: np.ndarray,
    spacing: float,
    top_slab_mm: float,
    mid_band_mm: float,
) -> tuple[StepResult, np.ndarray]:
    z_nonzero = np.where(bone_zyx.any(axis=(1, 2)))[0]
    if z_nonzero.size == 0:
        return StepResult("sagittal", 0.0, "empty_bone_mask", 0), np.zeros((1, 1), dtype=np.uint8)

    z_top = int(z_nonzero.min())
    slab = max(1, int(round(top_slab_mm / spacing)))
    z1 = min(z_top + slab, bone_zyx.shape[0])

    x_mid = bone_zyx.shape[2] // 2
    band = max(1, int(round(mid_band_mm / spacing / 2.0)))
    x0 = max(0, x_mid - band)
    x1 = min(bone_zyx.shape[2], x_mid + band + 1)

    # [z, y, x] -> collapse x band => [z, y]
    img = bone_zyx[z_top:z1, :, x0:x1].max(axis=2)

    raw_angle, pts = fit_major_axis_angle_deg(img)
    if raw_angle is None:
        return StepResult("sagittal", 0.0, "not_enough_points", int(pts.shape[0])), img

    angle_deg = normalize_angle_to_half_turn(-raw_angle)
    return StepResult("sagittal", angle_deg, "ok", int(pts.shape[0])), img


def save_qc_image(binary_2d: np.ndarray, step: StepResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(binary_2d, cmap="gray", origin="lower")
    ax.set_title(f"{step.step_name}: angle={step.angle_deg:.2f} deg ({step.status})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_study(path: Path, args: argparse.Namespace) -> tuple[str, list[StepResult], sitk.Image]:
    study_id = study_id_from_path(path)
    ct = preprocess_ct(path, spacing=args.spacing)

    step_results: list[StepResult] = []

    # Step 1: axial -> rotate around z
    bone = make_bone_mask(ct, threshold_hu=args.bone_threshold_hu)
    axial_step, axial_img = estimate_axial_angle(
        bone_zyx=bone,
        spacing=args.spacing,
        offset_mm=args.axial_offset_mm,
    )
    step_results.append(axial_step)
    if args.save_qc:
        save_qc_image(
            axial_img,
            axial_step,
            args.output_dir / study_id / "qc_axial.png",
        )
    ct = rotate_ct_single_axis(ct, angle_deg=axial_step.angle_deg, axis="z")

    # Step 2: coronal -> rotate around y
    bone = make_bone_mask(ct, threshold_hu=args.bone_threshold_hu)
    coronal_step, coronal_img = estimate_coronal_angle(
        bone_zyx=bone,
        spacing=args.spacing,
        top_slab_mm=args.top_slab_mm,
        mid_band_mm=args.mid_band_mm,
    )
    step_results.append(coronal_step)
    if args.save_qc:
        save_qc_image(
            coronal_img,
            coronal_step,
            args.output_dir / study_id / "qc_coronal.png",
        )
    ct = rotate_ct_single_axis(ct, angle_deg=coronal_step.angle_deg, axis="y")

    # Step 3: sagittal -> rotate around x
    bone = make_bone_mask(ct, threshold_hu=args.bone_threshold_hu)
    sagittal_step, sagittal_img = estimate_sagittal_angle(
        bone_zyx=bone,
        spacing=args.spacing,
        top_slab_mm=args.top_slab_mm,
        mid_band_mm=args.mid_band_mm,
    )
    step_results.append(sagittal_step)
    if args.save_qc:
        save_qc_image(
            sagittal_img,
            sagittal_step,
            args.output_dir / study_id / "qc_sagittal.png",
        )
    ct = rotate_ct_single_axis(ct, angle_deg=sagittal_step.angle_deg, axis="x")

    return study_id, step_results, ct


def save_angles_csv(rows: list[dict[str, str | float | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "step_name",
                "angle_deg",
                "status",
                "n_points",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["study_id"],
                    row["step_name"],
                    row["angle_deg"],
                    row["status"],
                    row["n_points"],
                ]
            )


def save_final_angles_csv(rows: list[dict[str, str | float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "axial_deg",
                "coronal_deg",
                "sagittal_deg",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["study_id"],
                    row["axial_deg"],
                    row["coronal_deg"],
                    row["sagittal_deg"],
                ]
            )


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    nii_paths = collect_nifti_paths(input_path)
    if not nii_paths:
        raise FileNotFoundError(f"No .nii/.nii.gz found in: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    per_step_rows: list[dict[str, str | float | int]] = []
    per_study_rows: list[dict[str, str | float]] = []

    print(f"[INFO] Found studies: {len(nii_paths)}")
    for idx, path in enumerate(nii_paths, start=1):
        study_id = study_id_from_path(path)
        print(f"[INFO] [{idx}/{len(nii_paths)}] Processing: {study_id}")
        study_id, steps, ct_rot = process_study(path, args=args)

        # Save rotated volume.
        study_out = output_dir / study_id
        study_out.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(ct_rot, str(study_out / f"{study_id}_rotated_baseline.nii.gz"))

        step_map = {s.step_name: s for s in steps}
        per_study_rows.append(
            {
                "study_id": study_id,
                "axial_deg": step_map["axial"].angle_deg,
                "coronal_deg": step_map["coronal"].angle_deg,
                "sagittal_deg": step_map["sagittal"].angle_deg,
            }
        )
        for s in steps:
            per_step_rows.append(
                {
                    "study_id": study_id,
                    "step_name": s.step_name,
                    "angle_deg": s.angle_deg,
                    "status": s.status,
                    "n_points": s.n_points,
                }
            )

    save_angles_csv(per_step_rows, output_dir / "angles_per_step.csv")
    save_final_angles_csv(per_study_rows, output_dir / "angles_per_study.csv")
    print(f"[DONE] Saved baseline outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

