from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import resample_mask_to_isotropic
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
REQUIRED_PARTS = [
    "auditory_canal_left",
    "auditory_canal_right",
    "eye_left",
    "eye_right",
    "eye_lens_left",
    "eye_lens_right",
    "nasal_cavity_left",
    "nasal_cavity_right",
]


@dataclass
class LandmarkRow:
    study_id: str
    landmark_name: str
    x_mm: float
    y_mm: float
    z_mm: float
    method: str
    n_voxels_used: int
    source_part: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract pseudo-label landmarks from ROI masks with robust farthest-point "
            "strategy for auditory canals and nasal cavities."
        )
    )
    parser.add_argument(
        "--volumes-dir",
        type=Path,
        required=True,
        help="Directory with CT studies (.nii/.nii.gz).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help="Masks directory: <study_id>/<masks_group>/<part>.nii.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save landmarks CSVs.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing in mm (default: 1.0).",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=0.03,
        help=(
            "Fraction of farthest voxels to average for robust extreme points "
            "(recommended range 0.02..0.05, default: 0.03)."
        ),
    )
    return parser.parse_args()


def collect_volume_paths(volumes_dir: Path) -> list[Path]:
    nii = list(volumes_dir.rglob("*.nii"))
    nii_gz = list(volumes_dir.rglob("*.nii.gz"))
    return sorted({*nii, *nii_gz})


def study_id_from_volume_path(volume_path: Path) -> str:
    if volume_path.name.endswith(".nii.gz"):
        return volume_path.name[: -len(".nii.gz")]
    if volume_path.name.endswith(".nii"):
        return volume_path.name[: -len(".nii")]
    return volume_path.stem


def find_mask_path(study_mask_dir: Path, part_name: str) -> Path | None:
    gz_candidates = sorted(study_mask_dir.glob(f"*/{part_name}.nii.gz"))
    if gz_candidates:
        return gz_candidates[0]
    nii_candidates = sorted(study_mask_dir.glob(f"*/{part_name}.nii"))
    if nii_candidates:
        return nii_candidates[0]
    return None


def preprocess_ct(volume_path: Path, spacing: float) -> sitk.Image:
    ct = read_nifti(volume_path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    ct_iso = resample_ct_to_isotropic(ct, target_spacing=(spacing, spacing, spacing))
    return ct_iso


def preprocess_mask(mask_path: Path, ct_ref_iso: sitk.Image, spacing: float) -> sitk.Image:
    mask = read_nifti(mask_path)
    mask = sitk.DICOMOrient(mask, TARGET_ORIENTATION)
    mask_iso = resample_mask_to_isotropic(mask, target_spacing=(spacing, spacing, spacing))
    # Force exact CT grid for voxel-wise consistency.
    mask_aligned = sitk.Resample(
        mask_iso,
        ct_ref_iso,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        mask_iso.GetPixelID(),
    )
    return mask_aligned


def compute_head_center_mm(ct_iso: sitk.Image) -> np.ndarray:
    ct_np = sitk_image_to_numpy(ct_iso)  # [z, y, x]
    foreground = ct_np > -300.0
    coords_zyx = np.argwhere(foreground)

    if coords_zyx.size == 0:
        center_zyx = (np.array(ct_np.shape, dtype=np.float64) - 1.0) / 2.0
    else:
        center_zyx = coords_zyx.mean(axis=0)

    z, y, x = [float(v) for v in center_zyx]
    center_xyz_mm = ct_iso.TransformContinuousIndexToPhysicalPoint((x, y, z))
    return np.asarray(center_xyz_mm, dtype=np.float64)


def mask_points_mm(mask_image: sitk.Image) -> np.ndarray:
    mask_np = sitk_image_to_numpy(mask_image)
    coords_zyx = np.argwhere(mask_np > 0)
    if coords_zyx.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    points = np.empty((coords_zyx.shape[0], 3), dtype=np.float64)
    for i, (z, y, x) in enumerate(coords_zyx):
        points[i] = mask_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
    return points


def centroid_from_points(points_mm: np.ndarray) -> np.ndarray | None:
    if points_mm.shape[0] == 0:
        return None
    return points_mm.mean(axis=0)


def robust_farthest_mean(
    points_mm: np.ndarray,
    center_mm: np.ndarray,
    top_percent: float,
) -> tuple[np.ndarray | None, int]:
    if points_mm.shape[0] == 0:
        return None, 0

    distances = np.linalg.norm(points_mm - center_mm[None, :], axis=1)
    n = distances.shape[0]
    k = max(1, int(round(n * top_percent)))

    idx = np.argpartition(distances, -k)[-k:]
    selected = points_mm[idx]
    return selected.mean(axis=0), selected.shape[0]


def landmark_row(
    study_id: str,
    landmark_name: str,
    point_mm: np.ndarray | None,
    method: str,
    n_voxels_used: int,
    source_part: str,
    status: str,
) -> LandmarkRow:
    if point_mm is None:
        return LandmarkRow(
            study_id=study_id,
            landmark_name=landmark_name,
            x_mm=float("nan"),
            y_mm=float("nan"),
            z_mm=float("nan"),
            method=method,
            n_voxels_used=n_voxels_used,
            source_part=source_part,
            status=status,
        )
    return LandmarkRow(
        study_id=study_id,
        landmark_name=landmark_name,
        x_mm=float(point_mm[0]),
        y_mm=float(point_mm[1]),
        z_mm=float(point_mm[2]),
        method=method,
        n_voxels_used=n_voxels_used,
        source_part=source_part,
        status=status,
    )


def extract_study_landmarks(
    study_id: str,
    ct_iso: sitk.Image,
    part_to_mask: dict[str, sitk.Image],
    top_percent: float,
) -> list[LandmarkRow]:
    rows: list[LandmarkRow] = []
    head_center = compute_head_center_mm(ct_iso)
    rows.append(
        landmark_row(
            study_id=study_id,
            landmark_name="head_center",
            point_mm=head_center,
            method="ct_foreground_center_of_mass_hu_gt_-300",
            n_voxels_used=0,
            source_part="ct",
            status="ok",
        )
    )

    part_points: dict[str, np.ndarray] = {
        part: mask_points_mm(mask_img) for part, mask_img in part_to_mask.items()
    }

    def add_centroid(part_name: str, landmark_name: str) -> None:
        points = part_points.get(part_name, np.empty((0, 3), dtype=np.float64))
        center = centroid_from_points(points)
        status = "ok" if center is not None else "missing_or_empty_mask"
        rows.append(
            landmark_row(
                study_id=study_id,
                landmark_name=landmark_name,
                point_mm=center,
                method="centroid",
                n_voxels_used=int(points.shape[0] if center is not None else 0),
                source_part=part_name,
                status=status,
            )
        )

    def add_robust_far(part_name: str, landmark_name: str) -> np.ndarray | None:
        points = part_points.get(part_name, np.empty((0, 3), dtype=np.float64))
        robust_point, used = robust_farthest_mean(points, head_center, top_percent=top_percent)
        status = "ok" if robust_point is not None else "missing_or_empty_mask"
        rows.append(
            landmark_row(
                study_id=study_id,
                landmark_name=landmark_name,
                point_mm=robust_point,
                method=f"robust_farthest_top_{top_percent:.3f}_mean",
                n_voxels_used=used,
                source_part=part_name,
                status=status,
            )
        )
        return robust_point

    # Eyes and lenses by centroids.
    add_centroid("eye_left", "eye_left_centroid")
    add_centroid("eye_right", "eye_right_centroid")
    add_centroid("eye_lens_left", "eye_lens_left_centroid")
    add_centroid("eye_lens_right", "eye_lens_right_centroid")

    # Auditory canals by robust farthest top-percent mean.
    add_robust_far("auditory_canal_left", "auditory_canal_left_robust_far")
    add_robust_far("auditory_canal_right", "auditory_canal_right_robust_far")

    # Nasal cavities robust farthest, then midpoint as nose-tip proxy.
    nasal_left = add_robust_far("nasal_cavity_left", "nasal_cavity_left_robust_far")
    nasal_right = add_robust_far("nasal_cavity_right", "nasal_cavity_right_robust_far")

    if nasal_left is not None and nasal_right is not None:
        nose_tip_proxy = 0.5 * (nasal_left + nasal_right)
        rows.append(
            landmark_row(
                study_id=study_id,
                landmark_name="nose_tip_proxy_mid",
                point_mm=nose_tip_proxy,
                method="midpoint(nasal_left_robust_far,nasal_right_robust_far)",
                n_voxels_used=0,
                source_part="nasal_cavity_left+nasal_cavity_right",
                status="ok",
            )
        )
    else:
        rows.append(
            landmark_row(
                study_id=study_id,
                landmark_name="nose_tip_proxy_mid",
                point_mm=None,
                method="midpoint(nasal_left_robust_far,nasal_right_robust_far)",
                n_voxels_used=0,
                source_part="nasal_cavity_left+nasal_cavity_right",
                status="missing_inputs",
            )
        )

    return rows


def write_landmarks_csv(rows: list[LandmarkRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "landmark_name",
                "x_mm",
                "y_mm",
                "z_mm",
                "method",
                "n_voxels_used",
                "source_part",
                "status",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.study_id,
                    r.landmark_name,
                    r.x_mm,
                    r.y_mm,
                    r.z_mm,
                    r.method,
                    r.n_voxels_used,
                    r.source_part,
                    r.status,
                ]
            )


def write_missing_csv(missing: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "part_name"])
        for study_id, part_name in missing:
            writer.writerow([study_id, part_name])


def main() -> None:
    args = parse_args()
    volumes_dir: Path = args.volumes_dir
    masks_dir: Path = args.masks_dir
    output_dir: Path = args.output_dir
    spacing: float = args.spacing
    top_percent: float = args.top_percent

    if not (0.0 < top_percent <= 1.0):
        raise ValueError("--top-percent must be in (0, 1].")
    if top_percent < 0.02 or top_percent > 0.05:
        print(
            "[WARN] top-percent is outside suggested 0.02..0.05 range. "
            f"Current: {top_percent:.3f}"
        )

    if not volumes_dir.exists() or not volumes_dir.is_dir():
        raise NotADirectoryError(f"Invalid --volumes-dir: {volumes_dir}")
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise NotADirectoryError(f"Invalid --masks-dir: {masks_dir}")

    volume_paths = collect_volume_paths(volumes_dir)
    if not volume_paths:
        raise FileNotFoundError(f"No studies found in: {volumes_dir}")

    all_rows: list[LandmarkRow] = []
    missing_parts: list[tuple[str, str]] = []

    print(f"[INFO] Found studies: {len(volume_paths)}")
    for idx, volume_path in enumerate(volume_paths, start=1):
        study_id = study_id_from_volume_path(volume_path)
        print(f"[INFO] [{idx}/{len(volume_paths)}] {study_id}")

        ct_iso = preprocess_ct(volume_path, spacing=spacing)
        study_mask_dir = masks_dir / study_id
        part_to_mask: dict[str, sitk.Image] = {}

        if not study_mask_dir.exists():
            for part in REQUIRED_PARTS:
                missing_parts.append((study_id, part))
            all_rows.append(
                landmark_row(
                    study_id=study_id,
                    landmark_name="head_center",
                    point_mm=compute_head_center_mm(ct_iso),
                    method="ct_foreground_center_of_mass_hu_gt_-300",
                    n_voxels_used=0,
                    source_part="ct",
                    status="ok",
                )
            )
            print(f"[WARN] Missing study mask folder: {study_mask_dir}")
            continue

        for part in REQUIRED_PARTS:
            mask_path = find_mask_path(study_mask_dir, part)
            if mask_path is None:
                missing_parts.append((study_id, part))
                continue
            part_to_mask[part] = preprocess_mask(mask_path, ct_ref_iso=ct_iso, spacing=spacing)

        rows = extract_study_landmarks(
            study_id=study_id,
            ct_iso=ct_iso,
            part_to_mask=part_to_mask,
            top_percent=top_percent,
        )
        all_rows.extend(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_landmarks_csv(all_rows, output_dir / "landmarks_per_study.csv")
    write_missing_csv(missing_parts, output_dir / "missing_landmarks.csv")

    n_ok = sum(1 for r in all_rows if r.status == "ok")
    n_bad = len(all_rows) - n_ok
    print(f"[DONE] Landmark rows: {len(all_rows)}")
    print(f"[DONE] OK rows: {n_ok}, non-OK rows: {n_bad}")
    print(f"[DONE] Missing parts entries: {len(missing_parts)}")
    print(f"[DONE] Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
