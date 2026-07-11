from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import apply_brain_ct_window
from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
HIDDEN_LANDMARKS = {
    "head_center",
    "nasal_cavity_left_robust_far",
    "nasal_cavity_right_robust_far",
}


@dataclass
class Landmark:
    name: str
    xyz_mm: tuple[float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize extracted landmarks as red points over averaged CT projections."
        )
    )
    parser.add_argument(
        "--volumes-dir",
        type=Path,
        required=True,
        help="Directory with CT studies (.nii/.nii.gz).",
    )
    parser.add_argument(
        "--landmarks-csv",
        type=Path,
        required=True,
        help="CSV from scripts/extract_landmarks.py (landmarks_per_study.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory with per-study projection images.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing in mm (default: 1.0).",
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


def load_landmarks(csv_path: Path) -> dict[str, list[Landmark]]:
    by_study: dict[str, list[Landmark]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            landmark_name = row["landmark_name"]
            if landmark_name in HIDDEN_LANDMARKS:
                continue
            study_id = row["study_id"]
            x = float(row["x_mm"])
            y = float(row["y_mm"])
            z = float(row["z_mm"])
            lm = Landmark(
                name=landmark_name,
                xyz_mm=(x, y, z),
            )
            by_study.setdefault(study_id, []).append(lm)
    return by_study


def preprocess_ct(ct_path: Path, spacing: float) -> sitk.Image:
    ct = read_nifti(ct_path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    return resample_ct_to_isotropic(ct, target_spacing=(spacing, spacing, spacing))


def mean_projections(volume_zyx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "axial": volume_zyx.mean(axis=0),    # [y, x]
        "coronal": volume_zyx.mean(axis=1),  # [z, x]
        "sagittal": volume_zyx.mean(axis=2), # [z, y]
    }


def landmarks_to_indices(ct_ref: sitk.Image, landmarks: list[Landmark]) -> list[tuple[str, float, float, float]]:
    result: list[tuple[str, float, float, float]] = []
    for lm in landmarks:
        x, y, z = lm.xyz_mm
        ix, iy, iz = ct_ref.TransformPhysicalPointToContinuousIndex((x, y, z))
        result.append((lm.name, float(ix), float(iy), float(iz)))
    return result


def render_projection_with_landmarks(
    projection: np.ndarray,
    indices_xyz: list[tuple[str, float, float, float]],
    plane: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(projection, cmap="gray", origin="lower")

    for _, ix, iy, iz in indices_xyz:
        if plane == "axial":
            px, py = ix, iy
        elif plane == "coronal":
            px, py = ix, iz
        elif plane == "sagittal":
            px, py = iy, iz
        else:
            continue
        ax.scatter(px, py, c="red", s=12)

    ax.set_title(f"{plane} mean projection + landmarks")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    volumes_dir: Path = args.volumes_dir
    landmarks_csv: Path = args.landmarks_csv
    output_dir: Path = args.output_dir
    spacing: float = args.spacing

    if not volumes_dir.exists() or not volumes_dir.is_dir():
        raise NotADirectoryError(f"Invalid --volumes-dir: {volumes_dir}")
    if not landmarks_csv.exists():
        raise FileNotFoundError(f"Missing --landmarks-csv: {landmarks_csv}")

    volume_paths = collect_volume_paths(volumes_dir)
    if not volume_paths:
        raise FileNotFoundError(f"No studies found in: {volumes_dir}")

    landmarks_by_study = load_landmarks(landmarks_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found studies: {len(volume_paths)}")
    for idx, volume_path in enumerate(volume_paths, start=1):
        study_id = study_id_from_volume_path(volume_path)
        print(f"[INFO] [{idx}/{len(volume_paths)}] Study: {study_id}")

        ct_iso = preprocess_ct(volume_path, spacing=spacing)
        ct_hu = sitk_image_to_numpy(ct_iso)
        ct_windowed = apply_brain_ct_window(
            ct_hu,
            window_level=40.0,
            window_width=90.0,
            output_range=(0.0, 255.0),
        )
        projections = mean_projections(ct_windowed)
        lm_indices = landmarks_to_indices(ct_iso, landmarks_by_study.get(study_id, []))

        study_out = output_dir / study_id
        render_projection_with_landmarks(
            projections["axial"],
            lm_indices,
            plane="axial",
            out_path=study_out / "mean_projection_landmarks_axial_0_0.png",
        )
        render_projection_with_landmarks(
            projections["coronal"],
            lm_indices,
            plane="coronal",
            out_path=study_out / "mean_projection_landmarks_coronal_0_1.png",
        )
        render_projection_with_landmarks(
            projections["sagittal"],
            lm_indices,
            plane="sagittal",
            out_path=study_out / "mean_projection_landmarks_sagittal_0_2.png",
        )

    print(f"[DONE] Saved visualizations to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
