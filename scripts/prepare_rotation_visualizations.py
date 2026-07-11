from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import apply_brain_ct_window
from utils import image_center_physical
from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import resample_image_with_transform
from utils import rotation_matrix_to_transform
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
PLANE_COLS = [("axial", 0), ("coronal", 1), ("sagittal", 2)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare per-study 2-row visualization tiles for HTML: "
            "row 0 original CT mean projections, row 1 rotated CT mean projections."
        )
    )
    parser.add_argument(
        "--volumes-dir",
        type=Path,
        required=True,
        help="Directory with CT studies (.nii/.nii.gz).",
    )
    parser.add_argument(
        "--angles-csv",
        type=Path,
        required=True,
        help="CSV from estimate_head_angles.py with rotation matrices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory with per-study images.",
    )
    parser.add_argument(
        "--option",
        type=str,
        default="gpa",
        choices=["gpa", "axis", "reference"],
        help="Which canonical option to visualize (default: gpa).",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing in mm (default: 1.0).",
    )
    parser.add_argument(
        "--window-level",
        type=float,
        default=40.0,
        help="CT window level for visualization (default: 40).",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=90.0,
        help="CT window width for visualization (default: 90).",
    )
    return parser.parse_args()


def collect_volume_paths(volumes_dir: Path) -> list[Path]:
    nii = list(volumes_dir.rglob("*.nii"))
    nii_gz = list(volumes_dir.rglob("*.nii.gz"))
    return sorted({*nii, *nii_gz})


def study_id_from_path(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    if path.name.endswith(".nii"):
        return path.name[: -len(".nii")]
    return path.stem


def load_rotations(angles_csv: Path, option: str) -> dict[str, np.ndarray]:
    mapping: dict[str, np.ndarray] = {}
    with angles_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("option") != option:
                continue
            if row.get("status") != "ok":
                continue
            study_id = row["study_id"]
            R = np.array(
                [
                    [float(row["r11"]), float(row["r12"]), float(row["r13"])],
                    [float(row["r21"]), float(row["r22"]), float(row["r23"])],
                    [float(row["r31"]), float(row["r32"]), float(row["r33"])],
                ],
                dtype=np.float64,
            )
            mapping[study_id] = R
    return mapping


def mean_projections(volume_zyx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "axial": volume_zyx.mean(axis=0),
        "coronal": volume_zyx.mean(axis=1),
        "sagittal": volume_zyx.mean(axis=2),
    }


def prepare_for_display(image_2d: np.ndarray) -> np.ndarray:
    return np.flipud(image_2d)


def save_png(image_2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_img = sitk.GetImageFromArray(np.clip(image_2d, 0, 255).astype(np.uint8))
    sitk.WriteImage(sitk_img, str(out_path))


def preprocess_ct(ct_path: Path, spacing: float) -> sitk.Image:
    ct = read_nifti(ct_path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    return resample_ct_to_isotropic(ct, target_spacing=(spacing, spacing, spacing))


def window_ct(ct_img: sitk.Image, wl: float, ww: float) -> np.ndarray:
    ct_hu = sitk_image_to_numpy(ct_img)
    return apply_brain_ct_window(
        ct_hu,
        window_level=wl,
        window_width=ww,
        output_range=(0.0, 255.0),
    )


def rotate_ct(ct_img: sitk.Image, R_input_to_canonical: np.ndarray) -> sitk.Image:
    transform = rotation_matrix_to_transform(
        matrix=R_input_to_canonical,
        center=image_center_physical(ct_img),
        invert=True,  # R maps input->canonical, resampler expects output->input
    )
    return resample_image_with_transform(
        image=ct_img,
        transform=transform,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        expand_to_fit=True,
    )


def main() -> None:
    args = parse_args()
    volumes_dir: Path = args.volumes_dir
    angles_csv: Path = args.angles_csv
    output_dir: Path = args.output_dir

    if not volumes_dir.exists() or not volumes_dir.is_dir():
        raise NotADirectoryError(f"Invalid --volumes-dir: {volumes_dir}")
    if not angles_csv.exists():
        raise FileNotFoundError(f"Missing --angles-csv: {angles_csv}")

    volume_paths = collect_volume_paths(volumes_dir)
    if not volume_paths:
        raise FileNotFoundError(f"No studies found in: {volumes_dir}")

    rotations = load_rotations(angles_csv, option=args.option)
    if not rotations:
        raise RuntimeError(
            f"No valid rotations found in {angles_csv} for option={args.option}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found studies: {len(volume_paths)}")
    print(f"[INFO] Rotations loaded for option '{args.option}': {len(rotations)}")

    for idx, study_path in enumerate(volume_paths, start=1):
        study_id = study_id_from_path(study_path)
        print(f"[INFO] [{idx}/{len(volume_paths)}] {study_id}")

        ct_iso = preprocess_ct(study_path, spacing=args.spacing)
        original_windowed = window_ct(ct_iso, wl=args.window_level, ww=args.window_width)
        original_proj = mean_projections(original_windowed)

        study_out = output_dir / study_id
        for plane_name, col_idx in PLANE_COLS:
            file_name = f"mean_projection_original_{plane_name}_0_{col_idx}.png"
            save_png(prepare_for_display(original_proj[plane_name]), study_out / file_name)

        R = rotations.get(study_id)
        if R is None:
            print(f"[WARN] Missing rotation for {study_id}; skipping rotated row.")
            continue

        ct_rot = rotate_ct(ct_iso, R_input_to_canonical=R)
        rotated_windowed = window_ct(ct_rot, wl=args.window_level, ww=args.window_width)
        rotated_proj = mean_projections(rotated_windowed)

        for plane_name, col_idx in PLANE_COLS:
            file_name = f"mean_projection_rotated_{plane_name}_1_{col_idx}.png"
            save_png(prepare_for_display(rotated_proj[plane_name]), study_out / file_name)

    print(f"[DONE] Visualization tiles saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
