from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import apply_brain_ct_window  # noqa: E402
from utils import read_nifti  # noqa: E402
from utils import resample_ct_to_isotropic  # noqa: E402
from utils import sitk_image_to_numpy  # noqa: E402


PLANE_COLS = [("axial", 0), ("coronal", 1), ("sagittal", 2)]
TARGET_ORIENTATION = "LPS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create visualization tiles for a folder of NIfTI studies. "
            "Output structure is compatible with scripts/build_image_grid_html.py."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Folder with NIfTI studies.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder. Each study will have its own subfolder.",
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
        help="CT brain window level (default: 40).",
    )
    parser.add_argument(
        "--window-width",
        type=float,
        default=90.0,
        help="CT brain window width for visualization (default: 90).",
    )
    return parser.parse_args()


def collect_nifti_paths(input_dir: Path) -> list[Path]:
    nii_paths = sorted(input_dir.rglob("*.nii"))
    nii_gz_paths = sorted(input_dir.rglob("*.nii.gz"))
    return sorted({*nii_paths, *nii_gz_paths})


def study_name_from_path(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[: -len(".nii.gz")]
    if path.name.endswith(".nii"):
        return path.name[: -len(".nii")]
    return path.stem


def compute_head_center(hu_volume: np.ndarray) -> tuple[int, int, int]:
    # Heuristic brain/head foreground threshold in HU to ignore surrounding air.
    foreground = hu_volume > -300.0
    coords = np.argwhere(foreground)

    if coords.size == 0:
        zc, yc, xc = np.array(hu_volume.shape) // 2
        return int(zc), int(yc), int(xc)

    zc, yc, xc = coords.mean(axis=0)
    return int(round(float(zc))), int(round(float(yc))), int(round(float(xc)))


def clip_index(index: int, size: int) -> int:
    return max(0, min(size - 1, index))


def center_cuts(volume: np.ndarray, center_zyx: tuple[int, int, int]) -> dict[str, np.ndarray]:
    zc, yc, xc = center_zyx
    zc = clip_index(zc, volume.shape[0])
    yc = clip_index(yc, volume.shape[1])
    xc = clip_index(xc, volume.shape[2])

    return {
        "axial": volume[zc, :, :],     # [y, x]
        "coronal": volume[:, yc, :],   # [z, x]
        "sagittal": volume[:, :, xc],  # [z, y]
    }


def mean_projections(volume: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "axial": volume.mean(axis=0),    # [y, x], mean over z
        "coronal": volume.mean(axis=1),  # [z, x], mean over y
        "sagittal": volume.mean(axis=2), # [z, y], mean over x
    }


def to_uint8(image_2d: np.ndarray) -> np.ndarray:
    return np.clip(image_2d, 0, 255).astype(np.uint8)


def save_png(image_2d: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk_img = sitk.GetImageFromArray(image_2d)
    sitk.WriteImage(sitk_img, str(out_path))


def get_orientation(image: sitk.Image) -> str:
    return sitk.DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        image.GetDirection()
    )


def prepare_for_display(plane_name: str, image_2d: np.ndarray) -> np.ndarray:
    """
    Convert slice/projection from array coordinates to a more natural display view.

    PNG output does not carry medical orientation metadata, so we apply explicit
    flips for consistent visual appearance in standard image viewers.
    """
    _ = plane_name  # kept for future plane-specific display rules
    return np.flipud(image_2d)


def process_study(
    study_path: Path,
    output_root: Path,
    target_spacing: float,
    window_level: float,
    window_width: float,
) -> None:
    print(f"[INFO] Processing: {study_path}")
    ct_image = read_nifti(study_path)
    before_orientation = get_orientation(ct_image)
    ct_image = sitk.DICOMOrient(ct_image, TARGET_ORIENTATION)
    after_orientation = get_orientation(ct_image)
    if before_orientation != after_orientation:
        print(f"[INFO] Orientation: {before_orientation} -> {after_orientation}")

    ct_iso = resample_ct_to_isotropic(
        ct_image,
        target_spacing=(target_spacing, target_spacing, target_spacing),
    )

    ct_hu = sitk_image_to_numpy(ct_iso)
    ct_windowed = apply_brain_ct_window(
        ct_hu,
        window_level=window_level,
        window_width=window_width,
        output_range=(0, 255),
    )

    center = compute_head_center(ct_hu)
    cuts = center_cuts(ct_windowed, center)
    projections = mean_projections(ct_windowed)

    study_output_dir = output_root / study_name_from_path(study_path)
    study_output_dir.mkdir(parents=True, exist_ok=True)

    # Row 0: orthogonal center cuts.
    row_idx = 0
    for plane_name, col_idx in PLANE_COLS:
        filename = f"center_cut_{plane_name}_{row_idx}_{col_idx}.png"
        display_image = prepare_for_display(plane_name, cuts[plane_name])
        save_png(to_uint8(display_image), study_output_dir / filename)

    # Row 1: axis-wise mean intensity projections.
    row_idx = 1
    for plane_name, col_idx in PLANE_COLS:
        filename = f"mean_projection_{plane_name}_{row_idx}_{col_idx}.png"
        display_image = prepare_for_display(plane_name, projections[plane_name])
        save_png(to_uint8(display_image), study_output_dir / filename)

    print(f"[INFO] Saved: {study_output_dir}")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory is invalid: {input_dir}")

    studies = collect_nifti_paths(input_dir)
    if not studies:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in: {input_dir}")

    print(f"[INFO] Found studies: {len(studies)}")
    for study_path in studies:
        process_study(
            study_path=study_path,
            output_root=output_dir,
            target_spacing=args.spacing,
            window_level=args.window_level,
            window_width=args.window_width,
        )

    print(f"[DONE] Visualization tiles are saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
