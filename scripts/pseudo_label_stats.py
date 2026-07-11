from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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
TARGET_PARTS = [
    "frontal_lobe",
    "occipital_lobe",
    "parietal_lobe",
    "temporal_lobe",
    "auditory_canal_left",
    "auditory_canal_right",
    "eye_left",
    "eye_lens_left",
    "eye_lens_right",
    "eye_right",
    "nasal_cavity_left",
    "nasal_cavity_right",
]


@dataclass
class PartStats:
    study_id: str
    part_name: str
    mask_path: Path
    voxel_count: int
    volume_mm3: float
    surface_mm2: float
    volume_surface_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute pseudo-label quality stats for selected body parts from masks: "
            "volume and surface distributions per part."
        )
    )
    parser.add_argument(
        "--volumes-dir",
        type=Path,
        required=True,
        help="Directory with CT studies in NIfTI format.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        required=True,
        help=(
            "Directory with masks structure: "
            "<study_id>/<masks_group>/<part_of_body>.nii.gz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save CSV summary and per-part histograms.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing for preprocessing (default: 1.0).",
    )
    parser.add_argument(
        "--parts",
        nargs="+",
        default=TARGET_PARTS,
        help="Optional custom list of part names to analyze.",
    )
    return parser.parse_args()


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with:\n"
            "  uv add matplotlib\n"
            "or\n"
            "  pip install matplotlib"
        ) from exc
    return plt


def study_id_from_volume_path(volume_path: Path) -> str:
    if volume_path.name.endswith(".nii.gz"):
        return volume_path.name[: -len(".nii.gz")]
    if volume_path.name.endswith(".nii"):
        return volume_path.name[: -len(".nii")]
    return volume_path.stem


def collect_volume_paths(volumes_dir: Path) -> list[Path]:
    nii = list(volumes_dir.rglob("*.nii"))
    nii_gz = list(volumes_dir.rglob("*.nii.gz"))
    return sorted({*nii, *nii_gz})


def find_mask_for_part(study_mask_dir: Path, part_name: str) -> Path | None:
    candidates = sorted(study_mask_dir.glob(f"*/{part_name}.nii.gz"))
    if candidates:
        return candidates[0]
    candidates = sorted(study_mask_dir.glob(f"*/{part_name}.nii"))
    if candidates:
        return candidates[0]
    return None


def preprocess_ct_and_mask(
    ct_path: Path,
    mask_path: Path,
    target_spacing: float,
) -> tuple[sitk.Image, sitk.Image]:
    ct = read_nifti(ct_path)
    mask = read_nifti(mask_path)

    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    mask = sitk.DICOMOrient(mask, TARGET_ORIENTATION)

    ct_iso = resample_ct_to_isotropic(
        ct,
        target_spacing=(target_spacing, target_spacing, target_spacing),
    )
    mask_iso = resample_mask_to_isotropic(
        mask,
        target_spacing=(target_spacing, target_spacing, target_spacing),
    )

    # Force mask to exact CT grid to avoid tiny differences in size/origin.
    mask_aligned = sitk.Resample(
        mask_iso,
        ct_iso,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0.0,
        mask_iso.GetPixelID(),
    )
    return ct_iso, mask_aligned


def compute_surface_mm2(mask_zyx: np.ndarray, spacing_xyz: tuple[float, float, float]) -> float:
    """
    Approximate surface area by counting boundary faces in a binary voxel grid.
    """
    binary = (mask_zyx > 0).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0

    sx, sy, sz = spacing_xyz
    area_x_face = sy * sz
    area_y_face = sx * sz
    area_z_face = sx * sy

    padded = np.pad(binary, ((1, 1), (1, 1), (1, 1)), mode="constant")
    dz = np.abs(np.diff(padded, axis=0)).sum()  # interfaces orthogonal to z
    dy = np.abs(np.diff(padded, axis=1)).sum()  # interfaces orthogonal to y
    dx = np.abs(np.diff(padded, axis=2)).sum()  # interfaces orthogonal to x

    # z-axis in numpy corresponds to physical z, etc. Face areas map accordingly.
    return float(dz * area_z_face + dy * area_y_face + dx * area_x_face)


def compute_part_stats(
    study_id: str,
    part_name: str,
    mask_path: Path,
    mask_image: sitk.Image,
) -> PartStats:
    mask_np = sitk_image_to_numpy(mask_image)  # [z, y, x]
    voxel_count = int((mask_np > 0).sum())
    spacing_xyz = tuple(float(v) for v in mask_image.GetSpacing())  # (x, y, z)
    voxel_volume_mm3 = spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2]

    volume_mm3 = voxel_count * voxel_volume_mm3
    surface_mm2 = compute_surface_mm2(mask_np, spacing_xyz=spacing_xyz)
    volume_surface_ratio = 0.0 if surface_mm2 <= 0.0 else float(volume_mm3 / surface_mm2)

    return PartStats(
        study_id=study_id,
        part_name=part_name,
        mask_path=mask_path,
        voxel_count=voxel_count,
        volume_mm3=float(volume_mm3),
        surface_mm2=float(surface_mm2),
        volume_surface_ratio=volume_surface_ratio,
    )


def save_csv(rows: list[PartStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "part_name",
                "mask_path",
                "voxel_count",
                "volume_mm3",
                "surface_mm2",
                "volume_surface_ratio",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.study_id,
                    row.part_name,
                    str(row.mask_path),
                    row.voxel_count,
                    row.volume_mm3,
                    row.surface_mm2,
                    row.volume_surface_ratio,
                ]
            )


def save_part_histograms(rows: list[PartStats], output_dir: Path) -> None:
    plt = ensure_matplotlib()
    per_part: dict[str, list[PartStats]] = defaultdict(list)
    for row in rows:
        per_part[row.part_name].append(row)

    hist_dir = output_dir / "histograms"
    hist_dir.mkdir(parents=True, exist_ok=True)

    for part_name, part_rows in sorted(per_part.items()):
        ratios = np.array([r.volume_surface_ratio for r in part_rows], dtype=np.float64)

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        ax.hist(ratios, bins=20)
        ax.set_title(f"{part_name}: volume/surface")
        ax.set_xlabel("mm")
        ax.set_ylabel("count")

        fig.tight_layout()
        fig.savefig(hist_dir / f"{part_name}_ratio_hist.png", dpi=160)
        plt.close(fig)


def save_part_summary(rows: list[PartStats], out_path: Path) -> None:
    per_part: dict[str, list[PartStats]] = defaultdict(list)
    for row in rows:
        per_part[row.part_name].append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "part_name",
                "n",
                "volume_mean_mm3",
                "volume_std_mm3",
                "surface_mean_mm2",
                "surface_std_mm2",
                "ratio_mean_mm",
                "ratio_std_mm",
            ]
        )
        for part_name, part_rows in sorted(per_part.items()):
            vols = np.array([r.volume_mm3 for r in part_rows], dtype=np.float64)
            surfs = np.array([r.surface_mm2 for r in part_rows], dtype=np.float64)
            ratios = np.array([r.volume_surface_ratio for r in part_rows], dtype=np.float64)
            writer.writerow(
                [
                    part_name,
                    len(part_rows),
                    float(vols.mean()) if len(vols) else 0.0,
                    float(vols.std()) if len(vols) else 0.0,
                    float(surfs.mean()) if len(surfs) else 0.0,
                    float(surfs.std()) if len(surfs) else 0.0,
                    float(ratios.mean()) if len(ratios) else 0.0,
                    float(ratios.std()) if len(ratios) else 0.0,
                ]
            )


def save_global_summary(rows: list[PartStats], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vols = np.array([r.volume_mm3 for r in rows], dtype=np.float64)
    surfs = np.array([r.surface_mm2 for r in rows], dtype=np.float64)
    ratios = np.array([r.volume_surface_ratio for r in rows], dtype=np.float64)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "n"])
        writer.writerow(
            [
                "volume_mm3",
                float(vols.mean()) if len(vols) else 0.0,
                float(vols.std()) if len(vols) else 0.0,
                len(vols),
            ]
        )
        writer.writerow(
            [
                "surface_mm2",
                float(surfs.mean()) if len(surfs) else 0.0,
                float(surfs.std()) if len(surfs) else 0.0,
                len(surfs),
            ]
        )
        writer.writerow(
            [
                "volume_surface_ratio_mm",
                float(ratios.mean()) if len(ratios) else 0.0,
                float(ratios.std()) if len(ratios) else 0.0,
                len(ratios),
            ]
        )


def main() -> None:
    args = parse_args()
    volumes_dir: Path = args.volumes_dir
    masks_dir: Path = args.masks_dir
    output_dir: Path = args.output_dir
    target_spacing: float = args.spacing
    parts: list[str] = list(dict.fromkeys(args.parts))

    if not volumes_dir.exists() or not volumes_dir.is_dir():
        raise NotADirectoryError(f"Invalid --volumes-dir: {volumes_dir}")
    if not masks_dir.exists() or not masks_dir.is_dir():
        raise NotADirectoryError(f"Invalid --masks-dir: {masks_dir}")

    volume_paths = collect_volume_paths(volumes_dir)
    if not volume_paths:
        raise FileNotFoundError(f"No NIfTI studies found in: {volumes_dir}")

    all_rows: list[PartStats] = []
    missing_masks: list[tuple[str, str]] = []

    print(f"[INFO] Found studies: {len(volume_paths)}")
    for idx, volume_path in enumerate(volume_paths, start=1):
        study_id = study_id_from_volume_path(volume_path)
        study_mask_dir = masks_dir / study_id
        print(f"[INFO] [{idx}/{len(volume_paths)}] Study: {study_id}")

        if not study_mask_dir.exists():
            for part_name in parts:
                missing_masks.append((study_id, part_name))
            print(f"[WARN] No mask directory for study: {study_mask_dir}")
            continue

        for part_name in parts:
            mask_path = find_mask_for_part(study_mask_dir, part_name)
            if mask_path is None:
                missing_masks.append((study_id, part_name))
                continue

            _, mask_aligned = preprocess_ct_and_mask(
                ct_path=volume_path,
                mask_path=mask_path,
                target_spacing=target_spacing,
            )
            row = compute_part_stats(
                study_id=study_id,
                part_name=part_name,
                mask_path=mask_path,
                mask_image=mask_aligned,
            )
            all_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(all_rows, output_dir / "per_study_part_stats.csv")
    save_part_summary(all_rows, output_dir / "per_part_summary.csv")
    save_global_summary(all_rows, output_dir / "global_summary.csv")
    save_part_histograms(all_rows, output_dir)

    missing_csv = output_dir / "missing_masks.csv"
    with missing_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["study_id", "part_name"])
        for study_id, part_name in missing_masks:
            writer.writerow([study_id, part_name])

    print(f"[DONE] Saved stats rows: {len(all_rows)}")
    print(f"[DONE] Missing masks entries: {len(missing_masks)}")
    if all_rows:
        ratio_values = np.array([r.volume_surface_ratio for r in all_rows], dtype=np.float64)
        print(
            "[DONE] Global ratio mean/std (mm): "
            f"{float(ratio_values.mean()):.6f} / {float(ratio_values.std()):.6f}"
        )
    print(f"[DONE] Output directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
