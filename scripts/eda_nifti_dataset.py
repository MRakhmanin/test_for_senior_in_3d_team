from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


@dataclass
class StudyInfo:
    study_path: Path
    spacing_x: float
    spacing_y: float
    spacing_z: float
    size_x: int
    size_y: int
    size_z: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EDA over a directory of NIfTI studies and save histograms."
    )
    parser.add_argument("input_dir", type=Path, help="Folder with .nii/.nii.gz studies.")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder for EDA artifacts (plots + CSV summary).",
    )
    return parser.parse_args()


def collect_nifti_paths(input_dir: Path) -> list[Path]:
    nii_paths = list(input_dir.rglob("*.nii"))
    nii_gz_paths = list(input_dir.rglob("*.nii.gz"))
    return sorted({*nii_paths, *nii_gz_paths})


def read_study_info(study_path: Path) -> StudyInfo:
    image = sitk.ReadImage(str(study_path))
    spacing = image.GetSpacing()  # (x, y, z)
    size = image.GetSize()  # (x, y, z)
    return StudyInfo(
        study_path=study_path,
        spacing_x=float(spacing[0]),
        spacing_y=float(spacing[1]),
        spacing_z=float(spacing[2]),
        size_x=int(size[0]),
        size_y=int(size[1]),
        size_z=int(size[2]),
    )


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with:\n"
            "  uv add matplotlib\n"
            "or\n"
            "  pip install matplotlib"
        ) from exc
    return plt


def plot_numeric_hist(values_xyz: np.ndarray, title: str, out_path: Path, bins: int = 30) -> None:
    plt = ensure_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axis_names = ["x", "y", "z"]

    for i, axis_name in enumerate(axis_names):
        axes[i].hist(values_xyz[:, i], bins=bins)
        axes[i].set_title(f"{title} ({axis_name})")
        axes[i].set_xlabel(axis_name)
        axes[i].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_summary_csv(studies: list[StudyInfo], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_path",
                "spacing_x",
                "spacing_y",
                "spacing_z",
                "size_x",
                "size_y",
                "size_z",
            ]
        )
        for info in studies:
            writer.writerow(
                [
                    str(info.study_path),
                    info.spacing_x,
                    info.spacing_y,
                    info.spacing_z,
                    info.size_x,
                    info.size_y,
                    info.size_z,
                ]
            )


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Invalid input directory: {input_dir}")

    study_paths = collect_nifti_paths(input_dir)
    if not study_paths:
        raise FileNotFoundError(f"No .nii/.nii.gz studies found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Found studies: {len(study_paths)}")

    studies: list[StudyInfo] = []
    for idx, study_path in enumerate(study_paths, start=1):
        print(f"[INFO] Reading {idx}/{len(study_paths)}: {study_path}")
        studies.append(read_study_info(study_path))

    save_summary_csv(studies, output_dir / "study_summary.csv")

    spacing_xyz = np.array([[s.spacing_x, s.spacing_y, s.spacing_z] for s in studies], dtype=np.float64)
    size_xyz = np.array([[s.size_x, s.size_y, s.size_z] for s in studies], dtype=np.float64)
    plot_numeric_hist(
        spacing_xyz,
        title="Spacing Distribution",
        out_path=output_dir / "spacing_distribution.png",
    )
    plot_numeric_hist(
        size_xyz,
        title="Size Distribution",
        out_path=output_dir / "size_distribution.png",
    )

    print(f"[DONE] EDA artifacts are saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
