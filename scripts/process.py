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

from baseline.sequential_ellipse_baseline import process_study as baseline_process_study
from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import resample_image_with_transform
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process studies by selected method: "
            "baseline (sequential ellipse) or model (3D CNN angle regressor)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="NIfTI file or directory with NIfTI studies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for rotated studies and summary CSV.",
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline", "model"],
        help="Processing method.",
    )
    parser.add_argument(
        "--study-ids",
        nargs="+",
        default=None,
        help="Optional list of study IDs to process. If omitted, process all.",
    )
    parser.add_argument(
        "--mock-save",
        action="store_true",
        help=(
            "Save tiny placeholder NIfTI files instead of full rotated volumes. "
            "Useful for fast pipeline checks."
        ),
    )

    # Shared preprocessing.
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Target isotropic spacing in mm before processing (default: 1.0).",
    )

    # Baseline args.
    parser.add_argument("--bone-threshold-hu", type=float, default=300.0)
    parser.add_argument(
        "--axial-offset-mm",
        type=float,
        default=15.0,
        help="Offset from lowest bone slice for baseline axial step (default: 15).",
    )
    parser.add_argument("--top-slab-mm", type=float, default=50.0)
    parser.add_argument("--mid-band-mm", type=float, default=10.0)
    parser.add_argument("--save-qc", action="store_true")

    # Model args.
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (.pt). Required when --method model.",
    )
    parser.add_argument(
        "--input-shape",
        type=str,
        default="128,128,128",
        help="Model input D,H,W for inference resize (default: 128,128,128).",
    )
    parser.add_argument("--window-level", type=float, default=40.0)
    parser.add_argument("--window-width", type=float, default=150.0)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda (default: auto).",
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


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    vals = [int(v.strip()) for v in shape_str.split(",")]
    if len(vals) != 3 or any(v <= 0 for v in vals):
        raise ValueError("--input-shape should be D,H,W, e.g. 128,128,128")
    return vals[0], vals[1], vals[2]


def run_model_inference(
    ct_path: Path,
    checkpoint_path: Path,
    spacing: float,
    input_shape_dhw: tuple[int, int, int],
    window_level: float,
    window_width: float,
    device_arg: str,
) -> tuple[np.ndarray, sitk.Image]:
    import torch
    from torch.nn import functional as F

    from model.angle_regressor_3d import AngleRegressor3D

    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    base_channels = int(ckpt_args.get("base_channels", 24))
    norm = str(ckpt_args.get("norm", "group"))
    dropout = float(ckpt_args.get("dropout", 0.1))

    model = AngleRegressor3D(
        in_channels=1,
        base_channels=base_channels,
        norm=norm,
        dropout_p=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ct = read_nifti(ct_path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    ct_iso = resample_ct_to_isotropic(ct, target_spacing=(spacing, spacing, spacing))

    ct_hu = sitk_image_to_numpy(ct_iso).astype(np.float32, copy=False)  # [z, y, x]
    lower = float(window_level - window_width / 2.0)
    upper = float(window_level + window_width / 2.0)
    ct_norm = np.clip(ct_hu, lower, upper)
    ct_norm = (ct_norm - lower) / (upper - lower)
    ct_norm = np.clip(ct_norm, 0.0, 1.0).astype(np.float32, copy=False)

    x = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    x = F.interpolate(x, size=input_shape_dhw, mode="trilinear", align_corners=False)
    x = x.to(device, non_blocking=True)

    with torch.no_grad():
        pred = model(x)[0].detach().cpu().numpy().astype(np.float64)
    return pred, ct_iso


def rotate_ct_with_model_angles(ct_iso: sitk.Image, angles_deg: np.ndarray) -> sitk.Image:
    tx = sitk.Euler3DTransform()
    size = np.array(ct_iso.GetSize(), dtype=np.float64)
    center_idx = (size - 1.0) / 2.0
    center = ct_iso.TransformContinuousIndexToPhysicalPoint([float(v) for v in center_idx])
    tx.SetCenter(center)

    roll, pitch, yaw = [float(v) for v in angles_deg]
    tx.SetRotation(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))

    return resample_image_with_transform(
        image=ct_iso,
        transform=tx,
        interpolator=sitk.sitkLinear,
        default_value=-1024.0,
        expand_to_fit=True,
    )


def save_summary(rows: list[dict[str, str | float]], out_csv: Path, method: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if method == "baseline":
            writer.writerow(
                [
                    "study_id",
                    "roll_deg",
                    "pitch_deg",
                    "yaw_deg",
                    "axial_deg",
                    "coronal_deg",
                    "sagittal_deg",
                    "save_mode",
                ]
            )
            for r in rows:
                writer.writerow(
                    [
                        r["study_id"],
                        r["roll_deg"],
                        r["pitch_deg"],
                        r["yaw_deg"],
                        r["axial_deg"],
                        r["coronal_deg"],
                        r["sagittal_deg"],
                        r["save_mode"],
                    ]
                )
        else:
            writer.writerow(["study_id", "roll_deg", "pitch_deg", "yaw_deg", "save_mode"])
            for r in rows:
                writer.writerow(
                    [
                        r["study_id"],
                        r["roll_deg"],
                        r["pitch_deg"],
                        r["yaw_deg"],
                        r["save_mode"],
                    ]
                )


def save_mock_nifti(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.Image([1, 1, 1], sitk.sitkFloat32)
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(img, str(out_path))


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input does not exist: {args.input}")

    if args.method == "model" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --method model")
    if args.method == "model" and args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    all_paths = collect_nifti_paths(args.input)
    if not all_paths:
        raise FileNotFoundError(f"No .nii/.nii.gz found in: {args.input}")

    selected_ids = set(args.study_ids) if args.study_ids else None
    paths = [p for p in all_paths if selected_ids is None or study_id_from_path(p) in selected_ids]
    if not paths:
        raise RuntimeError("No studies matched the requested --study-ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Method: {args.method}")
    print(f"[INFO] Studies to process: {len(paths)}")

    rows: list[dict[str, str | float]] = []
    for idx, path in enumerate(paths, start=1):
        study_id = study_id_from_path(path)
        print(f"[INFO] [{idx}/{len(paths)}] Processing {study_id}")
        study_out = args.output_dir / study_id
        study_out.mkdir(parents=True, exist_ok=True)

        if args.method == "baseline":
            sid, steps, ct_rot = baseline_process_study(path, args)
            step_map = {s.step_name: s for s in steps}
            out_path = study_out / f"{sid}_rotated_baseline.nii.gz"
            if args.mock_save:
                save_mock_nifti(out_path)
                save_mode = "mock"
            else:
                sitk.WriteImage(ct_rot, str(out_path))
                save_mode = "full"
            rows.append(
                {
                    "study_id": sid,
                    # Match estimate_head_angles.py notation:
                    # roll(X), pitch(Y), yaw(Z).
                    # Baseline steps are:
                    # axial -> rotate around Z
                    # coronal -> rotate around Y
                    # sagittal -> rotate around X
                    "roll_deg": float(step_map["sagittal"].angle_deg),
                    "pitch_deg": float(step_map["coronal"].angle_deg),
                    "yaw_deg": float(step_map["axial"].angle_deg),
                    "axial_deg": float(step_map["axial"].angle_deg),
                    "coronal_deg": float(step_map["coronal"].angle_deg),
                    "sagittal_deg": float(step_map["sagittal"].angle_deg),
                    "save_mode": save_mode,
                }
            )
        else:
            assert args.checkpoint is not None
            pred_angles, ct_iso = run_model_inference(
                ct_path=path,
                checkpoint_path=args.checkpoint,
                spacing=args.spacing,
                input_shape_dhw=parse_shape(args.input_shape),
                window_level=args.window_level,
                window_width=args.window_width,
                device_arg=args.device,
            )
            ct_rot = rotate_ct_with_model_angles(ct_iso, pred_angles)
            out_path = study_out / f"{study_id}_rotated_model.nii.gz"
            if args.mock_save:
                save_mock_nifti(out_path)
                save_mode = "mock"
            else:
                sitk.WriteImage(ct_rot, str(out_path))
                save_mode = "full"
            rows.append(
                {
                    "study_id": study_id,
                    "roll_deg": float(pred_angles[0]),
                    "pitch_deg": float(pred_angles[1]),
                    "yaw_deg": float(pred_angles[2]),
                    "save_mode": save_mode,
                }
            )

    save_summary(rows, args.output_dir / f"processed_{args.method}_summary.csv", method=args.method)
    print(f"[DONE] Saved outputs to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

