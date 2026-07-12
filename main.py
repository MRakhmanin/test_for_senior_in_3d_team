from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F

from model.angle_regressor_3d import AngleRegressor3D
from utils import read_nifti, resample_ct_to_isotropic, sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
DEFAULT_SPACING = 2.0
DEFAULT_WL = 40.0
DEFAULT_WW = 150.0
DEFAULT_INPUT_SHAPE = (128, 128, 128)  # D, H, W
DEFAULT_CHECKPOINT = Path("artifacts/angle_regressor_axis_cv3_bs32_clip2_fold0_best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process one NIfTI study: run CNN inference, save rotated NIfTI, "
            "print predicted angles and rotation matrix."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input .nii/.nii.gz study.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to output rotated .nii.gz (default: <input_stem>_rotated.nii.gz).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Path to model checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto|cpu|cuda (default: auto).",
    )
    return parser.parse_args()


def build_model(checkpoint_path: Path, device: torch.device) -> AngleRegressor3D:
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
    return model


def preprocess_ct(input_path: Path) -> tuple[sitk.Image, torch.Tensor]:
    ct = read_nifti(input_path)
    ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
    ct_iso = resample_ct_to_isotropic(
        ct,
        target_spacing=(DEFAULT_SPACING, DEFAULT_SPACING, DEFAULT_SPACING),
    )

    ct_hu = sitk_image_to_numpy(ct_iso).astype(np.float32, copy=False)
    lower = float(DEFAULT_WL - DEFAULT_WW / 2.0)
    upper = float(DEFAULT_WL + DEFAULT_WW / 2.0)
    ct_clip = np.clip(ct_hu, lower, upper)
    ct_norm = (ct_clip - lower) / (upper - lower)
    ct_norm = np.clip(ct_norm, 0.0, 1.0)

    x = torch.from_numpy(ct_norm).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
    x = F.interpolate(x, size=DEFAULT_INPUT_SHAPE, mode="trilinear", align_corners=False)
    return ct_iso, x


@torch.no_grad()
def predict_angles(model: AngleRegressor3D, x: torch.Tensor, device: torch.device) -> np.ndarray:
    pred = model(x.to(device, non_blocking=True))[0].detach().cpu().numpy().astype(np.float64)
    return pred


def rotate_ct(ct_img: sitk.Image, angles_deg: np.ndarray) -> tuple[sitk.Image, np.ndarray]:
    tx = sitk.Euler3DTransform()
    size = np.array(ct_img.GetSize(), dtype=np.float64)
    center_idx = (size - 1.0) / 2.0
    center = ct_img.TransformContinuousIndexToPhysicalPoint([float(v) for v in center_idx])
    tx.SetCenter(center)

    roll, pitch, yaw = [float(v) for v in angles_deg]
    tx.SetRotation(np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw))

    matrix = np.array(tx.GetMatrix(), dtype=np.float64).reshape(3, 3)

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(ct_img.GetSpacing())
    resampler.SetSize(ct_img.GetSize())
    resampler.SetOutputDirection(ct_img.GetDirection())
    resampler.SetOutputOrigin(ct_img.GetOrigin())
    resampler.SetTransform(tx)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(-1024.0)
    resampler.SetOutputPixelType(ct_img.GetPixelID())
    rotated = resampler.Execute(ct_img)
    return rotated, matrix


def default_output_path(input_path: Path) -> Path:
    if input_path.name.endswith(".nii.gz"):
        stem = input_path.name[: -len(".nii.gz")]
    elif input_path.suffix == ".nii":
        stem = input_path.stem
    else:
        stem = input_path.stem
    return input_path.with_name(f"{stem}_rotated.nii.gz")


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not (args.input.name.endswith(".nii") or args.input.name.endswith(".nii.gz")):
        raise ValueError("Input must be .nii or .nii.gz")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = build_model(args.checkpoint, device=device)
    ct_iso, x = preprocess_ct(args.input)
    angles_deg = predict_angles(model, x, device=device)
    rotated_ct, matrix = rotate_ct(ct_iso, angles_deg)

    output_path = args.output if args.output is not None else default_output_path(args.input)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(rotated_ct, str(output_path))

    payload = {
        "output_path": str(output_path.resolve()),
        "angles_deg": {
            "roll": float(angles_deg[0]),
            "pitch": float(angles_deg[1]),
            "yaw": float(angles_deg[2]),
        },
        "rotation_matrix_3x3": matrix.tolist(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
