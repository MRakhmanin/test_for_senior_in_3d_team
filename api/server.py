from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from torch.nn import functional as F

from model.angle_regressor_3d import AngleRegressor3D
from utils import read_nifti, resample_ct_to_isotropic, sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
DEFAULT_SPACING = 2.0
DEFAULT_WL = 40.0
DEFAULT_WW = 150.0
DEFAULT_INPUT_SHAPE = (128, 128, 128)  # D, H, W


class InferenceService:
    def __init__(self) -> None:
        device_env = os.getenv("DEVICE", "auto")
        if device_env == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_env)
        print(f"[INFO] Inference device: {self.device}")

        checkpoint_path = os.getenv("MODEL_CHECKPOINT")
        if checkpoint_path is None:
            raise RuntimeError(
                "MODEL_CHECKPOINT env var is required and must point to .pt checkpoint."
            )
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise RuntimeError(f"MODEL_CHECKPOINT not found: {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        ckpt_args = checkpoint.get("args", {})
        base_channels = int(ckpt_args.get("base_channels", 24))
        norm = str(ckpt_args.get("norm", "group"))
        dropout = float(ckpt_args.get("dropout", 0.1))

        self.model = AngleRegressor3D(
            in_channels=1,
            base_channels=base_channels,
            norm=norm,
            dropout_p=dropout,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print("[INFO] Model loaded and ready")

    def preprocess_ct(self, path: Path) -> tuple[sitk.Image, torch.Tensor]:
        ct = read_nifti(path)
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
        return ct_iso, x.to(self.device, non_blocking=True)

    @torch.no_grad()
    def predict_angles(self, x: torch.Tensor) -> np.ndarray:
        pred = self.model(x)[0].detach().cpu().numpy().astype(np.float64)
        return pred

    @staticmethod
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


app = FastAPI(title="CT Head Orientation Normalization API", version="1.0.0")

SERVICE: InferenceService | None = None


@app.on_event("startup")
def startup_event() -> None:
    global SERVICE
    SERVICE = InferenceService()


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/process")
async def process(file: UploadFile = File(...)) -> StreamingResponse:
    if SERVICE is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    file_name = file.filename or "input.nii.gz"
    if not (file_name.endswith(".nii") or file_name.endswith(".nii.gz")):
        raise HTTPException(status_code=400, detail="Only .nii/.nii.gz files are supported.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / file_name
        input_bytes = await file.read()
        input_path.write_bytes(input_bytes)

        ct_iso, model_input = SERVICE.preprocess_ct(input_path)
        angles_deg = SERVICE.predict_angles(model_input)
        rotated_ct, matrix = SERVICE.rotate_ct(ct_iso, angles_deg)

        rotated_path = tmp_path / "rotated.nii.gz"
        sitk.WriteImage(rotated_ct, str(rotated_path))

        payload = {
            "angles_deg": {
                "roll": float(angles_deg[0]),
                "pitch": float(angles_deg[1]),
                "yaw": float(angles_deg[2]),
            },
            "rotation_matrix_3x3": matrix.tolist(),
        }

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("result.json", json.dumps(payload, indent=2))
            zf.write(rotated_path, arcname="rotated.nii.gz")
        zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="result.zip"'},
    )

