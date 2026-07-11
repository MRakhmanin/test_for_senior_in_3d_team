from __future__ import annotations

import argparse
import csv
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import SimpleITK as sitk
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.angle_regressor_3d import AngleRegressor3D
from model.angle_regressor_3d import angle_regression_loss
from utils import read_nifti
from utils import resample_ct_to_isotropic
from utils import sitk_image_to_numpy


TARGET_ORIENTATION = "LPS"
DEFAULT_WL = 40.0
DEFAULT_WW = 150.0


@dataclass
class Sample:
    study_id: str
    volume_path: Path
    angles_deg: np.ndarray  # [3]


class IntensityAugment3D:
    """
    Small intensity-only augmentations on normalized [0, 1] volume.
    """

    def __init__(
        self,
        p: float = 0.8,
        brightness_delta: float = 0.08,
        contrast_range: tuple[float, float] = (0.9, 1.1),
        gamma_range: tuple[float, float] = (0.9, 1.1),
        noise_std: float = 0.02,
    ) -> None:
        self.p = p
        self.brightness_delta = brightness_delta
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return x

        out = x.astype(np.float32, copy=True)

        # Contrast around mean.
        if random.random() < 0.8:
            c = random.uniform(self.contrast_range[0], self.contrast_range[1])
            m = float(out.mean())
            out = (out - m) * c + m

        # Brightness shift.
        if random.random() < 0.6:
            b = random.uniform(-self.brightness_delta, self.brightness_delta)
            out = out + b

        # Gamma.
        if random.random() < 0.6:
            g = random.uniform(self.gamma_range[0], self.gamma_range[1])
            out = np.power(np.clip(out, 0.0, 1.0), g, dtype=np.float32)

        # Additive noise.
        if random.random() < 0.5:
            noise = np.random.normal(0.0, self.noise_std, size=out.shape).astype(np.float32)
            out = out + noise

        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


class NiftiAngleDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        target_spacing: float,
        output_shape_dhw: tuple[int, int, int],
        window_level: float = DEFAULT_WL,
        window_width: float = DEFAULT_WW,
        augment: IntensityAugment3D | None = None,
    ) -> None:
        self.samples = samples
        self.target_spacing = target_spacing
        self.output_shape_dhw = output_shape_dhw
        self.window_level = window_level
        self.window_width = window_width
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _load_preprocess_ct(self, volume_path: Path) -> np.ndarray:
        ct = read_nifti(volume_path)
        ct = sitk.DICOMOrient(ct, TARGET_ORIENTATION)
        ct_iso = resample_ct_to_isotropic(
            ct,
            target_spacing=(
                self.target_spacing,
                self.target_spacing,
                self.target_spacing,
            ),
        )
        ct_hu = sitk_image_to_numpy(ct_iso)  # [z, y, x]
        lower = float(self.window_level - self.window_width / 2.0)
        upper = float(self.window_level + self.window_width / 2.0)
        # Explicitly clip HU values in brain window before normalization.
        ct_clip = np.clip(ct_hu.astype(np.float32, copy=False), lower, upper)
        ct_norm = (ct_clip - lower) / (upper - lower)
        return np.clip(ct_norm, 0.0, 1.0).astype(np.float32, copy=False)

    def _resize_to_shape(self, volume_zyx: np.ndarray) -> Tensor:
        x = torch.from_numpy(volume_zyx).unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        x = F.interpolate(
            x,
            size=self.output_shape_dhw,
            mode="trilinear",
            align_corners=False,
        )
        return x.squeeze(0)  # [1,D,H,W]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        sample = self.samples[idx]
        ct = self._load_preprocess_ct(sample.volume_path)
        if self.augment is not None:
            ct = self.augment(ct)
        x = self._resize_to_shape(ct).contiguous()  # [1,D,H,W]
        y = torch.from_numpy(sample.angles_deg.astype(np.float32, copy=False))  # [3]
        return x, y


def parse_shape(shape_str: str) -> tuple[int, int, int]:
    parts = [int(v.strip()) for v in shape_str.split(",")]
    if len(parts) != 3 or any(v <= 0 for v in parts):
        raise ValueError("--input-shape must be three positive ints, e.g. 128,128,128")
    return parts[0], parts[1], parts[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 3D CNN to regress roll/pitch/yaw from NIfTI volumes."
    )
    parser.add_argument("--volumes-dir", type=Path, required=True, help="Directory with CT NIfTI studies.")
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        required=True,
        help="Annotation CSV from estimate_head_angles.py (single option csv recommended).",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and logs.")

    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--spacing", type=float, default=2.0, help="Target isotropic spacing in mm (default: 2.0).")
    parser.add_argument(
        "--input-shape",
        type=str,
        default="128,128,128",
        help="Model input shape D,H,W after resizing (default: 128,128,128).",
    )
    parser.add_argument("--window-level", type=float, default=DEFAULT_WL)
    parser.add_argument("--window-width", type=float, default=DEFAULT_WW)

    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--norm", type=str, default="group", choices=["group", "instance", "batch"])
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_volume_paths(volumes_dir: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted({*volumes_dir.rglob("*.nii"), *volumes_dir.rglob("*.nii.gz")}):
        name = path.name
        if name.endswith(".nii.gz"):
            study_id = name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            study_id = name[: -len(".nii")]
        else:
            study_id = path.stem
        mapping[study_id] = path
    return mapping


def load_samples(annotations_csv: Path, volume_map: dict[str, Path]) -> list[Sample]:
    samples: list[Sample] = []
    with annotations_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("status", "ok")
            if status != "ok":
                continue
            study_id = row["study_id"]
            if study_id not in volume_map:
                continue
            angles = np.array(
                [
                    float(row["roll_deg"]),
                    float(row["pitch_deg"]),
                    float(row["yaw_deg"]),
                ],
                dtype=np.float32,
            )
            samples.append(
                Sample(
                    study_id=study_id,
                    volume_path=volume_map[study_id],
                    angles_deg=angles,
                )
            )
    return samples


def build_kfold_indices(n_samples: int, n_folds: int, seed: int) -> list[list[int]]:
    if n_folds < 2:
        raise ValueError("--num-folds must be >= 2")
    if n_samples < n_folds:
        raise RuntimeError(
            f"Not enough samples ({n_samples}) for {n_folds}-fold CV. "
            "Reduce --num-folds or provide more data."
        )

    indices = list(range(n_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1

    folds: list[list[int]] = []
    start = 0
    for size in fold_sizes:
        folds.append(indices[start : start + size])
        start += size
    return folds


def train_one_epoch(
    model: torch.nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = angle_regression_loss(pred, y)
        loss.backward()
        optimizer.step()

        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        total_n += bsz
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = angle_regression_loss(pred, y)
        mae = (pred - y).abs().mean()

        bsz = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bsz
        total_mae += float(mae.detach().cpu()) * bsz
        total_n += bsz
    denom = max(total_n, 1)
    return {"loss": total_loss / denom, "mae_deg": total_mae / denom}


def save_metrics_csv(metrics_rows: list[dict[str, float | int]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_mae_deg"])
        for row in metrics_rows:
            writer.writerow(
                [
                    row["epoch"],
                    row["train_loss"],
                    row["val_loss"],
                    row["val_mae_deg"],
                ]
            )


def save_cv_summary(
    fold_rows: list[dict[str, float | int]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    val_losses = np.array([float(r["best_val_loss"]) for r in fold_rows], dtype=np.float64)
    val_maes = np.array([float(r["best_val_mae_deg"]) for r in fold_rows], dtype=np.float64)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "n_train", "n_val", "best_epoch", "best_val_loss", "best_val_mae_deg"])
        for row in fold_rows:
            writer.writerow(
                [
                    row["fold"],
                    row["n_train"],
                    row["n_val"],
                    row["best_epoch"],
                    row["best_val_loss"],
                    row["best_val_mae_deg"],
                ]
            )

        writer.writerow([])
        writer.writerow(["metric", "mean", "std"])
        writer.writerow(["best_val_loss", float(val_losses.mean()), float(val_losses.std())])
        writer.writerow(["best_val_mae_deg", float(val_maes.mean()), float(val_maes.std())])


def run_single_fold(
    fold_idx: int,
    train_samples: list[Sample],
    val_samples: list[Sample],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    fold_dir = output_dir / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    input_shape = parse_shape(args.input_shape)
    train_ds = NiftiAngleDataset(
        train_samples,
        target_spacing=args.spacing,
        output_shape_dhw=input_shape,
        window_level=args.window_level,
        window_width=args.window_width,
        augment=IntensityAugment3D(),
    )
    val_ds = NiftiAngleDataset(
        val_samples,
        target_spacing=args.spacing,
        output_shape_dhw=input_shape,
        window_level=args.window_level,
        window_width=args.window_width,
        augment=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = AngleRegressor3D(
        in_channels=1,
        base_channels=args.base_channels,
        norm=args.norm,
        dropout_p=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics_rows: list[dict[str, float | int]] = []
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_ckpt = fold_dir / "best.pt"
    last_ckpt = fold_dir / "last.pt"

    print(
        f"[FOLD {fold_idx}] train/val samples: "
        f"{len(train_samples)}/{len(val_samples)}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae_deg": val_metrics["mae_deg"],
        }
        metrics_rows.append(row)
        print(
            f"[FOLD {fold_idx} | EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.5f} "
            f"val_loss={val_metrics['loss']:.5f} "
            f"val_mae_deg={val_metrics['mae_deg']:.4f}"
        )

        if val_metrics["mae_deg"] < best_val_mae:
            best_val_mae = val_metrics["mae_deg"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_mae_deg": best_val_mae,
                    "args": vars(args),
                },
                best_ckpt,
            )

    torch.save(
        {
            "fold": fold_idx,
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": metrics_rows[-1]["val_loss"],
            "val_mae_deg": metrics_rows[-1]["val_mae_deg"],
            "args": vars(args),
        },
        last_ckpt,
    )
    save_metrics_csv(metrics_rows, fold_dir / "metrics.csv")

    print(
        f"[FOLD {fold_idx}] best: epoch={best_epoch}, "
        f"val_loss={best_val_loss:.5f}, val_mae_deg={best_val_mae:.4f}"
    )

    return {
        "fold": fold_idx,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_mae_deg": float(best_val_mae),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.volumes_dir.exists() or not args.volumes_dir.is_dir():
        raise NotADirectoryError(f"Invalid --volumes-dir: {args.volumes_dir}")
    if not args.annotations_csv.exists():
        raise FileNotFoundError(f"Missing --annotations-csv: {args.annotations_csv}")

    volume_map = collect_volume_paths(args.volumes_dir)
    if not volume_map:
        raise FileNotFoundError(f"No .nii/.nii.gz files found in {args.volumes_dir}")

    samples = load_samples(args.annotations_csv, volume_map)
    if len(samples) < args.num_folds:
        raise RuntimeError(
            f"Not enough valid samples ({len(samples)}) for {args.num_folds}-fold CV. "
            "Check annotation CSV, volume IDs, or reduce --num-folds."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = build_kfold_indices(len(samples), n_folds=args.num_folds, seed=args.seed)
    print(f"[INFO] Samples total: {len(samples)}")
    print(f"[INFO] Running {args.num_folds}-fold cross-validation")

    fold_summaries: list[dict[str, float | int]] = []
    for fold_idx in range(args.num_folds):
        val_idx = set(folds[fold_idx])
        train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
        val_samples = [s for i, s in enumerate(samples) if i in val_idx]
        summary = run_single_fold(
            fold_idx=fold_idx,
            train_samples=train_samples,
            val_samples=val_samples,
            args=args,
            output_dir=output_dir,
            device=device,
        )
        fold_summaries.append(summary)

    save_cv_summary(fold_summaries, output_dir / "cv_summary.csv")
    mae_values = np.array([float(r["best_val_mae_deg"]) for r in fold_summaries], dtype=np.float64)
    print(
        "[DONE] CV finished. best_val_mae_deg mean/std: "
        f"{float(mae_values.mean()):.4f}/{float(mae_values.std()):.4f}"
    )
    print(f"[DONE] CV summary: {(output_dir / 'cv_summary.csv').resolve()}")


if __name__ == "__main__":
    main()

