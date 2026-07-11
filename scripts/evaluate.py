from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


POSSIBLE_ANGLE_KEYS = [
    ("roll_deg", "pitch_deg", "yaw_deg"),
    ("axial_deg", "coronal_deg", "sagittal_deg"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate angle prediction CSV against annotation CSV by study_id. "
            "Computes MAE and RMSE per angle and overall."
        )
    )
    parser.add_argument(
        "--pred-csv",
        type=Path,
        required=True,
        help="Prediction CSV path.",
    )
    parser.add_argument(
        "--target-csv",
        type=Path,
        required=True,
        help="Ground-truth annotation CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation artifacts.",
    )
    parser.add_argument(
        "--study-id-col",
        type=str,
        default="study_id",
        help="Study ID column name in both CSV files (default: study_id).",
    )
    parser.add_argument(
        "--angle-cols",
        nargs=3,
        default=None,
        metavar=("A1", "A2", "A3"),
        help=(
            "Optional explicit angle column names. If omitted, script tries "
            "['roll_deg','pitch_deg','yaw_deg'] then "
            "['axial_deg','coronal_deg','sagittal_deg']."
        ),
    )
    return parser.parse_args()


def infer_angle_columns(fieldnames: list[str], angle_cols: list[str] | None) -> tuple[str, str, str]:
    if angle_cols is not None:
        missing = [c for c in angle_cols if c not in fieldnames]
        if missing:
            raise ValueError(f"Missing explicit angle columns in CSV: {missing}")
        return angle_cols[0], angle_cols[1], angle_cols[2]

    for cols in POSSIBLE_ANGLE_KEYS:
        if all(c in fieldnames for c in cols):
            return cols
    raise ValueError(
        "Could not infer angle columns. Provide --angle-cols explicitly."
    )


def load_csv_map(
    path: Path,
    study_id_col: str,
    angle_cols: list[str] | None,
) -> tuple[dict[str, np.ndarray], tuple[str, str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if study_id_col not in reader.fieldnames:
            raise ValueError(f"Missing study id column '{study_id_col}' in {path}")

        a1, a2, a3 = infer_angle_columns(reader.fieldnames, angle_cols)
        out: dict[str, np.ndarray] = {}
        for row in reader:
            sid = row.get(study_id_col, "")
            if sid == "":
                continue
            try:
                angles = np.array(
                    [float(row[a1]), float(row[a2]), float(row[a3])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError):
                continue
            if np.any(np.isnan(angles)):
                continue
            out[sid] = angles
    return out, (a1, a2, a3)


def mae(x: np.ndarray) -> float:
    return float(np.mean(np.abs(x)))


def rmse(x: np.ndarray) -> float:
    return float(math.sqrt(float(np.mean(np.square(x)))))


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_map, pred_cols = load_csv_map(
        args.pred_csv,
        study_id_col=args.study_id_col,
        angle_cols=args.angle_cols,
    )
    target_map, target_cols = load_csv_map(
        args.target_csv,
        study_id_col=args.study_id_col,
        angle_cols=args.angle_cols,
    )

    common_ids = sorted(set(pred_map.keys()) & set(target_map.keys()))
    if not common_ids:
        raise RuntimeError("No overlapping study IDs between prediction and target CSV.")

    errors = []
    for sid in common_ids:
        err = pred_map[sid] - target_map[sid]
        errors.append(err)
    errors_np = np.stack(errors, axis=0)  # [N, 3]

    per_axis_mae = np.mean(np.abs(errors_np), axis=0)
    per_axis_rmse = np.sqrt(np.mean(np.square(errors_np), axis=0))
    overall_mae = mae(errors_np.reshape(-1))
    overall_rmse = rmse(errors_np.reshape(-1))
    vector_l2_rmse = float(np.sqrt(np.mean(np.sum(np.square(errors_np), axis=1))))

    # Save per-study errors.
    per_study_csv = output_dir / "per_study_errors.csv"
    with per_study_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                f"pred_{pred_cols[0]}",
                f"pred_{pred_cols[1]}",
                f"pred_{pred_cols[2]}",
                f"target_{target_cols[0]}",
                f"target_{target_cols[1]}",
                f"target_{target_cols[2]}",
                f"err_{pred_cols[0]}",
                f"err_{pred_cols[1]}",
                f"err_{pred_cols[2]}",
                "abs_err_mean",
                "l2_err",
            ]
        )
        for sid in common_ids:
            pred = pred_map[sid]
            target = target_map[sid]
            err = pred - target
            writer.writerow(
                [
                    sid,
                    pred[0],
                    pred[1],
                    pred[2],
                    target[0],
                    target[1],
                    target[2],
                    err[0],
                    err[1],
                    err[2],
                    float(np.mean(np.abs(err))),
                    float(np.sqrt(np.sum(np.square(err)))),
                ]
            )

    # Save summary.
    summary_csv = output_dir / "metrics_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["n_common_studies", len(common_ids)])
        writer.writerow([f"mae_{pred_cols[0]}", float(per_axis_mae[0])])
        writer.writerow([f"mae_{pred_cols[1]}", float(per_axis_mae[1])])
        writer.writerow([f"mae_{pred_cols[2]}", float(per_axis_mae[2])])
        writer.writerow([f"rmse_{pred_cols[0]}", float(per_axis_rmse[0])])
        writer.writerow([f"rmse_{pred_cols[1]}", float(per_axis_rmse[1])])
        writer.writerow([f"rmse_{pred_cols[2]}", float(per_axis_rmse[2])])
        writer.writerow(["overall_mae", overall_mae])
        writer.writerow(["overall_rmse", overall_rmse])
        writer.writerow(["vector_l2_rmse", vector_l2_rmse])

    print(f"[DONE] Common studies: {len(common_ids)}")
    print(
        "[DONE] MAE (3 angles): "
        f"{per_axis_mae[0]:.4f}, {per_axis_mae[1]:.4f}, {per_axis_mae[2]:.4f}"
    )
    print(
        "[DONE] RMSE (3 angles): "
        f"{per_axis_rmse[0]:.4f}, {per_axis_rmse[1]:.4f}, {per_axis_rmse[2]:.4f}"
    )
    print(f"[DONE] Overall MAE/RMSE: {overall_mae:.4f}/{overall_rmse:.4f}")
    print(f"[DONE] Saved per-study errors: {per_study_csv.resolve()}")
    print(f"[DONE] Saved summary metrics: {summary_csv.resolve()}")


if __name__ == "__main__":
    main()

