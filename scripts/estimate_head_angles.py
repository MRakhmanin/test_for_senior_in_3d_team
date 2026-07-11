from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_LANDMARKS = [
    "auditory_canal_left_robust_far",
    "auditory_canal_right_robust_far",
    "eye_left_centroid",
    "eye_right_centroid",
    "eye_lens_left_centroid",
    "eye_lens_right_centroid",
    "nose_tip_proxy_mid",
]
YAW_SHIFT_DEG = -180.0


@dataclass
class AngleRow:
    study_id: str
    option: str
    status: str
    n_points: int
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    r11: float
    r12: float
    r13: float
    r21: float
    r22: float
    r23: float
    r31: float
    r32: float
    r33: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate head rotation (roll/pitch/yaw + matrix) with three canonical options: "
            "GPA template, axis method, and reference study."
        )
    )
    parser.add_argument(
        "--landmarks-csv",
        type=Path,
        required=True,
        help="Path to landmarks_per_study.csv from extract_landmarks.py",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Where to save estimated angles and rotation matrices.",
    )
    parser.add_argument(
        "--reference-study",
        type=str,
        default="CQ500CT6",
        help="Study ID to use as canonical reference for reference-based option.",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=DEFAULT_LANDMARKS,
        help="Landmark names to use for GPA and reference options.",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=3,
        help="Minimum corresponding landmarks required for Kabsch (default: 3).",
    )
    parser.add_argument(
        "--gpa-iters",
        type=int,
        default=30,
        help="Maximum GPA iterations (default: 30).",
    )
    parser.add_argument(
        "--gpa-tol",
        type=float,
        default=1e-6,
        help="GPA convergence tolerance (default: 1e-6).",
    )
    return parser.parse_args()


def load_landmarks(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    by_study: dict[str, dict[str, np.ndarray]] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status", "ok") != "ok":
                continue
            sid = row["study_id"]
            name = row["landmark_name"]
            point = np.array(
                [float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])],
                dtype=np.float64,
            )
            by_study.setdefault(sid, {})[name] = point
    return by_study


def center_points(points: np.ndarray) -> np.ndarray:
    return points - points.mean(axis=0, keepdims=True)


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Row-vector convention:
    Find R minimizing || P @ R - Q ||_F with det(R)=+1.
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def euler_zyx_degrees(R: np.ndarray) -> tuple[float, float, float]:
    """
    Return roll(X), pitch(Y), yaw(Z) from rotation matrix using ZYX convention.
    """
    if abs(R[2, 0]) < 1.0 - 1e-9:
        pitch = -math.asin(R[2, 0])
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.pi / 2 if R[2, 0] <= -1 else -math.pi / 2
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def rotation_z_deg(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def apply_canonical_yaw_shift(R_input_to_canonical: np.ndarray, yaw_shift_deg: float) -> np.ndarray:
    """
    Apply fixed yaw rotation to canonical space.
    Row-vector convention: p_canonical = p_input @ R
    New canonical after yaw shift F: p_new = p_input @ (R @ F)
    """
    F = rotation_z_deg(yaw_shift_deg)
    R_new = R_input_to_canonical @ F
    return R_new


def normalize(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def common_points(
    study_pts: dict[str, np.ndarray],
    template_pts: dict[str, np.ndarray],
    preferred_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    names = [n for n in preferred_names if n in study_pts and n in template_pts]
    if not names:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    P = np.stack([study_pts[n] for n in names], axis=0)
    Q = np.stack([template_pts[n] for n in names], axis=0)
    return P, Q


def build_gpa_template(
    studies: dict[str, dict[str, np.ndarray]],
    landmark_names: list[str],
    min_points: int,
    max_iters: int,
    tol: float,
) -> dict[str, np.ndarray]:
    valid_ids = [
        sid
        for sid, lm in studies.items()
        if sum(1 for n in landmark_names if n in lm) >= min_points
    ]
    if not valid_ids:
        raise RuntimeError("No studies with enough landmarks for GPA.")

    common_names = [
        n for n in landmark_names
        if all(n in studies[sid] for sid in valid_ids)
    ]
    if len(common_names) < min_points:
        raise RuntimeError(
            f"Not enough landmarks common across studies for GPA. "
            f"Found {len(common_names)}, need >= {min_points}."
        )

    init_sid = valid_ids[0]
    T = np.stack([studies[init_sid][n] for n in common_names], axis=0)
    T = center_points(T)

    for _ in range(max_iters):
        aligned_shapes: list[np.ndarray] = []
        for sid in valid_ids:
            P = np.stack([studies[sid][n] for n in common_names], axis=0)
            P0 = center_points(P)
            R = kabsch_rotation(P0, T)
            aligned_shapes.append(P0 @ R)

        mean_shape = np.mean(np.stack(aligned_shapes, axis=0), axis=0)
        mean_shape = center_points(mean_shape)
        delta = float(np.linalg.norm(mean_shape - T))
        T = mean_shape
        if delta < tol:
            break

    return {name: T[i] for i, name in enumerate(common_names)}


def axis_method_rotation(study_pts: dict[str, np.ndarray]) -> np.ndarray | None:
    required = [
        "eye_left_centroid",
        "eye_right_centroid",
        "auditory_canal_left_robust_far",
        "auditory_canal_right_robust_far",
        "nose_tip_proxy_mid",
    ]
    if any(name not in study_pts for name in required):
        return None

    eye_l = study_pts["eye_left_centroid"]
    eye_r = study_pts["eye_right_centroid"]
    ear_l = study_pts["auditory_canal_left_robust_far"]
    ear_r = study_pts["auditory_canal_right_robust_far"]
    nose = study_pts["nose_tip_proxy_mid"]

    x_eye = eye_r - eye_l
    x_ear = ear_r - ear_l
    x = normalize(x_eye + x_ear)  # requested: mean of eyes/ears left-right lines
    if x is None:
        return None

    ear_mid = 0.5 * (ear_l + ear_r)
    y_tmp = normalize(nose - ear_mid)  # anterior direction
    if y_tmp is None:
        return None

    z = normalize(np.cross(x, y_tmp))
    if z is None:
        return None
    y = normalize(np.cross(z, x))
    if y is None:
        return None

    B = np.column_stack([x, y, z])  # axes of current head in world space
    R = B.T  # rotate current basis to canonical identity basis
    if np.linalg.det(R) < 0:
        return None
    return R


def make_row(study_id: str, option: str, status: str, n_points: int, R: np.ndarray | None) -> AngleRow:
    if R is None:
        nan = float("nan")
        return AngleRow(study_id, option, status, n_points, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan)

    R = apply_canonical_yaw_shift(R, yaw_shift_deg=YAW_SHIFT_DEG)
    roll, pitch, yaw = euler_zyx_degrees(R)
    return AngleRow(
        study_id=study_id,
        option=option,
        status=status,
        n_points=n_points,
        roll_deg=float(roll),
        pitch_deg=float(pitch),
        yaw_deg=float(yaw),
        r11=float(R[0, 0]),
        r12=float(R[0, 1]),
        r13=float(R[0, 2]),
        r21=float(R[1, 0]),
        r22=float(R[1, 1]),
        r23=float(R[1, 2]),
        r31=float(R[2, 0]),
        r32=float(R[2, 1]),
        r33=float(R[2, 2]),
    )


def estimate_rows(args: argparse.Namespace) -> list[AngleRow]:
    studies = load_landmarks(args.landmarks_csv)
    if not studies:
        raise RuntimeError("No valid landmarks loaded from CSV.")

    landmark_names = list(dict.fromkeys(args.landmarks))
    gpa_template = build_gpa_template(
        studies=studies,
        landmark_names=landmark_names,
        min_points=args.min_points,
        max_iters=args.gpa_iters,
        tol=args.gpa_tol,
    )

    if args.reference_study not in studies:
        raise RuntimeError(f"Reference study not found: {args.reference_study}")
    ref_template = {
        name: point
        for name, point in studies[args.reference_study].items()
        if name in landmark_names
    }

    rows: list[AngleRow] = []
    for sid, pts in sorted(studies.items()):
        # Option 1: GPA
        P, Q = common_points(pts, gpa_template, landmark_names)
        if P.shape[0] < args.min_points:
            rows.append(make_row(sid, "gpa", "not_enough_points", int(P.shape[0]), None))
        else:
            R = kabsch_rotation(center_points(P), center_points(Q))
            rows.append(make_row(sid, "gpa", "ok", int(P.shape[0]), R))

        # Option 2: Axis method
        R_axis = axis_method_rotation(pts)
        if R_axis is None:
            rows.append(make_row(sid, "axis", "missing_required_landmarks", 0, None))
        else:
            rows.append(make_row(sid, "axis", "ok", 5, R_axis))

        # Option 3: Reference
        P_ref, Q_ref = common_points(pts, ref_template, landmark_names)
        if P_ref.shape[0] < args.min_points:
            rows.append(make_row(sid, "reference", "not_enough_points", int(P_ref.shape[0]), None))
        else:
            R_ref = kabsch_rotation(center_points(P_ref), center_points(Q_ref))
            rows.append(make_row(sid, "reference", "ok", int(P_ref.shape[0]), R_ref))

    return rows


def save_rows(rows: list[AngleRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "option",
                "status",
                "n_points",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "r11",
                "r12",
                "r13",
                "r21",
                "r22",
                "r23",
                "r31",
                "r32",
                "r33",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.study_id,
                    r.option,
                    r.status,
                    r.n_points,
                    r.roll_deg,
                    r.pitch_deg,
                    r.yaw_deg,
                    r.r11,
                    r.r12,
                    r.r13,
                    r.r21,
                    r.r22,
                    r.r23,
                    r.r31,
                    r.r32,
                    r.r33,
                ]
            )


def save_option_rows(rows: list[AngleRow], out_csv_base: Path, option: str) -> Path:
    """
    Save a single-option annotation CSV for direct train-script usage.
    File naming:
      <base_stem>_<option>.csv
    """
    option_rows = [r for r in rows if r.option == option]
    out_path = out_csv_base.with_name(f"{out_csv_base.stem}_{option}{out_csv_base.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "study_id",
                "status",
                "n_points",
                "roll_deg",
                "pitch_deg",
                "yaw_deg",
                "r11",
                "r12",
                "r13",
                "r21",
                "r22",
                "r23",
                "r31",
                "r32",
                "r33",
            ]
        )
        for r in option_rows:
            writer.writerow(
                [
                    r.study_id,
                    r.status,
                    r.n_points,
                    r.roll_deg,
                    r.pitch_deg,
                    r.yaw_deg,
                    r.r11,
                    r.r12,
                    r.r13,
                    r.r21,
                    r.r22,
                    r.r23,
                    r.r31,
                    r.r32,
                    r.r33,
                ]
            )
    return out_path


def main() -> None:
    args = parse_args()
    rows = estimate_rows(args)
    save_rows(rows, args.output_csv)
    out_gpa = save_option_rows(rows, args.output_csv, "gpa")
    out_axis = save_option_rows(rows, args.output_csv, "axis")
    out_ref = save_option_rows(rows, args.output_csv, "reference")
    n_ok = sum(1 for r in rows if r.status == "ok")
    print(f"[DONE] Saved rows: {len(rows)} -> {args.output_csv.resolve()}")
    print(f"[DONE] Saved option annotations: {out_gpa.resolve()}")
    print(f"[DONE] Saved option annotations: {out_axis.resolve()}")
    print(f"[DONE] Saved option annotations: {out_ref.resolve()}")
    print(f"[DONE] OK rows: {n_ok}, non-OK rows: {len(rows) - n_ok}")


if __name__ == "__main__":
    main()
