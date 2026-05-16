#!/usr/bin/env python3
"""
Generate labeled digit training crops from a Sudoku image.

Usage:
  python3 generate_train_data.py sudoku_photo.png
  python3 generate_train_data.py sudoku_photo.png --out-dir generated_data

Workflow:
  1) Detect and crop the Sudoku board (same core strategy as main.cpp)
  2) Split into 9x9 cells
  3) Show each candidate cell
  4) Prompt label: 1-9 to save, Enter/0 to skip, q to quit
"""
from __future__ import annotations

import argparse
import math
import os
from datetime import datetime

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None
import numpy as np

DIGIT_FEATURE_SIDE = 32


def remove_noise(src: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned


def convert_to_binary(src: np.ndarray) -> np.ndarray:
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )


def order_corners(corners: list[np.ndarray]) -> np.ndarray:
    pts = np.array(corners, dtype=np.float32)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 1] - pts[:, 0]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]   # top-left
    ordered[2] = pts[np.argmax(sums)]   # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def crop_sudoku_board(src: np.ndarray) -> np.ndarray:
    binary = convert_to_binary(src)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_corners: list[np.ndarray] = []
    best_area = 0.0
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_corners = [p[0] for p in approx]

    if len(best_corners) != 4:
        raise RuntimeError("Could not find a rectangular sudoku board.")

    ordered = order_corners(best_corners)
    side = int(math.sqrt(best_area))
    dst = np.array(
        [[0, 0], [side, 0], [side, side], [0, side]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(src, matrix, (side, side))


def isolate_center_cluster(digit_binary: np.ndarray) -> np.ndarray:
    n_labels, labels = cv2.connectedComponents(digit_binary)
    if n_labels <= 1:
        return digit_binary.copy()

    center = np.array([digit_binary.shape[1] / 2.0, digit_binary.shape[0] / 2.0], dtype=np.float32)
    min_distance = float("inf")
    center_label = -1

    for label in range(1, n_labels):
        ys, xs = np.where(labels == label)
        if xs.size == 0:
            continue
        centroid = np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)
        distance = float(np.linalg.norm(centroid - center))
        if distance < min_distance:
            min_distance = distance
            center_label = label

    if center_label < 0:
        return digit_binary.copy()

    result = np.zeros_like(digit_binary)
    result[labels == center_label] = digit_binary[labels == center_label]
    return result


def normalize_digit(digit_binary: np.ndarray, side: int = DIGIT_FEATURE_SIDE) -> np.ndarray:
    contours, _ = cv2.findContours(digit_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros((side, side), dtype=np.uint8)

    areas = [cv2.contourArea(c) for c in contours]
    best_idx = int(np.argmax(areas))
    best_area = areas[best_idx]
    if best_area < 20.0:
        return np.zeros((side, side), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(contours[best_idx])
    x = max(0, x - 2)
    y = max(0, y - 2)
    w = min(digit_binary.shape[1] - x, w + 4)
    h = min(digit_binary.shape[0] - y, h + 4)
    roi = digit_binary[y:y + h, x:x + w]

    max_dim = max(w, h)
    square = np.zeros((max_dim, max_dim), dtype=np.uint8)
    ox = (max_dim - w) // 2
    oy = (max_dim - h) // 2
    square[oy:oy + h, ox:ox + w] = roi
    return cv2.resize(square, (side, side), interpolation=cv2.INTER_AREA)


def is_mostly_empty(cell: np.ndarray) -> bool:
    border_x = max(1, cell.shape[1] // 5)
    border_y = max(1, cell.shape[0] // 5)
    inner = cell[border_y:cell.shape[0] - border_y, border_x:cell.shape[1] - border_x]
    if inner.size == 0:
        return True
    return cv2.countNonZero(inner) < int(inner.size * 0.01)


def save_sample(out_dir: str, label: int, sample: np.ndarray) -> str:
    digit_dir = os.path.join(out_dir, str(label))
    os.makedirs(digit_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(digit_dir, f"{label}_{stamp}.png")
    cv2.imwrite(path, sample)
    return path


def process_image(image_path: str, out_dir: str) -> None:
    if cv2 is None:
        raise RuntimeError(
            "Missing dependency: cv2 (OpenCV for Python). "
            "Install it with: pip install opencv-python"
        )

    src = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if src is None:
        raise RuntimeError(f"Could not read input image: {image_path}")

    cleaned = remove_noise(src)
    cropped = crop_sudoku_board(cleaned)
    binary_board = convert_to_binary(cropped)

    cell_w = binary_board.shape[1] // 9
    cell_h = binary_board.shape[0] // 9

    cv2.namedWindow("Cell", cv2.WINDOW_NORMAL)
    saved = 0
    skipped = 0

    print("Labeling controls: [1-9]=save, [Enter or 0]=skip, [q]=quit")
    for row in range(9):
        for col in range(9):
            y0, y1 = row * cell_h, (row + 1) * cell_h
            x0, x1 = col * cell_w, (col + 1) * cell_w
            cell = binary_board[y0:y1, x0:x1].copy()
            likely_empty = is_mostly_empty(cell)

            centered = isolate_center_cluster(cell)
            normalized = normalize_digit(centered, DIGIT_FEATURE_SIDE)

            preview = cv2.resize(normalized, (256, 256), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                preview,
                f"r{row+1} c{col+1}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255,),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Cell", preview)
            cv2.waitKey(1)

            while True:
                hint = " [likely empty]" if likely_empty else ""
                user = input(
                    f"Cell r{row+1} c{col+1}{hint} label (1-9 save, Enter/0 skip, q quit): "
                ).strip().lower()
                if user == "q":
                    print("Stopping labeling early by request.")
                    cv2.destroyAllWindows()
                    print(f"Saved {saved} samples, skipped {skipped}.")
                    return
                if user in {"", "0"}:
                    skipped += 1
                    break
                if user in {"1", "2", "3", "4", "5", "6", "7", "8", "9"}:
                    label = int(user)
                    path = save_sample(out_dir, label, normalized)
                    saved += 1
                    print(f"Saved -> {path}")
                    break
                print("Invalid input. Use 1-9, Enter/0, or q.")

    cv2.destroyAllWindows()
    print(f"Done. Saved {saved} samples, skipped {skipped}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labeled training digits from Sudoku images.")
    parser.add_argument("image_path", help="Path to input Sudoku image, e.g. sudoku_photo.png")
    parser.add_argument(
        "--out-dir",
        default="generated_data",
        help="Output directory for labeled samples (default: generated_data)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        process_image(args.image_path, args.out_dir)
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
