"""YOLOv8-seg based climbing hold detection: masks, boxes, centroids, color, and overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "yolov8n-seg.pt"

# Hue ranges (OpenCV HSV: H in [0, 179]) for the primary climbing hold colors. Black/white
# are handled separately since they're distinguished by value/saturation, not hue. Red wraps
# around hue 0, so it gets two ranges; "pink" is treated as red, "teal" as blue.
_HUE_COLOR_RANGES: list[tuple[str, int, int]] = [
    ("red", 0, 10),
    ("orange", 11, 25),
    ("yellow", 26, 35),
    ("green", 36, 85),
    ("blue", 86, 130),
    ("purple", 131, 169),
    ("red", 170, 179),
]

# BGR swatches used to render each classified color in overlays.
COLOR_NAME_TO_BGR: dict[str, tuple[int, int, int]] = {
    "red": (0, 0, 255),
    "orange": (0, 140, 255),
    "yellow": (0, 220, 220),
    "green": (0, 170, 0),
    "blue": (220, 130, 0),
    "purple": (200, 0, 160),
    "black": (60, 60, 60),
    "white": (230, 230, 230),
    "unknown": (160, 160, 160),
}


@dataclass
class HoldDetection:
    """A single detected hold: its mask, polygon, box, color, and derived geometry."""

    mask: np.ndarray  # bool, HxW, same size as the source frame
    polygon: np.ndarray  # float32, Nx2, contour points in source-frame coordinates
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    centroid: tuple[float, float]
    area: float
    color: str = "unknown"


def classify_hold_color(hsv_frame: np.ndarray, mask: np.ndarray) -> str:
    """Classify a hold's dominant color from the HSV pixels under its mask.

    Uses the median hue/saturation/value over the masked region (robust to shadow and
    highlight outliers within the patch) to bucket into one of the primary climbing hold
    colors, or "black"/"white"/"unknown".
    """
    pixels = hsv_frame[mask]
    if pixels.size == 0:
        return "unknown"

    hue, saturation, value = np.median(pixels, axis=0)

    if value < 50:
        return "black"
    if saturation < 40 and value > 180:
        return "white"

    for name, low, high in _HUE_COLOR_RANGES:
        if low <= hue <= high:
            return name

    return "unknown"


def compute_centroid_and_area(polygon: np.ndarray) -> tuple[tuple[float, float], float]:
    """Compute the (x, y) centroid and enclosed area of a polygon via image moments."""
    contour = polygon.astype(np.float32).reshape(-1, 1, 2)
    moments = cv2.moments(contour)
    area = cv2.contourArea(contour)

    if moments["m00"] == 0:
        centroid = (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))
    else:
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    return centroid, area


def filter_by_confidence(detections: list[HoldDetection], min_confidence: float) -> list[HoldDetection]:
    """Return only detections with confidence >= min_confidence."""
    return [d for d in detections if d.confidence >= min_confidence]


def filter_by_color(detections: list[HoldDetection], color: str) -> list[HoldDetection]:
    """Return only detections classified as `color`, to isolate a single boulder problem."""
    return [d for d in detections if d.color == color]


class HoldDetector:
    """Wraps a YOLOv8-seg model to detect climbing holds in frames."""

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, device: str | None = None):
        self.model = YOLO(str(weights))
        if device is not None:
            self.model.to(device)

    def detect(
        self,
        frames: np.ndarray | list[np.ndarray],
        conf: float = 0.25,
    ) -> list[HoldDetection] | list[list[HoldDetection]]:
        """Run inference on a single frame or a batch, returning HoldDetections per frame."""
        is_single = isinstance(frames, np.ndarray)
        images = [frames] if is_single else frames

        results = self.model.predict(images, conf=conf, verbose=False)
        parsed = [self._parse_result(result, image) for result, image in zip(results, images)]

        return parsed[0] if is_single else parsed

    @staticmethod
    def _parse_result(result, frame: np.ndarray) -> list[HoldDetection]:
        if result.boxes is None:
            return []

        orig_h, orig_w = result.orig_shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if result.masks is not None:
            return HoldDetector._parse_segmentation(result, result.boxes, orig_h, orig_w, hsv_frame)
        return HoldDetector._parse_boxes_only(result.boxes, orig_h, orig_w, hsv_frame)

    @staticmethod
    def _parse_segmentation(result, boxes, orig_h: int, orig_w: int, hsv_frame: np.ndarray) -> list[HoldDetection]:
        detections: list[HoldDetection] = []
        raw_masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
        polygons = result.masks.xy  # list of (N_i, 2) arrays in original-image coordinates

        for i in range(len(polygons)):
            polygon = np.asarray(polygons[i], dtype=np.float32)
            if polygon.shape[0] < 3:
                continue

            mask = cv2.resize(raw_masks[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) > 0.5
            bbox = tuple(boxes.xyxy[i].cpu().numpy().tolist())
            confidence = float(boxes.conf[i])
            class_id = int(boxes.cls[i])
            centroid, area = compute_centroid_and_area(polygon)

            detections.append(
                HoldDetection(
                    mask=mask,
                    polygon=polygon,
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    centroid=centroid,
                    area=area,
                    color=classify_hold_color(hsv_frame, mask),
                )
            )

        return detections

    @staticmethod
    def _parse_boxes_only(boxes, orig_h: int, orig_w: int, hsv_frame: np.ndarray) -> list[HoldDetection]:
        """Synthesize a rectangular polygon/mask/centroid/area from each box.

        Lets HoldDetector accept plain (non -seg) YOLOv8 detection checkpoints, which have
        no mask output, in addition to YOLOv8-seg models.
        """
        detections: list[HoldDetection] = []

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().tolist()
            polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            mask = np.zeros((orig_h, orig_w), dtype=bool)
            xi1, yi1 = max(0, round(x1)), max(0, round(y1))
            xi2, yi2 = min(orig_w, round(x2)), min(orig_h, round(y2))
            mask[yi1:yi2, xi1:xi2] = True

            detections.append(
                HoldDetection(
                    mask=mask,
                    polygon=polygon,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(boxes.conf[i]),
                    class_id=int(boxes.cls[i]),
                    centroid=centroid,
                    area=area,
                    color=classify_hold_color(hsv_frame, mask),
                )
            )

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: list[HoldDetection],
    draw_masks: bool = False,
    draw_outlines: bool = True,
    draw_centroids: bool = True,
    mask_alpha: float = 0.25,
    outline_thickness: int = 2,
) -> np.ndarray:
    """Return a copy of `frame` with clean, color-coded hold outlines (and optional mask fills).

    Holds are colored by their classified `color` (COLOR_NAME_TO_BGR) rather than class id, and
    labeled with subtle contour outlines instead of boxes with confidence text.
    """
    overlay = frame.copy()
    mask_layer = frame.copy()

    for det in detections:
        color = COLOR_NAME_TO_BGR.get(det.color, COLOR_NAME_TO_BGR["unknown"])

        if draw_masks:
            mask_layer[det.mask] = color

        if draw_outlines:
            contour = det.polygon.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [contour], isClosed=True, color=color, thickness=outline_thickness, lineType=cv2.LINE_AA)

        if draw_centroids:
            cx, cy = round(det.centroid[0]), round(det.centroid[1])
            cv2.circle(overlay, (cx, cy), 3, color, -1, cv2.LINE_AA)

    if draw_masks:
        overlay = cv2.addWeighted(mask_layer, mask_alpha, overlay, 1 - mask_alpha, 0)

    return overlay


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image to run hold detection on")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8-seg weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", type=Path, default=Path("hold_detections.jpg"), help="Overlay output path")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    detector = HoldDetector(weights=args.weights)
    detections = detector.detect(frame, conf=args.conf)
    overlay = draw_detections(frame, detections)
    cv2.imwrite(str(args.output), overlay)
    print(f"Detected {len(detections)} holds. Overlay saved to {args.output}")


if __name__ == "__main__":
    main()
