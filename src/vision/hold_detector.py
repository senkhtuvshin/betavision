"""YOLOv8-seg based climbing hold detection: masks, boxes, centroids, and overlays."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_WEIGHTS = "yolov8n-seg.pt"


@dataclass
class HoldDetection:
    """A single detected hold: its mask, polygon, box, and derived geometry."""

    mask: np.ndarray  # bool, HxW, same size as the source frame
    polygon: np.ndarray  # float32, Nx2, contour points in source-frame coordinates
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    centroid: tuple[float, float]
    area: float


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
        parsed = [self._parse_result(result) for result in results]

        return parsed[0] if is_single else parsed

    @staticmethod
    def _parse_result(result) -> list[HoldDetection]:
        detections: list[HoldDetection] = []
        if result.masks is None or result.boxes is None:
            return detections

        orig_h, orig_w = result.orig_shape
        raw_masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
        polygons = result.masks.xy  # list of (N_i, 2) arrays in original-image coordinates
        boxes = result.boxes

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
                )
            )

        return detections


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per class_id so the same class renders consistently."""
    digest = hashlib.md5(str(class_id).encode()).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def draw_detections(
    frame: np.ndarray,
    detections: list[HoldDetection],
    draw_masks: bool = True,
    draw_boxes: bool = True,
    draw_centroids: bool = True,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """Return a copy of `frame` with hold masks, boxes, and centroids overlaid."""
    overlay = frame.copy()
    mask_layer = frame.copy()

    for det in detections:
        color = _color_for_class(det.class_id)

        if draw_masks:
            mask_layer[det.mask] = color

        if draw_boxes:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                f"{det.confidence:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if draw_centroids:
            cx, cy = int(det.centroid[0]), int(det.centroid[1])
            cv2.circle(overlay, (cx, cy), 4, color, -1)

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
