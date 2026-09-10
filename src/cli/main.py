"""End-to-end pipeline: video/image -> hold detection -> pose tracking -> beta path.

Climber `--height`/`--arm-span` are given in meters. Reach constraints in
reach_graph.ClimberProfile are expressed in pixels (the same unit as hold
centroids), so this module calibrates a pixels-per-meter scale from the
detected pose's shoulder-to-ankle span before building the profile. Without a
detected pose that calibration isn't possible, so an uncalibrated fallback
scale is used and a warning is logged.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from src.kinematics.pose_tracker import (
    ClimberPose,
    PoseTracker,
    compute_reach_spans,
    draw_pose,
    estimate_center_of_mass,
)
from src.kinematics.reach_graph import (
    START_NODE_ID,
    ClimberProfile,
    ReachabilityGraph,
    draw_beta_path,
    find_beta_path,
)
from src.vision.hold_detector import HoldDetection, HoldDetector, draw_detections, filter_by_confidence
from src.vision.video_loader import download_video, extract_frames

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_WEIGHTS = "yolov8n-seg.pt"

# Shoulder-to-ankle midpoint span as a fraction of standing height; used to turn a
# detected pose's pixel dimensions into a pixels-per-meter calibration.
SHOULDER_TO_ANKLE_HEIGHT_FRACTION = 0.82
UNCALIBRATED_PIXELS_PER_METER = 100.0  # fallback used when no pose is detected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-url", help="YouTube URL of the climbing clip")
    source.add_argument("--image", type=Path, help="Path to a single still image of the wall/climber")

    parser.add_argument("--start", default="0", help='Clip start timestamp, e.g. "1:23" (with --video-url)')
    parser.add_argument("--end", default=None, help='Clip end timestamp, e.g. "1:45" (with --video-url)')

    parser.add_argument("--height", type=float, default=1.75, help="Climber height in meters (default 1.75)")
    parser.add_argument("--arm-span", type=float, default=1.78, help="Climber arm span in meters (default 1.78)")

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8-seg weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Hold detection confidence threshold")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output artifacts")
    parser.add_argument("--save-vis", action="store_true", help="Save an annotated diagnostic overlay image")

    return parser


def get_input_frame(args: argparse.Namespace) -> np.ndarray:
    """Load a single BGR frame from --image, or the first frame of --video-url's clip window."""
    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame

    video_path = download_video(args.video_url)
    frames = extract_frames(video_path, args.start, args.end, sample_fps=1.0)
    try:
        _, frame = next(frames)
    except StopIteration:
        raise RuntimeError(f"No frames extracted between {args.start} and {args.end}")
    return frame


def estimate_pixels_per_meter(pose: ClimberPose | None, height_m: float) -> tuple[float, bool]:
    """Return (pixels_per_meter, calibrated). Falls back to a flat scale if no pose is available."""
    if pose is not None:
        shoulder_mid = (
            (pose.pixel["LEFT_SHOULDER"][0] + pose.pixel["RIGHT_SHOULDER"][0]) / 2,
            (pose.pixel["LEFT_SHOULDER"][1] + pose.pixel["RIGHT_SHOULDER"][1]) / 2,
        )
        ankle_mid = (
            (pose.pixel["LEFT_ANKLE"][0] + pose.pixel["RIGHT_ANKLE"][0]) / 2,
            (pose.pixel["LEFT_ANKLE"][1] + pose.pixel["RIGHT_ANKLE"][1]) / 2,
        )
        pixel_span = math.dist(shoulder_mid, ankle_mid)
        if pixel_span > 0:
            return pixel_span / (SHOULDER_TO_ANKLE_HEIGHT_FRACTION * height_m), True

    return UNCALIBRATED_PIXELS_PER_METER, False


def average_visibility(pose: ClimberPose) -> float:
    return sum(pose.visibility.values()) / len(pose.visibility)


def select_endpoint_hold_indices(detections: list[HoldDetection]) -> tuple[int, int]:
    """Return (bottom_index, top_index): holds with the largest/smallest centroid y."""
    bottom_idx = max(range(len(detections)), key=lambda i: detections[i].centroid[1])
    top_idx = min(range(len(detections)), key=lambda i: detections[i].centroid[1])
    return bottom_idx, top_idx


def summarize_path(graph: ReachabilityGraph, path: list[int]) -> tuple[list[tuple[int, float]], float]:
    """Return [(hold_id, raw_move_distance), ...] and the total A*-optimized path cost."""
    moves = []
    total_cost = 0.0
    for a_id, b_id in zip(path, path[1:]):
        distance = math.dist(graph.nodes[a_id].centroid, graph.nodes[b_id].centroid)
        cost = dict(graph.neighbors(a_id))[b_id]
        total_cost += cost
        moves.append((b_id, distance))
    return moves, total_cost


def run_pipeline(args: argparse.Namespace) -> None:
    frame = get_input_frame(args)
    logger.info("Loaded frame: %dx%d", frame.shape[1], frame.shape[0])

    detector = HoldDetector(weights=args.weights)
    detections = filter_by_confidence(detector.detect(frame, conf=args.conf), args.conf)
    logger.info("Detected %d holds", len(detections))

    with PoseTracker(static_image_mode=True) as tracker:
        pose = tracker.process(frame)

    pixels_per_meter, calibrated = estimate_pixels_per_meter(pose, args.height)
    if not calibrated:
        logger.warning("No climber pose detected; using an uncalibrated reach scale (%.0f px/m)", pixels_per_meter)

    climber = ClimberProfile(
        height=args.height * pixels_per_meter,
        arm_span=args.arm_span * pixels_per_meter,
    )

    com = estimate_center_of_mass(pose) if pose is not None else None

    overlay = draw_detections(frame, detections)
    if pose is not None:
        overlay = draw_pose(overlay, pose, center_of_mass=com)

    path, graph, total_cost, moves = None, None, None, []
    if len(detections) > 0:
        bottom_idx, top_idx = select_endpoint_hold_indices(detections)

        if com is not None:
            graph = ReachabilityGraph.build(detections, climber, start_position=com)
            start_id = START_NODE_ID
        else:
            graph = ReachabilityGraph.build(detections, climber)
            start_id = bottom_idx

        path = find_beta_path(graph, start_id=start_id, goal_id=top_idx)
        if path is not None:
            moves, total_cost = summarize_path(graph, path)
            overlay = draw_beta_path(overlay, graph, path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "beta_overlay.jpg"
    if args.save_vis:
        cv2.imwrite(str(output_path), overlay)

    print_summary(detections, pose, path, moves, total_cost, output_path if args.save_vis else None)


def print_summary(
    detections: list[HoldDetection],
    pose: ClimberPose | None,
    path: list[int] | None,
    moves: list[tuple[int, float]],
    total_cost: float | None,
    output_path: Path | None,
) -> None:
    print("\n=== BetaVision Pipeline Summary ===")
    print(f"Holds detected: {len(detections)}")

    if pose is not None:
        print(f"Climber pose confidence (avg landmark visibility): {average_visibility(pose):.2%}")
        spans = compute_reach_spans(pose)
        print(f"Wingspan: {spans.wingspan:.1f}px  |  Reach: L {spans.left_reach_distance:.1f}px, "
              f"R {spans.right_reach_distance:.1f}px")
    else:
        print("Climber pose confidence: no climber detected in frame")

    if path is None:
        print("Beta path: none found (no holds, unreachable goal, or no route)")
    else:
        print(f"Beta sequence ({len(path)} holds): {' -> '.join(str(h) for h in path)}")
        for hold_id, distance in moves:
            print(f"  -> hold {hold_id}: {distance:.1f}px")
        print(f"Total path cost: {total_cost:.1f}")

    if output_path is not None:
        print(f"Annotated overlay saved to {output_path}")
    print("====================================\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.video_url is not None and args.end is None:
        parser.error("--end is required when using --video-url")

    run_pipeline(args)


if __name__ == "__main__":
    main()
