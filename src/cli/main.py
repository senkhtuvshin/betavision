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
import shutil
import subprocess
from collections import Counter
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
    find_contact_holds,
)
from src.vision.hold_detector import (
    HoldDetection,
    HoldDetector,
    draw_detections,
    filter_by_color,
    filter_by_confidence,
)
from src.vision.video_loader import download_video, extract_frames, parse_timestamp

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_WEIGHTS = "yolov8n-seg.pt"

# Shoulder-to-ankle midpoint span as a fraction of standing height; used to turn a
# detected pose's pixel dimensions into a pixels-per-meter calibration.
SHOULDER_TO_ANKLE_HEIGHT_FRACTION = 0.82
UNCALIBRATED_PIXELS_PER_METER = 100.0  # fallback used when no pose is detected

# Fraction of the climber's max reach used as the "currently gripping this hold" threshold.
# Box-only detections center on the hold's bounding box, not necessarily the exact grip point,
# and pose landmarks have their own pixel noise, so this needs to be more forgiving than it
# might seem: on a real test photo the actual gripping wrist sat ~30px from its hold's centroid
# against a ~112px max reach radius (a 0.15 fraction, i.e. ~17px, missed every real contact).
CONTACT_RADIUS_FRACTION = 0.35

# Codecs tried in order for annotated video export; avc1 (H.264) is what QuickTime plays
# natively, but not every OpenCV build has a working H.264 encoder, so mp4v is the fallback.
VIDEO_CODEC_CANDIDATES = ("avc1", "mp4v")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-url", help="YouTube URL of the climbing clip")
    source.add_argument("--video-path", type=Path, help="Path to a local climbing video file")
    source.add_argument("--image", type=Path, help="Path to a single still image of the wall/climber")

    parser.add_argument("--start", default="0", help='Clip start timestamp, e.g. "1:23" (with --video-url/--video-path)')
    parser.add_argument("--end", default=None, help='Clip end timestamp, e.g. "1:45" (with --video-url/--video-path)')
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Frames per second to sample from the source video (default: every decoded frame)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Process the full clip frame-by-frame and export an annotated video "
        "(requires --video-url or --video-path)",
    )

    parser.add_argument("--height", type=float, default=1.75, help="Climber height in meters (default 1.75)")
    parser.add_argument("--arm-span", type=float, default=1.78, help="Climber arm span in meters (default 1.78)")

    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8-seg weights path")
    parser.add_argument("--conf", type=float, default=0.25, help="Hold detection confidence threshold")

    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output artifacts")
    parser.add_argument("--save-vis", action="store_true", help="Save an annotated diagnostic overlay image")

    parser.add_argument("--route-color", default=None, help="Filter candidate holds to this color (e.g. 'blue')")
    parser.add_argument(
        "--auto-route",
        action="store_true",
        help="Auto-detect the route color from holds nearest the climber's hands/feet",
    )

    return parser


def resolve_video_path(args: argparse.Namespace) -> Path:
    """Return a local video file path, downloading it first if --video-url was given."""
    if args.video_path is not None:
        if not args.video_path.exists():
            raise FileNotFoundError(f"Could not find video: {args.video_path}")
        return args.video_path
    return download_video(args.video_url)


def get_input_frame(args: argparse.Namespace) -> np.ndarray:
    """Load a single BGR frame from --image, or the first frame of the video clip window."""
    if args.image is not None:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")
        return frame

    video_path = resolve_video_path(args)
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


def determine_dominant_route_color(
    holds: list[HoldDetection],
    contacts: dict[str, int | None],
    pose: ClimberPose | None = None,
) -> str | None:
    """Infer a route's color, preferring holds the climber is actually contacting.

    Falls back to whichever hold is nearest each limb regardless of distance only if no limb
    registered a real contact - a much noisier signal, since "nearest" can be a hold nowhere
    near actually gripped (e.g. tens of pixels away, on an ungrounded limb).
    """
    contacted_colors = [holds[idx].color for idx in contacts.values() if idx is not None]
    if contacted_colors:
        return Counter(contacted_colors).most_common(1)[0][0]

    if pose is None or not holds:
        return None

    limb_names = ("LEFT_WRIST", "RIGHT_WRIST", "LEFT_ANKLE", "RIGHT_ANKLE")
    nearest_colors = []
    for limb in limb_names:
        limb_position = pose.pixel[limb]
        nearest_hold = min(holds, key=lambda h: math.dist(limb_position, h.centroid))
        nearest_colors.append(nearest_hold.color)

    color, _count = Counter(nearest_colors).most_common(1)[0]
    return color


def select_endpoint_hold_indices(detections: list[HoldDetection]) -> tuple[int, int]:
    """Return (bottom_index, top_index): holds with the largest/smallest centroid y."""
    bottom_idx = max(range(len(detections)), key=lambda i: detections[i].centroid[1])
    top_idx = min(range(len(detections)), key=lambda i: detections[i].centroid[1])
    return bottom_idx, top_idx


def compute_beta_path(
    holds: list[HoldDetection],
    climber: ClimberProfile,
    pose: ClimberPose | None,
    com: tuple[float, float] | None,
    contact_radius: float,
) -> tuple[ReachabilityGraph | None, list[int] | None, dict[str, int | None]]:
    """Build the reachability graph and A* path for the current holds/climber state.

    Prefers starting from holds the climber is actively gripping (limb contact); falls back
    to the climber's CoM as a virtual start, then to the bottom-most hold if no pose exists.
    """
    contacts = find_contact_holds(pose, holds, contact_radius) if pose is not None else {}
    if not holds:
        return None, None, contacts

    contact_hold_ids = sorted({idx for idx in contacts.values() if idx is not None})
    bottom_idx, top_idx = select_endpoint_hold_indices(holds)

    if contact_hold_ids:
        graph = ReachabilityGraph.build(holds, climber)
        start_id = contact_hold_ids
    elif com is not None:
        graph = ReachabilityGraph.build(holds, climber, start_position=com)
        start_id = START_NODE_ID
    else:
        graph = ReachabilityGraph.build(holds, climber)
        start_id = bottom_idx

    path = find_beta_path(graph, start_id=start_id, goal_id=top_idx)
    return graph, path, contacts


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

    contact_radius = climber.max_reach_radius * CONTACT_RADIUS_FRACTION

    route_color = args.route_color
    if args.auto_route:
        pre_filter_contacts = find_contact_holds(pose, detections, contact_radius) if pose is not None else {}
        auto_color = determine_dominant_route_color(detections, pre_filter_contacts, pose)
        if auto_color is not None:
            route_color = auto_color
            logger.info("Auto-detected route color: %s", route_color)
        else:
            logger.warning("--auto-route requested but no pose/holds were available to infer a color")

    if route_color is not None:
        route_filtered = filter_by_color(detections, route_color)
        if route_filtered:
            detections = route_filtered
            logger.info("Filtered to %d '%s' holds", len(detections), route_color)
        else:
            logger.warning("No holds matched route color '%s'; keeping all %d holds", route_color, len(detections))

    com = estimate_center_of_mass(pose) if pose is not None else None

    overlay = draw_detections(frame, detections)
    if pose is not None:
        overlay = draw_pose(overlay, pose, center_of_mass=com)

    # Recompute against `detections` as it stands now (post route-color filtering, if any),
    # since hold indices shift once the list is filtered.
    graph, path, contacts = compute_beta_path(detections, climber, pose, com, contact_radius)
    moves, total_cost = ([], None)
    if path is not None:
        moves, total_cost = summarize_path(graph, path)
        overlay = draw_beta_path(overlay, graph, path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "beta_overlay.jpg"
    if args.save_vis:
        cv2.imwrite(str(output_path), overlay)

    print_summary(
        detections, pose, path, moves, total_cost, output_path if args.save_vis else None, contacts, route_color
    )


def print_summary(
    detections: list[HoldDetection],
    pose: ClimberPose | None,
    path: list[int] | None,
    moves: list[tuple[int, float]],
    total_cost: float | None,
    output_path: Path | None,
    contacts: dict[str, int | None] | None = None,
    route_color: str | None = None,
) -> None:
    print("\n=== BetaVision Pipeline Summary ===")
    if route_color is not None:
        print(f"Route color filter: {route_color}")
    print(f"Holds detected: {len(detections)}")

    if pose is not None:
        print(f"Climber pose confidence (avg landmark visibility): {average_visibility(pose):.2%}")
        spans = compute_reach_spans(pose)
        print(f"Wingspan: {spans.wingspan:.1f}px  |  Reach: L {spans.left_reach_distance:.1f}px, "
              f"R {spans.right_reach_distance:.1f}px")
    else:
        print("Climber pose confidence: no climber detected in frame")

    if contacts:
        contact_str = ", ".join(
            f"{limb}: hold {idx}" if idx is not None else f"{limb}: none" for limb, idx in contacts.items()
        )
        print(f"Limb contacts: {contact_str}")

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


def probe_video_metadata(video_path: Path) -> tuple[float, float]:
    """Return (source_fps, total_frame_count) read from the video file's container metadata."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return fps, frame_count


def open_video_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    """Open a cv2.VideoWriter at `path`, trying each codec in VIDEO_CODEC_CANDIDATES in turn."""
    for codec in VIDEO_CODEC_CANDIDATES:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(path), fourcc, fps, frame_size)
        if writer.isOpened():
            logger.info("Opened VideoWriter with codec '%s'", codec)
            return writer
        writer.release()

    raise RuntimeError(f"Could not open a VideoWriter for {path} with any of {VIDEO_CODEC_CANDIDATES}")


def remux_for_quicktime(raw_path: Path, final_path: Path) -> bool:
    """Remux `raw_path` to H.264/yuv420p at `final_path` for QuickTime compatibility.

    OpenCV's VideoWriter output isn't always playable in QuickTime even with the 'avc1'
    fourcc (pixel format / container details vary by platform and OpenCV build), so this
    shells out to ffmpeg for a known-compatible remux when it's available. Returns True if
    the remux ran (raw_path is removed); False if ffmpeg isn't installed, in which case
    raw_path is simply renamed to final_path and kept as-is.
    """
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        logger.warning("ffmpeg not found; keeping direct OpenCV output (may not play in QuickTime)")
        raw_path.replace(final_path)
        return False

    subprocess.run(
        [ffmpeg_path, "-y", "-i", str(raw_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(final_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    raw_path.unlink()
    logger.info("Remuxed to QuickTime-compatible H.264/yuv420p: %s", final_path)
    return True


def print_progress(current: int, total: int, bar_width: int = 30) -> None:
    """Render a single-line, carriage-return progress bar: [####----] frame X/Y."""
    total_display = max(total, current, 1)
    filled = int(bar_width * current / total_display)
    bar = "#" * filled + "-" * (bar_width - filled)
    print(f"\r[{bar}] frame {current}/{total_display}", end="", flush=True)


def run_video_pipeline(args: argparse.Namespace) -> None:
    """Process every (sampled) frame of a video clip, caching hold detection from frame one
    but re-running pose tracking, limb-contact grounding, and beta-path planning every frame,
    then encode the annotated sequence to --output-dir/annotated_beta.mp4.

    Caching holds from frame 1 assumes a fixed camera position for the whole clip, not just a
    physically static wall: if the camera pans, zooms, or shakes, the cached hold positions
    will drift out of alignment with the wall as the clip progresses. Fine for a locked-down
    tripod shot; a handheld clip would need periodic re-detection or frame registration to
    compensate, which this does not do.
    """
    video_path = resolve_video_path(args)
    start_seconds = parse_timestamp(args.start)
    end_seconds = parse_timestamp(args.end)

    source_fps, _ = probe_video_metadata(video_path)
    output_fps = args.sample_fps if args.sample_fps else source_fps
    estimated_frames = max(1, round((end_seconds - start_seconds) * output_fps))

    detector = HoldDetector(weights=args.weights)
    holds: list[HoldDetection] | None = None
    climber: ClimberProfile | None = None
    calibrated = False
    contact_radius = 0.0
    route_color = args.route_color

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "annotated_beta.mp4"
    raw_output_path = output_path.with_suffix(".raw.mp4")
    writer: cv2.VideoWriter | None = None

    frame_count = 0
    visibility_scores: list[float] = []

    with PoseTracker(static_image_mode=False) as tracker:
        for _timestamp, frame in extract_frames(video_path, start_seconds, end_seconds, sample_fps=args.sample_fps):
            frame_count += 1

            if holds is None:
                holds = filter_by_confidence(detector.detect(frame, conf=args.conf), args.conf)
                logger.info("Cached %d holds from frame 1 (assumed static wall)", len(holds))

            pose = tracker.process(frame)

            if not calibrated:
                pixels_per_meter, calibrated = estimate_pixels_per_meter(pose, args.height)
                climber = ClimberProfile(
                    height=args.height * pixels_per_meter,
                    arm_span=args.arm_span * pixels_per_meter,
                )
                contact_radius = climber.max_reach_radius * CONTACT_RADIUS_FRACTION
                if calibrated:
                    logger.info("Calibrated climber profile from frame %d", frame_count)

            active_holds = holds
            if route_color is None and args.auto_route and pose is not None:
                probe_contacts = find_contact_holds(pose, holds, contact_radius)
                auto_color = determine_dominant_route_color(holds, probe_contacts, pose)
                if auto_color is not None:
                    route_color = auto_color
                    logger.info("Auto-detected route color: %s", route_color)

            if route_color is not None:
                color_filtered = filter_by_color(holds, route_color)
                if color_filtered:
                    active_holds = color_filtered

            com = estimate_center_of_mass(pose) if pose is not None else None

            overlay = draw_detections(frame, active_holds)
            if pose is not None:
                overlay = draw_pose(overlay, pose, center_of_mass=com)
                visibility_scores.append(average_visibility(pose))

            graph, path, _contacts = compute_beta_path(active_holds, climber, pose, com, contact_radius)
            if path is not None:
                overlay = draw_beta_path(overlay, graph, path)

            if writer is None:
                height, width = overlay.shape[:2]
                writer = open_video_writer(raw_output_path, output_fps, (width, height))

            writer.write(overlay)
            print_progress(frame_count, estimated_frames)

    if writer is not None:
        writer.release()
        remux_for_quicktime(raw_output_path, output_path)
    else:
        logger.warning("No frames were processed; no video written")

    print()
    print("\n=== BetaVision Video Pipeline Summary ===")
    print(f"Frames processed: {frame_count}")
    print(f"Holds cached from frame 1: {len(holds) if holds else 0}")
    if route_color is not None:
        print(f"Route color filter: {route_color}")
    if visibility_scores:
        print(f"Avg climber pose confidence: {sum(visibility_scores) / len(visibility_scores):.2%}")
    print(f"Annotated video saved to {output_path}")
    print("==========================================\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    is_video_source = args.video_url is not None or args.video_path is not None
    if is_video_source and args.end is None:
        parser.error("--end is required when using --video-url or --video-path")

    if args.save_video and not is_video_source:
        parser.error("--save-video requires --video-url or --video-path")

    if args.save_video:
        run_video_pipeline(args)
        return

    run_pipeline(args)


if __name__ == "__main__":
    main()
