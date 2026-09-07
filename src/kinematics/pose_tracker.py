"""MediaPipe Pose based climber tracking: landmarks, center of mass, and reach spans."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmarks relevant to climbing technique analysis (wrists, elbows, shoulders, hips, knees, ankles).
KEY_LANDMARKS: dict[str, "mp_pose.PoseLandmark"] = {
    "LEFT_SHOULDER": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "LEFT_ELBOW": mp_pose.PoseLandmark.LEFT_ELBOW,
    "RIGHT_ELBOW": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "LEFT_WRIST": mp_pose.PoseLandmark.LEFT_WRIST,
    "RIGHT_WRIST": mp_pose.PoseLandmark.RIGHT_WRIST,
    "LEFT_HIP": mp_pose.PoseLandmark.LEFT_HIP,
    "RIGHT_HIP": mp_pose.PoseLandmark.RIGHT_HIP,
    "LEFT_KNEE": mp_pose.PoseLandmark.LEFT_KNEE,
    "RIGHT_KNEE": mp_pose.PoseLandmark.RIGHT_KNEE,
    "LEFT_ANKLE": mp_pose.PoseLandmark.LEFT_ANKLE,
    "RIGHT_ANKLE": mp_pose.PoseLandmark.RIGHT_ANKLE,
}


@dataclass
class ClimberPose:
    """Key landmark coordinates for one detected climber in a single frame."""

    normalized: dict[str, tuple[float, float, float]]  # name -> (x, y, z) in [0, 1], z is relative depth
    pixel: dict[str, tuple[int, int]]  # name -> (x, y) in pixel coordinates
    visibility: dict[str, float]  # name -> visibility score in [0, 1]
    frame_shape: tuple[int, int]  # (height, width)
    raw_landmarks: object  # mediapipe NormalizedLandmarkList, kept for drawing the full skeleton


@dataclass
class ReachSpans:
    """Dynamic reach geometry derived from a ClimberPose."""

    wingspan: float  # pixel distance between the two wrists
    left_reach_vector: tuple[float, float]  # left hip -> left wrist, in pixels
    right_reach_vector: tuple[float, float]  # right hip -> right wrist, in pixels
    left_reach_distance: float
    right_reach_distance: float


class PoseTracker:
    """Wraps MediaPipe Pose, handling both static-image and streaming-video modes."""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self._pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame: np.ndarray) -> ClimberPose | None:
        """Run pose estimation on a single BGR frame, returning None if no climber is found."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._pose.process(rgb)

        if not result.pose_landmarks:
            return None

        return self._to_climber_pose(result.pose_landmarks, frame.shape)

    @staticmethod
    def _to_climber_pose(landmark_list, frame_shape: tuple[int, ...]) -> ClimberPose:
        height, width = frame_shape[:2]
        normalized: dict[str, tuple[float, float, float]] = {}
        pixel: dict[str, tuple[int, int]] = {}
        visibility: dict[str, float] = {}

        for name, landmark_id in KEY_LANDMARKS.items():
            landmark = landmark_list.landmark[landmark_id.value]
            normalized[name] = (landmark.x, landmark.y, landmark.z)
            pixel[name] = (round(landmark.x * width), round(landmark.y * height))
            visibility[name] = landmark.visibility

        return ClimberPose(
            normalized=normalized,
            pixel=pixel,
            visibility=visibility,
            frame_shape=(height, width),
            raw_landmarks=landmark_list,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def estimate_center_of_mass(
    pose: ClimberPose,
    hip_weight: float = 0.6,
    shoulder_weight: float = 0.4,
) -> tuple[float, float]:
    """Approximate torso center of mass as a weighted average of hip and shoulder midpoints.

    Hips are weighted more heavily by default since the pelvis sits closer to the body's
    true center of mass than the shoulder line.
    """
    if abs(hip_weight + shoulder_weight - 1.0) > 1e-6:
        raise ValueError("hip_weight and shoulder_weight must sum to 1.0")

    height, width = pose.frame_shape
    hip_x = (pose.normalized["LEFT_HIP"][0] + pose.normalized["RIGHT_HIP"][0]) / 2
    hip_y = (pose.normalized["LEFT_HIP"][1] + pose.normalized["RIGHT_HIP"][1]) / 2
    shoulder_x = (pose.normalized["LEFT_SHOULDER"][0] + pose.normalized["RIGHT_SHOULDER"][0]) / 2
    shoulder_y = (pose.normalized["LEFT_SHOULDER"][1] + pose.normalized["RIGHT_SHOULDER"][1]) / 2

    com_x = hip_weight * hip_x + shoulder_weight * shoulder_x
    com_y = hip_weight * hip_y + shoulder_weight * shoulder_y
    return com_x * width, com_y * height


def compute_reach_spans(pose: ClimberPose) -> ReachSpans:
    """Compute wrist-to-wrist wingspan and per-side hip-to-wrist reach vectors, in pixels."""
    left_wrist, right_wrist = pose.pixel["LEFT_WRIST"], pose.pixel["RIGHT_WRIST"]
    left_hip, right_hip = pose.pixel["LEFT_HIP"], pose.pixel["RIGHT_HIP"]

    left_vector = (left_wrist[0] - left_hip[0], left_wrist[1] - left_hip[1])
    right_vector = (right_wrist[0] - right_hip[0], right_wrist[1] - right_hip[1])

    return ReachSpans(
        wingspan=math.dist(left_wrist, right_wrist),
        left_reach_vector=left_vector,
        right_reach_vector=right_vector,
        left_reach_distance=math.hypot(*left_vector),
        right_reach_distance=math.hypot(*right_vector),
    )


def draw_pose(
    frame: np.ndarray,
    pose: ClimberPose,
    center_of_mass: tuple[float, float] | None = None,
) -> np.ndarray:
    """Return a copy of `frame` with the pose skeleton and, if given, a CoM marker overlaid."""
    overlay = frame.copy()

    mp_drawing.draw_landmarks(
        overlay,
        pose.raw_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
    )

    if center_of_mass is not None:
        cx, cy = int(center_of_mass[0]), int(center_of_mass[1])
        cv2.circle(overlay, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(overlay, "CoM", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return overlay


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to an image of a climber")
    parser.add_argument("--output", type=Path, default=Path("pose_overlay.jpg"), help="Overlay output path")
    args = parser.parse_args()

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    with PoseTracker(static_image_mode=True) as tracker:
        pose = tracker.process(frame)

    if pose is None:
        print("No climber pose detected.")
        return

    com = estimate_center_of_mass(pose)
    spans = compute_reach_spans(pose)
    overlay = draw_pose(frame, pose, center_of_mass=com)
    cv2.imwrite(str(args.output), overlay)

    print(f"CoM: {com}")
    print(f"Wingspan: {spans.wingspan:.1f}px")
    print(f"Left reach: {spans.left_reach_distance:.1f}px, Right reach: {spans.right_reach_distance:.1f}px")
    print(f"Overlay saved to {args.output}")


if __name__ == "__main__":
    main()
