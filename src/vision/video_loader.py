"""Download a YouTube clip and extract frames from a trimmed time range for inference."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yt_dlp

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def parse_timestamp(value: str | float) -> float:
    """Convert a numeric-seconds value or "HH:MM:SS"/"MM:SS" string to seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid timestamp: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + float(part)
    return seconds


def download_video(url: str, output_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download the best-quality mp4 for `url` into `output_dir`, returning its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    if not video_path.exists():
        raise FileNotFoundError(f"Expected downloaded video at {video_path}, but it is missing")

    logger.info("Downloaded %s -> %s", url, video_path)
    return video_path


def extract_frames(
    video_path: Path,
    start_time: str | float = 0.0,
    end_time: str | float | None = None,
    sample_fps: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_seconds, frame) tuples for `video_path` between start_time and end_time.

    `sample_fps` optionally subsamples the source frame rate (e.g. 5.0 to grab 5 frames/sec
    from a 30fps source) instead of yielding every decoded frame.
    """
    start_seconds = parse_timestamp(start_time)
    end_seconds = parse_timestamp(end_time) if end_time is not None else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    try:
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_stride = max(1, round(source_fps / sample_fps)) if sample_fps else 1

        cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if end_seconds is not None and timestamp > end_seconds:
                break

            if frame_index % frame_stride == 0:
                yield timestamp, frame

            frame_index += 1
    finally:
        cap.release()


def save_frames(
    frames: Iterator[tuple[float, np.ndarray]],
    output_dir: Path,
    prefix: str = "frame",
) -> list[Path]:
    """Write (timestamp, frame) pairs to `output_dir` as JPEGs named by timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for timestamp, frame in frames:
        frame_path = output_dir / f"{prefix}_{timestamp:08.3f}.jpg"
        cv2.imwrite(str(frame_path), frame)
        saved_paths.append(frame_path)

    logger.info("Saved %d frames -> %s", len(saved_paths), output_dir)
    return saved_paths


def load_clip_frames(
    url: str,
    start_time: str | float,
    end_time: str | float,
    sample_fps: float | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> Iterator[tuple[float, np.ndarray]]:
    """Download `url` if needed and yield frames from the [start_time, end_time] window."""
    video_path = download_video(url, output_dir=raw_dir)
    yield from extract_frames(video_path, start_time, end_time, sample_fps=sample_fps)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube URL of the climbing clip")
    parser.add_argument("start", help='Start timestamp, e.g. "1:23" or seconds')
    parser.add_argument("end", help='End timestamp, e.g. "1:45" or seconds')
    parser.add_argument("--sample-fps", type=float, default=None, help="Frames per second to sample")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Download directory")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_PROCESSED_DIR, help="Directory to save extracted frames"
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args()

    frames = load_clip_frames(
        args.url,
        args.start,
        args.end,
        sample_fps=args.sample_fps,
        raw_dir=args.raw_dir,
    )
    save_frames(frames, args.output_dir)


if __name__ == "__main__":
    main()
