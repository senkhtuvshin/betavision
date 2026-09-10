"""Fine-tune YOLOv8-seg for climbing hold segmentation, or fetch pretrained weights instead.

Three ways to get real hold weights into models/, fastest first:

  1. Bypass training entirely with a community checkpoint:
       python scripts/train_hold_detector.py --download-only

     This pulls jwlarocque/yolov8n-freeclimbs-detect-2 from Hugging Face into
     models/pretrained_holds.pt. IMPORTANT: that checkpoint is a *detection* model
     (bounding boxes, single class named "rock") — not instance segmentation. It loads
     fine via ultralytics and finds real holds (confirmed: 49 boxes at conf>=0.15 on a
     real gym photo), but src/vision/hold_detector.py reads `result.masks`, which is
     None for a detection-only model, so it currently yields zero detections through the
     existing pipeline. Useful for evaluating box/centroid quality now, or as a base to
     fine-tune further into a -seg model; not a drop-in mask/polygon source as-is.

  2. Fine-tune on a public Roboflow segmentation dataset (needs a free API key from
     https://app.roboflow.com/settings/api and `pip install roboflow`):
       python scripts/train_hold_detector.py \\
           --roboflow-workspace climb-ai --roboflow-project hold-detector-rnvkl \\
           --roboflow-version 1 --roboflow-api-key $ROBOFLOW_API_KEY

     "climb-ai/hold-detector-rnvkl" (https://universe.roboflow.com/climb-ai/hold-detector-rnvkl)
     is a public instance-segmentation dataset distinguishing holds from volumes; confirm the
     version number and exact class list on the project page before a long training run, since
     Roboflow projects get re-versioned over time. Browse universe.roboflow.com/search?q=class:hold
     for alternatives.

  3. Fine-tune on your own exported dataset:
       python scripts/train_hold_detector.py --data path/to/data.yaml

Training runs 25-30 epochs by default on Apple Silicon MPS (falls back to CPU if MPS
isn't available), and copies the resulting best.pt to models/best_holds.pt.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DEFAULT_BASE_WEIGHTS = "yolov8n-seg.pt"
DEFAULT_EPOCHS = 30
DEFAULT_OUTPUT_WEIGHTS = MODELS_DIR / "best_holds.pt"

# Verified (2026-09-10) real, working checkpoint that bypasses local training. See the
# module docstring for the detection-vs-segmentation caveat before relying on it.
PRETRAINED_HOLD_WEIGHTS_URL = (
    "https://huggingface.co/jwlarocque/yolov8n-freeclimbs-detect-2/resolve/main/"
    "yolov8n-freeclimbs-detect-2.pt"
)


def resolve_device() -> str:
    """Prefer Apple Silicon MPS, falling back to CPU when it isn't available."""
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def download_pretrained_weights(
    url: str = PRETRAINED_HOLD_WEIGHTS_URL,
    dest: Path = MODELS_DIR / "pretrained_holds.pt",
) -> Path:
    """Download an existing publicly hosted hold-detection checkpoint, bypassing local training."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading pretrained hold weights from %s", url)
    urllib.request.urlretrieve(url, dest)
    logger.info("Saved to %s", dest)
    return dest


def download_roboflow_dataset(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    format_: str = "yolov8",
) -> Path:
    """Download a Roboflow dataset version and return the path to its data.yaml."""
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise ImportError(
            "The 'roboflow' package is required for automatic dataset download. "
            "Install it with `pip install roboflow`."
        ) from exc

    rf = Roboflow(api_key=api_key)
    rf_version = rf.workspace(workspace).project(project).version(version)
    dataset = rf_version.download(format_)
    return Path(dataset.location) / "data.yaml"


def train(
    data_yaml: Path,
    base_weights: str,
    epochs: int,
    device: str,
    output_weights: Path,
) -> Path:
    """Fine-tune `base_weights` on `data_yaml`, copying the resulting best.pt to output_weights."""
    from ultralytics import YOLO

    model = YOLO(base_weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        device=device,
        project=str(output_weights.parent),
        name="hold_seg_run",
        exist_ok=True,
    )

    best_checkpoint = model.trainer.best if model.trainer.best.exists() else model.trainer.last
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_checkpoint, output_weights)
    logger.info("Best weights copied to %s", output_weights)
    return output_weights


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--data", type=Path, help="Path to an existing data.yaml (skips Roboflow download)")
    parser.add_argument("--roboflow-workspace", default="climb-ai", help="Roboflow workspace slug")
    parser.add_argument("--roboflow-project", default="hold-detector-rnvkl", help="Roboflow project slug")
    parser.add_argument("--roboflow-version", type=int, default=1, help="Roboflow dataset version number")
    parser.add_argument(
        "--roboflow-api-key",
        default=os.environ.get("ROBOFLOW_API_KEY"),
        help="Roboflow API key (defaults to the ROBOFLOW_API_KEY env var)",
    )
    parser.add_argument("--roboflow-format", default="yolov8", help="Roboflow export format")

    parser.add_argument("--base-weights", default=DEFAULT_BASE_WEIGHTS, help="Base YOLOv8-seg checkpoint to fine-tune")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs (recommended 25-30)")
    parser.add_argument("--device", default=None, help="Force a device (mps/cpu); default: auto-detect")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_WEIGHTS, help="Where to copy the best weights")

    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Skip training; download the public pretrained checkpoint instead",
    )
    parser.add_argument("--pretrained-url", default=PRETRAINED_HOLD_WEIGHTS_URL, help="URL used by --download-only")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.download_only:
        path = download_pretrained_weights(args.pretrained_url, MODELS_DIR / "pretrained_holds.pt")
        print(f"Pretrained (detection-only) weights saved to {path}")
        return

    if args.data is not None:
        data_yaml = args.data
    else:
        if not args.roboflow_api_key:
            parser.error("--roboflow-api-key (or the ROBOFLOW_API_KEY env var) is required when --data is not given")
        data_yaml = download_roboflow_dataset(
            args.roboflow_workspace,
            args.roboflow_project,
            args.roboflow_version,
            args.roboflow_api_key,
            args.roboflow_format,
        )

    device = args.device or resolve_device()
    logger.info("Training %s on %s for %d epochs using %s", args.base_weights, device, args.epochs, data_yaml)
    output = train(data_yaml, args.base_weights, args.epochs, device, args.output)
    print(f"Best weights saved to {output}")


if __name__ == "__main__":
    main()
