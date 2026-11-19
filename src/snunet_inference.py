import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from snunet_model import load_snunet_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "pretrained"
    / "snunet_c16_256x256_40k_levircd.pth"
)
IMAGE_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
INFERENCE_SCALE = (1024, 1024)
SIZE_DIVISOR = 32


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _resize_for_inference(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    target_width, target_height = INFERENCE_SCALE
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, int(width * scale + 0.5))
    resized_height = max(1, int(height * scale + 0.5))
    return cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )


def _prepare_image(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
    return (tensor - IMAGE_MEAN) / IMAGE_STD


def _make_overlay(after: np.ndarray, changed: np.ndarray) -> np.ndarray:
    overlay = after.copy()
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    overlay[changed] = (
        0.55 * overlay[changed] + 0.45 * red[changed]
    ).astype(np.uint8)
    return overlay


def run_inference(
    image_t1_path: str | Path,
    image_t2_path: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    image_t1_path = Path(image_t1_path)
    image_t2_path = Path(image_t2_path)
    output_dir = Path(output_dir)

    image_t1 = _read_rgb(image_t1_path)
    image_t2 = _read_rgb(image_t2_path)
    if image_t1.shape != image_t2.shape:
        raise ValueError(
            "The T1 and T2 images must have identical dimensions; "
            f"got {image_t1.shape[:2]} and {image_t2.shape[:2]}"
        )

    image_t1 = _resize_for_inference(image_t1)
    image_t2 = _resize_for_inference(image_t2)
    t1_height, t1_width = image_t1.shape[:2]
    t2_height, t2_width = image_t2.shape[:2]
    if (t1_height, t1_width) != (t2_height, t2_width):
        raise ValueError("The resized T1 and T2 images must have identical dimensions")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_snunet_model(CHECKPOINT_PATH, device)
    x_t1 = _prepare_image(image_t1).to(device)
    x_t2 = _prepare_image(image_t2).to(device)

    pad_height = (-t1_height) % SIZE_DIVISOR
    pad_width = (-t1_width) % SIZE_DIVISOR
    if pad_height or pad_width:
        x_t1 = F.pad(x_t1, (0, pad_width, 0, pad_height), value=0)
        x_t2 = F.pad(x_t2, (0, pad_width, 0, pad_height), value=0)

    with torch.inference_mode():
        logits = model(x_t1, x_t2)
        logits = logits[:, :, :t1_height, :t1_width]
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    changed = prediction == 1
    mask = changed.astype(np.uint8) * 255
    overlay = _make_overlay(image_t2, changed)
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_name = image_t1_path.stem
    if pair_name.endswith("_T1"):
        pair_name = pair_name[:-3]
    mask_path = output_dir / f"{pair_name}_snunet_change_mask.png"
    visualization_path = output_dir / f"{pair_name}_snunet_comparison.png"
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Unable to write change mask: {mask_path}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].imshow(image_t1)
    axes[0, 0].set_title("Before (T1)")
    axes[0, 1].imshow(image_t2)
    axes[0, 1].set_title("After (T2)")
    axes[1, 0].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("SNUNet-CD predicted change mask")
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("Prediction overlaid on T2")
    for axis in axes.flat:
        axis.axis("off")
    figure.suptitle("Pretrained SNUNet-CD inference; no ground-truth overlay")
    figure.savefig(visualization_path, dpi=150)
    plt.close(figure)

    changed_pixels = int(changed.sum())
    total_pixels = int(changed.size)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Device: {device}")
    print(f"Output size: {t1_width}x{t1_height}")
    print(f"Predicted changed pixels: {changed_pixels}/{total_pixels}")
    print(f"Change mask saved to {mask_path}")
    print(f"Comparison visualization saved to {visualization_path}")
    return mask_path, visualization_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the separate pretrained SNUNet-CD LEVIR-CD inference path."
    )
    parser.add_argument("--img_t1", required=True, type=Path)
    parser.add_argument("--img_t2", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "output")
    args = parser.parse_args()
    run_inference(args.img_t1, args.img_t2, args.output_dir)


if __name__ == "__main__":
    main()
