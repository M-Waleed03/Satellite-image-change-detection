from pathlib import Path

import torch
import segmentation_models_pytorch as smp
from config import DEVICE, MODEL_PATH

def load_model():
    checkpoint_path = Path(MODEL_PATH)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Trained change-detection checkpoint not found: "
            f"{checkpoint_path.resolve()}"
        )

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=6,
        classes=1,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected a model state dictionary in {checkpoint_path}, "
            f"got {type(checkpoint).__name__}"
        )

    model.load_state_dict(checkpoint)
    model.to(DEVICE).eval()
    return model

