import torch
import numpy as np
from PIL import Image

def load_image(path, size=256):
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size))
    img = np.array(img).transpose(2,0,1)/255.0
    return torch.tensor(img, dtype=torch.float32)  # <-- now works

def save_mask(mask, path):
    mask_img = (mask > 0.5).astype(np.uint8) * 255
    Image.fromarray(mask_img).save(path)

def overlay_mask(img_path, mask_path, output_path, alpha=0.5):
    img = Image.open(img_path).convert("RGB").resize((256,256))
    mask = Image.open(mask_path).convert("L").resize((256,256))

    overlay = Image.new("RGB", img.size)
    overlay.paste((255,0,0), mask=mask)  # red overlay for change
    blended = Image.blend(img, overlay, alpha)
    blended.save(output_path)
    print(f"Overlay saved to {output_path}")

