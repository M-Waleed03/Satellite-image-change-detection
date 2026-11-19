import torch
from model import load_model
from utils import load_image, save_mask, overlay_mask
from config import DEVICE, IMAGE_SIZE

def main(img_t1, img_t2, output_path):
    model = load_model()

    img1 = load_image(img_t1, IMAGE_SIZE)
    img2 = load_image(img_t2, IMAGE_SIZE)

    x = torch.cat([img1, img2], dim=0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(x)
        mask = pred.squeeze(0).squeeze(0).sigmoid().cpu().numpy()

    # Save binary change mask
    save_mask(mask, output_path)
    print(f"Change mask saved to {output_path}")

    # --- Overlay mask on top of T2 image ---
    overlay_output = output_path.replace(".png", "_overlay.png")
    overlay_mask(
        img_path=img_t2,
        mask_path=output_path,
        output_path=overlay_output
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_t1", required=True)
    parser.add_argument("--img_t2", required=True)
    parser.add_argument("--output", default="output/change_mask.png")
    args = parser.parse_args()
    main(args.img_t1, args.img_t2, args.output)

