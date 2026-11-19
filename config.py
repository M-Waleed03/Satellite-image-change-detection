import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 256
MODEL_PATH = "models/pretrained/changeformer_levir.pt"  # ChangeFormer pretrained

