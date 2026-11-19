# Satellite Image Change Detection

This checkout preserves the original SMP U-Net path and adds a separate, pretrained SNUNet-CD inference path. The SNUNet-CD implementation is attributed to its original authors; this project does not claim to have developed or trained that model.

## Verified demonstration

The separate SNUNet-CD path was run on the supplied 1024×1024 A image pair. It strictly loaded the published checkpoint and produced:

- Binary model prediction: `output/A_snunet_change_mask.png` (1024×1024, 12,339 predicted changed pixels out of 1,048,576).
- Before/after/mask/overlay visualization: `output/A_snunet_comparison.png`.
- Input-only preview: `output/A_input_pair_preview.png` (not a model result).

The predicted pixel count describes this output only; it is not an accuracy or performance metric. The sample’s source and ground-truth label are not documented, so this output is not quantitatively validated.

## Model paths

### Original SMP U-Net path

The existing `src/model.py` and `src/inference.py` remain the original project path: a six-channel SMP U-Net with a ResNet34 encoder and one output channel. The required `models/pretrained/snunet_cd.pth` checkpoint is absent, and no verifiable checkpoint matching that exact U-Net configuration was found. The loader requires a checkpoint and will not produce an untrained prediction.

The root `config.py` refers to a different ChangeFormer checkpoint, while `src/config.py` refers to `snunet_cd.pth`; neither file exists. The original inference scripts use `src/config.py`.

### Separate pretrained SNUNet-CD path

Run the working path without changing or substituting weights in the U-Net implementation:

```powershell
.\.venv\Scripts\python.exe src\snunet_inference.py --img_t1 data\samples\A_T1.png --img_t2 data\samples\A_T2.png --output_dir output
```

The implementation in `src/snunet_model.py` uses the SNUNet_ECAM architecture with base channel 16 and its two-image interface. It strictly loads all 236 state-dictionary entries from the checkpoint. The output head produces two-class logits; the mask is formed from the model’s per-pixel argmax (class 1 = changed), not from a hand-created or pixel-difference mask.

The preprocessing follows the published Open-CD LEVIR-CD config: paired RGB inputs, the configured per-channel mean/std, resize-to-fit 1024×1024 while preserving aspect ratio, and padding to a multiple of 32. The supplied A pair is already 1024×1024. The checkpoint config uses 256×256 training crops and a 1024×1024 test resize.

### Checkpoint provenance

- Source: [Open-CD Model Zoo: LEVIR-CD](https://huggingface.co/likyoo/Open-CD_Model_Zoo/tree/main/LEVIR-CD)
- File: `snunet_c16_256x256_40k_levircd.pth`
- Size: 40,792,765 bytes
- Verified SHA-256: `984bfd07e0a30bbea5cd5a2fec918c994208c63b6c69e67a6592e06871811572`
- The model card declares Apache-2.0. The associated Open-CD config specifies SNUNet_ECAM, two separate three-channel inputs, and a two-class output.
- The checkpoint is ignored by Git (`*.pth`); it was downloaded only after approval.

The checkpoint is **not** compatible with the SMP U-Net and is used only by the separate SNUNet-CD path.

## Original repository and attribution

The source project is [RamyaShankarkumar/LevirCD](https://github.com/RamyaShankarkumar/LevirCD). Its history contains nine commits, no releases or tags, and no checkpoint in `models/pretrained/` (only `.gitkeep`). The initial inference commit introduced a simplified network named `SNUNetCD`; a later commit replaced it with the SMP U-Net currently present. The repository’s README is its only substantive project documentation; `notebooks/` is empty apart from its placeholder.

SNUNet-CD is the work of Fang et al.; the reference implementation is [likyoo/Siam-NestedUNet](https://github.com/likyoo/Siam-NestedUNet). The added SNUNet implementation is separately attributed under the MIT license in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The downloaded checkpoint’s model-card license is separate from the original repository’s licensing status.

## Data and evaluation

The checkout contains four paired images under `data/samples/` but no ground-truth masks, dataset split, or sample-level provenance. The complete LEVIR-CD dataset was not downloaded. No evaluation script or labels are present, so precision, recall, F1, IoU, and other performance metrics cannot be validly calculated and are not reported.

## License and reuse

Neither this checkout nor the original repository’s `main` branch contains a `LICENSE` file. No explicit reuse or redistribution permission for the original project’s code is stated. Preserve the author’s attribution and obtain permission before redistributing or reusing it beyond rights that may otherwise apply. Do not treat the third-party licenses listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) as a license for the original repository. This is a repository status note, not legal advice.

## Environment setup (Windows)

Verified with 64-bit Python 3.11.9, CPU-only PyTorch, and the versions pinned in `requirements.txt`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

PyTorch 2.6.0 and torchvision 0.21.0 are a compatible pair; the earlier README’s torchvision 0.16.0 recommendation was not compatible with PyTorch 2.6.0. For a CUDA installation, select the appropriate PyTorch wheel for the local driver and CUDA setup before installing the remaining requirements. CUDA was unavailable and not tested in this environment.

## My Contributions

- Set up and verified an isolated Python environment and corrected dependency pins.
- Added a separately attributed SNUNet-CD inference path using a verified pretrained LEVIR-CD checkpoint, while preserving the original SMP U-Net path.
- Ran inference on the included A image pair and saved the genuine model mask and visualization.
- Documented the repository history, checkpoint provenance, licensing status, and evaluation limitations.

I did not develop SNUNet-CD, train a model, or claim model performance from the unlabelled sample.
