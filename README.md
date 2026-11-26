<div align="center">

# SGSR
### A Saliency-Guided Image Super-Resolution Network

**ICIP 2023** · Dayeon Kim and Munchurl Kim · KAIST

[![Paper](https://img.shields.io/badge/Paper-IEEE_Xplore-00629B)](https://ieeexplore.ieee.org/document/10222146)
[![Conference](https://img.shields.io/badge/ICIP-2023-536DFE)](https://doi.org/10.1109/ICIP49359.2023.10222146)

[Paper](https://ieeexplore.ieee.org/document/10222146) · [Poster](SGSR_eposter_ICIP2023_revised.pdf) · [Installation](#installation) · [Inference](#inference) · [Training](#training) · [Citation](#citation)

</div>

Official PyTorch implementation of **SGSR: A Saliency-Guided Image Super-Resolution Network**.

SGSR uses saliency features to guide **×4 image super-resolution**, emphasizing salient objects while reducing unnecessary structures in the background. The model combines an RRDB image restoration backbone, a saliency network, saliency-guided attention, and a saliency-weighted training loss.

<p align="center">
  <a href="SGSR_eposter_ICIP2023_revised.pdf">
    <img src="SGSR_eposter_ICIP2023.png" alt="SGSR ICIP 2023 poster: method, architecture, and qualitative results" width="960">
  </a>
</p>

> **Pretrained weights:** SGSR and SalUNet checkpoints are not currently included. Inference requires both files; the commands below use placeholders for weights you have trained or obtained separately. The original processed training data and complete raw-COCO preprocessing pipeline are also not included. See [training prerequisites](docs/training.md).

## Installation

Run all commands from the repository root. Use a separate Python environment; Python **3.12** with PyTorch **2.5.1** / torchvision **0.20.1** is the reference setup for the CPU smoke checks.

```bash
git clone https://github.com/dayna-k/SGSR.git
cd SGSR
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CUDA 12.1 build for a compatible NVIDIA driver.
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

For CPU inference, replace the PyTorch installation command with:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

For other CUDA builds, choose a matching pair from the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). SGSR training uses CUDA; the standalone inference command also supports CPU. The original ICIP experiment environment was not recorded in a dependency lockfile, and the CUDA training recipe has not been revalidated end to end.

## Inference

Upscale a single image or a folder of low-resolution images. **Ground-truth images and saliency masks are not required**: the SalUNet checkpoint supplies the saliency features.

```bash
python inference.py \
  --input /path/to/low_resolution_images \
  --output results/demo \
  --checkpoint /path/to/generator.pt \
  --saliency-checkpoint /path/to/saliency.pth \
  --device cuda:0
```

Use `--device cpu` for CPU inference. Results are saved as PNG images at four times the input width and height. The default network has 16 RRDB blocks and 64 feature channels. Existing output files are protected unless `--overwrite` is supplied.

See **[the inference guide](docs/inference.md)** for checkpoint formats, supported inputs, memory considerations, and the separate legacy evaluation workflow.

## Training

Training has three stages. Prepare paired LR/HR images and continuous saliency score maps before starting; the guide specifies the directory layout, image sizes, and naming rules.

| Stage | Entry point | Purpose |
| --- | --- | --- |
| 1. Saliency learning | `python -m SalUNet.unet_train` | Train the saliency feature network. |
| 2. SR pretraining | `python main_train_G_only.py` | Train SGSR with pixel and saliency-weighted reconstruction losses. |
| 3. GAN training | `python main_train_GAN.py` | Initialize from stage 2 and train with perceptual and adversarial losses. |

**[Follow the training guide →](docs/training.md)** for copyable commands for every stage, data preparation requirements, hyperparameters, saved checkpoints, and resume behavior.

Select GPUs through the environment, for example `CUDA_VISIBLE_DEVICES=0` for one GPU or `CUDA_VISIBLE_DEVICES=0,1` for two GPUs. The trainers respect this setting. All training examples describe the released code; they are not a claim of reproduced paper results.

## Evaluation

[`test.py`](test.py) preserves the original COCO crop evaluation workflow. It requires **HR images, LR images, score maps, both model checkpoints, and LPIPS weights**. It uses fixed 384×384 HR crops and selects a subset of images based on mask area, so its results must be interpreted with that protocol.

Use `inference.py` to generate images. Consult [evaluation requirements and caveats](docs/inference.md#legacy-evaluation) before running `test.py` or comparing its numbers with another implementation.

## Repository structure

```text
SGSR/
├── inference.py                 # Single-image and folder inference
├── main_train_G_only.py         # SR reconstruction pretraining
├── main_train_GAN.py            # Adversarial training
├── test.py                      # Legacy crop evaluation
├── data.py                      # Paired HR/LR/score-map loading
├── model/                       # SGSR and ESRGAN architectures
├── SalUNet/                     # Saliency model, loader, and trainer
├── LPIPS_models/, LPIPS_util/   # Legacy perceptual metric code
├── make_saliency_score_map.py   # Historical preprocessing component
├── docs/                        # Training and inference guides
└── requirements.txt             # Python dependencies (excluding PyTorch)
```

## Development checks

The smoke tests use synthetic images and randomly initialized models; they do not
measure restoration quality or reproduce paper results.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{kim2023sgsr,
  author    = {Kim, Dayeon and Kim, Munchurl},
  title     = {{SGSR}: A Saliency-Guided Image Super-Resolution Network},
  booktitle = {2023 IEEE International Conference on Image Processing (ICIP)},
  year      = {2023},
  pages     = {980--984},
  doi       = {10.1109/ICIP49359.2023.10222146}
}
```

## Acknowledgments

The released implementation includes components based on [ESRGAN](https://github.com/xinntao/ESRGAN), [PyTorch U-Net](https://github.com/milesial/Pytorch-UNet), and [LPIPS](https://github.com/richzhang/PerceptualSimilarity). Please also consult their respective repositories and terms.

## Contact and usage

For questions, open a [GitHub issue](https://github.com/dayna-k/SGSR/issues) or contact **Dayeon Kim** at [dyk4501@kaist.ac.kr](mailto:dyk4501@kaist.ac.kr).

The source codes and datasets can be freely used for research and education only. Any commercial use should get formal permission first. Third-party code and datasets remain subject to their own terms.
