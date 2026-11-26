# Inference and evaluation

[Back to README](../README.md)

## Checkpoints

Inference needs **two trained models**: the SGSR generator and the saliency feature
extractor (`UNetAttenModel`). The public repository does not bundle either weight
file or a verified download URL. `result/ckpt/place_holder.txt` and the logs under
`SalUNet/UNet_atten_new/` are not checkpoints. Use checkpoints you have obtained
from the authors or produced with the [training workflow](training.md).

The new `inference.py` accepts:

- A raw generator state dictionary, as saved by the SR training scripts.
- A saliency checkpoint containing a `model` state dictionary, as saved by
  `SalUNet/unet_train.py`.
- Either state dictionary wrapped in `state_dict`, with or without DataParallel's
  `module.` prefix.

Weights are loaded strictly: the generator must use 3 channels, 64 features,
16 residual blocks by default, and two ×2 upsampling stages. `--num-blocks` may be
changed only to match a checkpoint trained with a different block count. This
entrypoint supports **×4 SR**.

## Upscale an image or directory

From the repository root, after installing the dependencies:

```bash
python inference.py \
  --input /path/to/LR/image.png \
  --output results/demo \
  --checkpoint /path/to/generator.pt \
  --saliency-checkpoint /path/to/saliency.pth \
  --device cuda:0
```

Replace `--input` with a directory to process all supported images directly inside
it. The directory scan is non-recursive. PNG, JPEG, BMP, TIFF, and WebP inputs are
supported when their codec is available in OpenCV. Each result is an 8-bit,
three-channel PNG named `<input-stem>_x4.png`. Grayscale images become three-channel
images; alpha channels and image metadata are not retained.

Use `--device cpu` for CPU execution. The default `--device auto` selects CUDA if
available, otherwise CPU. To select a physical GPU explicitly:

```bash
CUDA_VISIBLE_DEVICES=2 python inference.py \
  --input /path/to/LR \
  --output results/demo \
  --checkpoint /path/to/generator.pt \
  --saliency-checkpoint /path/to/saliency.pth \
  --device cuda:0
```

With `CUDA_VISIBLE_DEVICES=2`, `cuda:0` refers to that selected GPU. No HR images or
saliency maps are needed: the saliency features are predicted from each LR input.
Existing output files are protected unless `--overwrite` is passed. Input images
with duplicate stems, such as `image.jpg` and `image.png`, must be renamed or
processed separately.

### Image size and preprocessing

The loader preserves the original training code's OpenCV **BGR** channel order and
normalizes pixels to `[-1, 1]`. Outputs are converted back to BGR for OpenCV to save.
Do not add an RGB conversion when using the released training convention.

The attention block requires even LR dimensions. `inference.py` replicates the
last row or column for odd inputs and crops the SR result back to exactly four
times the original height and width. Even-sized inputs are passed through without
padding. Padding is a convenience for user images, not the historical evaluation
protocol.

SGSR uses global attention with a matrix of approximately
`(H/2 × W/2)²` elements. Memory usage grows quickly with image size. Start with a
small LR crop (the original evaluation expects 96 × 96), and reduce the input
size if memory is exhausted. This entrypoint does not implement tiled inference;
tiling changes the attention context and would require separate validation.

## Legacy evaluation

`test.py` is the original dataset evaluation entrypoint. It requires paired LR,
HR, and saliency-score maps plus the two model checkpoints. It additionally uses
the bundled LPIPS implementation and its learned calibration weights. Obtain the
AlexNet v0.1 calibration file from the [official LPIPS repository](https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/weights/v0.1/alex.pth)
and supply its path with `--lpips_weights`. Torchvision also downloads the AlexNet
backbone on first use unless it is already cached. See
[installation](../README.md#installation) for evaluation dependencies.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py \
  --val_GTdataDir /path/to/validation/HR \
  --val_INdataDir /path/to/validation/LR \
  --val_SEGdataDir /path/to/validation/saliency \
  --pretrained_model /path/to/generator.pt \
  --pretrained_sal_path /path/to/saliency.pth \
  --lpips_weights /path/to/LPIPS/alex.pth \
  --results_folder results/evaluation \
  --gpu 0
```

HR filenames determine the pairs: `000001.jpg` or `000001.png` is paired with
`LR/000001.png` and `saliency/000001.png`. Saliency maps are single-channel PNGs at
HR resolution, encoding scores in `[0, 255]`. Evaluation expects **96 × 96 LR and
384 × 384 HR/maps**.

The script writes `config.txt` and `log_new.txt`. Its image-saving block is
commented out; use `inference.py` to save SR images.

### Interpreting the legacy metrics

The historical script removes an 8-pixel HR border and selects only samples for
which the maximum-score foreground (`map == 255`) and the zero-score background
each occupy more than 10% of the cropped 368 × 368 region. Its averages are over
that selected subset, not every supplied image.

The formulas are preserved for compatibility with the original script, including
its channel and masking conventions. In particular, the function named `rgb2y`
is applied directly to BGR arrays; its result is not a standard RGB-to-Y metric.
SSIM uses the generated image's range, and masked LPIPS uses the script's original
masking and area scaling. Treat these as historical diagnostics, not an independent
verification of the paper's reported scores. Reproducing those scores requires the
original dataset preparation, splits, trained weights, and evaluation protocol.
