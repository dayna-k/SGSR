# Training SGSR

This guide follows the released code. The training stages are saliency-network training, generator pretraining, and adversarial fine-tuning. Run all commands from the repository root after [installation](../README.md#installation).

**Data and pretrained weights are not included.** You need prepared HR/LR image pairs and saliency score maps. The public repository does not contain the full pipeline that produced the paper's training split and normalized score maps, so these commands are a guide to training the released implementation, not a claim of exact paper reproduction.

## 1. Prepare data

For the released ×4 setup, prepare aligned 384 × 384 HR images, 96 × 96 LR images, and 384 × 384 grayscale saliency score maps. For saliency-network training, also prepare aligned 96 × 96 score maps. Keep the original train/validation split separate.

```text
datasets/sgsr/
├── train/
│   ├── HR/                 # e.g. 000000000009.jpg (384 × 384)
│   ├── LR/                 # e.g. 000000000009.png (96 × 96)
│   ├── scores/             # e.g. 000000000009.png (384 × 384)
│   └── scores_lr/          # e.g. 000000000009.png (96 × 96)
└── val/
    ├── HR/
    ├── LR/
    ├── scores/
    └── scores_lr/
```

- Pair files by their filename stem. `image.jpg` in `HR/` needs `image.png` in `LR/` and `scores/`. Saliency training pairs each LR image with a same-stem PNG in `scores_lr/`.
- Store score maps as 8-bit grayscale PNGs. SR training divides values by 255; its saliency-weighted loss requires a nonzero score sum for every training image. An all-zero map is rejected.
- HR dimensions must be exactly four times LR dimensions. LR dimensions must be even and at least 16 pixels. Use the 96/384 square crops above for the documented training setup.
- All training images in a batch must have the same shape. `--patchSize` and `--need_patch` do **not** perform cropping; prepare crops before training. The loader applies random flips and 90-degree rotations.
- The SR training loader uses OpenCV BGR images normalized to `[-1, 1]`. The standalone saliency trainer uses RGB with ImageNet normalization and scales each score map by its maximum. These historical preprocessing conventions are retained in the code; changing them requires retraining and revalidation.

### What is included for score-map preparation?

[`make_saliency_score_map.py`](../make_saliency_score_map.py) is a legacy utility, not an end-to-end COCO preprocessing command. Its `make_score` function requires precomputed instance-mask JPEGs and a custom `instance_info.json` with image and instance-area/ROI statistics. It writes top-score masks under `mask_top_score/` and updates that JSON. It does not produce the `mask_score_norm` PNGs used by the training scripts. The required intermediate annotations and the full normalized-score-map preparation pipeline are not included.

Use the original prepared score maps when available. Substituting binary masks or a different degradation/score-generation procedure changes the training setup and should be reported with any results.

## 2. Train the saliency network

Skip this stage if you already have a compatible `UNetAttenModel` checkpoint. The SR scripts expect a checkpoint dictionary containing a `model` state dictionary.

```bash
CUDA_VISIBLE_DEVICES=0 python -m SalUNet.unet_train \
  --train_images datasets/sgsr/train/LR \
  --train_masks datasets/sgsr/train/scores_lr \
  --val_images datasets/sgsr/val/LR \
  --val_masks datasets/sgsr/val/scores_lr \
  --base_save_path result/saliency \
  --start_epoch 0 \
  --epochs 400 \
  --bs 8 \
  --img_size 96 \
  --lr 0.0004 \
  --n_worker 8
```

The saliency trainer uses one GPU. It can also run on CPU with `--use_gpu false`. Its input and mask paths are explicit; no machine-specific paths need to be edited.

Outputs are saved under:

```text
result/saliency/wbce_w0-1.0_w1-1.15/
├── config.txt
├── train_log.txt
├── weights/
│   └── best-model_epoch-..._mae-..._loss-....pth
└── optimizers/
    └── best-opt_epoch-..._mae-..._loss-....pth
```

Select a saved validation checkpoint and use its **actual filename** below. The directory suffix comes from legacy arguments; the current `SalBCELoss` implementation uses fixed weights `1.0` and `1.12`. The `--wbce_w0`, `--wbce_w1`, and `--alpha_loss` arguments do not change that loss.

Saliency training can restore both weights and optimizer state:

```bash
CUDA_VISIBLE_DEVICES=0 python -m SalUNet.unet_train \
  --train_images datasets/sgsr/train/LR \
  --train_masks datasets/sgsr/train/scores_lr \
  --val_images datasets/sgsr/val/LR \
  --val_masks datasets/sgsr/val/scores_lr \
  --base_save_path result/saliency \
  --start_epoch 100 --epochs 400 \
  --res_mod /path/to/model_epoch-099_mae-..._loss-....pth \
  --res_opt /path/to/opt_epoch-099_mae-..._loss-....pth
```

Epoch numbers in saved filenames are zero-based. Set `--start_epoch` to the next epoch. The best-MAE tracker resets when resuming; random-number and data-loader states are not saved.

## 3. Pretrain the SGSR generator

This stage optimizes pixel and saliency-weighted L1 losses. The saliency network stays frozen. Without an optional ESRGAN initialization, the generator starts from its default random initialization.

First set the checkpoint path, replacing the example with an existing file:

```bash
SAL_CHECKPOINT=/path/to/best-model_epoch-..._mae-..._loss-....pth
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 python main_train_G_only.py \
  --G_model SGSR \
  --GTdataDir datasets/sgsr/train/HR \
  --INdataDir datasets/sgsr/train/LR \
  --SEGdataDir datasets/sgsr/train/scores \
  --val_GTdataDir datasets/sgsr/val/HR \
  --val_INdataDir datasets/sgsr/val/LR \
  --val_SEGdataDir datasets/sgsr/val/scores \
  --pretrained_sal_path "$SAL_CHECKPOINT" \
  --results result/sgsr_pretrain \
  --scale 4 --nBlock 16 --nFeat 64 --nChannel 3 \
  --batchSize 8 --patchSize 384 --nThreads 8 \
  --epochs 60 --lr_G 0.0001 --lrDecay 5 \
  --lambda_pixel 0.5 --lambda_obj 0.5
```

For compatible plain ESRGAN generator weights, `--finetuned true --finetune_path /path/to/esrgan.pt` initializes shared generator parameters. The checkpoint must match the released `model/esrgan.py` architecture; arbitrary ESRGAN checkpoints are not interchangeable. This initialization expects an unwrapped state dictionary.

## 4. Fine-tune with adversarial and perceptual losses

Use the SGSR generator checkpoint from stage 3. With the 60-epoch command above, the last epoch is `generator_59.pt`.

```bash
CUDA_VISIBLE_DEVICES=0 python main_train_GAN.py \
  --G_model SGSR --D_model RelaD \
  --GTdataDir datasets/sgsr/train/HR \
  --INdataDir datasets/sgsr/train/LR \
  --SEGdataDir datasets/sgsr/train/scores \
  --val_GTdataDir datasets/sgsr/val/HR \
  --val_INdataDir datasets/sgsr/val/LR \
  --val_SEGdataDir datasets/sgsr/val/scores \
  --pretrained_sal_path "$SAL_CHECKPOINT" \
  --finetuned true \
  --finetune_G_path result/sgsr_pretrain/generator_59.pt \
  --results result/sgsr_gan \
  --scale 4 --nBlock 16 --nFeat 64 --nChannel 3 \
  --batchSize 8 --patchSize 384 --nThreads 8 \
  --epochs 60 --lr_G 0.0001 --lr_D 0.000001 --lrDecay 5 \
  --lambda_pixel 0.5 --lambda_obj 0.5 \
  --lambda_adv 0.1 --lambda_vgg 0.5
```

The discriminator starts from random initialization. `--finetune_D_path` is a retained legacy argument and is not used to load a discriminator. The VGG19 feature extractor uses torchvision pretrained weights; the first run may download those weights into the torch cache.

Both SR stages require CUDA. Set `CUDA_VISIBLE_DEVICES=0,1` to use two visible GPUs with `DataParallel`; the batch size is the total across those GPUs. GPU selection is controlled by the environment. Legacy `--gpu` and `--multi-processing` arguments do not select or disable devices.

The learning rates above are the script defaults. With `--decayType step --lrDecay 5`, the implementation uses `initial_lr / 2**((epoch + 1) // 5)` for zero-based epoch indices.

## 5. Outputs and restarting SR training

Each SR run writes the following under `--results`:

| File | Contents |
| --- | --- |
| `config.txt` | Parsed arguments, appended on each invocation |
| `log.txt` | Training losses and validation diagnostics |
| `<iteration>_<index>.png` | Validation image grids |
| `generator_<epoch>_<iteration>.pt` | Generator state at checkpoint intervals |
| `generator_<epoch>.pt` | Generator state at each epoch end |
| `discriminator_<epoch>*.pt` | Corresponding discriminator states in GAN training |

`--sample_interval` defaults to 1,000 optimizer iterations; `--checkpoint_interval` defaults to 15,000. Validation also runs before the training loop. Use separate result directories for separate experiments.

To restart at epoch 10, repeat the same stage command with the same `--results` and add `--start_epoch 10`. It loads `generator_9.pt`; GAN training also loads `discriminator_9.pt`. Set `--epochs` to the desired total epoch count, not the number of additional epochs.

**SR checkpoints contain model weights only.** Adam optimizer state and random-number states are not restored, so this is a weights-based restart rather than an exact continuation. The generator-only script additionally accepts `--start_with_iter true --start_epoch E --start_iter I` for a `generator_E_I.pt` checkpoint when `E` is nonzero; `I` is the global iteration index, not the batch index within the epoch.

The validation PSNR printed during SR training is a legacy diagnostic: it uses an 8-pixel border crop and the original channel conversion on BGR-loaded images. Do not equate it with a separately standardized benchmark protocol or claim paper-level reproduction from that log alone.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing saliency or generator checkpoint | Weights are not bundled. Supply your own compatible checkpoint with the relevant path argument. |
| Missing LR image or score map | Match the HR stem and use `.png` for the paired LR image and map. |
| Zero score map error | Saliency normalization and the SR saliency-weighted loss need nonzero scores. Check preprocessing. |
| Batch collation shape error | Pre-crop aligned square image/map pairs to a common size; `--patchSize` does not crop. |
| No optimizer steps | There must be at least `--batchSize` training images; SR loaders use `drop_last=True`. |
| Out of GPU memory | Reduce `--batchSize`; start with the documented 96 × 96 LR crops. Attention memory grows rapidly with image size. |
| Boolean option ignored | Pass explicit `true` or `false`, e.g. `--finetuned true`. |
| Unexpected LR/HR scaling | The documented and supported CLI configuration is ×4 with 64 features and three channels. |
