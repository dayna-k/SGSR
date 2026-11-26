"""Run SGSR x4 on low-resolution images without ground truth or saliency masks."""

import argparse
from collections.abc import Mapping
from pathlib import Path


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Input image or directory of images (non-recursive).')
    parser.add_argument('--output', type=Path, default=Path('results/inference'),
                        help='Output directory; images are saved as <stem>_x4.png.')
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Trained SGSR generator state dict (.pt/.pth).')
    parser.add_argument('--saliency-checkpoint', type=Path, required=True,
                        help='Trained UNetAttenModel checkpoint (.pth).')
    parser.add_argument('--device', default='auto',
                        help='auto (CUDA if available), cpu, or cuda:N.')
    parser.add_argument('--num-blocks', type=int, default=16,
                        help='Generator residual blocks; must match the checkpoint (default: 16).')
    parser.add_argument('--overwrite', action='store_true',
                        help='Allow replacing existing output images.')
    return parser


def check_checkpoint_file(path):
    if not path.is_file():
        raise ValueError(f'Checkpoint not found: {path}. The repository does not bundle '
                         'trained SGSR or saliency weights; provide both checkpoints.')
    with path.open('rb') as handle:
        if handle.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'{path} is a Git LFS pointer, not model weights. '
                             'Fetch the corresponding LFS object before running inference.')


def collect_jobs(input_path, output_dir, overwrite=False):
    if input_path.is_dir():
        if input_path.resolve() == output_dir.resolve():
            raise ValueError('Use separate input and output directories.')
        images = sorted(path for path in input_path.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    elif input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        images = [input_path]
    else:
        raise ValueError(f'Input is not a supported image or directory: {input_path}')
    if not images:
        raise ValueError(f'No supported images found in {input_path}')
    jobs = [(path, output_dir / f'{path.stem}_x4.png') for path in images]
    outputs = [dest.resolve() for _, dest in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Input images share a filename stem and would overwrite one another. '
                         'Rename them or process them separately.')
    inputs = {path.resolve() for path in images}
    for _, dest in jobs:
        if dest.resolve() in inputs:
            raise ValueError(f'Output would replace an input image: {dest}')
        if dest.exists() and not overwrite:
            raise ValueError(f'Output already exists: {dest}. Use another output directory '
                             'or pass --overwrite.')
    return jobs


def load_weights(path):
    """Accept the original raw generator and {'model': ...} saliency formats."""
    import torch

    check_checkpoint_file(path)
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot load tensor checkpoint {path}: {exc}') from exc
    if isinstance(checkpoint, Mapping):
        for key in ('state_dict', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], Mapping):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ValueError(f'{path} does not contain a nonempty model state dict.')
    state_dict = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(f'{path} contains a non-tensor state entry; '
                             'supply a raw state dict or a model/state_dict checkpoint.')
        name = key.removeprefix('module.')
        if name in state_dict:
            raise ValueError(f'Duplicate state key after removing module. in {path}: {name}')
        state_dict[name] = value
    return state_dict


def select_device(name):
    import torch

    if name == 'auto':
        name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f'Invalid device {name!r}; use cpu or cuda:N.') from exc
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Supported devices are cpu and cuda:N.')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Use --device cpu or install a CUDA-enabled PyTorch build.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {device.index} is not visible.')
    return device


def image_to_tensor(image, device):
    """Keep OpenCV BGR ordering, matching data.RGB_np2Tensor despite its name."""
    import numpy as np
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
    return ((tensor / 255.0 - 0.5) * 2.0).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    import numpy as np

    image = (tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255.0
    return image.clip(0, 255).astype(np.uint8)


def super_resolve(image, generator, saliency, device):
    import torch
    import torch.nn.functional as functional

    height, width = image.shape[:2]
    tensor = image_to_tensor(image, device)
    # Guided_Attn downsamples by two and adds the upsampled result to its input.
    # Pad only odd dimensions; preserve the original path for even-sized images.
    if height % 2 or width % 2:
        tensor = functional.pad(tensor, (0, width % 2, 0, height % 2), mode='replicate')
    with torch.inference_mode():
        _, features = saliency(tensor)
        prediction = generator(tensor, features)
    return tensor_to_image(prediction[:, :, :height * 4, :width * 4])


def run(args):
    if args.num_blocks < 1:
        raise ValueError('--num-blocks must be positive.')
    for path in (args.checkpoint, args.saliency_checkpoint):
        check_checkpoint_file(path)
    jobs = collect_jobs(args.input, args.output, args.overwrite)

    # Keep --help and path validation available before installing ML dependencies.
    import cv2
    from model.sgsrnet import GeneratorRRDB_SGSR
    from SalUNet.unet_new import UNetAttenModel

    device = select_device(args.device)
    generator = GeneratorRRDB_SGSR(channels=3, filters=64,
                                  num_res_blocks=args.num_blocks, num_upsample=2)
    saliency = UNetAttenModel()
    for model, path in ((generator, args.checkpoint), (saliency, args.saliency_checkpoint)):
        try:
            model.load_state_dict(load_weights(path), strict=True)
        except RuntimeError as exc:
            raise ValueError(f'Checkpoint architecture mismatch for {path}: {exc}') from exc
        model.to(device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f'SGSR x4 | device: {device} | images: {len(jobs)}')
    for index, (source, destination) in enumerate(jobs, start=1):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'Cannot decode image: {source}')
        result = super_resolve(image, generator, saliency, device)
        if not cv2.imwrite(str(destination), result):
            raise OSError(f'Cannot write output image: {destination}')
        print(f'[{index}/{len(jobs)}] {source.name} -> {destination} '
              f'({result.shape[1]} x {result.shape[0]})')


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except ImportError as exc:
        parser.exit(1, f'Missing or incompatible dependency: {exc}\n'
                       'Install the dependencies described in README.md.\n')
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
