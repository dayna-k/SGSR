"""CPU checks for the image entrypoint; no trained weights or downloads needed."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from inference import (check_checkpoint_file, collect_jobs, image_to_tensor,
                       load_weights, super_resolve, tensor_to_image)
from model.sgsrnet import GeneratorRRDB_SGSR
from SalUNet.unet_new import UNetAttenModel


class InferenceTests(unittest.TestCase):
    def test_bgr_normalization_roundtrip(self):
        # Distinct blue/red values expose accidental RGB channel conversion.
        image = np.array([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8)
        tensor = image_to_tensor(image, torch.device('cpu'))
        self.assertEqual(tuple(tensor.shape), (1, 3, 1, 2))
        self.assertEqual(tensor[0, 0, 0, 0].item(), -1)
        self.assertEqual(tensor[0, 2, 0, 0].item(), 1)
        np.testing.assert_allclose(tensor_to_image(tensor), image, atol=1)

    def test_wrapped_and_plain_weights_preserve_unprefixed_keys(self):
        weights = {'weight': torch.tensor([[1., 2.]])}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            for payload in (weights, {'model': {'module.weight': weights['weight']}},
                            {'state_dict': weights}):
                torch.save(payload, path)
                result = load_weights(path)
                self.assertEqual(list(result), ['weight'])
                torch.testing.assert_close(result['weight'], weights['weight'])

    def test_odd_image_runs_real_network_and_crops_to_exact_x4(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        generator = GeneratorRRDB_SGSR(3, 64, num_res_blocks=1).eval()
        saliency = UNetAttenModel().eval()
        image = np.zeros((5, 7, 3), dtype=np.uint8)
        output = super_resolve(image, generator, saliency, torch.device('cpu'))
        self.assertEqual(output.shape, (20, 28, 3))
        self.assertEqual(output.dtype, np.uint8)

    def test_jobs_reject_collisions_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'input'
            output = Path(folder) / 'output'
            source.mkdir()
            output.mkdir()
            (source / 'sample.png').touch()
            (source / 'sample.jpg').touch()
            with self.assertRaisesRegex(ValueError, 'share a filename stem'):
                collect_jobs(source, output)
            (source / 'sample.jpg').unlink()
            (output / 'sample_x4.png').touch()
            with self.assertRaisesRegex(ValueError, 'already exists'):
                collect_jobs(source, output)
            self.assertEqual(len(collect_jobs(source, output, overwrite=True)), 1)
            with self.assertRaisesRegex(ValueError, 'separate input and output'):
                collect_jobs(source, source)

    def test_missing_and_lfs_checkpoint_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'weights.pt'
            with self.assertRaisesRegex(ValueError, 'Checkpoint not found'):
                check_checkpoint_file(path)
            path.write_text('version https://git-lfs.github.com/spec/v1\n')
            with self.assertRaisesRegex(ValueError, 'Git LFS pointer'):
                check_checkpoint_file(path)


if __name__ == '__main__':
    unittest.main()
