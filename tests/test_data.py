"""CPU checks for paired training data and historical preprocessing."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from data import SR_test, SR_train
from SalUNet.unet_dataloader import SODLoader


class TrainingDataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ('HR', 'LR', 'scores', 'scores_lr'):
            (self.root / folder).mkdir()
        self.write_pair('sample')
        self.args = SimpleNamespace(
            scale=4, nChannel=3, patchSize=64,
            GTdataDir=str(self.root / 'HR'), INdataDir=str(self.root / 'LR'),
            SEGdataDir=str(self.root / 'scores'),
            val_GTdataDir=str(self.root / 'HR'), val_INdataDir=str(self.root / 'LR'),
            val_SEGdataDir=str(self.root / 'scores'))

    def tearDown(self):
        self.tmp.cleanup()

    def write_pair(self, stem):
        for folder, size in [('HR', 64), ('LR', 16)]:
            image = np.zeros((size, size, 3), dtype=np.uint8)
            image[:] = (0, 128, 255)  # BGR
            cv2.imwrite(str(self.root / folder / (stem + '.png')), image)
        for folder, size in [('scores', 64), ('scores_lr', 16)]:
            cv2.imwrite(str(self.root / folder / (stem + '.png')),
                        np.full((size, size), 128, dtype=np.uint8))

    def saliency_loader(self):
        return SODLoader(image_dir=str(self.root / 'LR'),
                         mask_dir=str(self.root / 'scores_lr'), target_size=16)

    def test_nonnumeric_pair_and_bgr_preprocessing(self):
        (self.root / 'HR' / 'notes.txt').write_text('not an image')
        for dataset in (SR_train(self.args), SR_test(self.args)):
            hr, lr, score = dataset[0]
            self.assertEqual(len(dataset), 1)
            self.assertEqual(tuple(hr.shape), (3, 64, 64))
            self.assertEqual(tuple(lr.shape), (3, 16, 16))
            torch.testing.assert_close(lr[:, 0, 0], torch.tensor([-1., 128 / 127.5 - 1, 1.]))
            torch.testing.assert_close(score.mean(), torch.tensor(128 / 255.))

    def test_missing_pair_is_actionable(self):
        (self.root / 'LR' / 'sample.png').unlink()
        with self.assertRaisesRegex(ValueError, 'Cannot read image:.*sample.png'):
            SR_train(self.args)[0]

    def test_misaligned_score_map_rejected(self):
        cv2.imwrite(str(self.root / 'scores' / 'sample.png'), np.ones((32, 32), np.uint8))
        with self.assertRaisesRegex(ValueError, 'Score map must match HR dimensions'):
            SR_train(self.args)[0]

    def test_empty_scores_rejected_before_normalization(self):
        cv2.imwrite(str(self.root / 'scores' / 'sample.png'), np.zeros((64, 64), np.uint8))
        cv2.imwrite(str(self.root / 'scores_lr' / 'sample.png'), np.zeros((16, 16), np.uint8))
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            SR_train(self.args)[0]
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            self.saliency_loader()[0]

    def test_saliency_pairs_by_stem_and_keeps_rgb_normalization(self):
        self.write_pair('another')
        dataset = self.saliency_loader()
        self.assertEqual([Path(path).stem for path in dataset.inp_files], ['another', 'sample'])
        self.assertEqual([Path(path).stem for path in dataset.out_files], ['another', 'sample'])
        image, mask = dataset[0]
        expected = (torch.tensor([1., 128 / 255., 0.]) - torch.tensor([.485, .456, .406])) / torch.tensor([.229, .224, .225])
        torch.testing.assert_close(image[:, 0, 0], expected)
        torch.testing.assert_close(mask, torch.ones((1, 16, 16)))


if __name__ == '__main__':
    unittest.main()
