import unittest

import os

import numpy as np

from disentanglement_datasets import DSprites


class TestDSprites(unittest.TestCase):
    def setUp(self):
        self.dataset = DSprites(root="data", download=True)

    def test_download(self):
        dataset = DSprites(root="data", download=True)
        downloaded_path = dataset.resource_path("dataset")
        self.assertTrue(os.path.exists(downloaded_path), "dataset did not download")

    def test_length(self):
        self.assertEqual(len(self.dataset), 737280)

    def test_shapes(self):
        item = self.dataset[0]
        for key in item:
            self.assertEqual(item[key].shape, self.dataset.shapes[key])

    def test_corrupted_fails_validation(self):
        downloaded_path = self.dataset.resource_path("dataset")
        np.savez(downloaded_path, bad=np.array([1., 2., 3.]))
        with self.assertRaises(RuntimeError):
            dataset = DSprites(root="data", download=False)
