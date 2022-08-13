import unittest

import os

import numpy as np

from disentanglement_datasets import MPI3DToy


@unittest.skipUnless(os.getenv("RUN_BIG_TESTS"), "don't run tests for big datasets")
class TestMPI3DToy(unittest.TestCase):
    def setUp(self):
        self.dataset = MPI3DToy(root="data", download=True)

    def test_download(self):
        dataset = MPI3DToy(root="data", download=True)
        downloaded_path = dataset.resource_path("images")
        self.assertTrue(os.path.exists(downloaded_path), "dataset did not download")

    def test_length(self):
        self.assertEqual(len(self.dataset), 1036800)

    def test_shapes(self):
        item = self.dataset[0]
        for key in item:
            self.assertEqual(item[key].shape, self.dataset.shapes[key])
