"""DeepMind's DSprites dataset."""

from typing import Dict

import torch

import numpy as np

from .base import BaseDisentanglementDataset
from .resource import Resource


class DSprites(BaseDisentanglementDataset):
    """
    DSprites is a dataset designed for evaluating disentanglement models. It
    consists of three shapes which vary in position and rotation. The first
    latent factor, color, is constant (white).

    [1] https://github.com/deepmind/dsprites-dataset
    """

    # The entire dataset is offered in a numpy zip archive on GitHub.
    resources = {
        "dataset": Resource(
            filename="dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            url="https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true",
            md5="7da33b31b13a06f4b04a70402ce90c2e",
        )
    }

    shapes = {"input": (64, 64), "latent": (6,)}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.load_dataset()

    def load_dataset(self):
        """
        Load the numpy archive and convert to torch tensors.
        """
        raw = np.load(self.resource_path("dataset"))
        self.images = torch.from_numpy(raw["imgs"])
        self.latent_factors = torch.from_numpy(raw["latents_values"])

    def length(self):
        """Number of images."""
        return self.images.shape[0]

    def get_item(self, idx) -> Dict[str, torch.Tensor]:
        return {"input": self.images[idx, :], "latent": self.latent_factors[idx, :]}
