"""MPI3D datasets."""

from typing import Dict

import torch

import numpy as np

from .base import BaseDisentanglementDataset
from .resource import Resource


class MPI3DToy(BaseDisentanglementDataset):
    """
    A 3D dataset designed for evaluating disentangled representation learning
    algorithms.

    This dataset consists of simplistic simulated images.

    [1] https://github.com/rr-learning/disentanglement_dataset
    """

    resources = {
        "images": Resource(
            filename="mpi3d_toy.npz",
            url="https://storage.googleapis.com/disentanglement_dataset/Final_Dataset/mpi3d_toy.npz",
            md5="55889cb7c7dfc655d6e0277beee88868",
        )
    }

    shapes = {"input": (64, 64, 3), "latent": (7,)}

    # Correct shape for the leading dimensions to extract the underlying factors
    # of variation.
    _latent_factor_shape = [6, 6, 2, 3, 3, 40, 40]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.load_dataset()

    def load_dataset(self):
        """
        Load the numpy archive and convert to torch tensor.
        """
        self.images = torch.from_numpy(
            np.load(self.resource_path("images"))["images"].reshape(
                [*self._latent_factor_shape, *self.shapes["input"]]
            )
        )

    def length(self):
        """Compute the length."""
        return np.prod(self._latent_factor_shape)

    def get_item(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get the `idx`th image and latent factor values.
        """
        shaped_idx = np.unravel_index(idx, self._latent_factor_shape)

        return {
            "input": self.images[tuple(shaped_idx)],
            "latent": torch.from_numpy(np.array(shaped_idx, dtype=int)),
        }
