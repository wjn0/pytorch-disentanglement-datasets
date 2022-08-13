"""Base dataset."""

from typing import Callable, Dict

from abc import abstractmethod

import os

import torch
from torch.utils.data import Dataset


class BaseDisentanglementDataset(Dataset):
    """
    Handles the downloading and verification of resources, and transforming
    elements of the dataset.
    """

    # Downloadable resources. Maps the name of the resource to the Resource
    # object.
    resources = {}

    def __init__(
        self, root: str, transform: Callable = None, download: bool = False
    ) -> None:
        """
        Params:
            root: The root directory of all datasets.
            transform: A Callable that will transform each item in the dataset
                       when indexed.
            download: Whether to download the dataset if it's not available in
                      `root`.
        """
        self.root = root
        self.transform = transform

        # Validate resources, downloading them if necessary.
        self.validate_resources(download=download)

    @abstractmethod
    def length(self):
        """Length of the dataset, should be overridden in child."""

    @abstractmethod
    def get_item(self, idx):
        """Get the `idx`th item of the dataset, should be overridden in child."""

    def validate_resources(self, download: bool) -> None:
        """
        Validate the existence and integrity of resources.

        Params:
            download: Whether to download the resources if they don't exist or
                      the integrity check fails.
        """
        for resource in self.resources.values():
            if not resource.validate(self.path):
                if download:
                    resource.download(self.path)
                else:
                    raise RuntimeError(
                        "Resource for dataset not available, pass download=True"
                    )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Gets and transforms the item.

        Always returns a dictionary mapping strings to tensors.

        Params:
            idx: The index in the dataset, must be less than the length.

        Returns:
            item: The transformed element of the dataset.
        """
        item = self.get_item(idx)

        if self.transform is not None:
            item = self.transform(item)

        return item

    def __len__(self) -> int:
        """The length of the dataset."""
        return self.length()

    def resource_path(self, resource_name):
        """The path to the resource with the given name on disk."""
        return os.path.join(self.path, self.resources[resource_name].filename)

    @property
    def path(self):
        """The path to this dataset's resources."""
        return os.path.join(self.root, self.__class__.__name__)
