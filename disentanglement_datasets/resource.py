"""Web-to-disk model."""

from dataclasses import dataclass

import os

from torchvision.datasets.utils import check_integrity, download_url


@dataclass
class Resource:
    """
    Represents a downloadable web resource.
    """

    filename: str
    url: str
    md5: str

    def validate(self, base_path):
        """
        Validate that the resource exists.

        Params:
            base_path: The directory in which the resource should exist.

        Returns:
            A boolean indicating whether the resource exists and passes the
            integrity check.
        """
        file_path = os.path.join(base_path, self.filename)
        if not os.path.exists(file_path):
            return False
        if not self._validate_md5(file_path, self.md5):
            return False

        return True

    def _validate_md5(self, filepath, md5):
        return check_integrity(filepath, md5)

    def download(self, base_path):
        """
        Download this resource.

        Params:
            base_path: The directory to download the resource into.
        """
        download_url(self.url, base_path, self.filename, self.md5)
