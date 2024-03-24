import os
from typing import Tuple, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class AerialDroneSemanticSegmentationDataset(Dataset):

    def __init__(
        self,
        root_path,
        transforms=None,
    ) -> None:
        self.root_path = root_path
        self.transforms = transforms

        self.image_paths, self.masks_paths = self._get_paths()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = Image.open((self.image_paths[index]))
        mask = Image.open((self.masks_paths[index]))

        torch_image = ToTensor()(image)
        torch_mask = ToTensor()(mask)

        if self.transforms is not None:
            for transform in self.transforms:
                torch_image = transform(torch_image)
                torch_mask = transform(torch_mask)

        return (torch_image, torch_mask)

    def _get_paths(self) -> Tuple[List[str], List[str]]:

        images_paths = []
        masks_paths = []

        for root, _, files in os.walk(self.root_path):
            for file in files:
                file_path = f"{root}/{file}"
                # differentiating between images and masks is easy
                # because masks are *.png and images are *.jpg
                if file.endswith(".png"):
                    masks_paths.append(file_path)
                elif file.endswith(".jpg"):
                    images_paths.append(file_path)

        return (images_paths, masks_paths)
