import os
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor, RandomCrop


class AerialDroneSemanticSegmentationDataset(Dataset):

    def __init__(
        self,
        root_path,
        transforms=None,
        image_height: int = 400,
        image_width: int = 600,
    ) -> None:
        self.root_path = root_path
        self.transforms = transforms
        self.image_height = image_height
        self.image_width = image_width

        self.image_paths, self.masks_paths = self._get_paths()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = Image.open((self.image_paths[index]))
        mask = Image.open((self.masks_paths[index]))

        torch_image = ToTensor()(image)
        torch_mask = PILToTensor()(mask)

        if self.transforms is not None:
            for transform in self.transforms:
                # apply transforms to both images simultaneously,
                # so that random transforms (like RandomCrop) are consistent
                # https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914/3?u=ssgosh
                if isinstance(transform, RandomCrop):
                    i, j, h, w = transform.get_params(
                        torch_image, output_size=(self.image_height, self.image_width)
                    )
                    torch_image = TF.crop(torch_image, i, j, h, w)
                else:
                    torch_image = transform(torch_image)

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
