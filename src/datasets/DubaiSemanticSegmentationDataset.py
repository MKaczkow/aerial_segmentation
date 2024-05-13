import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor
from src.datasets.utils.ConvertDubaiMasks import ConvertDubaiMasks


class DubaiSemanticSegmentationDataset(Dataset):

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

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.masks_paths[index])

        pil_to_tensor_transform = PILToTensor()
        torch_mask = pil_to_tensor_transform(mask)

        if mask.mode != 'P':
            convert_dubai_masks_transform = ConvertDubaiMasks()
            torch_mask = convert_dubai_masks_transform(torch_mask).unsqueeze(0)

        to_tensor_transform = ToTensor()
        torch_image = to_tensor_transform(image)

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
