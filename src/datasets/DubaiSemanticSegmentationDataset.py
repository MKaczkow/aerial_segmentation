import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, ToTensor


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
        print(self.masks_paths[index])
        mask = Image.open(self.masks_paths[index], mode='P')
        print(mask.size)
        print(mask.mode)
        print(mask.getextrema())
        print(type(mask))

        mask = mask.convert('P')
        print(mask.size)
        print(mask.mode)
        print(mask.getextrema())
        print(type(mask))
        # for some weird reason, this is needed ^

        torch_image = ToTensor()(image)
        pil_to_tensor_transform = PILToTensor()
        torch_mask = pil_to_tensor_transform(mask)
        # torch_mask = PILToTensor()(mask)

        import numpy as np
        print(type(torch_mask))
        print(torch_mask.max())
        print(torch_mask.min())
        unique, counts = np.unique(torch_mask.to('cpu'), return_counts=True)
        print(dict(zip(unique, counts)))

        if self.transforms is not None:
            print("Transforming")
            for transform in self.transforms:
                torch_image = transform(torch_image)
                torch_mask = transform(torch_mask)
        else:
            print("No Transform")
            
        print(type(torch_mask))
        print(torch_mask.max())
        print(torch_mask.min())
        unique, counts = np.unique(torch_mask.to('cpu'), return_counts=True)
        print(dict(zip(unique, counts)))

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
