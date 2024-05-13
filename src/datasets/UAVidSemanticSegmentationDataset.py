import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, PILToTensor
from src.datasets.utils.ConvertUAVidMasks import ConvertUAVidMasks


class UAVidSemanticSegmentationDataset(Dataset):

    def __init__(self, root_path, transforms=None, split="train") -> None:
        self.root_path = root_path
        self.split = split

        if split not in ["train", "valid", "test"]:
            raise ValueError("split must be one of [train, valid, test]")

        self.root_path = os.path.join(
            root_path, split, split
        )  # weirdly, datapath contains this path twice
        self.transforms = transforms

        self.image_paths, self.masks_paths = self._get_paths()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = Image.open((self.image_paths[index]))
        torch_image = ToTensor()(image)

        if self.transforms is not None:
            for transform in self.transforms:
                torch_image = transform(torch_image)

        if self.split != "test":

            mask = Image.open((self.masks_paths[index]))
            pil_to_tensor_transform = PILToTensor()
            torch_mask = pil_to_tensor_transform(mask)

            convert_uavid_masks_transform = ConvertUAVidMasks()
            torch_mask = convert_uavid_masks_transform(torch_mask).unsqueeze(0)

            if self.transforms is not None:
                for transform in self.transforms:
                    torch_mask = transform(torch_mask)

            return (torch_image, torch_mask)

        return torch_image

    def _get_paths(self) -> Tuple[List[str], List[str]]:

        images_paths = []
        masks_paths = []

        for root, _, files in os.walk(self.root_path):
            for file in files:
                file_path = f"{root}/{file}"
                if file.endswith(".png"):
                    if "Labels" in file_path:
                        masks_paths.append(file_path)
                    elif "Images" in file_path:
                        images_paths.append(file_path)
                    else:
                        raise ValueError("Unknown file path")

        return (images_paths, masks_paths)
