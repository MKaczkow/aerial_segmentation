import os
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor, RandomCrop, ToTensor


class INRIAAerialImageLabellingDataset(Dataset):

    def __init__(
        self, root_path, transforms=None, split="train", image_size: int = 576
    ) -> None:
        self.root_path = root_path
        self.split = split
        self.image_size = image_size

        if split not in ["train", "test"]:
            raise ValueError("split must be one of [train, test]")

        self.root_path = os.path.join(root_path, split)
        self.transforms = transforms
        self.image_paths, self.gt_paths = self._get_paths()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        image = Image.open((self.image_paths[index]))
        torch_image = ToTensor()(image)

        if self.transforms is not None:
            for transform in self.transforms:
                # apply transforms to both images simultaneously,
                # so that random transforms (like RandomCrop) are consistent
                # https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914/3?u=ssgosh
                if isinstance(transform, RandomCrop):
                    i, j, h, w = transform.get_params(
                        torch_image, output_size=(self.image_size, self.image_size)
                    )
                    torch_image = TF.crop(torch_image, i, j, h, w)
                else:
                    torch_image = transform(torch_image)

        if self.split != "test":
            gt = Image.open((self.gt_paths[index]))
            torch_gt = PILToTensor()(gt)

            if self.transforms is not None:
                for transform in self.transforms:
                    # apply transforms to both images simultaneously,
                    # if random transforms (like RandomCrop) are used
                    # else apply transforms normally
                    if isinstance(transform, RandomCrop):
                        torch_gt = TF.crop(torch_gt, i, j, h, w)
                    else:
                        torch_gt = transform(torch_gt)

            return (torch_image, torch_gt)

        return torch_image

    def _get_paths(self) -> Tuple[List[str], List[str]]:

        images_paths = []
        gt_paths = []

        for root, _, files in os.walk(self.root_path):
            for file in files:
                file_path = f"{root}/{file}"
                if file.endswith(".tif"):
                    if "gt" in file_path:
                        gt_paths.append(file_path)
                    elif "images" in file_path:
                        images_paths.append(file_path)
                    else:
                        raise ValueError("Unknown file path")

        return (images_paths, gt_paths)
