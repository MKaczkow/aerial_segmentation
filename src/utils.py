import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image


def view_and_save_images_shapes(
    data_loader: torch.utils.data.DataLoader = None,
    filename: str = "image_shapes",
    verbose: bool = False,
    train: bool = True,
) -> None:
    """Utility function to view and save the shapes of images and masks.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader to iterate over.
        train (bool): If True, the DataLoader should return (images, masks) pairs.
    """
    print("Opening file", filename)
    with open(f"{filename}.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if verbose:
            print("Saving image shapes to CSV file...", file)

        writer.writerow(["Batch Index", "Image Shape", "Mask Shape"])  # Write header
        if verbose:
            print("Header written to CSV file...", file)

        # Iterate over the DataLoader
        if train:
            for batch_idx, (images, masks) in enumerate(data_loader):
                if verbose:
                    print(f"Batch Index: {batch_idx}")
                    print(f"Image Shape: {images.shape}")
                    print(f"Mask Shape: {masks.shape}")
                # Get the shape of images in the current batch
                image_shape = list(images.shape)
                mask_shape = list(masks.shape)

                # Write the shape to the CSV file
                writer.writerow([batch_idx, image_shape, mask_shape])
                if verbose:
                    print("Writting row", batch_idx)
        else:
            for batch_idx, images in enumerate(data_loader):
                if verbose:
                    print(f"Batch Index: {batch_idx}")
                    print(f"Image Shape: {images.shape}")
                # Get the shape of images in the current batch
                image_shape = list(images.shape)

                # Write the shape to the CSV file
                writer.writerow([batch_idx, image_shape])
                if verbose:
                    print("Writting row", batch_idx)

    print(f"Image shapes saved to {filename}.csv")


def save_patch_inria(patch, path, prefix, index):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{prefix}_{index}.png")
    save_image(patch, filename)


def save_patch_uavid(patch, path, prefix, index):
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"{prefix}_{index}.png")
    array = patch.numpy().astype(np.uint8)
    print("save patch uavid")
    print("array")
    print(type(array))
    print(array.shape)
    print(array.dtype)
    unique, counts = np.unique(array, return_counts=True)
    print(dict(zip(unique, counts)))
    print()
    Image.fromarray(array.squeeze()).save(filename)


def pad_to_patch_size(image, mask, patch_size):
    _, img_height, img_width = image.shape
    patch_height, patch_width = patch_size

    # Calculate padding sizes
    pad_height = (patch_height - img_height % patch_height) % patch_height
    pad_width = (patch_width - img_width % patch_width) % patch_width

    # Apply padding
    padding = (0, pad_width, 0, pad_height)  # (left, right, top, bottom)
    padded_image = F.pad(image, padding, mode="constant", value=0)
    if mask is not None:
        padded_mask = F.pad(mask, padding, mode="constant", value=0)
        return padded_image, padded_mask

    return padded_image, None


def slice_and_save_uavid(
    image,
    mask,
    patch_size=(500, 500),
    image_save_path="image_patches",
    mask_save_path="mask_patches",
    prefix="patch",
    train=True,
):
    # Pad image and mask to ensure they can be evenly divided into patches
    print("slice and save uavid")
    # print("image")
    # print(type(image))
    # print(image.shape)
    # print(image.dtype)
    print("mask")
    print(type(mask))
    print(mask.shape)
    print(mask.dtype)
    unique, counts = torch.unique(mask, return_counts=True)
    print(dict(zip(unique, counts)))

    if train:
        image, mask = pad_to_patch_size(image, mask, patch_size)
        print("mask after padding")
        print(type(mask))
        print(mask.shape)
        print(mask.dtype)
        unique, counts = torch.unique(mask, return_counts=True)
        print(dict(zip(unique, counts)))
    else:
        image, _ = pad_to_patch_size(image, None, patch_size)

    # Unpack the patch size
    patch_height, patch_width = patch_size

    # Get the dimensions of the image and mask
    _, img_height, img_width = image.shape

    # Calculate the number of patches along each dimension
    num_patches_vertical = img_height // patch_height
    num_patches_horizontal = img_width // patch_width

    patch_index = 0

    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # Calculate the starting and ending indices for the current patch
            start_h = i * patch_height
            end_h = start_h + patch_height
            start_w = j * patch_width
            end_w = start_w + patch_width

            # Slice the image and mask to create the patch
            image_patch = image[:, start_h:end_h, start_w:end_w]
            if train:
                mask_patch = mask[:, start_h:end_h, start_w:end_w]

            # Save the patches
            # not a mistake, just handle normal data
            save_patch_inria(
                image_patch, image_save_path, f"{prefix}_image", patch_index
            )
            if train:
                print("mask_patch")
                print(type(mask_patch))
                print(mask_patch.shape)
                print(mask_patch.dtype)
                unique, counts = torch.unique(mask_patch, return_counts=True)
                print(dict(zip(unique, counts)))
                print()
                save_patch_uavid(
                    mask_patch, mask_save_path, f"{prefix}_mask", patch_index
                )

            patch_index += 1


def slice_and_save_inria(
    image,
    mask,
    patch_size=(500, 500),
    image_save_path="image_patches",
    mask_save_path="mask_patches",
    prefix="patch",
    train=True,
):
    # Unpack the patch size
    patch_height, patch_width = patch_size

    # Get the dimensions of the image and mask
    print("slice and save")
    print("image")
    print(type(image))
    print(image.shape)
    print(image.dtype)
    _, img_height, img_width = image.shape

    # Calculate the number of patches along each dimension
    num_patches_vertical = img_height // patch_height
    num_patches_horizontal = img_width // patch_width

    patch_index = 0

    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # Calculate the starting and ending indices for the current patch
            start_h = i * patch_height
            end_h = start_h + patch_height
            start_w = j * patch_width
            end_w = start_w + patch_width

            # Slice the image and mask to create the patch
            image_patch = image[:, start_h:end_h, start_w:end_w]
            if train:
                mask_patch = mask[:, start_h:end_h, start_w:end_w]

            # Save the patches
            save_patch_inria(
                image_patch, image_save_path, f"{prefix}_image", patch_index
            )
            if train:
                save_patch_inria(
                    mask_patch, mask_save_path, f"{prefix}_mask", patch_index
                )

            patch_index += 1


def process_inria_dataloader_and_save(
    dataloader,
    patch_size=(500, 500),
    image_save_path="image_patches",
    mask_save_path="mask_patches",
    train=True,
):

    if train:
        for batch_idx, (image, mask) in enumerate(dataloader):
            for img_idx in range(image.size(0)):  # iterate over batch dimension
                # print(type(mask))
                # print(mask.shape)
                # print(mask.dtype)
                # unique, counts = torch.unique(mask, return_counts=True)
                # print(dict(zip(unique, counts)))
                slice_and_save_inria(
                    image[img_idx],
                    torch.div(mask[img_idx], 255),
                    patch_size,
                    image_save_path,
                    mask_save_path,
                    prefix=f"batch_{batch_idx}_img_{img_idx}",
                )
    else:
        for batch_idx, image in enumerate(dataloader):
            for img_idx in range(image.size(0)):  # iterate over batch dimension
                # print(type(mask))
                # print(mask.shape)
                # print(mask.dtype)
                # unique, counts = torch.unique(mask, return_counts=True)
                # print(dict(zip(unique, counts)))
                slice_and_save_inria(
                    image[img_idx],
                    None,
                    patch_size,
                    image_save_path,
                    None,
                    prefix=f"batch_{batch_idx}_img_{img_idx}",
                    train=train,
                )


def process_uavid_dataloader_and_save(
    dataloader,
    patch_size=(500, 500),
    image_save_path="image_patches",
    mask_save_path="mask_patches",
    train=True,
):
    if train:
        for batch_idx, (image, mask) in enumerate(dataloader):
            for img_idx in range(image.size(0)):  # iterate over batch dimension
                print()
                print("mask[img_idx]")
                print(type(mask[img_idx]))
                print(mask[img_idx].shape)
                print(mask[img_idx].dtype)
                unique, counts = torch.unique(mask[img_idx], return_counts=True)
                print(dict(zip(unique, counts)))
                slice_and_save_uavid(
                    image[img_idx],
                    mask[img_idx],
                    patch_size,
                    image_save_path,
                    mask_save_path,
                    prefix=f"batch_{batch_idx}_img_{img_idx}",
                    train=train,
                )
        else:
            for batch_idx, image in enumerate(dataloader):
                for img_idx in range(image.size(0)):  # iterate over batch dimension
                    slice_and_save_uavid(
                        image[img_idx],
                        None,
                        patch_size,
                        image_save_path,
                        mask_save_path,
                        prefix=f"batch_{batch_idx}_img_{img_idx}",
                        train=train,
                    )
