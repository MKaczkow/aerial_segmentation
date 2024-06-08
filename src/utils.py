import csv
import torch


def view_and_save_images_shapes(data_loader: torch.utils.data.DataLoader=None, filename: str='image_shapes', verbose: bool=False) -> None:
    """Utility function to view and save the shapes of images and masks.

    Args:
        data_loader (torch.utils.data.DataLoader): DataLoader to iterate over.
    """    
    print("Opening file", filename)
    with open(f'{filename}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if verbose:
            print("Saving image shapes to CSV file...", file)

        writer.writerow(['Batch Index', 'Image Shape', 'Mask Shape'])  # Write header
        if verbose:
            print("Header written to CSV file...", file)

        # Iterate over the DataLoader
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
                print("Writting rov", batch_idx)

    print(f"Image shapes saved to {filename}.csv")