import torch
from torch.nn import Module


class ConvertDubaiMasks(Module):

    def __init__(self, color_classes_dict: dict | None = None):
        super().__init__()
        if color_classes_dict is not None:
            self.color_classes_dict = color_classes_dict
        else:
            self.color_classes_dict = {
                (60, 17, 152): 0,   # Building
                (132, 41, 246): 1,  # Land (unpaved area)
                (110, 193, 228): 2,  # Road
                (254, 221, 58): 3,  # Vegetation
                (226, 169, 41): 4,  # Water
                (155, 155, 155): 5  # Unlabeled
            }

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Function to map RGB values to class numbers
        def map_to_class(rgb_value):
            # Default to 0 if color not found
            return self.color_classes_dict.get(tuple(rgb_value.tolist()), 0)

        # Convert tensor to numpy array
        # Assuming tensor is in CHW format
        numpy_array = torch.permute(tensor, (1, 2, 0)).cpu().numpy()

        # Convert the image pixels using the color classes dictionary
        p_image_pixels = [map_to_class(rgb)
                          for rgb in numpy_array.reshape(-1, 3)]

        # Convert list to tensor
        p_tensor = torch.tensor(p_image_pixels, dtype=torch.uint8).reshape(
            tensor.size(1), tensor.size(2))

        return p_tensor
