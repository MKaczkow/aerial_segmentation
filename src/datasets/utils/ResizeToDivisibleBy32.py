from torch.nn import Module
from torchvision.transforms import Resize


class ResizeToDivisibleBy32(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        height = tensor.shape[1]
        width = tensor.shape[2]

        new_height = (height + 31) // 32 * 32
        new_width = (width + 31) // 32 * 32

        transform = Resize([new_height, new_width])
        return transform(tensor)
