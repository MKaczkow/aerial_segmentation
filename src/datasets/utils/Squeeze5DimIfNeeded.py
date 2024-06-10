from torch.nn import Module


class Squeeze5DimIfNeeded(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        tensor = tensor.squeeze(0)
        return tensor
