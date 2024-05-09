import torch


class BaselineModel(torch.nn.Module):
    """Model generating random predictions with shape given by number of classes.

    Number of classes should be passed as an argument to the constructor.
    """

    def __init__(self, classes: int = 1):
        """Constructor for BaselineModel.

        Args:
            classes (int, optional): Number of classes to "predict" - implicates dim=1 for output tensor. Defaults to 1.
        """
        super().__init__()
        self.classes = classes

    def forward(self, x) -> torch.Tensor:
        return torch.rand(x.shape[0], self.classes, x.shape[2], x.shape[3])
