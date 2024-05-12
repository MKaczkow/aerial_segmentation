import segmentation_models_pytorch as smp
import torch


def evaluate_result(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "binary",
    num_classes: int | None = None,
) -> dict:
    """Evaluate model output against target labels.

    Prediction is first squeezed and moved to CPU.
    Target is only squeezed.
    Both prediction and target tensors are then converted to integers to calculate metrics.
    Supported modes are "binary" and "multiclass".

    Args:
        pred (torch.Tensor): Model prediction tensor. Assuming shape (batch_size, num_classes, height, width).
        target (torch.Tensor): Target tensor.

    Returns:
        dict: Dictionary containing evaluation metrics.

    Raises:
        ValueError: If mode is not "binary" or "multiclass".
    """

    if mode not in ["binary", "multiclass"]:
        raise ValueError(
            f"Mode {mode} not supported. Choose either 'binary' or 'multiclass'."
        )

    if mode == "multiclass":
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred.squeeze().to("cpu").int(),
            target.squeeze().int(),
            mode=mode,
            num_classes=num_classes,
        )
    else:
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred.squeeze().to("cpu").int(), target.squeeze().int(), mode=mode
        )

    iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
    f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
    accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
    recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")

    return {"iou": iou_score, "f1": f1_score, "accuracy": accuracy, "recall": recall}
