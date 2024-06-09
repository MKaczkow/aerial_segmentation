import os
import random

import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback


class SaveTestPreds(Callback):
    def __init__(self, save_dir, num_samples=4):
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.test_dataloaders))
        images = batch

        num_samples = min(self.num_samples, len(images))

        for _ in range(num_samples):
            idx = random.randint(0, len(images) - 1)
            image = images[idx]
            pred = pl_module(image.unsqueeze(0))

            self.save_image_mask_pred(image, pred, trainer.current_epoch, idx)

    def save_image_mask_pred(self, image, pred, epoch, idx):
        image = image.cpu().numpy().transpose(1, 2, 0)

        _, ax = plt.subplots(1, 2)
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(pred.squeeze().cpu(), cmap="gray")
        ax[1].set_title("Pred")
        ax[1].axis("off")

        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch}_idx_{idx}.png"))
        plt.close()
