from pytorch_lightning.callbacks import Callback
import random
import os
import matplotlib.pyplot as plt


class SaveTestPredsDubai(Callback):
    def __init__(self, save_dir, num_samples=4):
        super().__init__()
        self.save_dir = save_dir
        self.num_samples = num_samples
        os.makedirs(self.save_dir, exist_ok=True)

    def on_test_epoch_end(self, trainer, pl_module):
        batch = next(iter(trainer.test_dataloaders))
        images, masks = batch
        print('images')
        print(images.shape)
        print('masks')
        print(masks.shape)

        num_samples = min(self.num_samples, len(images))

        for i in range(num_samples):
            idx = random.randint(0, len(images) - 1)
            image, mask = images[idx], masks[idx]
            pred = pl_module(image.unsqueeze(0))
            print('pred')
            print(pred.shape)

            self.save_image_mask_pred(image, mask, pred, trainer.current_epoch, idx)

    def save_image_mask_pred(self, image, mask, pred, epoch, idx):
        image = image.cpu().numpy().transpose(1, 2, 0)
        mask = mask.cpu().numpy()

        # for Dubai dataset (and possibly others, with non-binary problem)
        if pred.shape[1] > 1:
            pred = pred.argmax(dim=1)

        _, ax = plt.subplots(1, 3)
        ax[0].imshow(image)
        ax[0].set_title("Image")
        ax[0].axis("off")

        ax[1].imshow(mask.squeeze(), cmap="gray")
        ax[1].set_title("Mask")
        ax[1].axis("off")

        ax[2].imshow(pred.squeeze().cpu(), cmap="gray")
        ax[2].set_title("Pred")
        ax[2].axis("off")

        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch}_idx_{idx}.png"))
        plt.close()
