from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import f1_score


class SimpsonsClassifier(pl.LightningModule):
    """
    Module for training and evaluation models
    for the classification task
    """

    def __init__(self, model, mode):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.mode = mode
        self.criterion = nn.CrossEntropyLoss()
        self.metric = f1_score

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use.
        """
        if self.mode == "Base":
            optimizer = torch.optim.Adam(self.parameters())
            return {"optimizer": optimizer}
        elif self.mode == "Better":
            optimizer = torch.optim.Adam(self.parameters())
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.5)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise ValueError("Mode should be one of 'Base' or 'Better'.")

    def training_step(self, batch: Any):
        """
        Here you compute and return the training loss for e.g. the progress bar or logger.

        Args:
            batch: The output of Class `~torch.utils.data.DataLoader`.
                A tensor, tuple or list.

        Return:
            The loss
        """

        input, label = batch
        output = self.forward(input)
        loss = self.criterion(output, label)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: any):
        """
        Operates on a single batch of data from the validation set.
        In this step calculate accuracy.

        Args:
            batch: The output of Class`~torch.utils.data.DataLoader`.

        Return:
            dict: A dictionary. Include keys "val_loss" and "val_acc"
            with loss and accuracy in validation step.
        """
        input, label = batch
        output = self.forward(input)
        loss = self.criterion(output, label)

        pred = torch.argmax(output, 1)
        acc = (pred == label).float().mean()

        metric = self.metric(pred, label)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_F1", metric, prog_bar=True, on_step=False, on_epoch=True)

        return {"test_loss": loss, "test_acc": acc, "test_F1": metric}

    def test_step(self, batch: any):
        """
        Operates on a single batch of data from the test set.
        In this calculate accuracy and F1-score.

        Args:
            batch: The output of Class`~torch.utils.data.DataLoader`.

        Return:
            dict: A dictionary. Include keys "test_loss", "test_acc",
            "test_F1" with loss, accuracy and F1-score in test step.
        """
        pass

    def predict_step(self, input: Any):
        """
        Step function called during :meth:`Trainer.predict`.

        Args:
            batch: Current batch.

        Return:
            Predicted output
        """
        logit = self.forward(input)
        return torch.nn.functional.softmax(logit, dim=-1).numpy()
