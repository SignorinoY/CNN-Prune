import os

import torch
from lightning import LightningDataModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm

from src.utils.pruning import (
    apply_pruning,
    filter_parameters_to_prune,
    log_sparsity_stats,
    remove_pruning
)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        learning_rate: float = 0.001,
        patience: int = 5,
        max_epochs: int = 25,
        device: str = "cpu",
        dirpath: str = "checkpoints/",
    ):
        device = torch.device(device)
        self.device = device

        self.model = model.to(device)

        self.criterion = criterion
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, mode="min", patience=min(patience // 2 + 1, 3)
        )
        self.patience = patience
        self.max_epochs = max_epochs
        self.dirpath = dirpath

        # metric objects for calculating and averaging accuracy across batches
        num_classes = model.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

        # for averaging loss across batches
        self.train_loss = MeanMetric().to(device)
        self.val_loss = MeanMetric().to(device)
        self.test_loss = MeanMetric().to(device)

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def step(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        outputs = self.model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = self.criterion(outputs, labels)
        return loss, preds, labels

    def training_step(self, batch):
        loss, preds, labels = self.step(batch)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.train_loss(loss)
        self.train_acc(preds, labels)

    def validation_step(self, batch):
        loss, preds, labels = self.step(batch)

        self.val_loss(loss)
        self.val_acc(preds, labels)

    def test_step(self, batch):
        loss, preds, labels = self.step(batch)

        self.test_loss(loss)
        self.test_acc(preds, labels)

    def fit(self, datamodule: LightningDataModule):
        datamodule.prepare_data()
        datamodule.setup(stage="fit")

        train_steps = len(datamodule.train_dataloader())
        val_steps = len(datamodule.val_dataloader())

        n_epoch_no_improvement = 0
        val_acc_best = 0

        train_pbar = tqdm(
            range(train_steps),
            postfix={
                "train/loss": 0.0,
                "train/acc": 0.0,
            },
        )
        val_pbar = tqdm(
            range(val_steps),
            postfix={
                "val/loss": 0.0,
                "val/acc": 0.0,
                "val/acc_best": 0.0,
                "no_improve": 0,
            },
        )

        for epoch in range(self.max_epochs):
            train_pbar.set_description(f"{epoch + 1}/{self.max_epochs}")

            self.model.train()
            for batch in datamodule.train_dataloader():
                self.training_step(batch)
                train_pbar.update()

            train_pbar.set_postfix(
                {
                    "train/loss": self.train_loss.compute().item(),
                    "train/acc": self.train_acc.compute().item(),
                }
            )
            train_pbar.reset()

            self.train_loss.reset()
            self.train_acc.reset()

            val_pbar.set_description(f"{epoch + 1}/{self.max_epochs}")
            self.model.eval()
            for batch in datamodule.val_dataloader():
                self.validation_step(batch)
                val_pbar.update()

            val_loss = self.val_loss.compute()
            val_acc = self.val_acc.compute()
            self.scheduler.step(val_loss)
            self.val_acc_best(val_acc)

            if val_acc.item() > val_acc_best:
                n_epoch_no_improvement = 0
                self.save()
            else:
                n_epoch_no_improvement += 1
            val_acc_best = self.val_acc_best.compute().item()

            val_pbar.set_postfix(
                {
                    "val/loss": val_loss.item(),
                    "val/acc": val_acc.item(),
                    "val/acc_best": val_acc_best,
                    "no_improve": n_epoch_no_improvement,
                }
            )
            val_pbar.reset()
            self.val_loss.reset()
            self.val_acc.reset()

            if n_epoch_no_improvement > self.patience:
                break

        train_pbar.close()
        val_pbar.close()
        datamodule.teardown(stage="fit")

    def test(self, datamodule: LightningDataModule):
        self.load(os.path.join(self.dirpath, "last.pth"))
        datamodule.setup(stage="test")
        test_steps = len(datamodule.test_dataloader())

        pbar = tqdm(range(test_steps), leave=False)

        self.model.eval()
        for batch in datamodule.test_dataloader():
            self.test_step(batch)
            pbar.update()

        datamodule.teardown(stage="test")
        pbar.close()
        return {
            "test/loss": self.test_loss.compute().item(),
            "test/acc": self.test_acc.compute().item(),
        }

    def prune(self, amount: float = 0.5, type: str = "unstructured"):
        parameters_to_prune = filter_parameters_to_prune(self.model)
        apply_pruning(parameters_to_prune, amount=amount, type=type)
        log_sparsity_stats(parameters_to_prune)
        remove_pruning(self.model)

    def load(self, ckpt_path: str):
        self.model.load_state_dict(torch.load(ckpt_path))

    def save(self):
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        torch.save(self.model.state_dict(), os.path.join(self.dirpath, "last.pth"))
