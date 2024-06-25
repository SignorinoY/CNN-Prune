import logging
import os

import torch
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm


def train(
    model,
    datamodule,
    criterion,
    optimizer,
    scheduler,
    num_epochs: int = 25,
    dirpath: str = "checkpoints",
    device: str = "mps",
):
    device = torch.device(device)

    num_classes = datamodule.num_classes
    # metric objects for calculating and averaging accuracy across batches
    train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    # for averaging loss across batches
    train_loss = MeanMetric().to(device)
    val_loss = MeanMetric().to(device)

    # for tracking best so far validation accuracy
    val_acc_best = MaxMetric()

    model.to(device)
    datamodule.setup(stage="fit")

    train_steps = len(datamodule.train_dataloader())
    val_steps = len(datamodule.val_dataloader())

    pbar = tqdm(
        range(train_steps),
        postfix={
            "train/loss": 0.0,
            "val/acc": 0.0,
            "val/acc_best": 0.0,
        },
        leave=False,
    )

    for epoch in range(num_epochs):
        pbar.set_description(f"{epoch + 1}/{num_epochs}")
        model.train()

        for batch in datamodule.train_dataloader():
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss(loss)
            train_acc(preds, labels)

            pbar.update()
            if epoch > 0:
                pbar.set_postfix(
                    {
                        "train/loss": train_loss.compute().item(),
                        "val/acc": val_acc.compute().item(),
                        "val/acc_best": val_acc_best.compute().item(),
                    }
                )
            else:
                pbar.set_postfix({"train/loss": train_loss.compute().item()})
        pbar.reset()

        model.eval()
        for batch in datamodule.val_dataloader():
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_loss(loss)
            val_acc(preds, labels)

        acc = val_acc.compute()

        val_acc_best(acc)
        scheduler.step(acc)

    print("Training complete.")

    datamodule.teardown(stage="fit")


def eval(model, datamodule, criterion, device: str = "mps"):
    device = torch.device(device)
    model.to(device)
    datamodule.setup(stage="test")

    test_acc = Accuracy(task="multiclass", num_classes=datamodule.num_classes).to(device)
    test_loss = MeanMetric().to(device)

    model.eval()
    for batch in datamodule.test_dataloader():
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        test_loss(loss)
        test_acc(preds, labels)

    print(f"Test accuracy: {test_acc.compute().item()}")
    datamodule.teardown(stage="test")
