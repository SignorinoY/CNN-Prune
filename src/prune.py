import lightning.pytorch as pl
import rootutils
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data import MNISTDataModule
from src.models.components import LeNet5
from src.utils.pruning import (
    apply_pruning,
    filter_parameters_to_prune,
    log_sparsity_stats,
    remove_pruning,
)
from src.utils.utils import eval, train

pl.seed_everything(42)


dm = MNISTDataModule(batch_size=64)

num_classes = dm.num_classes
model = LeNet5(
    num_classes=num_classes,
    grayscale=True,
)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3)

train(
    model=model,
    datamodule=dm,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=2,
)
eval(model=model, datamodule=dm, criterion=criterion)
amount = 0.9
total_steps = 10

amount_per_step = 1 - (1 - amount) ** (1 / total_steps)

for _ in range(total_steps):
    parameters_to_prune = filter_parameters_to_prune(model)
    apply_pruning(parameters_to_prune, amount=amount_per_step)
    log_sparsity_stats(parameters_to_prune)

    train(
        model=model,
        datamodule=dm,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=2,
    )
    eval(model=model, datamodule=dm, criterion=criterion)
remove_pruning(model)
