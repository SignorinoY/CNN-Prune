import torch
from torch import nn


class LeNet5(nn.Module):
    """ "Implement the LeNet-5 architecture."""

    def __init__(self, num_classes: int = 10, gray_scale: bool = False) -> None:
        """Initialize a `LeNet5` module.

        :param num_classes: The number of classes in the dataset. Defaults to `10`.
        :param gray_scale: Whether to use grayscale images. Defaults to `False`.
        """
        super().__init__()

        self.gray_scale = gray_scale
        self.num_classes = num_classes

        if self.gray_scale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6 * in_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6 * in_channels, 16 * in_channels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5 * in_channels, 120 * in_channels),
            nn.ReLU(),
            nn.Linear(120 * in_channels, 84 * in_channels),
            nn.ReLU(),
            nn.Linear(84 * in_channels, num_classes),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    _ = LeNet5()
