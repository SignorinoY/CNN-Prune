from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, gray_scale=False):
        super().__init__()
        self.num_classes = num_classes
        self.gray_scale = gray_scale

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if gray_scale:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
