from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes=10, gray_scale=False):
        super().__init__()
        self.num_classes = num_classes
        self.gray_scale = gray_scale

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        if gray_scale:
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
