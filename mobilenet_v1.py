import math
import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, depth_filter_size = (3,3), padding=1, alpha=1) -> None:
        super(DepthwiseSeparableConv, self).__init__()
        self.kernel_size = depth_filter_size
        in_channels = int(alpha * in_channels)
        out_channels = int(alpha * out_channels)

        pad = 'same' if stride == 1 else padding
        self.depth_conv = nn.Conv2d(in_channels, in_channels, depth_filter_size, padding=pad, stride=stride, groups=in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        self.point_conv = nn.Conv2d(in_channels, out_channels, (1,1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


"""
Dropout is not part of the official implementation
but training results on CIFAR-10 show overfitting
"""
class MobileNetV1(torch.nn.Module):
    def __init__(self, num_classes=1000, width_multiplier = 1, resolution_multiplier = 1) -> None:
        super(MobileNetV1, self).__init__()
        self.alpha = width_multiplier
        self.rho = resolution_multiplier

        self.layers = nn.Sequential(
            nn.Conv2d(3, int(32 * self.alpha), 3, stride=2, padding=1),
            DepthwiseSeparableConv(32, 64, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(64, 128, stride=2, alpha=self.alpha),
            DepthwiseSeparableConv(128, 128, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(128, 256, stride=2, alpha=self.alpha),
            DepthwiseSeparableConv(256, 256, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(256, 512, stride=2, alpha=self.alpha),
            DepthwiseSeparableConv(512, 512, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(512, 512, alpha=self.alpha),
            DepthwiseSeparableConv(512, 512, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(512, 512, alpha=self.alpha),
            DepthwiseSeparableConv(512, 512, alpha=self.alpha),
            nn.Dropout2d(0.2),
            DepthwiseSeparableConv(512, 1024, stride=2, alpha=self.alpha),
            DepthwiseSeparableConv(1024, 1024, stride=2, padding=4, alpha=self.alpha),
            nn.AdaptiveAvgPool2d((1,1)), # Using Adaptive pooling instead of `AvgPool2d`
        )

        self.linear = nn.Linear(int(1024 * self.alpha), num_classes)

        # Skipping `softmax` here as this would be applied during post-processing for inference
        # and during training crossentropy loss expects the logits (raw values) and not the normalised scores

    def forward(self, batch) -> torch.Tensor: # batch is of size BxCxHxW
        batch = self.resolution_multiply(batch)
        x = self.layers(batch)
        x = x.squeeze()
        x = self.linear(x)
        return x


    """
    Apply the resolution multiplier `rho` to the input height and width
    """
    def resolution_multiply(self, batch):
        hw = batch.size()[2:]
        new_size = tuple([math.floor(i*self.rho) for i in hw])
        return tf.resize(batch[:,:], new_size )
