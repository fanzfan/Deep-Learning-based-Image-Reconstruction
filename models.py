import torch.nn as nn
import torch.nn.functional as F

"""
    Author : Vela Yang
    Last edited : 30th, July, 2021
    Framework : PyTorch
    This .py file implements the Residual Block of ResNet and the AVS3 in-loop filter for luminance channel(Y channel, Y of YUV)
"""

# single residual block of the ResNet
# !REMIND : a relu function should be followed by this block
class ResBlock(nn.Module):
    def __init__(self, kernel_size=3, n_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        return x


# Neutral network for AVS3 coding in-loop filter
class AVS3Filter(nn.Module):
    def __init__(self, n_input=1, n_output=1, kernel_size=3, n_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(n_input, n_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.res1 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res2 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res3 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res4 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res5 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res6 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res7 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res8 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res9 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res10 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res11 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res12 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res13 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res14 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res15 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res16 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.res17 = ResBlock(kernel_size=kernel_size, n_channels=n_channels)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.conv3 = nn.Conv2d(n_channels, n_output, kernel_size=kernel_size, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(n_channels)

    def forward(self, x):
        rec = x  # reconstructed frame, in Fig.2
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.res1(x))
        x = F.relu(self.res2(x))
        x = F.relu(self.res3(x))
        x = F.relu(self.res4(x))
        x = F.relu(self.res5(x))
        x = F.relu(self.res6(x))
        x = F.relu(self.res7(x))
        x = F.relu(self.res8(x))
        x = F.relu(self.res9(x))
        x = F.relu(self.res10(x))
        x = F.relu(self.res11(x))
        x = F.relu(self.res12(x))
        x = F.relu(self.res13(x))
        x = F.relu(self.res14(x))
        x = F.relu(self.res15(x))
        x = F.relu(self.res16(x))
        x = F.relu(self.res17(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + rec
        return x
