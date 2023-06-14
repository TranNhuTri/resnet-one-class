from torch import nn
from torch.nn import functional


class PreActivationBlock(nn.Module):
    def __init__(self, in_planes, planes, stride, *args, **kwargs):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.bn1(x)
        out = functional.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = functional.relu(out)
        out = self.conv2(out)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out += shortcut
        return out
