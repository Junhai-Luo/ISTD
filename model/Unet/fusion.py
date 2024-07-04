import torch.nn as nn


class AddFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(AddFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        out = xh + xl

        return out


class BottomUpLocalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(BottomUpLocalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,h,w -> bz,cl/r,h,w
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,h,w -> bz,cl,h,w
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        bottomup_w = self.bottomup(xl)  # bz,cl,h,w -> bz,cl,h,w
        out = bottomup_w * xh + xl

        return out


class BottomUpGlobalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(BottomUpGlobalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.bottomup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,h,w -> bz,cl/r,h,w
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,h,w -> bz,cl,h,w
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        bottomup_w = self.bottomup(xl)  # bz,cl,h,w -> bz,cl,h,w
        out = bottomup_w * xh + xl

        return out


class TopDownGlobalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(TopDownGlobalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 bz,cl,h,w -> bz,cl,1,1
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,1,1 -> bz,cl/r,1,1
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,1,1 -> bz,cl,1,1
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        topdown_w = self.topdown(xh)  # bz,cl,h,w -> bz,cl,1,1
        out = topdown_w * xl + xh  # bz,cl,h,w -> bz,cl,h,w

        return out


class TopDownLocalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(TopDownLocalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,1,1 -> bz,cl/r,1,1
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,1,1 -> bz,cl,1,1
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        topdown_w = self.topdown(xh)  # bz,cl,h,w -> bz,cl,1,1
        out = topdown_w * xl + xh  # bz,cl,h,w -> bz,cl,h,w

        return out


# bz,cl,h,w
#           } => bz,cl,h,w
# bz,ch,h,w
class BiAsyFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(BiAsyFuseLayer, self).__init__()

        # 越低的层通道数越少，使用相加融合时应使通道数与低层一致
        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),  # bz,ch,h,w -> bz,cl,h,w
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        # 自顶向下的全局调制
        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化 bz,cl,h,w -> bz,cl,1,1
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,1,1 -> bz,cl/r,1,1
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,1,1 -> bz,cl,1,1
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        # 自底向上的局部调制
        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),  # bz,cl,h,w -> bz,cl/r,h,w
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),  # bz,cl/r,h,w -> bz,cl,h,w
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)  # bz,ch,h,w -> bz,cl,h,w

        topdown_w = self.topdown(xh)  # bz,cl,h,w -> bz,cl,1,1
        bottomup_w = self.bottomup(xl)  # bz,cl,h,w -> bz,cl,h,w
        out = topdown_w * xl + bottomup_w * xh  # bz,cl,h,w -> bz,cl,h,w

        return out


class BiLocalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(BiLocalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_w = self.topdown(xh)
        bottomup_w = self.bottomup(xl)
        out = topdown_w * xl + bottomup_w * xh

        return out


class BiGlobalFuseLayer(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels, r=4):
        super(BiGlobalFuseLayer, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_w = self.topdown(xh)
        bottomup_w = self.bottomup(xl)
        out = topdown_w * xl + bottomup_w * xh

        return out
