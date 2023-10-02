import functools

from utils.registry import ARCH_REGISTRY

from .module_util import *


## Channel Attention (CA) Layer
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DegradationInfModule(nn.Module):
    def __init__(self, in_channels=64, num_residual_blocks=16, num_conv_layers=4):
        super(DegradationInfModule, self).__init__()
        self.in_channels = in_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_conv_layers = num_conv_layers

        layers = []
        layers += [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(num_residual_blocks):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

        for _ in range(num_conv_layers):
            layers += [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            ]

        layers += [
            ChannelAttention(in_channels),
            nn.Conv2d(in_channels, int(in_channels/2), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/2), int(in_channels/4), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/4), int(in_channels/8), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/8), int(in_channels/16), kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels/16),1, kernel_size=3, stride=1, padding=1),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        out = self.model(x)
        return out

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # initialization
        initialize_weights(
            [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1
        )

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block"""

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
        super(RRDBNet, self).__init__()
        self.upscale = upscale
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)
        self.degInf= DegradationInfModule(64,16,4)
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB_block_f, nb)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        if upscale == 4:
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x,deg_strength=None):
        deg_strength_map=self.degInf(deg_strength)
        x = torch.cat((x, deg_strength_map), dim=1)
        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        if self.upscale == 2 or self.upscale == 3:
            fea = self.lrelu(
                self.conv_up1(
                    F.interpolate(fea, scale_factor=self.upscale, mode="nearest")
                )
            )
        if self.upscale == 4:
            fea = self.lrelu(
                self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
            fea = self.lrelu(
                self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return out
@ARCH_REGISTRY.register()
class RRDBNet_channel3(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4):
        super(RRDBNet, self).__init__()
        self.upscale = upscale
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = make_layer(RRDB_block_f, nb)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        if upscale == 4:
            self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        fea = self.conv_first(x)
        trunk = self.conv_body(self.body(fea))
        fea = fea + trunk

        if self.upscale == 2 or self.upscale == 3:
            fea = self.lrelu(
                self.conv_up1(
                    F.interpolate(fea, scale_factor=self.upscale, mode="nearest")
                )
            )
        if self.upscale == 4:
            fea = self.lrelu(
                self.conv_up1(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
            fea = self.lrelu(
                self.conv_up2(F.interpolate(fea, scale_factor=2, mode="nearest"))
            )
        out = self.conv_last(self.lrelu(self.conv_hr(fea)))

        return out
