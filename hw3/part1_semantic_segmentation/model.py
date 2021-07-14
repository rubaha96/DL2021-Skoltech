# model
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        # TODO
        self.num_down_blocks = num_down_blocks

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace=True)
        )

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv0 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True)
        )
        
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True)
        )

        self.upsample2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.upsample3 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_classes, kernel_size=1),
            nn.Softmax()
        )

    def forward(self, inputs):

        # interpolate to the nearest 2**num_down_blocks
        x = F.interpolate(inputs, size=2**self.num_down_blocks)

        # encoder
        e0 = self.enc_conv0(x)
        p0 = self.pool0(e0)
        e1 = self.enc_conv1(p0)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(p3)

        # decoder
        d0 = self.dec_conv0(torch.cat((e3, self.upsample0(b)), dim=1))
        d1 = self.dec_conv1(torch.cat((e2, self.upsample1(d0)), dim=1))
        d2 = self.dec_conv2(torch.cat((e1, self.upsample2(d1)), dim=1))
        logits = self.dec_conv3(torch.cat((e0, self.upsample3(d2)), dim=1))
        # logits = None # TODO

        # interpolate logits to the orig img
        logits = F.interpolate(logits, size=(inputs.shape[2], inputs.shape[3]))

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.init_backbone()

        if aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])

        self.head = DeepLabHead(self.out_features, num_classes)

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-2]))
            # TODO: number of output features in the backbone
            self.out_features = 512

        elif self.backbone == 'vgg11_bn':
            self.model = models.vgg11_bn(pretrained=True).features
            # TODO
            self.out_features = 512

        elif self.backbone == 'mobilenet_v3_small':
            self.model = models.mobilenet_v3_small(pretrained=True).features
            # TODO
            self.out_features = 576

    def _forward(self, x):
        # TODO: forward pass through the backbone
        if self.backbone == 'resnet18':
            x = self.model(x)

        elif self.backbone == 'vgg11_bn':
            x = self.model(x)

        elif self.backbone == 'mobilenet_v3_small':
            x = self.model(x)

        return x

    def forward(self, inputs):
        x = self._forward(inputs) # TODO
        if self.aspp:
            logits = self.head.forward(self.aspp.forward(x))
        else:
            logits = self.head.forward(x)
        up = torch.nn.Upsample(size=(inputs.shape[2], inputs.shape[3]), mode='bilinear')
        logits = up(logits)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()
        
        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True)
        )
        self.conv330 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, dilation=atrous_rates[0], padding=atrous_rates[0]),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True)
        )
        self.conv331 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, dilation=atrous_rates[1], padding=atrous_rates[1]),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True)
        )
        self.conv332 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, dilation=atrous_rates[2], padding=atrous_rates[2]),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            # nn.AveragePool2d(),
            nn.Conv2d(in_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels*5, in_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # TODO: forward pass through the ASPP module
        avg_pool = nn.AvgPool2d((x.shape[2], x.shape[3]))
        up = torch.nn.Upsample(size=(x.shape[2], x.shape[3]))
        conv11 = self.conv11(x)
        conv330 = self.conv330(x)
        conv331 = self.conv331(x)
        conv332 = self.conv332(x)
        avg_pool = up(self.pool(avg_pool(x)))
        res = torch.cat((conv11, conv330,
                         conv331, conv332, avg_pool), dim=1)
        res = self.conv(res)
        res = self.dropout(res)
        
        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        return res