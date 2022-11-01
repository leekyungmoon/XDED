import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from dassl.modeling.ops import MixStyle, ReplaceStyle

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

from .resnet import ResNet

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""
"""
Standard residual networks
"""
@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L1(pretrained=True, **kwargs):
    rep_layers = ['layer1']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L2(pretrained=True, **kwargs):
    rep_layers = ['layer2']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L3(pretrained=True, **kwargs):
    rep_layers = ['layer3']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L4(pretrained=True, **kwargs):
    rep_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L12(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer2']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L13(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer3']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L14(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L23(pretrained=True, **kwargs):
    rep_layers = ['layer2', 'layer3']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L24(pretrained=True, **kwargs):
    rep_layers = ['layer2', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L34(pretrained=True, **kwargs):
    rep_layers = ['layer3', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L123(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer2', 'layer3']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L124(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer2', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L234(pretrained=True, **kwargs):
    rep_layers = ['layer2', 'layer3', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_replacestyle_L1234(pretrained=True, **kwargs):
    rep_layers = ['layer1', 'layer2', 'layer3', 'layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   rep_layers=rep_layers)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model
