import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from dassl.modeling.ops import MixStyle
from dassl.modeling.ops.replacestyle import *

from .build import BACKBONE_REGISTRY
from .resnet import ResNet, Backbone, conv3x3, BasicBlock, Bottleneck, init_pretrained_weights

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

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
def resnet18_whiten_cov_proxy_replaceall_cos_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   whiten_cov=True,
                   replace_layers=replace_layers,
                   is_proxy=True,
                   replacer=ReplaceAllProxyCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet18_whiten_proxy_replaceall_cos_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   is_proxy=True,
                   replacer=ReplaceAllProxyCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_whiten_proxy_replace_cos_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   is_proxy=True,
                   replacer=ReplaceProxyCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_whiten_replace_cos_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   replacer=ReplaceCosFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_whiten_replace_cos_unnorm_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   replacer=ReplaceCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_whiten_interpol_cos_unnorm_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   replacer=InterpolCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model

@BACKBONE_REGISTRY.register()
def resnet18_whiten_cov_replace_cos_unnorm_flatten_L12_4(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    replace_layers = ['layer4']
    model = ResNet(block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   whiten_layers=whiten_layers,
                   replace_layers=replace_layers,
                   whiten_cov = True,
                   replacer=ReplaceCosUnNormFlatten)

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model
