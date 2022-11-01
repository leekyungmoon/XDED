import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from dassl.modeling.ops import MixStyle, ReplaceStyle

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from .resnet import conv3x3, BasicBlock, Bottleneck, ResNet

import pdb
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class ResNet_Penultimate(Backbone):
    def __init__(
        self,
        block, layers,
        whiten_layers=[],
        replace_layers=[],
        replacer=None,
        whiten_cov=False,
        is_proxy=False,
        shape_mix_layers=[],
        **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        self._out_features = 512 * block.expansion

        self.penultimate_sig = torch.nn.Embedding(self._out_features, 1, 1).cuda()
        self.penultimate_mean = torch.nn.Embedding(self._out_features, 1, 1).cuda()
        """
        self.penultimate_sig = torch.randn(self._out_features, 1, 1,
                                           requires_grad=True).cuda()
        self.penultimate_mean = torch.randn(self._out_features, 1, 1,
                                            requires_grad=True).cuda()
        """
        #nn.init.kaiming_normal_(self.penultimate_sig, mode='fan_out')
        #nn.init.kaiming_normal_(self.penultimate_mean, mode='fan_out')

        self.is_proxy = is_proxy
        self.whiten_cov = whiten_cov

        self.whiten_layers = whiten_layers
        self.replace_layers = replace_layers

        if replacer is not None:
            self.replacer=replacer(replace_layers)

        print('ResNet BACKBONE INITIALIZED')
        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x, y=None):
        sz_batch = x.size(0)
        f_dict={}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f_dict['layer0'] = x.detach()
        if 'layer0' in self.whiten_layers:
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        x = self.layer1(x)
        f_dict['layer1'] = x.detach()
        if 'layer1' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer1')
            else:
                print('TEST --> ReplaceSTYLE LAYER1 EXECUTED')
                x = self.replacer.replace(x, 'layer1')
        if 'layer1' in self.whiten_layers:
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        x = self.layer2(x)
        f_dict['layer2'] = x.detach()
        if 'layer2' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer2')
            else:
                print('TEST --> ReplaceSTYLE LAYER2 EXECUTED')
                x = self.replacer.replace(x, 'layer2')
        if 'layer2' in self.whiten_layers:
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        x = self.layer3(x)
        f_dict['layer3'] = x.detach()
        if 'layer3' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer3')
            else:
                print('TEST --> ReplaceSTYLE LAYER3 EXECUTED')
                x = self.replacer.replace(x, 'layer3')
        if 'layer3' in self.whiten_layers:
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        x = self.layer4(x)
        f_dict['layer4'] = x
        """
        penultimate_mu = x.mean(dim=[2,3], keepdim=True)
        penultimate_var = x.var(dim=[2,3], keepdim=True)
        penultimate_sig = (penultimate_var+1e-6).sqrt()
        penultimate_normed = (x-penultimate_mu)/penultimate_sig

        pdb.set_trace()
        x = penultimate_normed * self.penultimate_sig + self.penultimate_mean
        """
        return x, f_dict

    def forward(self, x, y=None):
        f, f_dict = self.featuremaps(x, y)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1), f_dict
