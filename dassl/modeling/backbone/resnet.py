import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from dassl.modeling.ops import MixStyle, ReplaceStyle

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

import pdb
import numpy as np

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


class ResNet(Backbone):
    def __init__(
        self,
        block, layers,
        ms_layers=[], ms_p=0.5, ms_a=0.1,
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

        self.is_proxy = is_proxy
        self.whiten_cov = whiten_cov

        self.shape_mix_layers = shape_mix_layers
        self.whiten_layers = whiten_layers
        self.replace_layers = replace_layers

        if replacer is not None:
            self.replacer=replacer(replace_layers)
        """
        if len(rep_layers) > 0:
            self.replacestyle = ReplaceStyle(rep_layers)
            for layer_name in rep_layers:
                assert layer_name in ['layer0', 'layer1', 'layer2',
                                      'layer3', 'layer4']
            print(f'Insert ReplaceStyle after {rep_layers}')
        self.rep_layers = rep_layers
        """

        self.mixstyle = None
        if ms_layers:
            self.mixstyle = MixStyle(p=ms_p, alpha=ms_a)
            for layer_name in ms_layers:
                assert layer_name in ['layer1', 'layer2', 'layer3']
            print(f'Insert MixStyle after {ms_layers}')
        self.ms_layers = ms_layers

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

    def vanilla_featuremaps(self, x):
        g_dict={}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        g_dict['layer0'] = x.detach()
        x = self.layer1(x)
        g_dict['layer1'] = x.detach()
        x = self.layer2(x)
        g_dict['layer2'] = x.detach()
        x = self.layer3(x)
        g_dict['layer3'] = x.detach()
        x = self.layer4(x)
        g_dict['layer4'] = x.detach()
        return x, g_dict


    def featuremaps(self, x, y=None):
        beta_alpha = 1.0
        if y is not None and len(self.shape_mix_layers) > 0:
            cls_set = torch.unique(y)
            cls_idx = {}
            cls_perm = {}
            perm_idx = torch.arange(0, y.shape[0], dtype=torch.int64).cuda()
            for cls_elem in cls_set:
                idx  = (y==cls_elem).nonzero(as_tuple=True)[0]
                shuffle_idx = idx.detach().cpu().numpy()
                np.random.shuffle(shuffle_idx)
                shuffle_idx = torch.from_numpy( shuffle_idx ).cuda()
                #cls_idx[cls_elem.item()] = idx
                #cls_perm[cls_elem.item()] = shuffle_idx

                perm_idx[idx] = shuffle_idx
            #perm_idx = perm_idx.detach()


        f_dict={}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        f_dict['layer0'] = x.detach()
        if 'layer0' in self.whiten_layers:
            sz_batch = x.size(0)
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        x = self.layer1(x)
        #if 'layer1' in self.shape_mix_layers:
        #    if self.training:
        #        mix_lamb = np.random.beta(beta_alpha, beta_alpha)
        #        x = mix_lamb*x + (1-mix_lamb)*x[perm_idx]
        if 'layer1' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer1')
            else:
                print('TEST --> ReplaceSTYLE LAYER1 EXECUTED')
                x = self.replacer.replace(x, 'layer1')
        if 'layer1' in self.whiten_layers:
            sz_batch = x.size(0)
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig
        if 'layer1' in self.ms_layers:
            #print('mixstyle applied at layer1')
            x = self.mixstyle(x)
        if 'layer1' in self.shape_mix_layers:
            if self.training:
                mix_lamb = np.random.beta(beta_alpha, beta_alpha)
                x = mix_lamb*x + (1-mix_lamb)*x[perm_idx]
        f_dict['layer1'] = x.clone()


        x = self.layer2(x)
        #if 'layer2' in self.shape_mix_layers:
        #    if self.training:
        #        mix_lamb = np.random.beta(beta_alpha, beta_alpha)
        #        x = mix_lamb*x + (1-mix_lamb)*x[perm_idx]
        if 'layer2' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer2')
            else:
                print('TEST --> ReplaceSTYLE LAYER2 EXECUTED')
                x = self.replacer.replace(x, 'layer2')
        if 'layer2' in self.whiten_layers:
            sz_batch = x.size(0)
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig

        if 'layer2' in self.ms_layers:
            #print('mixstyle applied at layer2')
            x = self.mixstyle(x)

        if 'layer2' in self.shape_mix_layers:
            if self.training:
                mix_lamb = np.random.beta(beta_alpha, beta_alpha)
                x = mix_lamb*x + (1-mix_lamb)*x[perm_idx]
        f_dict['layer2'] = x.clone()


        x = self.layer3(x)
        if 'layer3' in self.replace_layers:
            if self.training:
                self.replacer(x, 'layer3')
            else:
                print('TEST --> ReplaceSTYLE LAYER3 EXECUTED')
                x = self.replacer.replace(x, 'layer3')
        if 'layer3' in self.whiten_layers:
            sz_batch = x.size(0)
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig
        if 'layer3' in self.ms_layers:
            #print('mixstyle applied at layer3')
            x = self.mixstyle(x)

        f_dict['layer3'] = x.clone()


        x = self.layer4(x)
        if 'layer4' in self.shape_mix_layers:
            if self.training:
                mix_lamb = np.random.beta(beta_alpha, beta_alpha)
                x = mix_lamb*x + (1-mix_lamb)*x[perm_idx]

        if 'layer4' in self.replace_layers:
            if self.is_proxy:
                if self.training:
                    pass
                else:
                    x = self.replacer.replace(x, self.proxies)
            else:
                if self.training:
                    self.replacer(x, 'layer4')
                else:
                    x = self.replacer.replace(x, 'layer4')
        if 'layer4' in self.whiten_layers:
            sz_batch = x.size(0)
            x_mu = x.mean(dim=[2,3,], keepdim=True)
            x_var = x.var(dim=[2,3], keepdim=True)
            x_sig = (x_var+1e-6).sqrt()
            if self.whiten_cov:
                x = x-x_mu
            else:
                x = (x-x_mu)/x_sig
        f_dict['layer4'] = x.clone()

        return x, f_dict

    def forward(self, x, y=None):
        f, f_dict = self.featuremaps(x, y)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1), f_dict


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
def resnet18(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet34(pretrained=True, **kwargs):
    model = ResNet(block=BasicBlock, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 6, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model


@BACKBONE_REGISTRY.register()
def resnet101(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 4, 23, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet101'])

    return model


@BACKBONE_REGISTRY.register()
def resnet152(pretrained=True, **kwargs):
    model = ResNet(block=Bottleneck, layers=[3, 8, 36, 3])

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet152'])

    return model


"""
Residual networks with mixstyle
"""


@BACKBONE_REGISTRY.register()
def resnet18_ms123(pretrained=True, **kwargs):
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])

    return model


@BACKBONE_REGISTRY.register()
def resnet50_ms123(pretrained=True, **kwargs):
    model = ResNet(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        ms_layers=['layer1', 'layer2', 'layer3']
    )

    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])

    return model
