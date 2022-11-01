import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

import pdb

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(Backbone):

    def __init__(self, whiten_layers=[]):
        super().__init__()
        #self.features = nn.Sequential(
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        #)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # Note that self.classifier outputs features rather than logits
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True)
        )

        self._out_features = 4096

        self.whiten_layers = whiten_layers

    def whiten(self, x):
        sz_batch = x.size(0)
        x_mu = x.mean(dim=[2,3,], keepdim=True)
        x_var = x.var(dim=[2,3,], keepdim=True)
        x_sig = (x_var+1e-6).sqrt()
        x = (x-x_mu)/x_sig
        return x

    def features(self, x):
        f_dict = {}
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        f_dict['layer0'] = x.detach()
        if 'layer0' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        f_dict['layer1'] = x.detach()
        if 'layer1' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv3(x)
        x = self.relu3(x)
        f_dict['layer2'] = x.detach()
        if 'layer2' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv4(x)
        x = self.relu4(x)
        f_dict['layer3'] = x.detach()
        if 'layer3' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        f_dict['layer4'] = x.detach()
        if 'layer4' in self.whiten_layers:
            x = self.whiten(x)
        return x, f_dict

    def forward(self, x, y=None):
        x, f_dict = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x), f_dict


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


@BACKBONE_REGISTRY.register()
def alexnet_whiten_L0(pretrained=True, **kwargs):
    whiten_layers = ['layer0']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L01(pretrained=True, **kwargs):
    whiten_layers = ['layer0', 'layer1']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L012(pretrained=True, **kwargs):
    whiten_layers = ['layer0', 'layer1', 'layer2']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L1(pretrained=True, **kwargs):
    whiten_layers = ['layer1']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L12(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L123(pretrained=True, **kwargs):
    whiten_layers = ['layer1', 'layer2', 'layer3']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model

@BACKBONE_REGISTRY.register()
def alexnet_whiten_L23(pretrained=True, **kwargs):
    whiten_layers = ['layer2', 'layer3']
    model = AlexNet(whiten_layers=whiten_layers)
    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])
    return model
