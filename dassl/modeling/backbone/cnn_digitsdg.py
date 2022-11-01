import torch.nn as nn
from torch.nn import functional as F

from dassl.utils import init_network_weights

from .build import BACKBONE_REGISTRY
from .backbone import Backbone


class Convolution(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=1, padding=1)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvNet(Backbone):

    def __init__(self, c_hidden=64, whiten_layers=[]):
        super().__init__()
        self.conv1 = Convolution(3, c_hidden)
        self.conv2 = Convolution(c_hidden, c_hidden)
        self.conv3 = Convolution(c_hidden, c_hidden)
        self.conv4 = Convolution(c_hidden, c_hidden)

        self._out_features = 2**2 * c_hidden

        self.whiten_layers = whiten_layers

    def _check_input(self, x):
        H, W = x.shape[2:]
        assert H == 32 and W == 32, \
            'Input to network must be 32x32, ' \
            'but got {}x{}'.format(H, W)

    def whiten(self, x):
        sz_batch = x.size(0)
        x_mu = x.mean(dim=[2,3,], keepdim=True)
        x_var = x.var(dim=[2,3], keepdim=True)
        x_sig = (x_var+1e-6).sqrt()
        x = (x-x_mu)/x_sig
        return x

    def forward(self, x, y=None):
        f_dict = {}
        self._check_input(x)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        f_dict['layer0'] = x.detach()
        if 'layer0' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        f_dict['layer1'] = x.detach()
        if 'layer1' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        f_dict['layer2'] = x.detach()
        if 'layer2' in self.whiten_layers:
            x = self.whiten(x)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        f_dict['layer3'] = x.detach()
        if 'layer3' in self.whiten_layers:
            x = self.whiten(x)
        return x.view(x.size(0), -1), f_dict


@BACKBONE_REGISTRY.register()
def cnn_digitsdg(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    model = ConvNet(c_hidden=64)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_whiten_L0(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    whiten_layers = ['layer0']
    model = ConvNet(c_hidden=64, whiten_layers=whiten_layers)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_whiten_L01(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    whiten_layers = ['layer0', 'layer1']
    model = ConvNet(c_hidden=64, whiten_layers=whiten_layers)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_whiten_L012(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    whiten_layers = ['layer0', 'layer1', 'layer2']
    model = ConvNet(c_hidden=64, whiten_layers=whiten_layers)
    init_network_weights(model, init_type='kaiming')
    return model

@BACKBONE_REGISTRY.register()
def cnn_digitsdg_whiten_L12(**kwargs):
    """
    This architecture was used for DigitsDG dataset in:

        - Zhou et al. Deep Domain-Adversarial Image Generation
        for Domain Generalisation. AAAI 2020.
    """
    whiten_layers = ['layer1', 'layer2']
    model = ConvNet(c_hidden=64, whiten_layers=whiten_layers)
    init_network_weights(model, init_type='kaiming')
    return model
