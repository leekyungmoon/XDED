import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class ReplaceStyle(nn.Module):
    def __init__(self, replace_layers):
        super().__init__()
        self.eps = 1e-6
        self.replace_layers = replace_layers
        self.mu_per_epoch = {}
        self.var_per_epoch = {}
        self.feat_per_epoch = {}
        self.reset_memory()

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def reset_memory(self):
        del self.mu_per_epoch, self.var_per_epoch, self.feat_per_epoch
        self.mu_per_epoch = {}
        self.var_per_epoch = {}
        self.feat_per_epoch = {}

        for rep_layer in self.replace_layers:
            self.mu_per_epoch[rep_layer] = []
            self.var_per_epoch[rep_layer] = []
            self.feat_per_epoch[rep_layer] = []

    def replace(self, test_feat, replace_layer):
        pass

    def forward(self, feat, replace_layer):
        pass


class ReplaceMean(ReplaceStyle):
    def replace(self, test_feat, replace_layer):
        B = test_feat.size(0)
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_normed = (test_feat-test_mu)/test_sig

        train_feat_epoch = torch.cat( self.feat_per_epoch[us_layer] )
        train_feat_epoch = train_feat_epoch.mean(dim=[0], keepdim=True)
        train_mu = train_feat_epoch.mean(dim=[2, 3], keepdim=True)
        train_var = train_feat_epoch.var(dim=[2, 3], keepdim=True)
        train_sig = (train_var + self.eps).sqrt()
        train_mu, train_sig = train_mu.detach(), train_sig.detach()
        return test_feat_normed*train_sig + train_mu

    def forward(self, feat, replace_layer):
        B = feat.size(0)
        mu = feat.mean(dim=[2, 3], keepdim=True)
        var = feat.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        self.mu_per_epoch[replace_layer].append(mu)
        self.var_per_epoch[replace_layer].append(sig)
        self.feat_per_epoch[replace_layer].append(feat.detach())

class ReplaceCosFlatten(ReplaceStyle):
    def replace(self, test_feat, replace_layer):
        B = test_feat.size(0)
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_normed = (test_feat-test_mu)/test_sig
        test_feat_normed_flatten = test_feat_normed.reshape(B, -1)
        test_feat_normed_flatten_l2norm = self.l2_norm(test_feat_normed_flatten)

        train_feat_epoch = torch.cat(self.feat_per_epoch[replace_layer])
        sz_train = train_feat_epoch.shape[0]
        train_mu = train_feat_epoch.mean(dim=[2, 3], keepdim=True)
        train_var = train_feat_epoch.var(dim=[2, 3], keepdim=True)
        train_sig = (train_var + self.eps).sqrt()
        train_mu, train_sig = train_mu.detach(), train_sig.detach()
        train_feat_normed = (train_feat_epoch-train_mu)/train_sig
        train_feat_normed_flatten = train_feat_normed.reshape(sz_train,
                                                              -1)
        train_feat_normed_flatten_l2norm = self.l2_norm(train_feat_normed_flatten)
        cos = F.linear(test_feat_normed_flatten_l2norm,
                       train_feat_normed_flatten_l2norm)
        cos_max_index = torch.topk(cos, 1)[1].reshape(-1)
        #train_feat_normed[cos_max[1].reshape(-1)]

        res = test_feat_normed*train_sig[cos_max_index] + train_mu[cos_max_index]
        #self.reset_memory()
        return res
        #res = test_feat_normed*train_sig[cos_max_index] + train_mu[cos_max_index]

    def forward(self, feat, replace_layer):
        self.feat_per_epoch[replace_layer].append(feat.detach())


class ReplaceCosUnNormFlatten(ReplaceCosFlatten):
    def replace(self, test_feat, replace_layer):
        B = test_feat.size(0)
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_flatten = test_feat.reshape(B, -1)
        test_feat_flatten_l2norm = self.l2_norm(test_feat_flatten)

        test_feat_normed = (test_feat-test_mu)/test_sig
        test_feat_normed_flatten = test_feat_normed.reshape(B, -1)
        test_feat_normed_flatten_l2norm = self.l2_norm(test_feat_normed_flatten)

        train_feat_epoch = torch.cat(self.feat_per_epoch[replace_layer])
        sz_train = train_feat_epoch.shape[0]
        train_mu = train_feat_epoch.mean(dim=[2, 3], keepdim=True)
        train_var = train_feat_epoch.var(dim=[2, 3], keepdim=True)
        train_sig = (train_var + self.eps).sqrt()
        train_mu, train_sig = train_mu.detach(), train_sig.detach()
        train_feat_flatten = train_feat_epoch.reshape(sz_train, -1)
        train_feat_flatten_l2norm = self.l2_norm(train_feat_flatten)

        train_feat_normed = (train_feat_epoch-train_mu)/train_sig
        train_feat_normed_flatten = train_feat_normed.reshape(sz_train,
                                                              -1)
        train_feat_normed_flatten_l2norm = self.l2_norm(train_feat_normed_flatten)
        cos = F.linear(test_feat_flatten_l2norm,
                       train_feat_flatten_l2norm)
        cos_max_index = torch.topk(cos, 1)[1].reshape(-1)
        #train_feat_normed[cos_max[1].reshape(-1)]

        res = test_feat_normed*train_sig[cos_max_index] + train_mu[cos_max_index]
        #self.reset_memory()
        return res
        #res = test_feat_normed*train_sig[cos_max_index] + train_mu[cos_max_index]

class ReplaceAllProxyCosUnNormFlatten(ReplaceCosFlatten):
    def replace(self, test_feat, proxies):
        test_sz_feat = test_feat.size()
        B = test_sz_feat[0]
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_flatten = test_feat.reshape(B, -1)
        test_feat_flatten_l2norm = self.l2_norm(test_feat_flatten)

        test_feat_normed = (test_feat-test_mu)/test_sig
        test_feat_normed_flatten = test_feat_normed.reshape(B, -1)
        test_feat_normed_flatten_l2norm = self.l2_norm(test_feat_normed_flatten)

        reshaped_proxies = proxies.reshape(-1,
                                           test_sz_feat[1],
                                           test_sz_feat[2],
                                           test_sz_feat[3])

        proxy_mu = reshaped_proxies.mean(dim=[2, 3], keepdim=True)
        proxy_var = reshaped_proxies.var(dim=[2, 3], keepdim=True)
        proxy_sig = (proxy_var + self.eps).sqrt()
        proxy_mu, proxy_sig = proxy_mu.detach(), proxy_sig.detach()
        proxy_l2norm = self.l2_norm(proxies)

        cos = F.linear(test_feat_flatten_l2norm,
                       proxy_l2norm)
        cos_max_index = torch.topk(cos, 1)[1].reshape(-1)

        res = reshaped_proxies[cos_max_index]#test_feat_normed*proxy_sig[cos_max_index] + proxy_mu[cos_max_index]
        return res

class ReplaceProxyCosUnNormFlatten(ReplaceCosFlatten):
    def replace(self, test_feat, proxies):
        test_sz_feat = test_feat.size()
        B = test_sz_feat[0]
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_flatten = test_feat.reshape(B, -1)
        test_feat_flatten_l2norm = self.l2_norm(test_feat_flatten)

        test_feat_normed = (test_feat-test_mu)/test_sig
        test_feat_normed_flatten = test_feat_normed.reshape(B, -1)
        test_feat_normed_flatten_l2norm = self.l2_norm(test_feat_normed_flatten)

        reshaped_proxies = proxies.reshape(-1,
                                           test_sz_feat[1],
                                           test_sz_feat[2],
                                           test_sz_feat[3])

        proxy_mu = reshaped_proxies.mean(dim=[2, 3], keepdim=True)
        proxy_var = reshaped_proxies.var(dim=[2, 3], keepdim=True)
        proxy_sig = (proxy_var + self.eps).sqrt()
        proxy_mu, proxy_sig = proxy_mu.detach(), proxy_sig.detach()
        proxy_l2norm = self.l2_norm(proxies)

        cos = F.linear(test_feat_flatten_l2norm,
                       proxy_l2norm)
        cos_max_index = torch.topk(cos, 1)[1].reshape(-1)

        res = test_feat_normed*proxy_sig[cos_max_index] + proxy_mu[cos_max_index]
        return res


class InterpolCosUnNormFlatten(ReplaceCosFlatten):
    def replace(self, test_feat, replace_layer):
        B = test_feat.size(0)
        test_mu = test_feat.mean(dim=[2,3], keepdim=True)
        test_var = test_feat.var(dim=[2,3], keepdim=True)
        test_sig = (test_var+self.eps).sqrt()
        test_feat_flatten = test_feat.reshape(B, -1)
        test_feat_flatten_l2norm = self.l2_norm(test_feat_flatten)

        test_feat_normed = (test_feat-test_mu)/test_sig

        train_feat_epoch = torch.cat(self.feat_per_epoch[replace_layer])
        sz_train = train_feat_epoch.shape[0]
        train_mu = train_feat_epoch.mean(dim=[2, 3], keepdim=True)
        train_var = train_feat_epoch.var(dim=[2, 3], keepdim=True)
        train_sig = (train_var + self.eps).sqrt()
        train_mu, train_sig = train_mu.detach(), train_sig.detach()
        train_feat_flatten = train_feat_epoch.reshape(sz_train, -1)
        train_feat_flatten_l2norm = self.l2_norm(train_feat_flatten)

        train_feat_normed = (train_feat_epoch-train_mu)/train_sig
        train_feat_normed_flatten = train_feat_normed.reshape(sz_train,
                                                              -1)
        train_feat_normed_flatten_l2norm = self.l2_norm(train_feat_normed_flatten)
        cos = F.linear(test_feat_flatten_l2norm,
                       train_feat_flatten_l2norm)
        cos_max = torch.topk(cos, 1)
        cos_max_value = F.relu( cos_max[0].reshape(-1) )
        cos_max_index = cos_max[1].reshape(-1)
        #train_feat_normed[cos_max[1].reshape(-1)]
        cos_train_sig = (train_sig[cos_max_index].reshape(100, 512) * cos_max_value.reshape(-1, 1) ).reshape( 100, 512, 1, 1)
        cos_train_mu = (train_mu[cos_max_index].reshape(100, 512) * cos_max_value.reshape(-1, 1) ).reshape( 100, 512, 1, 1)
        cos_test_sig = (test_sig.reshape(100, 512) * (1-cos_max_value).reshape(-1, 1) ).reshape( 100, 512, 1, 1)
        cos_test_mu = (test_mu.reshape(100, 512) * (1-cos_max_value).reshape(-1, 1) ).reshape( 100, 512, 1, 1)

        res = test_feat_normed*(cos_train_sig + cos_test_sig) + (cos_train_mu+cos_test_mu)
        #train_mu[cos_max_index]
        #self.reset_memory()
        return res
        #res = test_feat_normed*train_sig[cos_max_index] + train_mu[cos_max_index]
