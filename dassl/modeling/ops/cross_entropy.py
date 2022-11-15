import torch
from torch.nn import functional as F


def cross_entropy(input, target, label_smooth=0, reduction='mean'):
    """Cross entropy loss.

    Args:
        input (torch.Tensor): logit matrix with shape of (batch, num_classes).
        target (torch.LongTensor): int label matrix.
        label_smooth (float, optional): label smoothing hyper-parameter.
            Default is 0.
        reduction (str, optional): how the losses for a mini-batch
            will be aggregated. Default is 'mean'.
    """
    num_classes = input.shape[1]
    log_prob = F.log_softmax(input, dim=1)
    zeros = torch.zeros(log_prob.size())
    target = zeros.scatter_(1, target.unsqueeze(1).data.cpu(), 1)
    target = target.type_as(input)
    target = (1-label_smooth) * target + label_smooth/num_classes
    loss = (-target * log_prob).sum(1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError

class KDLoss(torch.nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

class pixelwise_XDEDLoss(torch.nn.Module):
    def __init__(self, temp_factor=2.0):
        super(pixelwise_XDEDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = torch.nn.KLDivLoss(reduction="sum")
        self.CLASS_NUM = 19

    def xded_loss(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

    def forward(self, main_out, gts):
        # main_out.shape : [batch, 19, 768, 768]
        # gts.shape : [batch, 768, 768]

        batch_size = main_out.shape[0]

        flat_gts = gts.reshape(-1) # [batch*768*768]
        flat_out = main_out.reshape(-1, self.CLASS_NUM)

        flat_targets = main_out.clone().detach().reshape(-1, self.CLASS_NUM)
        # [batch*768*768, 19]

        flat_gt_set = flat_gts.unique().tolist()
        ensemble_dict= {}

        for f_gt in flat_gt_set:
            if f_gt == 255:
                continue
            cur_gt_idx = flat_gts == f_gt # [False, True, ...]
            flat_targets[cur_gt_idx, :] = flat_out[cur_gt_idx].mean(0).detach()

        return self.xded_loss(flat_out, flat_targets)
