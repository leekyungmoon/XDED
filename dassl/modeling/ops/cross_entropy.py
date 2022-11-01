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
