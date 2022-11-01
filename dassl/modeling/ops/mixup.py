import torch
from torch.nn import functional as F

def mixup(x1, x2, y1, y2, beta, preserve_order=False):
    """Mixup.

    Args:
        x1 (torch.Tensor): data with shape of (b, c, h, w).
        x2 (torch.Tensor): data with shape of (b, c, h, w).
        y1 (torch.Tensor): label with shape of (b, n).
        y2 (torch.Tensor): label with shape of (b, n).
        beta (float): hyper-parameter for Beta sampling.
        preserve_order (bool): apply lmda=max(lmda, 1-lmda).
            Default is False.
    """
    lmda = torch.distributions.Beta(beta, beta).sample([x1.shape[0], 1, 1, 1])
    if preserve_order:
        lmda = torch.max(lmda, 1 - lmda)
    lmda = lmda.to(x1.device)
    xmix = x1*lmda + x2 * (1-lmda)
    lmda = lmda[:, :, 0, 0]
    ymix = y1*lmda + y2 * (1-lmda)
    return xmix, ymix

def mixup_hidden(x, rand_idx, lmda):
    mixed_x = lmda * x + (1 - lmda) * x[rand_idx,:]
    return mixed_x.cuda()


def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    lam = torch.max(lam, 1 - lam)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    if y is None:
        return mixed_x
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_cross_entropy(logit, label):
    output_x = F.softmax(logit, 1)
    loss = (-label * torch.log(output_x + 1e-5)).sum(1).mean()
    return loss
