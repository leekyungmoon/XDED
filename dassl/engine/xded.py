import torch
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.modeling.ops.mixup import mixup, mixup_cross_entropy
from dassl.modeling.ops.xbm import XBM
import pdb, time, datetime

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir) # dassl/engine
dassl_dir = parentdir.replace('/engine', '')
utils_dir = parentdir.replace('/engine', 'utils')
sys.path.append(parentdir)
sys.path.append(dassl_dir)
#sys.path.append(utils_dir)

from dassl.utils import MetricMeter, AverageMeter
from dassl.utils.torchtools import load_pretrained_weights, count_num_param
from dassl.optim.optimizer import build_optimizer, init_optim, build_proxy_optimizer
from dassl.optim.lr_scheduler import build_lr_scheduler
from dassl.engine.trainer import SimpleNet

from dassl.modeling.ops.cross_entropy import KDLoss

@TRAINER_REGISTRY.register()
class XDED(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.kd_loss = KDLoss(cfg.TRAINER.XDED.KD_TEMP)

    def xded_label(self, logit, replace=False):
        if replace:
            assert False
        nb_classes = int(self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE/self.cfg.IPC)
        label_repeat = self.cfg.IPC

        average_logits = []
        for cur_cls in range(nb_classes):
            cur_average_logit = logit[ cur_cls * self.cfg.IPC :(cur_cls+1) * self.cfg.IPC ].mean(0)
            average_logits.append(cur_average_logit)
        average_logits = torch.stack(average_logits)
        return torch.repeat_interleave(average_logits,
                                       repeats=self.cfg.IPC,
                                       dim=0).detach()

    def forward_backward(self, batch):
        input, label, domain = self.parse_batch_train(batch)
        output = self.model(input)
        cskd_label = self.xded_label(output,
                                     replace=self.cfg.TRAINER.XDED.REPLACE)
        loss = F.cross_entropy(output, label)
        xded_loss = self.kd_loss(output, cskd_label)
        total_loss = loss
        if self.epoch >= self.cfg.TRAINER.XDED.WARMUP:
            total_loss += self.cfg.TRAINER.XDED.LAMBDA * xded_loss
        self.model_backward_and_update(total_loss)

        loss_summary = {
            'loss': loss.item(),
            'xded_loss': xded_loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        domain = batch['domain']

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, domain
