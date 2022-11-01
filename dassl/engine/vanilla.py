from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy

import pdb

@TRAINER_REGISTRY.register()
class Vanilla(TrainerX):
    """Vanilla baseline."""

    def forward_backward(self, batch):
        input, label, domain, impath = self.parse_batch_train(batch)
        #pdb.set_trace()
        output = self.model(input)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        """
        self.label_list.append(label)
        self.domain_list.append(domain)
        self.impath_list.append(impath)
        pdb.set_trace()
        """

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']

        domain = batch['domain']
        impath = batch['impath']

        input = input.to(self.device)
        label = label.to(self.device)

        domain = domain.to(self.device)

        return input, label, domain, impath
