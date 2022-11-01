import torch
import pdb

class XBM:
    def __init__(self, cfg, nb_classes):
        self.cfg = cfg
        self.nb_classes = nb_classes
        self.K = cfg.TRAINER.CSKD.XBM.SIZE

        #self.feats = torch.zeros(self.K,
        #                         nb_classes).cuda()
        self.feats = {}
        #self.targets = {}
        self.ptr = {}
        for cls_elem in range(nb_classes):
            self.feats[cls_elem] = torch.zeros(self.K,
                                               nb_classes).cuda()
            #self.targets[cls_elem] = torch.zeros(self.K,
            #                                     dtype=torch.long).cuda()
            self.ptr[cls_elem] = 0

    def is_full(self, query_class):
        #return self.targets[-1].item() != 0
        return self.feats[query_class][-1].sum().item() != 0

    def get(self, targets):
        unique_targets = targets.unique()
        memory_feats = []
        memory_targets = []
        for target in unique_targets:
            #target_feat = None
            target = target.item()
            if self.is_full(target):
                target_feat = self.feats[target]
            else:
                target_feat = self.feats[target][:self.ptr[target]]
            memory_feats.append(target_feat)
            memory_targets.append( target+torch.zeros(target_feat.shape[0]) )
        return torch.cat(memory_feats), torch.cat(memory_targets)
        #if self.is_full:
        #    return self.feats, self.targets
        #else:
        #    return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue_per_class(self, query_cls, feats):
        q_size = feats.shape[0]
        query_cls = query_cls.item()
        if self.ptr[query_cls] + q_size > self.K:
            #pdb.set_trace()
            self.feats[query_cls][-q_size:] = feats
            self.ptr[query_cls] = 0
        else:
            self.feats[query_cls][ self.ptr[query_cls] : self.ptr[query_cls]+q_size ] = feats
            self.ptr[query_cls] += q_size

    def enqueue_dequeue(self, feats, targets):
        unique_targets = targets.unique()
        for target in unique_targets:
            self.enqueue_dequeue_per_class(target, feats[ targets == target ] )
        """
        q_size = len(targets)
        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            #self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            #self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size
        """
