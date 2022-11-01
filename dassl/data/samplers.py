import copy
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

import numpy as np

import pdb

class RandomDomainSampler(Sampler):
    """Random domain sampler.

    This sampler randomly samples N domains each with K
    images to form a minibatch.
    #LKM: NEED TO CODE RandomDomainSampler & IPC
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length

class BalancedSampler(Sampler):
    """
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = data_source.ys
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.data_source)
        self.num_classes = len(set(self.ys))
    """
    #def __init__(self, data_source, batch_size, n_domain):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.ys = []
        for i, item in enumerate(data_source):
            self.ys.append(item.label)

        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.data_source)
        #self.num_classes = len(set(self.ys))
        self.num_classes = len(set(self.ys))
        #self.ys = list(self.ys.keys())

        #self.length = len(list(self.__iter__()))

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=True)
            #pdb.set_trace()
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.ys) == sampled_classes[i])[0]
                class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)

    def __len__(self):
        return self.num_samples

class DomainBalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.ys = []
        self.domains = []
        for i, item in enumerate(data_source):
            self.ys.append(item.label)
            self.domains.append(item.domain)
        self.ys = np.array(self.ys)
        self.domains = np.array(self.domains)

        self.nb_domains = len( set(self.domains) )

        """
        self.domain_data_source =  []
        for cur_domain in range(self.nb_domains):
            self.domain_data_source_list.append(
                data_source[ self.domains == cur_domain ]
            )
        """

        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.data_source)
        self.num_classes = len(set(self.ys))

    def random_ipc_per_domain(self):
        ipc_per_domain = self.num_instances // self.nb_domains
        domain_ipc = [ ipc_per_domain for _ in range(self.nb_domains) ]
        if (self.num_instances % self.nb_domains) != 0:
            remainder = self.num_instances % self.nb_domains
            # remainder == 1 or 2
            # 1 <= remainder < self.nb_domains
            while remainder > 0:
                ran_domain_idx = np.random.choice(self.nb_domains, 1)[0]
                domain_ipc[ran_domain_idx] += 1
                remainder -= 1
        return domain_ipc


    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=True)
            domain_ipc = self.random_ipc_per_domain()
            for sampled_class in sampled_classes:
                class_sel = []
                for cur_domain in range(self.nb_domains):
                    cur_domain_ith_class_idxs = np.nonzero(
                        np.logical_and(self.ys == sampled_class, self.domains == cur_domain)
                    )[0]
                    cur_domain_class_sel = np.random.choice(cur_domain_ith_class_idxs,
                                                            size=domain_ipc[cur_domain], replace=False)
                    class_sel.append(cur_domain_class_sel)
                ret.extend(np.random.permutation(np.concatenate(class_sel)))
            num_batches -= 1
        return iter(ret)

    def __len__(self):
        return self.num_samples
    #def __len__(self):
    #    return self.length


    """
    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)
    """
    """
    def __len__(self):
        return self.length

    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = data_source.ys
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.data_source)
        self.num_classes = len(set(self.ys))

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.ys) == sampled_classes[i])[0]
                class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)
    """


def build_sampler(
    sampler_type, cfg=None, data_source=None, batch_size=32, n_domain=0, images_per_class = -1
):
    if sampler_type == 'RandomSampler':
        #pdb.set_trace()
        return RandomSampler(data_source)

    elif sampler_type == 'SequentialSampler':
        return SequentialSampler(data_source)

    elif sampler_type == 'RandomDomainSampler':
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == 'BalancedSampler':
        assert images_per_class > -1
        #pdb.set_trace()
        return BalancedSampler(data_source, batch_size, images_per_class)

    elif sampler_type == 'DomainBalancedSampler':
        assert images_per_class > -1
        #pdb.set_trace()
        return DomainBalancedSampler(data_source, batch_size, images_per_class)
    else:
        raise ValueError('Unknown sampler type: {}'.format(sampler_type))
