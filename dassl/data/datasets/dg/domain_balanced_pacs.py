import os.path as osp

from ..build import DATASET_REGISTRY
from ..base_dataset import Datum, DatasetBase

import pdb

@DATASET_REGISTRY.register()
class DomainBalancedPACS(DatasetBase):
    dataset_dir = 'pacs'
    domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    data_url = 'https://drive.google.com/uc?id=1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE'
    # the following images contain errors and should be ignored
    _error_paths = ['sketch/dog/n02103406_4068-1.png']

    def __init__(self, cfg):
        root = osp.abspath(osp.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, 'images')
        self.split_dir = osp.join(self.dataset_dir, 'splits')

        if not osp.exists(self.dataset_dir):
            dst = osp.join(root, 'pacs.zip')
            self.download_data(self.data_url, dst, from_gdrive=True)

        self.check_input_domains(
            cfg.DATASET.SOURCE_DOMAINS, cfg.DATASET.TARGET_DOMAINS
        )

        #train = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'train')
        train_list = []
        for s_domain in cfg.DATASET.SOURCE_DOMAINS:
            train_list.append(
                self._read_data([s_domain], 'train')
            )
        val = self._read_data(cfg.DATASET.SOURCE_DOMAINS, 'crossval')
        test = self._read_data(cfg.DATASET.TARGET_DOMAINS, 'all')

        super().__init__(train_x=train_list, val=val, test=test)
