import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import options as options

import os, pdb

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    cfg.DATASET.ROOT = '/root/pytorch_datasets'

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    cfg.DATASET.SOURCE_DOMAINS = ['art_painting', 'cartoon', 'photo']
    cfg.DATASET.TARGET_DOMAINS = ['sketch']

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.hsic_interval:
        cfg.TRAINER.HSIC_INTERVAL = args.hsic_interval
    if args.hsic_warmup > -1:
        cfg.TRAINER.HSIC_WARMUP = args.hsic_warmup

    cfg.remark = args.remark
    if args.output_dir:
        cfg.OUTPUT_DIR = os.path.join(
            args.output_dir,
            cfg.DATASET.TARGET_DOMAINS[0],
            cfg.TRAINER.NAME+'_'+cfg.remark
        )
    cfg.IPC = args.IPC


def setup_cfg(args):
    cfg = get_cfg_default()
    #pdb.set_trace()
    reset_cfg(cfg, args)
    #pdb.set_trace()
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    else:
        cfg.merge_from_file('configs/datasets/pacs.yaml')
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)

    #cfg.OUTPUT_DIR = os.path.join(
    LOG_DIR = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.DATASET.TARGET_DOMAINS[0],
        #cfg.TRAINER.NAME+'_'+cfg.MODEL.BACKBONE.NAME+'_'+cfg.remark,:6
        cfg.TRAINER.NAME+'_'+cfg.remark,
    )

    if not os.path.exists('{}'.format(cfg.OUTPUT_DIR)):
        os.makedirs('{}'.format(cfg.OUTPUT_DIR))
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    assert args.gpu_id > -1
    torch.cuda.set_device(args.gpu_id)

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    args = options.Options().initialize(argparse.ArgumentParser())
    main(args)
