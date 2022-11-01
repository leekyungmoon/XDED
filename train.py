import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

import options as options

import pdb, os

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

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

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

    cfg.remark = args.remakr
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
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_cfg(args)
    #print('setup_cfg'); pdb.set_trace()
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    #print_args(args, cfg)
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
