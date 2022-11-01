import argparse, os

class Options():

    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=str, default='', help='path to dataset')
        parser.add_argument(
            '--output-dir',
            type=str,
            default='logs',
            help='output directory'
        )

        parser.add_argument('--LOG_DIR',
                            default='../logs',
                            help='Path to log folder'
        )

        parser.add_argument(
            '--resume',
            type=str,
            default='',
            help='checkpoint directory (from which the training resumes)'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=1,
            help='only positive value enables a fixed seed'
        )
        parser.add_argument(
            '--source-domains',
            type=str,
            nargs='+',
            help='source domains for DA/DG'
        )
        parser.add_argument(
            '--target-domains',
            type=str,
            nargs='+',
            help='target domains for DA/DG'
        )
        parser.add_argument(
            '--transforms', type=str, nargs='+', help='data augmentation methods'
        )
        parser.add_argument(
            '--config-file', type=str, default='', help='path to config file'
        )
        parser.add_argument(
            '--dataset-config-file',
            type=str,
            default='',
            help='path to config file for dataset setup'
        )
        parser.add_argument(
            '--trainer', type=str, default='', help='name of trainer'
        )
        parser.add_argument(
            '--backbone', type=str, default='', help='name of CNN backbone'
        )
        parser.add_argument('--head', type=str, default='', help='name of head')
        parser.add_argument(
            '--eval-only', action='store_true', help='evaluation only'
        )
        parser.add_argument(
            '--model-dir',
            type=str,
            default='',
            help='load model from this directory for eval-only mode'
        )
        parser.add_argument(
            '--load-epoch',
            type=int,
            help='load model weights at this epoch for evaluation'
        )
        parser.add_argument(
            '--no-train', action='store_true', help='do not call trainer.train()'
        )
        parser.add_argument(
            'opts',
            default=None,
            nargs=argparse.REMAINDER,
            help='modify config options using the command-line'
        )
        parser.add_argument(
            '--hsic-warmup',
            type=int,
            default=5,
            help='hsic warmup epoch'
        )
        parser.add_argument(
            '--hsic-interval',
            type=int,
            default=1,
            help='hsic interval epoch'
        )
        parser.add_argument(
            '--gpu-id',
            default = -1,
            type = int,
            help = 'ID of GPU that is used for training.'
        )
        parser.add_argument(
            '--remark',
            type=str,
            default='',
        )

        parser.add_argument(
            '--optimizer',
            default='adamw'
        )
        parser.add_argument(
            '--lr',
            default=1e-3,
            type=float
        )
        parser.add_argument(
            '--epochs',
            default=50,
            type=int
        )
        #parser.add_argument(
        parser.add_argument(
            '--scheduler',
            type=str,
            default='cosine',
            help='cosine, step, exp'
        )
        parser.add_argument(
            '--IPC',
            default=4,
            type=int
        )
        parser.add_argument(
            '--save-every', action='store_true', help='evaluation only'
        )

        return parser.parse_args()
