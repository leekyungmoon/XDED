from dassl.utils import Registry, check_availability
import pdb
TRAINER_REGISTRY = Registry('TRAINER')


def build_trainer(cfg):
    avai_trainers = TRAINER_REGISTRY.registered_names()
    print(avai_trainers)
    check_availability(cfg.TRAINER.NAME, avai_trainers)
    if cfg.VERBOSE:
        print('Loading trainer: {}'.format(cfg.TRAINER.NAME))
    res = TRAINER_REGISTRY.get(cfg.TRAINER.NAME)(cfg)
    #print('build_trainer'); pdb.set_trace()
    return res
