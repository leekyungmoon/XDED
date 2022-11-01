from dassl.utils import Registry, check_availability
import pdb

BACKBONE_REGISTRY = Registry('BACKBONE')


def build_backbone(name, verbose=True, **kwargs):
    avai_backbones = BACKBONE_REGISTRY.registered_names()
    check_availability(name, avai_backbones)
    if verbose:
        print('Backbone: {}'.format(name))
    res = BACKBONE_REGISTRY.get(name)(**kwargs)
    #print('build_backbone'); pdb.set_trace()
    return res
