from dassl.utils import Registry, check_availability
import pdb
NETWORK_REGISTRY = Registry('NETWORK')


def build_network(name, verbose=True, **kwargs):
    avai_models = NETWORK_REGISTRY.registered_names()
    check_availability(name, avai_models)
    if verbose:
        print('Network: {}'.format(name))
    res = NETWORK_REGISTRY.get(name)(**kwargs)
    print('build_network'); pdb.set_trace()
    return res
