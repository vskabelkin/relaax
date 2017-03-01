from ..common import network


def make(config, sess):
    return network.make_mlps(config, sess)


def make_head(config, pnet, vnet, sess):
    return network.make_wrappers(config, pnet, vnet, sess)


def make_ppo(config, policy, sess):
    return network.PpoLbfgsUpdater(config, policy, sess)