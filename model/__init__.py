import logging

logger = logging.getLogger('base')


def create_model(opt, choice):
    from .model import DDPM as M
    m = M(opt, choice)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m


def create_fusionNet(opt):
    from .fusionModel import fusionModel as M2
    m = M2(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
