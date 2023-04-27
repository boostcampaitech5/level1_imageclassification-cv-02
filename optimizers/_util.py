from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from loss import F1Loss,FocalLoss, LabelSmoothingLoss
import torch.nn as nn


OPTIMIZER_ENTRYPOINTS = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adam': Adam,
}


CRITERION_ENTRYPOINTS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'bce': nn.BCEWithLogitsLoss,
}


SCHEDULER_ENTRYPOINTS = {
    'steplr': StepLR,
    'lambdalr': LambdaLR,
    'exponentiallr': ExponentialLR,
    'cosineannealinglr': CosineAnnealingLR,
    'cycliclr': CyclicLR,
    'reducelronplateau': ReduceLROnPlateau
}


def optimizer_entrypoint(optimizer_name):
    return OPTIMIZER_ENTRYPOINTS[optimizer_name]


def is_optimizer(optimizer_name):
    return optimizer_name in OPTIMIZER_ENTRYPOINTS


def scheduler_entrypoint(scheduler_name):
    return SCHEDULER_ENTRYPOINTS[scheduler_name]


def is_scheduler(scheduler_name):
    return scheduler_name in SCHEDULER_ENTRYPOINTS


def criterion_entrypoint(criterion_name):
    return CRITERION_ENTRYPOINTS[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in CRITERION_ENTRYPOINTS


def create_criterion(criterion_name, **kwargs):
    """_summary_

    Args:
        criterion_name (str): ['cross_entropy', 'focal', 'label_smoothing', 'f1', 'bce'] 사용가능

    Raises:
        RuntimeError: 해당 하는 loss가 없다면 raise error

    Returns:
        loss (Module): 해당 하는 loss return
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion

