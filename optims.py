from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, CyclicLR, ReduceLROnPlateau

_optimizer_entrypoints = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adam': Adam,
}

_scheduler_entrypoints = {
    'steplr': StepLR,
    'lambdalr': LambdaLR,
    'exponentiallr': ExponentialLR,
    'cosineannealinglr': CosineAnnealingLR,
    'cycliclr': CyclicLR,
    'reducelronplateau': ReduceLROnPlateau
}

def optimizer_entrypoint(optimizer_name):
    return _optimizer_entrypoints[optimizer_name]


def is_optimizer(optimizer_name):
    return optimizer_name in _optimizer_entrypoints


def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]


def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoints


def create_optimizer(optimizer_name, parameter, args):
    """_summary_

    Args:
        optimizer_name (str): [] 사용가능

    Raises:
        RuntimeError: 해당 하는 optimizer가 없다면 raise error

    Returns:
        optimizer (Module): 해당 하는 optimizer return
    """
    if is_optimizer(optimizer_name):
        create_fn = optimizer_entrypoint(optimizer_name)
        if optimizer_name in 'sgd':
            optimizer = create_fn(
                parameter,
                lr= args.lr,
                momentum = args.momentum,
                weight_decay = args.weight_decay
                )

        else:
            optimizer = create_fn(
                parameter,
                lr= args.lr,
                weight_decay = args.weight_decay
                )
    else:
        raise RuntimeError('Unknown optimizer (%s)' % optimizer_name)
    return optimizer


def create_scheduler(scheduler_name, optimizer, args):
    """_summary_

    Args:
        scheduler_name (str): [] 사용가능

    Raises:
        RuntimeError: 해당 하는 scheduler가 없다면 raise error

    Returns:
        scheduler (Module): 해당 하는 scheduler return
    """
    if is_scheduler(scheduler_name):
        create_fn = scheduler_entrypoint(scheduler_name)
        kargs = {}
        if scheduler_name =="steplr":
            kargs['step_size'] = args.lr_decay_step
            kargs['gamma'] = args.gamma
        
        elif scheduler_name =="lambdalr":
            kargs['lr_lambda'] = args.lr_lambda
        
        elif scheduler_name =="exponentialLR":
            kargs['gamma'] = args.gamma
        
        elif scheduler_name == "cosineannealinglr":
            kargs['T_max'] = args.tmax
            kargs['eta_min'] = args.lr*0.01
        
        elif scheduler_name == "cycliclr":
            kargs['base_lr'] = args.lr
            kargs['max_lr'] = args.maxlr
            kargs['step_size_up'] = args.tmax
            kargs['mode'] = args.mode
        
        elif scheduler_name == "reducelronplateau":
            kargs['mode'] = "max"
            kargs['factor'] = args.factor
            kargs['patience'] = args.patience
            kargs['threshold'] = args.threshold

        scheduler = create_fn(optimizer,**kargs)
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name)
    return scheduler