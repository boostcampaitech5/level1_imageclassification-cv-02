from torch.optim import SGD, Adagrad, Adam


OPTIMIZER_ENTRYPOINTS = {
    'sgd': SGD,
    'adagrad': Adagrad,
    'adam': Adam,
}

def optimizer_entrypoint(optimizer_name):
    """_summary_
    입력된 optimizer_name를 알맞은 대소문자로 바꿔줌
    OPTIMIZER_ENTRYPOINTS(dict)에 해당하는 key를 넣으면 value를 출력

    Args:
        optimizer_name (str): [sgd, adagrad, adam] 사용가능

    Returns:
        알맞은 대소문자로 바꾼 optimizer_name(str) return
    """
    return OPTIMIZER_ENTRYPOINTS[optimizer_name]


def is_optimizer(optimizer_name):
    """_summary_
    입력된 optimizer_name에 해당하는 지 확인
    OPTIMIZER_ENTRYPOINTS(dict)에 해당하는 key가 있는 확인

    Args:
        optimizer_name(str): [sgd, adagrad, adam] 사용가능

    Returns:
        Boolean : 해당하는 값이 있으면 True, 없으면 False
    """
    return optimizer_name in OPTIMIZER_ENTRYPOINTS


def create_optimizer(optimizer_name, parameter, args):
    """_summary_

    Args:
        optimizer_name (str): [sgd, adagrad, adam] 사용가능

    Raises:
        RuntimeError: 해당 하는 optimizer가 없다면 raise error

    Returns:
        optimizer (Module): 해당 하는 optimizer(str) return
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