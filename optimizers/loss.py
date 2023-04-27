import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        """_summary_

        Args:
            weight (list, optional): 가중치를 더할 list. Defaults to None.
            gamma (float, optional): (1-p)**gamma . Defaults to 2..
            reduction (str, optional): mean,sum 등의 return할 loss 계산. Defaults to 'mean'.
        """
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
    

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, label):
        cosine = F.linear(F.normalize(inputs), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=inputs.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.1, dim=-1):
        """_summary_

        Args:
            classes (int, optional): classes개수 만큼 smoothing. Defaults to 3.
            smoothing (float, optional): smoothing 할 비율. Defaults to 0.1.
            dim (int, optional): Defaults to -1.
        """
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    """_summary_
    label과 predict로 f1_score를 계산하여 Loss로 계산
    return : 1-f1core
    """
    def __init__(self, classes=3, epsilon=1e-7):
        """_summary_

        Args:
            classes (int, optional): classes 개수. Defaults to 3.
            epsilon (float, optional): _description_. Defaults to 1e-7.
        """
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

CRITERION_ENTRYPOINTS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'bce': nn.BCEWithLogitsLoss,
}

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

CRITERION_ENTRYPOINTS = {
    'cross_entropy': nn.CrossEntropyLoss,
    'focal': FocalLoss,
    'label_smoothing': LabelSmoothingLoss,
    'f1': F1Loss,
    'bce': nn.BCEWithLogitsLoss,
}

def criterion_entrypoint(criterion_name):
    return CRITERION_ENTRYPOINTS[criterion_name]


def is_criterion(criterion_name):
    return criterion_name in CRITERION_ENTRYPOINTS