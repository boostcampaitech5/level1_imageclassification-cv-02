import torch
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop,\
ColorJitter, RandomRotation, RandomHorizontalFlip
from ._util import MaskLabels, GenderLabels, AgeLabels
import numpy as np

class BaseAugmentation:
    """_summary_
    Validation or Test Data augmentation 
    CenterCrop, Resize, ToTensor, Normalize 를 사용합니다.

    retrun (tensor) : (3,resize[0], resize[1]) 모양의 tensor를 return 합니다.
    """
    def __init__(self, resize,  mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.transform = Compose([
            CenterCrop((320, 256)),
            Resize(resize),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class CustomAugmentation:
    """_summary_
    Train Data augmentation 
    CenterCrop, RandomHorizontalFlip, ToTensor, Normalize 를 사용합니다.

    retrun (tensor) : (3,resize[0], resize[1]) 모양의 tensor를 return 합니다.
    """
    def __init__(self, resize,  mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        """_summary_

        Args:
            resize (tuple): Resize image parameter
            mean (tuple, optional): Normalize mean parameter. Defaults to (0.548, 0.504, 0.479).
            std (tuple, optional): Noramlize std parameter. Defaults to (0.237, 0.247, 0.246).
        """
        self.transform = Compose([
            CenterCrop((320, 256)),
            Resize(resize),
            # ColorJitter(0.1,0.1,0.1,0.1),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
       이미지에 Gausszian Noisze 추가
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    

def mixup_collate_fn(batch):
    """_summary_
    DataLoader에 사용할 collate function 

    배치를 셔플하여 원래의 배치와 선형 결합.

    Args:
        batch (tensor): input
    Returns:
        batch (tensor): mixup input
    """
    indice = torch.randperm(len(batch))
    value = np.random.beta(0.1,0.1)
    t = type(batch[0][1])
    if t == AgeLabels:
        num_classes = 3
    elif t == GenderLabels:
        num_classes = 2
    elif t == MaskLabels:
        num_classes = 3
    else:
        num_classes = 18

    if len(batch[0])==2:
        img = []
        label = []
        for a,b in batch:
            temp = torch.tensor([0]*num_classes)
            if num_classes==18:
                temp[b] =1
            else:
                temp[b.value] = 1
            img.append(a)
            label.append(temp)
        img = torch.stack(img)
        label = torch.stack(label)
        shuffle_label = label[indice]

        label = value * label + (1 - value) * shuffle_label
    else:
        img = torch.stack(batch)    
    shuffle_img = img[indice]

    img = value * img + (1 - value) * shuffle_img

    if len(batch[0])==2:
        return img, label
    else:
        return img