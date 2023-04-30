import torch.nn as nn
import torch.nn.functional as F

import timm
import torch
import cv2
import numpy as np
from optimizers.loss import ArcMarginProduct

class MaskResnet18(nn.Module):
    """_summary_
    
    ResNet-18 기반의 마스크 착용 여부 분류 모델 클래스

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet18',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet34(nn.Module):
    """_summary_
    
    ResNet-34 기반의 마스크 착용 여부 분류 모델 클래스

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet34',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet50(nn.Module):
    """_summary_
    
    ResNet-50 기반의 마스크 착용 여부 분류 모델 클래스
    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet50',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet101(nn.Module):
    """_summary_
    
    ResNet-101 기반의 마스크 착용 여부 분류 모델 클래스

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet101',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskEfficientNet_b0(nn.Module):
    """_summary_
    
    EfficientNet b0 기반의 마스크 착용 여부 분류 모델 클래스
    input image size를 224*224일 때 최적화되어 있음
    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b1(nn.Module):
    """_summary_
    
    EfficientNet b2 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 260*260일 때 최적화되어 있음

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b1',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b2(nn.Module):
    """_summary_
    
    EfficientNet b3 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 300*300일 때 최적화되어 있음

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskEfficientNet_b3(nn.Module):
    """_summary_
    
    EfficientNet b4 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 380*380일 때 최적화되어 있음

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b4(nn.Module):
    """_summary_
    
    TinyVIT 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224일 때 최적화되어 있음

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskTinyVIT(nn.Module):
    """_summary_
    
    TinyVIT 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSmallVIT(nn.Module):
    """_summary_
    
    SmallVIT 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskSwinSmallWindow(nn.Module):
    """_summary_
    
    SwinSmallWindow 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_small_patch4_window7_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinSmall(nn.Module):
    """_summary_
    
    SwinSmall 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_s3_small_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinBase(nn.Module):
    """_summary_
    
    SwinBase 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_s3_base_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinBaseWindow(nn.Module):
    """_summary_
    
    SwinBaseWindow 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskMobileNet_125(nn.Module):
    """_summary_
    
    MobileNet_125 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 224*224, 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('mobilevitv2_125',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskMobileNet_150(nn.Module):
    """_summary_
    
    obileNet_150 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('mobilevitv2_150',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class Coatnet(nn.Module):
    """_summary_
    
    SwinBaseWindo 기반의 마스크 착용 여부 분류 모델 클래스
    input image size가 384*384이여야함

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.create_model("hf_hub:timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",num_classes=num_classes, pretrained=True)

    def forward(self,x):
        x = self.backbone(x)
        return x


class Canny(nn.Module):
    """_summary_
    
    입력된 이미지(x)에 Canny edge detection을 적용한 결과를 추가적인 채널로 생성하여,
    입력된 모델(backbone)을 통과시키는 모델

    Args:
        x (torch.Tensor): 크기가 (batch_size, channel=3+1, height, width)인 입력 이미지 텐서

    Returns:
        output (torch.Tensor): 모델의 순전파 연산 결과. 크기는 (batch_size, num_classes)
    """
    def __init__(self,backbone):
        super().__init__()
        self.add_canny = nn.Conv2d(4,3,1)
        self.backbone = backbone

    def forward(self,x):
        # batch, channel, h, w
        canny=[]
        s = np.uint8(x.detach().cpu().permute(0,2,3,1).numpy())
        for n in s:
            gray = cv2.cvtColor(n,cv2.COLOR_RGB2GRAY)
            gray = cv2.Canny(gray, 100,200)
            canny.append(torch.tensor(gray).float().unsqueeze(0)/255)
        canny = torch.stack(canny).cuda()

        x = torch.cat([canny,x],dim=1)
        x = self.add_canny(x)
        x = self.backbone(x)
        return x
    
    
class ArcfaceModel(nn.Module):
    """_summary_
    
    ArcFace 알고리즘을 사용하여 입력 이미지의 특징을 추출하고 분류를 수행하는 모델

    Args:
        backbone (nn.Module): 백본 신경망 모델
        num_features (int): 입력 이미지의 특징 벡터의 크기
        num_classes (int): 클래스 개수

    Returns:
        logits (torch.Tensor): ArcMarginProduct에 따라 margin 값을 계산한 output (batch_size, num_classes)
    """
    def __init__(self,backbone,num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(in_features=num_features, out_features=num_classes)

    def forward(self,x,labels):
        feature = self.backbone(x)
        logits = self.arcface(feature, labels)
        return logits


class ArcfaceModelInfer(nn.Module):
    """_summary_
    
    인퍼런스에 사용할 Arcface 모델

    Args:
        backbone (nn.Module): 백본 신경망 모델
        num_features (int): 입력 이미지의 특징 벡터의 크기
        num_classes (int): 클래스 개수

    Returns:
        logits (torch.Tensor): 분류 결과 output (batch_size, num_classes)
    """
    def __init__(self,backbone,num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = nn.Linear(in_features=num_features, out_features=num_classes,bias=False)

    def forward(self,x):
        x = self.backbone(x)
        x = F.normalize(x, dim=1)
        logits = self.arcface(x)
        return logits