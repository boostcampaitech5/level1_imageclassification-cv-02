import torch.nn as nn
import torch.nn.functional as F

import timm
import torch
import cv2
import numpy as np
from ..optimizers.loss import ArcMarginProduct

class MaskResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet18',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet34',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet50',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskResnet101(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('resnet101',num_classes =num_classes,  pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskEfficientNet_b0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b1',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b2',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskEfficientNet_b3(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b3',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskEfficientNet_b4(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b4',num_classes =num_classes, pretrained=True)

    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskTinyVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('vit_tiny_patch16_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSmallVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch16_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskSwinSmallWindowVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_small_patch4_window7_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinSmallVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_s3_small_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinBaseVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_s3_base_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskSwinBaseWindowVIT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('swin_base_patch4_window7_224',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class MaskMobileNet_125(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('mobilevitv2_125',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x


class MaskMobileNet_150(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model('mobilevitv2_150',num_classes =num_classes, pretrained=True)
        
    def forward(self, x):
        x = self.backbone(x)
        return x
    

class Coatnet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = timm.create_model("hf_hub:timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",num_classes=num_classes, pretrained=True)
        #self.backbone = timm.create_model("hf_hub:timm/maxvit_small_tf_384.in1k", pretrained=True)
        # self.backbone = timm.create_model("hf_hub:timm/convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=True)
        # self.backbone = timm.create_model("swin_base_patch4_window12_384_in22k",num_classes=num_classes, pretrained=True)
        # self.backbone = timm.create_model("hf_hub:timm/convnext_small.fb_in22k_ft_in1k_384", pretrained=True)
        # self.classifier = nn.Linear(1000,num_classes,bias=True)

    def forward(self,x):
        x = self.backbone(x)
        return x


class Canny(nn.Module):
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
    def __init__(self,backbone,num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = ArcMarginProduct(in_features=num_features, out_features=num_classes)

    def forward(self,x,labels):
        feature = self.backbone(x)
        logits = self.arcface(feature, labels)
        return logits


class ArcfaceModelInfer(nn.Module):
    def __init__(self,backbone,num_features, num_classes):
        super().__init__()
        self.backbone = backbone
        self.arcface = nn.Linear(in_features=num_features, out_features=num_classes,bias=False)

    def forward(self,x):
        x = self.backbone(x)
        x = F.normalize(x, dim=1)
        logits = self.arcface(x)
        return logits