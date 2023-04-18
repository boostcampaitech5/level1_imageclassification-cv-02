import torch.nn as nn
import torch.nn.functional as F

import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)
    
    #ResNet
    
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

    #EfficientNet

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

    #VIT

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
        self.backbone = timm.create_model('swin_base_patch4_window7_224',num_classes =num_classes, pretrained=True)
        

    def forward(self, x):
        x = self.backbone(x)
        
        return x


    #MaskMobileNet
    

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