import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    """_summary_
    
    기본적인 CNN을 이용한 마스크 착용 여부 분류 모델 클래스

    Args:
        num_classes (int): 분류할 클래스 수
        x (torch.Tensor): 모델에 입력할 이미지 데이터. 크기는 (batch_size, channel=3, height, width)

    Returns:
        output (torch.Tensor): foward 결과 output. 크기는 (batch_size, num_classes)
    """
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
    