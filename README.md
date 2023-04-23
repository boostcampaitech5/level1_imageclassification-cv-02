# 랩 업 리포트

## CV-2조 (함께가조)

강대호, 김서인, 이윤석, 정대훈

# 1. 프로젝트 개요

- **프로젝트 주제 -** 카메라로 촬영한 사람 얼굴 이미지의 마스크 착용 여부를 판단

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bad4b63c-88f6-4ee9-8433-098ad54ad47e/Untitled.png)

- **프로젝트 목표**
    - 다양한 방법 구현 및 실험을 통해 학습 과정 이해
    - 교육에서 학습한 것들을 모두 활용
    - 점수에 집착하지 않고, 배운 것들에 익숙해지고 새로운 시도들을 많이 해보는 것을 목표
- **프로젝트 개발 환경 및 협업 툴**
    - 개발 환경
        - Ubuntu 18.04.5 LTS
        - Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
        - Ram 90GB
        - Tesla V100 32GB
    - 협업 툴
        - Github
        - Zoom
        - Notion
        - TensorBoard
- **구조도(연관도)**
    - Dataset
        - 전체 사람 명 수 : 4,500 (train: 2,700, test: 1800)
        - 한 사람 당 사진의 개수: 7 [마스크 착용 5장, 잘못된 착용 1장, 미착용 1장]
        - 이미지 크기: (384, 512)
        - 클래스 : 마스크 착용 여부, 성별, 나이를 기준으로 총 18개의 클래스
    - 프로젝트 구조
        
        ```python
        project/
        ├── dataset.py         # Dataset 및 augmentation
        ├── model.py           # Load model
        ├── train.py           # Train 코드 
        ├── inference.py       # Inference 코드 및 Test 예측
        ├── loss.py            # loss 선언 코드
        ├── ensemble_csv.py    # Hard Voting
        ├── requirements.txt   # 환경 셋팅
        └── Data_EDA.ipynb     # Data EDA 결과 파일
        ```
        

# 2. 프로젝트 팀 구성 및 역할

- **강대호**
    - 이미지 분류 모델의 성능 향상을 위해 다양한 Augmentation 기법들을 적용하고, 그에 따른 training 결과를 비교 분석함.
- **김서인**
    - 다양한 Optimizer와 Scheduler를 사용하여 이미지 분류 모델의 성능을 비교하고, hyperparameter에 대한 실험을 진행함.
- **이윤석**
    - 팀을 이끌어가며 모델 학습에 필요한 여러 기능들을 구현하고, kfold, mixup, cutmix, canny 등을 적용해보며 모델 성능을 향상 시키는 실험을 진행함.
- **정대훈**
    - 다양한 pretrained 모델을 사용하여 실험을 진행하고, 이미지 분류 모델의 성능을 비교 분석함.

# 3. 프로젝트 수행 절차 및 방법

- 프로젝트 계획

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d36227d0-fe33-43be-985e-39bdf5d64530/Untitled.png)

1. **EDA**
    - 연령별 불균형 → 60세 이상이 매우 적음
    - 성별별 불균형 → 여자가 비교적 많음 다만(3:2로 꽤나 불균형한 편) 전체적인 분포는 유사함
    - incorrect mask는 다양한 방식이 존재
    - 데이터 labeling 오류 → normal과 incorrect 오류, ‘gender’, ‘age’ 잘못된 라벨링
    - Mask class 별로 RGB 통계값을 분석한 결과 클래스별로 뚜렷한 구분이 없었음
2. **성능 개선 아이디어 및 실험**
- Data
    - overfitting을 방지하도록 다양한 augmentation을 진행해 효과적인 augmentation 탐색
    - WeightedRandomSampler를 통해 class의 불균형 문제 해결 시도
    - StratifiedKFold cross validation 기법을 통해 불균형 문제 해결 시도
    - 다양한 augmentation 기법을 시도 ( pytorch.transform , mix up, cutmix 등 )
- Loss
    - Baseline의 4가지 loss인 focal, cross entropy, label_smoothing, f1 비교
    - 불균형 데이터를 방지 하기 위한 arcface loss 추가
- Models
    - Image classification task에 많이 사용되는 pretrained model 중 가벼운 것 위주로 사용
        - EfficientNet b1~b4, VIT, SWIN Transformer, Resnet18, Resnet34 , MobileNet
    - 같은 모델이더라도 모델 크기에 따라 성능 비교
    - 학습 시간을 줄이고 overfitting 방지를 위해 early stopping 사용
- Optimizer
    - SGD, Adam, Momentum, Adagrad 등의 다양한 optimizer를 비교하여 분석
    - weighted decay, lr, momentum 등 hyper parameter 탐색
- scheduler
    - ExponentialLR, lambdalr, CosineAnnealingLR, ReduceLROnPlateau 등 데이터와 모델에 맞는 최적화된 scheduler 탐색
- 기타
    - MaskSplitByProfileDataset, MaskBaseDataset 비교
    - 각각의 mask, gender, age를 따로 학습해 성능 실험
    - ensemble
        - 각각의 클래스별 모델 ensemble
        - StratifiedKFold
        - soft & hard voting
1. **프로젝트 관리 및 협업 툴**
- **Argparser**
    - 하나의 train코드로 argparse를 통해 다양한 실험을 할 수 있도록 코드를 작성

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5892ee7e-3699-4269-b655-f0471ec2b50c/Untitled.png)

- **Github**
    - Git flow를 이해하고, 멘토링 시간에 배운 협업 기법들을 최대한 사용하는 것을 목표로 함
        - develop, bugfix, 개인 branch를 사용하여 전체 프로젝트를 기능 별로 관리
    - Issue를 통해서 추가해야 할 코드를 알리고 버그를 알리면서 진행 상황을 관리
    - PR를 통해서 추가한 코드에 대한 설명 및 승인
    - Issue , PR 은 양식을 정해 놓고 버전이 한눈에 파악될 수 있도록 작성
- **Notion**
    - 각자 나눈 역할 정리
    - 수행한 실험 결과 정리
- **Zoom**
    - 아이디어 회의, 소통 및 의견 공유
- **TensorBoard**
    - 모델 실험을 기록하고 관리
    - 다양한 방법을 적용해보면서 성능을 비교

# 4. 프로젝트 수행 결과

- **결과** 최종 순위 9등
    - Public f1_score = 0.7556, acc = 81.1111 (9등)
    - Private f1_score = 0.7411, acc = 80.6825 (9등)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c3d58bdd-f2fd-413a-a673-bbd285c221ec/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/3cdf8609-231e-453d-acca-e127f71ffd86/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d1093079-062e-4f69-947c-aede79d03bc5/Untitled.png)

 ****✅: 성능향상에 도움이 됐던 아이디어

- **Data**
    - Augmentation
        - dataset을 확인하였을 때 test set에서 색이 다른 스카프를 쓰고 있는 데이터를 보고 train set에서는 하얀색의 스카프를 가지고 있어 color jitter를 추가하였으나 성능 저하로 사용하지 않음
        - ✅ 대부분의 데이터가 중앙에 얼굴이 위치하는 것으로 확인이 되어 CenterCrop을 사용함. 약간의 성능 향상이 있어 학습 시에 사용함 약간의 성능 향상이 있어 학습 시에 사용함
        - ✅ 다양한 augmentation 실험을 하면서 ColorJitter, RandomRotation, HorizontalFlip등을 해 보았지만 기본 이미지와 HorizontalFlip이 가장 좋은 성능을 보임
        - baseline 코드의 사진들의 mean, std 값을 3000개의 이미지로만 계산하여 전체 이미지의 값을 계산함. mean = (0.560,0.524,0.501), std = (0.6165, 0.587, 0.568) 로 데이터의 차이가 큰 것을 한번 더 확인 할 수 있었음.
            
            → ImageNet의 값들을 추가해 총 3개 가지의 파라미터로 실험하였으나 큰 차이는 없음.
            
        - ✅ Mixup, Cutmix를 통해 모델 일반화 능력을 향상 시키고 배경이나 옷 등의 불필요한 데이터 추출을 방지하기 위해 사용하였으나 Mixup은 성능 저하를 Cutmix는 약간의 성능 향상을 보였으나 학습 수렴을 하지 않음.
            
            → Cutmix를 사용하였을 때 학습 수렴을 위해 데이터 셋에 대한 이해가 더 필요해 보임
            
        - ✅ 이미지의 비율을 해치지 않고 학습하는 것이 더 좋을 것이라 판단하였으나 이미지 사이즈를 모델을 학습한 사이즈에 맞게 resize하였더니 더 좋은 성능 향상.
            
            → Crop → Resize 순서로 모델의 학습된 데이터 셋에 맞게 변경
            
    - split dataset
        - ✅기존의 MaskBaseDataset(마스크 기준)에서 동일 인물의 사진이 학습 셋과 검증 셋에 섞여 들어가는 오버 피팅이 발생하는 것을 방지하기 위해 MaskSplitByProfileDataset(사람 기준)을 사용하여 train와 test 사이의 gap을 줄임.
        - ✅Stratified kfold validation을 사용할 때 mask 분류 모델은 MaskBaseDataset, gender와 age 분류 모델은 MaskSplitByProfileDataset을 사용하여 train set이 validation set과 유사하지 않도록 분리
            
            Stratified kfold validation을 사용하면서 각각의 분리된 데이터의 모델 성능이 유사하지 않고 차이가 많이 나는 것으로 보아 데이터셋의 편차가 큰 것으로 확인
            
        - WeightedRandomSampler를 사용해 data Imbalance문제를 해결하려 했으나 60대 이상의 데이터가 너무 중복되어 학습하여 overfitting 발생
- **Loss**
    - ✅ base model에서 실행해본 결과 F1Loss와 FocalLoss가 CrossEntropy 보다 약간의 성능 향상을 보임. F1 score가 대회의 지표이므로 선택하였으나 학습에는 Loss의 값이 작아 이러한 부분을 보완한다면 더 좋은 학습이 될 것으로 생각이 됨
    - ✅ Mixup을 사용하기 위해 label을 one-hot encoding하여 사용하였으므로 Binary Cross Entropy Loss를 사용함
    - ✅ Cutmix를 사용할 때에는 섞인 이미지 각각을 분류할 수 있는 강력한 Loss를 사용하기 위해 Focal Loss를 사용함
    - Arcface loss 를 사용하려 했으나 대회 기간 내에 효과적인 조건을 찾지 못하여 보류함
- **Models**
    - 모델 실험
        - ✅ Image classification task에 많이 사용되는 pretrained model 중 모델 크기가 너무 크지도 않고  작지도 않을 때 성능이 잘나옴
        - ✅ ResNet18과 ResNet34 를 비교한 결과 ResNet34 가 성능이 조금 더 좋았음
        - EfficientNet b1~b4를 비교한 결과 EfficientNet b1~b2 성능이 좋았음
        - MobileNet 은 모델 크기가 너무 작아 성능이 잘 안 나옴
        - ✅ SwinSmallWindowVIT은 Resnet18 에 비하여 자체적으로 평가한 f1 스코어는 높지만 제출하여 얻는 f1 스코어는 낮아 Resnet18 가 일반화가 더 잘 되어 있다고 판단함.
            
            학습 시간이 오래 걸려 많은 실험을 하지 못하여 다음번에는 mixed precision을 사용하여 학습 시간을 줄인다면 더욱 효율적인 실험이 될 것으로 판단됨
            
    - 학습 방법
        - ✅ early stopping을 통해서 과도한 학습을 방지와 학습 시간을 절약함.
    - ✅ Edge Canny
        - Canny Edge Detection 기법을 통해 RGB 이미지 정보 뿐만 아닌 edge정보도 추가한다면 주름이나 마스크의 모양 등의 정보로 더욱 분류를 잘 할 수 있을 것이라 생각되어 구현
        - shape를 맞추기 위해 input 4 output 3 모양인 1x1 conv를 사용하였으나 조금 더 layer를 추가하여 확실하게 정보를 추출한다면 더 나은 성능을 보일 수 있을 것이라고 생각이 됨
        - mask,gender 분류에는 좋지 않았으나 age분류에는 성능 향상을 보임.
- **Hyperparameter Tuning**
    - SGD, Adam 비교
        
        SGD는 학습 속도는 느리지만 점진적으로 성능이 나아지는 그래프를, Adam은 학습 속도가 매우 빠르지만 쉽게 overfitting 현상이 나옴. 
        
        ✅시간이 부족하여 SGD 보다는 Adam을 사용하여 overfitting을 방지하는 다양한 기법들을 추가 하기로 함
        
    - weight decay
        
        ✅학습 시 accuracy와 loss가 크게 요동치는 것을 확인하여 weight decay를 늘여 안정된 학습을 하도록 parameter를 변경함
        
    - learning rate Scheduler 비교
        - ExponentialLR, StepLR, CosineAnnealingLR, ReduceLROnPlateau
            - ✅ StepLR을 사용하기에는 step과 factor의 더 나은 parameter를 찾지 못하여 ReduceLROnPlateau scheduler를 사용해 성능이 나아지지 않았을 때 lr를 줄임
            - ✅ CosineAnnealingLR은 lr의 0.001배의 최소값을 반복하도록 하였으며 최소값이 너무 낮을 경우 lr의 변화가 커 f1_score가 수렴하지 못하는 문제를 해결하도록 함
                
                작은 변화 폭을 주어 수렴도 하면서 local minima를 탈출하도록 조정함. 지속적으로 학습이 가능하나 시간이 오래 걸림
                
    - Batch size 비교
        
        처음 기본 batch size인 64로 사용하였으나 데이터 수가 적은 60대 이상의 라벨과 잘못된 마스크를 쓰는 라벨 등의 loss정보가 평균값을 취하면서 무뎌진다고 판단하여 Batch size를 줄이며 실험을 진행. 
        
        ✅ Batch size가 적을 때 더 나은 일반화 성능을 보여 낮은 Batch size로 최종 결과를 제출
        
- **Ensemble**
    - ✅ age, gender, mask를 따로 고려해서 각자 다른 모델로 학습하고 ensemble하여 soft voting을 통해 하나로 합침, 성능 향상이 있었음
    - ✅ [224, 224], [384,384], [512,384] 등 이미지의 크기와 nomalize mean,std 값을 변경하여 다양한 데이터로 학습한 모델들의 결과를 hard voting으로 ensemble을 하여 성능을 높임

# 5. 자체 평가 의견

### 잘한 점들

- Pytorch Template을 통해 모듈화된 프로젝트 구조를 학습했다.
- Github로 협업하며 각자의 역할을 분담해 협업 능력을 키운 것이 좋았습니다.
- 데일리 스크럼 및 피어세션 시간에 서로의 진행 사항을 공유하는 적극적인 의사소통을 통해 다양한 관점에서 문제를 해석하고 해결 방안을 고민하여 모델 학습에 대한 이해를 키웠습니다.
- 결과 보다는 개인의 역량 발전에 초점을 두어 작업한 것이 서로의 실력을 키운 좋은 기회가 된 것 같다.
- 점수보다는 다양한 시도와 실험 분석에 집중하여 다양한 관점을 키운 것이 도움이 많이 됐다.

### 아쉬웠던 점들

- 점수에 관심이 쏠려 더 다양한 아이디어와 해결 방안을 적용 시켜보지 못한 것이 아쉽다.
- 실험을 편하게 하기 위한 Baseline 작성이 먼저 되다 보니 제출을 많이 하지 못해 정확한 분석으로 이루어 지지 않은 것이 아쉬웠다.
- WandB를 이용했으면 효과적으로 실험 결과를 공유할 것 있었을 텐 데 이용하지 않았던 게 아쉽다.
- 실험 진행이 좀 더 원활하게 되도록 mixed precision으로 학습 시간 단축이나 json으로 실험 관리를 시도해보지 않은 것이 아쉽다.

### 프로젝트를 통해 배운점 또는 시사점

- 모델들의 성능 보다는 데이터에 대한 이해가 더 중요하다는 것을 알게 되었다. 가장 성능이 좋은 sota 보다도 모델에 대한 이해로 알맞은 모델을 선정하고 알맞은 조합을 찾는 것이 가장 중요하다고 생각되었다.
- 실험 결과에 대해 토론하면서 ML 성능을 향상 시키기 위한 아이디로 많이 얻을 수 있었다.
- 초기의 계획대로 점수보다 협업, 실험 관리, 전체적인 흐름을 배우고, 활용하는 것에 더 초점을 맞추니 전반적인 흐름 파악하는 것에 더욱 수월하였다.
- 다양한 의견들을 공유하는 것이 문제 해결에 많은 도움이 되며 때로는 특이한 아이디어가 가장 효과적인 해결 방법일 수 있다고 생각된다.

# 6. 레퍼런스

WeightedRandomSampler

- Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster R-CNN: Towards real-time object detection with region proposal networks. In Advances in neural information processing systems (pp. 91-99)

StratifiedKFold cross validation

- Kohavi, R. (1995). A study of cross-validation and bootstrap for accuracy estimation and model selection. In Ijcai (Vol. 14, No. 2, pp. 1137-1145)

Cutmix

- Sangdoo Yun, Seong Joon Oh, Sanghyuk Chun, Jongwon Kim. "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features." In Proceedings of the International Conference on Computer Vision (ICCV), 2019.

Edge Canny Detection

- Canny, J. (1986). A computational approach to edge detection. IEEE Transactions on pattern analysis and machine intelligence, (6), 679-698.
