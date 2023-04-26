import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop,\
ColorJitter, RandomRotation, RandomHorizontalFlip

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

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


class MaskLabels(int, Enum):
    """_summary_
    MASK = 0
    INCORRECT = 1
    NORMAL = 2
    """
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    """_summary_
    MALE, FEMALE의 라벨 
    MALE.value == 0
    FEMAEL.value == 1

    def from_str -> str type의 label을 0,1의 값으로 변경
    """
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        """_summary_

        Args:
            value (str): label (str)

        Raises:
            ValueError: label 맞는 value(str)이 들어오지 않으면

        Returns:
            int: label
        """
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    """_summary_
    YOUNG = 0
    MIDDEL = 1
    OLD = 2

    def from_number -> str type의 나이 숫자를 YOUNG, MIDDEL, OLD value로 변경
    """
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 29:
            return cls.YOUNG
        elif value < 58:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    """_summary_
    Dataset (Dataset): Train 폴더에 있는 모든 이미지 데이터를 val_ratio 비율로 분리
    
    """
    num_classes = 18

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }

    image_paths = []
    mask_labels = []
    gender_labels = []
    age_labels = []

    def __init__(self, data_dir, num_classes=18,category="multi", mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        """_summary_

        Args:
            data_dir (_type_): dataset path
            num_classes (int, optional): num_classes에 따른 getitem return 값 변경. Defaults to 18.
            category (str, optional): Category 별 getitem return 값 변경. Defaults to "multi".
            mean (tuple, optional): denormalize를 사용하기 위한 mean값. 없다면 3000개의 이미지 데이터로 계산. Defaults to (0.548, 0.504, 0.479).
            std (tuple, optional): denormalize를 사용하기 위한 mean값. 없다면 3000개의 이미지 데이터로 계산. Defaults to (0.237, 0.247, 0.246).
            val_ratio (float, optional): 전체 데이터셋을 train과 validation으로 나누기 위한 비율. Defaults to 0.2.
        """
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.transform = None
        self.category = category
        self.setup()
        self.calc_statistics()
        self.num_classes = num_classes

    def setup(self):
        """_summary_
            사람별로 모여있는 폴더안에 들어가서 image_paths, image_labels, gender_labels, age_labels 에 하나씩 저장
        """
        # 폴더 list 받기
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue
            # img_folder == 000004_male_Asian_54 
            img_folder = os.path.join(self.data_dir, profile)
            # 폴더안의 image list == [mask1.jpg, mask2.jpg, incorrect.jpg]
            for file_name in os.listdir(img_folder):
                # 확장자 제거
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                # 위 _file_names dict에서 해당하는 라벨 찾기 - > data/000004_male_Asian_54/mask1.jpg
                mask_label = self._file_names[_file_name]

                # 000004_male_Asian_54 -> 000004, male, Asian, 54
                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    def calc_statistics(self):
        """_summary_
        mean과 std 구하기, 3000개만 default mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
        전체 이미지 구할시 mean = [0.56019358 0.52410121 0.501457], std = [0.61664625 0.58719909 0.56828232]

        """
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255
            print(f"image mean = {self.mean}, std = {self.std}")

    def set_transform(self, transform):
        """_summary_
        getitem에 사용되는 transform 지정

        Args:
            transform (transform): Dataset tranform
        """
        self.transform = transform

    def __getitem__(self, index):
        """_summary_
        index 별 image와 label을 return 하는 함수

        Args:
            index (int): index 번호 

        Returns:
            img, label (tensor):
        """
        #assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # 각각의 라벨들로 0~18의 라벨 만들기
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if self.category=="multi":
            label = multi_class_label
        elif self.category =="age":
            label = age_label
        elif self.category == "gender":
            label = gender_label
        elif self.category =="mask":
            label = mask_label

        # transform 적용
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        """_summary_
        index에 해당 하는 mask label return
        """
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        """_summary_
        index에 해당 하는 gender label return
        """
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        """_summary_
        index에 해당 하는 age label return
        """
        return self.age_labels[index]

    def read_image(self, index):
        """_summary_
        index에 해당하는 image path에서 image 읽어오기.
        Args:
            index (int): imag_paths에 에서 선택할 index

        Returns:
            Image (PIL): PIL 타입의 Image
        """
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        """_summary_
        mask, gender, age를 합쳐서 0~17의 라벨 return

        Args:
            mask_label (MaskLabels): Mask label
            gender_label (GenderLabels): Gender label
            age_label (AgeLabels): Age label

        Returns:
            int: 하나의 합쳐진 label
        """
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        """_summary_
        합쳐진 라벨을 mask, gender, age label로 return
        Args:
            multi_class_label (int): 합쳐진 하나의 label

        Returns:
            Tuple[MaskLabels, GenderLabels, AgeLabels]: tuple로 된 3개의 라벨
        """
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        """_summary_
        transform을 통해 normalize 된 이미지 복원
        Args:
            image (PIL): normalize 된 이미지
            mean (tuple): 이미지 mean 값
            std (tuple): 이미지 std 값

        Returns:
            numpy: _description_
        """
        # 정규화한것을 다시 되돌리기
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        # 길이 만큼 random_split
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
        
        def :
            _split_profile(profiles, val_ratio) : 
            setup() : 
            split_dataset() -> List[Subset]:
    """


    def __init__(self, data_dir, num_classes,category="multi", mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        """_summary_

        Args:
            data_dir (str): train data가 있는 파일 경로
            num_classes (int): num_class 개수
            category (str, optional): category 지정. Defaults to "multi".
            mean (tuple, optional): denormalize를 위한 mean 값. Defaults to (0.548, 0.504, 0.479).
            std (tuple, optional): denormalize를 위한 std 값. Defaults to (0.237, 0.247, 0.246).
            val_ratio (float, optional): rain과 Validation을 나누기 위한 비율. Defaults to 0.2.
        """
        # 초기화된 dictionary 만들기 -> https://wikidocs.net/104993
        self.indices = defaultdict(list)
        self.balancing_dict = {}
        super().__init__(data_dir, num_classes, category, mean, std, val_ratio)


    @staticmethod
    def _split_profile(profiles, val_ratio):
        """_summary_
        val ratio 비율 기준으로 '사람 폴더' 별로 Trian, Val indice분리

        Args:
            profiles (list): 사람별 폴더 경로가 저장되있는 list
            val_ratio (float): validation을 나누기 위한 비율

        Returns:
            dict: {train indices, val indices} 
        """
        length = len(profiles)
        # profiles 길이에서 ratio 만큼 곱해 비율 측정
        n_val = int(length * val_ratio)

        # n_val에서 구한 크기 만큼 random sampling
        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    def setup(self):
        """_summary_
        _split_profile로 분리된 사람 별로 폴더안의 이미지와 라벨을 추가
        """
        profiles = os.listdir(self.data_dir)
        # data_dir 에서 "."으로 시작하지 않는 폴더 리스트 저장 -> 사람으로 random하게 split
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        # 랜덤하게 split
        split_profiles = self._split_profile(profiles, self.val_ratio)
        cnt = 0
        # phase - [train, val] , indices - [index 번호]
        for phase, indices in split_profiles.items():
            for _idx in indices:
                # profile - 폴더 리스트중에서 index번호에 해당하는 것 뽑기
                profile = profiles[_idx]
                # 폴더 파일 이름
                img_folder = os.path.join(self.data_dir, profile)
                # 폴더 파일 7개중에서 loop
                for file_name in os.listdir(img_folder):
                    # 파일확장자로 split -> 확장자 제거
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue
                    # 이미지 파일 path 만들기
                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    # 위 _file_names dict에서 해당하는 라벨 찾기 - > data/000004_male_Asian_54/mask1.jpg
                    mask_label = self._file_names[_file_name]

                    # 000004_male_Asian_54 -> 000004, male, Asian, 54
                    id, gender, race, age = profile.split("_")
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    # indices 해당하는 phase[train or val] 에 index번호 저장
                    self.indices[phase].append(cnt)
                    cnt += 1

    def split_dataset(self) -> List[Subset]:
        """_summary_
        Train과 Val로 분리된 indice로 Subset으로 만들어서 Return

        Returns:
            List[Subset]: [Train, val]
        """
        # subset 사용법 https://yeko90.tistory.com/entry/pytorch-how-to-use-Subset#1)_Subset_%EA%B8%B0%EB%B3%B8_%EC%BB%A8%EC%85%89
        # indice에 train, val 으로 해당하는 subset 분리
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class MySubset(Subset):
    """_summary_
        Train Dataset과 Validation Dataset에 각각 Trainsform을 적용하기 위한 Subset
    
    __getitem__(idx):
        
    
    """
    def __init__(self, subset,transform = None) -> None:
        """_summary_

        Args:
            subset (Subset): Subset을 받아 index 번호로 getitem
            transform (trnasform, optional): transform이 있다면 적용 Defaults to None.
        """
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        """_summary_
        Subset에 해당되는 idx를 return
        transform이 있다면 image에 적용시켜서 return

        Args:
            idx (int): _description_

        Returns:
            image, label (tensor): image, label return
        """
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x,y 

    def __len__(self):
        return len(self.subset)


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
    

class CustomDataset(Dataset):
    """_summary_
    Image_path와 label만들 받아 간단한 Dataset 
    label이 없다면 x만 return이 되어 TestDataset으로도 사용 가능
    """
    def __init__(self, image_paths, labels=None, transform= None):
        """_summary_

        Args:
            image_paths (list): image_path list
            labels (list, optional): label list. Defaults to None.
            transform (transoform, optional): . Defaults to None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        x = Image.open(self.image_paths[idx])

        if self.transform:
            x = self.transform(x)
        if self.labels:
            y = self.labels[idx]
            return x,y
        else:
            return x

    def __len__(self):
        return len(self.image_paths)