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


# 확률적 반올림 함수
# 실수를 확률적으로 정수로 바꿈 ex) 3.7 -> 30퍼 확률로 3, 70퍼 확률로 4
def probablity_work(num):
    if num-int(num) > np.random.uniform(0,1,1)[0]:
        return int(num)+1
    else:
        return int(num)
    

# 나이(10살 단위로), 성별로 데이터 밸런스를 맞춰주기 위한 dict
def balancing_10s_dict(data_dir):
    dic = {
        "male_10" : 0,
        "male_20" : 0,
        "male_30" : 0,
        "male_40" : 0,
        "male_50" : 0,
        "male_60" : 0,
        "female_10" : 0,
        "female_20" : 0,
        "female_30" : 0,
        "female_40" : 0,
        "female_50" : 0,
        "female_60" : 0,  
        }
    
    # 각 집단의 수를 구함
    profiles = os.listdir(data_dir)
    for profile in profiles:
        if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
            continue
        id, gender, race, age = profile.split("_")
        key = gender.lower()+"_"+str(int(age)//10 *10)
        dic[key] += 1
        max_value = max(dic.values())

    # 밸런싱
    for key_ in dic.keys():
        dic[key_] = max_value / dic[key_]
    
    return dic


# 나이(3 그룹으로), 성별로 데이터 밸런스를 맞춰주기 위한 dict
def balancing_gene_dict(data_dir):
    dic = {
        "male_young" : 0,
        "male_middle" : 0,
        "male_old" : 0,
        "female_young" : 0,
        "female_middle" : 0,
        "female_old" : 0,
        }
    # 각 집단의 수를 구함
    profiles = os.listdir(data_dir)
    for profile in profiles:
        if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
            continue
        id, gender, race, age = profile.split("_")
        age = int(age)
        if age >= 60:
            age = "old"
        elif age >= 30:
            age = "middle"
        else:
            age = "young"

        key = gender.lower()+"_"+age
        dic[key] += 1
        max_value = max(dic.values())
    # 밸런싱
    for key_ in dic.keys():
        dic[key_] = max_value / dic[key_]
    
    return dic


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class BaseAugmentation:
    def __init__(self, resize,  mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.transform = Compose([
            Resize(resize),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize,  mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.transform = Compose([
            Resize((320,256), Image.BILINEAR),
            CenterCrop(resize),
            # ColorJitter(0.05, 0.05, 0.05, 0.05),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if "mask" in value:
            return cls.MASK
        elif value == "incorrect":
            return cls.INCORRECT
        elif value == "normal":
            return cls.NORMAL
        else:
            raise ValueError(f"Mask value should be either 'mask1,2,3,4,5', 'incorrect' or 'normal', {value}")


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str, age_classes) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if age_classes == 3:
            if value < 30:
                return cls.YOUNG
            elif value < 60:
                return cls.MIDDLE
            else:
                return cls.OLD
        elif age_classes ==6:
            return value//10 -1


class MaskBaseDataset(Dataset):

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



    def __init__(self, data_dir, balancing_option, num_classes_list, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):

        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.transform = None

        # balancing_option 에 따라 나누는 방법 선택
        # None : 안함, 10s : 10살 별로, generation : young, middle, old로
        if balancing_option == "imbalance":
            self.balancing_dict == {}
        elif balancing_option == "10s":
            self.balancing_dict = balancing_10s_dict(self.data_dir)
        elif balancing_option == "generation":
            self.balancing_dict = balancing_gene_dict(self.data_dir)
        else:
            assert True, "check data balancing option"

        self.num_classes_list = num_classes_list
        self.setup()
        self.calc_statistics()


    def setup(self):
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
        # mean과 std 구하기, 3000개만
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

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        # 각각의 라벨들로 0~18의 라벨 만들기
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label, self.num_classes_list)
        # transform 적용
        image_transform = self.transform(image)
        return image_transform, age_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label, num_classes_list) -> int:
        return mask_label * num_classes_list[1]*num_classes_list[2] + gender_label * num_classes_list[2] + age_label

    @staticmethod
    def decode_multi_class(multi_class_label, num_classes_list) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // (num_classes_list[1]*num_classes_list[2])) % num_classes_list[0]
        gender_label = (multi_class_label // num_classes_list[2]) % num_classes_list[1]
        age_label = multi_class_label % num_classes_list[2]
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
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
    """


    def __init__(self, data_dir, balancing_option, num_classes_list, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), val_ratio=0.2):
        # 초기화된 dictionary 만들기 -> https://wikidocs.net/104993
        self.indices = defaultdict(list)
        self.balancing_dict = {}
        self.num_classes_list = []
        super().__init__(data_dir, balancing_option, num_classes_list, mean, std, val_ratio)


    @staticmethod
    def _split_profile(profiles, val_ratio):
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
                    # --categori에 따라서 class 숫자를 조절
                    if self.num_classes_list[0] == 1: mask_label =0
                    elif self.num_classes_list[0] == 3: mask_label = MaskLabels.from_str(_file_name)
                    else: raise ValueError(f"check --categoric of mask")

                    # 000004_male_Asian_54 -> 000004, male, Asian, 54
                    # --categori에 따라서 class 숫자를 조절
                    id, gender, race, age = profile.split("_")
                    if self.num_classes_list[1] == 1: gender_label=0
                    elif self.num_classes_list[1] == 2: gender_label = GenderLabels.from_str(gender)
                    else: raise ValueError(f"check --categoric of gender")
                
                    if self.num_classes_list[2] == 1: age_label = 0
                    elif self.num_classes_list[2] == 3 or self.num_classes_list[2] == 6:
                        age_label = AgeLabels.from_number(age, self.num_classes_list[2])
                    else: raise ValueError(f"check --categoric of age")

                    # 밸런싱 안할 떄
                    if len(self.balancing_dict.keys()) == 0:
                        num = 1

                    # 10살로 별로 나누었을 때
                    elif len(self.balancing_dict.keys()) == 12:
                        key = gender +"_" +str(int(age)//10 *10)
                        num = self.balancing_dict[key]
                        if mask_label:   # 마스크는 5개의 데이터가 있으므로
                            num = num*5

                    # young, middle, old로 나눌 때
                    else:
                        if int(age)>=60:
                            key = gender +"_old"
                        elif int(age)>=30:
                            key = gender +"_middle"
                        else:
                            key = gender +"_young"
                        num = self.balancing_dict[key]
                        if mask_label:  # 마스크는 5개의 데이터가 있으므로
                            num = num*5
                    
                    # 그 수를 확률적으로 반올림함
                    num = probablity_work(num)

                    # 그 수 만큼 라벨에 추가.
                    for _ in range(num):
                        self.image_paths.append(img_path)
                        self.mask_labels.append(mask_label)
                        self.gender_labels.append(gender_label)
                        self.age_labels.append(age_label)

                        # indices 해당하는 phase[train or val] 에 index번호 저장
                        self.indices[phase].append(cnt)
                        cnt += 1

    def split_dataset(self) -> List[Subset]:
        # subset 사용법 https://yeko90.tistory.com/entry/pytorch-how-to-use-Subset#1)_Subset_%EA%B8%B0%EB%B3%B8_%EC%BB%A8%EC%85%89
        # indice에 train, val 으로 해당하는 subset 분리
        return [Subset(self, indices) for phase, indices in self.indices.items()]


class TestAugmentation:
    def __init__(self, resize, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246), **args):
        self.img_paths = img_paths
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    def __call__(self, image):
        return self.transform(image)

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, transfrom=None, mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
        self.img_paths = img_paths
        if transfrom:
            self.transform = transfrom
        else:
            self.transform = Compose([
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
