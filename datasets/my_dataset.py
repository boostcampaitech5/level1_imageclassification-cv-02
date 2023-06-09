import os
from collections import defaultdict
from typing import Tuple, List

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from importlib import import_module

from collections import Counter
import multiprocessing

from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from datasets._util import GenderLabels, AgeLabels, FILE_NAME
from datasets.augmentation import mixup_collate_fn
from datasets.base_dataset import MaskBaseDataset


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
        """_summary_

        Returns:
            len (int): Subset data 크기
        """
        return len(self.subset)
    

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
        """_summary_

        Args:
            idx : _description_

        Returns:
            image ,(label) (tensor): label이 없다면 image 만 return
        """
        x = Image.open(self.image_paths[idx])

        if self.transform:
            x = self.transform(x)
        if self.labels:
            y = self.labels[idx]
            return x,y
        else:
            return x

    def __len__(self):
        """_summary_

        Returns:
            len (int): dataset 길이
        """
        return len(self.image_paths)


def make_loader(set, collate_fn, batch_size , shuffle):
    """_summary_

    Args:
        set (Dataset): loader에 사용할 Dataset
        collate_fn (func): collate_fn 함수
        batch_size (int): batch_size 
        shuffle (bool): loader를 shuffle 할지 안 할지

    Returns:
        Loader (DataLoader): Loader 
    """
    loader = DataLoader(
            set,
            batch_size = batch_size,
            num_workers = multiprocessing.cpu_count() // 2,
            shuffle = shuffle,
            pin_memory = torch.cuda.is_available(),
            collate_fn = collate_fn,
            drop_last = True,
        )
    return loader 


def make_dataset(data_dir,train_transform,val_transform,collate_fn,args):
    """_summary_
    Base dataset을 만드는 함수

    Args:
        data_dir (str): data가 있는 파일 경로
        train_transform (transform): train_set 에 적용시킬 transform 
        val_transform (transform): validation_set 에 적용시킬 transform
        collate_fn (func): loader에 적용 할 collate_fn 함수 
        args : train argument

    Returns:
        train_loader, val_loader (list): list에 담겨진 loader

    """
    train_dataloader = []
    val_dataloader = []

    dataset_module = getattr(import_module("datasets.base_dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        num_classes = args.num_classes,
        category = args.category,
        val_ratio = args.val_ratio
    )

    train_set, val_set = dataset.split_dataset()
    train_set = MySubset(train_set, transform = train_transform)
    val_set = MySubset(val_set, transform = val_transform)

    train_loader = make_loader(train_set, collate_fn, batch_size = args.batch_size, shuffle=True)

    val_loader = make_loader(val_set, collate_fn, batch_size = args.valid_batch_size, shuffle=False)

    train_dataloader.append(train_loader)
    val_dataloader.append(val_loader)
    
    return train_dataloader, val_dataloader


def make_kfold_dataset(data_dir,train_transform,val_transform,collate_fn,args):
    """_summary_
    Kfold_dataset을 만드는 함수 
    
    Args:
        data_dir (str): data가 있는 파일 경로
        train_transform (transform): train_set 에 적용시킬 transform 
        val_transform (transform): validation_set 에 적용시킬 transform
        collate_fn (func): loader에 적용 할 collate_fn 함수 
        args : train argument

    Returns:
        train_loader, val_loader (list): list에 담겨진 loader

    """
    train_dataloader = []
    val_dataloader = []

    image_paths = []
    image_labels = []
    # kfold 정의
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=False)
    profiles = os.listdir(data_dir)
    # data_dir 에서 "."으로 시작하지 않는 폴더 리스트 저장 
    profiles = [profile for profile in profiles if not profile.startswith(".")]

    if args.category in ["multi","mask"]:
        for profile in profiles:
            # img_folder == inputs/train/image/000004_male_Asian_54 
            img_folder = os.path.join(args.data_dir, profile)
            # 폴더안의 image list == [mask1.jpg, mask2.jpg, incorrect.jpg ...]
            for file_name in os.listdir(img_folder):
                # 확장자 제거, _file_name = mask1, ext=.jpg
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in FILE_NAME:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue
                
                # img_path = inputs/train/image/000004_male_Asian_54/mask1.jpg
                img_path = os.path.join(args.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                # 위 FILE_NAME dict에서 해당하는 라벨 찾기 - > mask1
                mask_label = FILE_NAME[_file_name]

                image_paths.append(img_path)
                id, gender, race, age = profile.split("_")
                if args.category == "multi":
                    age = AgeLabels.from_number(age)
                    gender = GenderLabels.from_str(gender)
                    image_labels.append(MaskBaseDataset.encode_multi_class(mask_label,gender,age))
                else:
                    image_labels.append(mask_label)
        # 마스크기준 kfold
        make_kfold = skf.split(image_paths, image_labels)
        
    else:
        # label을 찾아 그에 맞는 kfold
        temp_label = []
        for profile in profiles:
            # p = 000004_male_Asian_54
            id, gender, race, age = profile.split("_")
            if args.category == 'age':
                temp_label.append(AgeLabels.from_number(age))
            else:
                temp_label.append(GenderLabels.from_str(gender))
        
        # 사람별 kfold
        make_kfold = skf.split(profiles,temp_label)
    
    for k,(train_index, val_index) in enumerate(make_kfold):
        if args.category in ["multi", "mask"]:
            # 전체 이미지에서 train set 에 해당하는 index만 train_imgs 로 저장
            # train_list = [img1,img2,img3...img10]
            # [1,2,3,4,5,6,7,8] [9,10]
            # [1,2,3,4,5,6,9,10] [7,8]
            # [1,2,3,4,7,8,9,10] [5,6]
            # [1,2,5,6,7,8,9,10] [3,4]
            # [3,4,5,6,7,8,9,10] [1,2]
            train_imgs = [image_paths[i] for i in train_index]
            train_labels = [image_labels[i] for i in train_index]
            val_imgs = [image_paths[i] for i in val_index]
            val_labels = [image_labels[i] for i in val_index]
        else:
            # 사람별 이미지 폴더에서 train set에 해당하는 index 저장
            train_p = [profiles[i] for i in train_index]
            val_p = [profiles[i] for i in val_index]

            train_imgs = []
            train_labels =[]
            val_imgs = []
            val_labels = []
            # 폴더안의 7개의 이미지를 train_imgs에 저장
            for profile in train_p:
                id, gender, race, age = profile.split("_")
                age = AgeLabels.from_number(age)
                gender = GenderLabels.from_str(gender)
                img_folder = os.path.join(args.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in FILE_NAME:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(args.data_dir, profile, file_name)
                    train_imgs.append(img_path)
                    if args.category == "age":
                        train_labels.append(age)
                    else:
                        train_labels.append(gender)
            
            for profile in val_p:
                id, gender, race, age = profile.split("_")
                age = AgeLabels.from_number(age)
                gender = GenderLabels.from_str(gender)
                img_folder = os.path.join(args.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in FILE_NAME:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(args.data_dir, profile, file_name)
                    val_imgs.append(img_path)
                    if args.category == "age":
                        val_labels.append(age)
                    else:
                        val_labels.append(gender)

        train_set = CustomDataset(train_imgs,train_labels,transform=train_transform)
        val_set = CustomDataset(val_imgs,val_labels,transform=val_transform)
        
        # 클래스별 개수를 구하여 sampling
        class_counts = Counter(train_labels)
        if args.weightsampler:
            weights = torch.DoubleTensor([1./class_counts[i] for i in train_labels])
            weight_sampler = WeightedRandomSampler(weights,len(train_labels))
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=multiprocessing.cpu_count() // 2,
                sampler = weight_sampler,
                shuffle=False,
                pin_memory=torch.cuda.is_available(),
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            train_loader = make_loader(train_set, collate_fn, args.batch_size, shuffle=True)

        val_loader = make_loader(val_set, None, args.valid_batch_size, shuffle=False)

        train_dataloader.append(train_loader)
        val_dataloader.append(val_loader)

    return train_dataloader, val_dataloader



def make_dataloader(data_dir,args):
    """_summary_
    학습에 사용할 Dataloader를 만드는 전체 함수

    Args:
        data_dir (str): data가 있는 파일 경로
        args : train argument

    Returns:
        train_loader, val_loader (list): list에 담겨진 loader
    """
    train_transform_module = getattr(import_module("datasets.augmentation"), args.augmentation)  # default: CustomAugmentation
    val_transform_module = getattr(import_module("datasets.augmentation"), "BaseAugmentation")
    train_transform = train_transform_module(
        resize=args.resize,
    )
    val_transform = val_transform_module(
        resize=args.resize,
    )

    if args.mixup:
        collate_fn = mixup_collate_fn
        args.criterion = "bce"
    else:
        collate_fn = None

    if args.kfold:
        train_dataloader, val_dataloader = make_kfold_dataset(
            data_dir = data_dir, 
            train_transform = train_transform, 
            val_transform = val_transform, 
            collate_fn = collate_fn, 
            args = args
            )
    
    else:
        train_dataloader, val_dataloader = make_dataset(
            data_dir = data_dir, 
            train_transform = train_transform, 
            val_transform = val_transform, 
            collate_fn = collate_fn, 
            args = args
            )

    return train_dataloader, val_dataloader