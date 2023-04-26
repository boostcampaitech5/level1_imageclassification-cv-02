import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR, ExponentialLR, CosineAnnealingLR, CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from dataset import MaskBaseDataset, mixup_collate_fn, MySubset, CustomDataset
from loss import create_criterion
from optims import create_optimizer, create_scheduler
from collections import Counter

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes
    if args.category == "multi":
        num_classes = 18 # 18
    elif args.category == "gender":
        num_classes = 2
    else:
        num_classes = 3

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        num_classes = num_classes,
        category = args.category,
        val_ratio = args.val_ratio
    )
 
    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)  # default: CustomAugmentation
    val_transform_module = getattr(import_module("dataset"), "BaseAugmentation")
    train_transform = train_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    val_transform = val_transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    train_set = MySubset(train_set, transform = train_transform)
    val_set = MySubset(val_set, transform = val_transform)

    if args.mixup:
        collate_fn = mixup_collate_fn
        args.criterion = "bce"
    else:
        collate_fn = None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(
        num_classes=num_classes
    ).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion).to(device)

    optimizer = create_optimizer(
        args.optimizer, 
        filter(lambda p: p.requires_grad, model.parameters()), 
        args
        )
    
    scheduler = create_scheduler(args.scheduler,optimizer,args)
    
    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_val_score = 0
    early_stop = 0
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if args.cutmix:
                # generate mixed sample
                lam = np.random.beta(1.0, 1.0)
                rand_index = torch.randperm(inputs.size()[0]).to(device)
                labels_a = labels
                labels_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                # compute output
                outs = model(inputs)
                loss = criterion(outs, labels_a) * lam + criterion(outs, labels_b) * (1. - lam)
            else:
                outs = model(inputs)
                loss = criterion(outs, labels)
            preds = torch.argmax(outs, dim=-1)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            # mixup 을 사용할 경우 label이 [0,0,1] 이런 binary 형태로 나오기 때문에
            # argmax를 사용하여 가장 높은 값을 가진 label로 acc 측정
            if labels.dim() > 1:
                labels = torch.argmax(labels, dim=-1)
            matches += (preds == labels).sum().item()
            # interval 마다 loss, acc 계산
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0
            
        if args.scheduler != 'reducelronplateau':
            scheduler.step()

        # val loop
        trues = []
        predicts = []
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                # f1_score를 위한 label, predict 저장
                trues += labels.detach().cpu().numpy().tolist()
                predicts += preds.detach().cpu().numpy().tolist()
                
                if args.mixup:
                    t = []
                    for i in range(args.valid_batch_size):
                        temp = torch.tensor([0]*num_classes)
                        temp[labels[i]] = 1.0
                        t.append(temp)
                    labels = torch.stack(t).to(device).float()
                    loss_item = criterion(outs, labels).item()
                    labels = torch.argmax(labels,dim=-1)
                else:
                    loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_score = f1_score(trues, predicts, average='macro')
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_acc = max(best_val_acc, val_acc)

            if val_score > best_val_score:
                early_stop = 0
                print(f"New best model for f1_score : {val_score:4.2}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_score = val_score
            else:
                early_stop +=1
                if args.early_stopping_patience==-1:
                    pass
                elif early_stop > args.early_stopping_patience:
                    print("Early Stopping")
                    break
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1_score: {val_score:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_scalar("Val/f1_score",val_score,epoch)
            logger.add_figure("results", figure, epoch)
            print()
            if args.scheduler == 'reducelronplateau':
                scheduler.step(val_score)
    logger.close()

def ktrain(data_dir, model_dir, args):
    from dataset import MaskLabels, AgeLabels, GenderLabels
    from sklearn.model_selection import StratifiedKFold
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes
    if args.category == "multi":
        num_classes = 18 # 18
    elif args.category == "gender":
        num_classes = 2
    else:
        num_classes = 3
    
    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }
    # -- augmentation
    train_transform_module = getattr(import_module("dataset"), args.augmentation)  # default: CustomAugmentation
    val_transform_module = getattr(import_module("dataset"), "BaseAugmentation")
    train_transform = train_transform_module(
        resize=args.resize,
    )
    val_transform = val_transform_module(
        resize=args.resize,
    )

    # -- dataset
    image_paths = []
    image_labels = []
    # kfold 정의
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=False)
    profiles = os.listdir(data_dir)
    # data_dir 에서 "."으로 시작하지 않는 폴더 리스트 저장 
    profiles = [profile for profile in profiles if not profile.startswith(".")]

    if args.category=="mask":
        for profile in profiles:
            # img_folder == inputs/train/image/000004_male_Asian_54 
            img_folder = os.path.join(args.data_dir, profile)
            # 폴더안의 image list == [mask1.jpg, mask2.jpg, incorrect.jpg ...]
            for file_name in os.listdir(img_folder):
                # 확장자 제거, _file_name = mask1, ext=.jpg
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue
                
                # img_path = inputs/train/image/000004_male_Asian_54/mask1.jpg
                img_path = os.path.join(args.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                # 위 _file_names dict에서 해당하는 라벨 찾기 - > mask1
                mask_label = _file_names[_file_name]

                image_paths.append(img_path)
                image_labels.append(mask_label)
        # 마스크기준 kfold
        make_kfold = skf.split(image_paths, image_labels)
        
    else:
        # label을 찾아 그에 맞는 kfold
        temp_label = []
        for p in profiles:
            # p = 000004_male_Asian_54
            id, gender, race, age = p.split("_")
            if args.category == 'age':
                temp_label.append(AgeLabels.from_number(age))
            else:
                temp_label.append(GenderLabels.from_str(gender))
        
        # 사람별 kfold
        make_kfold = skf.split(profiles,temp_label)
    
    for k,(train_index, val_index) in enumerate(make_kfold):
        if args.category =="mask":
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
                    if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
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
                    if _file_name not in _file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(args.data_dir, profile, file_name)
                    val_imgs.append(img_path)
                    if args.category == "age":
                        val_labels.append(age)
                    else:
                        val_labels.append(gender)

        train_set = CustomDataset(train_imgs,train_labels,transform=train_transform)
        val_set = CustomDataset(val_imgs,val_labels,transform=val_transform)

        if args.mixup:
            collate_fn = mixup_collate_fn
            args.criterion = "bce"
        else:
            collate_fn = None
        
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
                pin_memory=use_cuda,
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                num_workers=multiprocessing.cpu_count() // 2,
                shuffle=True,
                pin_memory=use_cuda,
                collate_fn=collate_fn,
                drop_last=True,
            )


        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion).to(device)

        optimizer = create_optimizer(
            args.optimizer, 
            filter(lambda p: p.requires_grad, model.parameters()), 
            args
            )
        
        scheduler = create_scheduler(args.scheduler,optimizer,args)
        
        # -- logging
        save_dir = increment_path(os.path.join(model_dir, args.name+f"_{k+1}fold"))
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        best_val_score = 0
        early_stop = 0
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                # mixup 을 사용할 경우 label이 [0,0,1] 이런 binary 형태로 나오기 때문에
                # argmax를 사용하여 가장 높은 값을 가진 label로 acc 측정
                if labels.dim() > 1:
                    labels = torch.argmax(labels, dim=-1)
                matches += (preds == labels).sum().item()
                # interval 마다 loss, acc 계산
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    loss_value = 0
                    matches = 0
                
            if args.scheduler != 'reducelronplateau':
                scheduler.step()

            # val loop
            trues = []
            predicts = []
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    # f1_score를 위한 label, predict 저장
                    trues += labels.detach().cpu().numpy().tolist()
                    predicts += preds.detach().cpu().numpy().tolist()
                    
                    if args.mixup:
                        t = []
                        for i in range(args.valid_batch_size):
                            temp = torch.tensor([0]*num_classes)
                            temp[labels[i]] = 1.0
                            t.append(temp)
                        labels = torch.stack(t).to(device).float()
                        loss_item = criterion(outs, labels).item()
                        labels = torch.argmax(labels,dim=-1)
                    else:
                        loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                val_score = f1_score(trues, predicts, average='macro')
                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_acc = max(best_val_acc, val_acc)

                if val_score > best_val_score:
                    early_stop = 0
                    print(f"New best model for f1_score : {val_score:4.2}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                    best_val_score = val_score
                else:
                    early_stop +=1
                    if args.early_stopping_patience==-1:
                        pass
                    elif early_stop > args.early_stopping_patience:
                        print("Early Stopping")
                        break
                torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                print(
                    f"{k} [Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, f1_score: {val_score:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1_score",val_score,epoch)
                print()
                if args.scheduler == 'reducelronplateau':
                    scheduler.step(val_loss)
        logger.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 1)')
    
    # data
    parser.add_argument('--dataset', type=str, default='MaskSplitByProfileDataset', help='dataset augmentation type (default: MaskSplitByProfileDataset)')
    parser.add_argument('--augmentation', type=str, default='CustomAugmentation', help='train data augmentation type (default: CustomAugmentation)')
    parser.add_argument("--resize", nargs="+", type=int, default=[256, 192], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--mixup', action='store_true', help="use mixup 0.2")
    parser.add_argument('--kfold', type=int, help="using Kfold k")
    parser.add_argument('--weightsampler', action='store_true', help="using torch WeightedRamdomSampling")
    parser.add_argument('--cutmix', action='store_true', help='use cutmix')
    
    # model
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--category', type=str, default = "multi",choices=["multi","mask","gender","age"], help='choose labels type of multi,mask,gender,age')
    parser.add_argument('--early_stopping_patience', type=int, default = 5, help='input early stopping patience, It does not work if you input -1, default : 5')

    # optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--lr_decay_step', type=int, default=5, help='learning rate scheduler deacy step (default: 5)')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer such as sgd, momentum, adam, adagrad (default: sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay (default: 0.01)')
    
    # scheduler
    parser.add_argument('--scheduler', type=str, default='steplr', help='scheduler such as steplr, lambdalr, exponentiallr, cycliclr, reducelronplateau etc. (default: steplr)')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate scheduler gamma (default: 0.5)')
    parser.add_argument('--tmax', type=int, default=5, help='tmax used in CyclicLR and CosineAnnealingLR (default: 5)')
    parser.add_argument('--maxlr', type=float, default=0.1, help='maxlr used in CyclicLR (default: 0.1)')
    parser.add_argument('--mode', type=str, default='triangular', help='mode used in CyclicLR such as triangular, triangular2, exp_range (default: triangular)')
    parser.add_argument('--factor', type=float, default=0.5, help='mode used in ReduceLROnPlateau (default: 0.5)')
    parser.add_argument('--patience', type=int, default=4, help='mode used in ReduceLROnPlateau (default: 4)')
    parser.add_argument('--threshold', type=float, default=1e-4, help='mode used in ReduceLROnPlateau (default: 1e-4)')


    # loss
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    
    # log
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    if args.kfold:
        ktrain(data_dir, model_dir, args)
    else:
        train(data_dir, model_dir, args)
