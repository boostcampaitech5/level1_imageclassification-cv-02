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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from dataset import make_dataloader, MaskBaseDataset, mixup_collate_fn, MySubset, CustomDataset
from loss import create_criterion
from optims import create_optimizer, create_scheduler


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

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.category == "multi":
        num_classes = 18 # 18
    elif args.category == "gender":
        num_classes = 2
    else:
        num_classes = 3
    args.num_classes = num_classes

    # make dataloader
    t_loaders, v_loaders = make_dataloader(data_dir,args)

    for k,(train_loader, val_loader) in enumerate(zip(t_loaders,v_loaders)):
        # -- model
        model_module = getattr(import_module("model"), args.model)  # default: BaseModel
        model = model_module(
            num_classes=num_classes
        )
        if args.arcface:
            model_module = getattr(import_module("model"), "ArcfaceModel")
            model = model_module(
                model = model,
                num_classes = num_classes
            )    
        model = model.to(device)
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
        if k == 0:
            save_dir = increment_path(os.path.join(model_dir, args.name+f"_{k+1}"))
        else:
            save_dir = increment_path(os.path.join(model_dir, args.name+f"_{k+1}"))
            
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
                    if args.arcface:
                        outs = model(inputs,labels)
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

                    if args.arcface:
                        outs = model(inputs,labels)
                    else:
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
                val_acc = np.sum(val_acc_items) / len(val_loader)
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)

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
                logger.add_figure("results", figure, epoch)
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
    parser.add_argument('--arcface', action='store_true', help ="using arcface loss")

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

    train(data_dir, model_dir, args)
