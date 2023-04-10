import sys
import argparse
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
def get_args():
    parser = argparse.ArgumentParser(description="level1 classification")
    parser.add_argument("--seed", type=int, default=42, help="set random seed")
    parser.add_argument("--batch", type=int, default=8, help="set num of batch")
    parser.add_argument("--epochs", type=int, default=20, help="set num of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="set learning rate")
    return parser.parse_args()

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self,df,transform=None,train=True):
        self.df = df
        self.transform=transform
        self.train = train

        if self.train:
            self.f_path = "/opt/ml/input/data/train/images"
        else:
            self.f_path = "/opt/ml/input/data/eval/images"

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        i = self.df.iloc[index]
        f_path = os.path.join(self.f_path,i['path'])
        image_path = [p for p in os.listdir(f_path) if "_" not in p]
        image_path = os.path.join(f_path,random.choice(image_path))

        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)
        label = 0
        label += i['age']//30
        if i['gender']=="female":
            label+=3
        if "Incorrect" in image_path:
            label+=6
        elif "normal" in image_path:
            label+=12
        
        return image, label

if __name__=="__main__":
    args = get_args()
    
    set_random_seeds(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # load train.csv
    df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    
    # train, test split
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=args.seed)

    # data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((512,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((512,384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
    ])
    Model = resnet50(pretrained=True)
    Model.fc2 = torch.nn.Linear(1000,18)
    # print(Model)

    # set train, val dataset
    train_dataset = CustomDataset(train_df, train_transforms)
    val_dataset = CustomDataset(val_df, val_transforms)

    # set DataLoader
    train_loader = DataLoader(train_dataset, batch_size = args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = args.batch, shuffle=False)

    # Model

    # optimizer 
    optimizer = torch.optim.Adam(Model.parameters(), lr = args.lr, weight_decay=0.1)

    # scheduler
    schedulder = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.5, patience=3,verbose=True)

    # loss
    criterion = torch.nn.CrossEntropyLoss()

    # train and infer
    best_loss = np.Inf
    count = 0
    Model.to(device)
    for epoch in range(args.epochs):

        running_loss = 0
        running_corrects = 0

        # train
        Model.train()
        for imgs, labels in tqdm(iter(train_loader), leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = Model(imgs)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(logits,1)
            running_loss += loss.item() * labels.size(0)
            running_corrects += (preds==labels).sum().item()

        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100*running_corrects / len(train_loader.dataset)

        # infer
        Model.eval()
        trues= []
        predicts = []
        
        for imgs, labels in tqdm(iter(val_loader), leave=False):
            imgs, labels = imgs.to(device), labels.to(device)

            logits = Model(imgs)

            loss = criterion(logits,labels)
            _, preds = torch.max(logits,1)

            running_loss += loss.item() * labels.size(0)
            running_corrects += (preds==labels).sum().item()

            trues += labels.detach().cpu().numpy().tolist()
            predicts += preds.detach().cpu().numpy().tolist()

        val_loss = running_loss / len(val_loader.dataset)
        val_acc = 100*running_corrects / len(val_loader.dataset)
        val_score = f1_score(trues, predicts, average='macro')

        # 1 epoch 마다 scheduler 실행
        if schedulder is not None:
            schedulder.step(val_loss)
        
        print(f'Epoch [{epoch}], Train Loss : [{train_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 : [{val_score:.5f}]')

        if best_loss > val_loss:
            print("save model")
            count = 0
            best_loss = val_loss
            torch.save(Model.state_dict(),"checkpoint/test1.pth")
        else:
            count +=1
            if count > 10:
                print("Early Stopping")
                break
