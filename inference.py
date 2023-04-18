import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import TestDataset, MaskBaseDataset, CustomAugmentation
import json

def load_model(saved_model, num_classes, device):
    
    config_path = os.path.join(saved_model, 'config.json')
    with open(config_path, 'r') as f:
        json_data = json.load(f)
    
    model_cls = getattr(import_module("model"), json_data["model"])  # config.json 에 있는 파일
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    transform_cls = getattr(import_module("dataset"), args.augmentation)
    transform = transform_cls(
        resize = args.resize,
    )
    dataset = TestDataset(img_paths, args.resize,transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, args.name_csv+".csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

def ensemble_inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    transform_cls = getattr(import_module("dataset"), args.augmentation)
    transform = transform_cls(
        resize = args.resize,
    )
    dataset = TestDataset(img_paths, args.resize,transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    
    dirlist = sorted(os.listdir(model_dir)) # ./model/exp
    
    print("Calculating inference results..") 

    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            vote = {
                "age":np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "mask" : np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "gender" : np.array([[0.,0.] for _ in range(images.shape[0])])
                }
            
            for d in dirlist:
                model_path = os.path.join(model_dir,d) # model_path 받기 Age,Mask,Gender
                if "gender" in d.lower():
                    num_classes = 2
                else:
                    num_classes = 3

                model = load_model(model_path, num_classes, device).to(device) 
                model.eval()

                logit = model(images)
                vote[d.split("_")[0]] += logit.cpu().numpy()
            
            age = np.argmax(vote['age'],axis=-1)
            mask = np.argmax(vote['mask'],axis=-1)
            gender = np.argmax(vote['gender'],axis=-1)

            pred = mask * 6 + gender * 3 + age
            preds.extend(list(pred))

    info['ans'] = preds
    save_path = os.path.join(output_dir, args.name_csv+".csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument("--resize", nargs="+", type=int, default=[256, 192], help='resize size for image when training')
    parser.add_argument("--augmentation", type=str, default="TestAugmentation" , help="select augmentation (default: TestAugmentation)")
    parser.add_argument("--ensemble", action='store_true', help="use ensemble inference")

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    parser.add_argument('--name_csv', type=str, default="output")

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    if args.ensemble:
        ensemble_inference(data_dir, model_dir, output_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)
