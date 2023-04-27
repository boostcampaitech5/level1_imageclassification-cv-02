import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn

from datasets.my_dataset import CustomDataset
from utils.util import read_json, load_model, get_numclass

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    transform_cls = getattr(import_module("datasets.augmentation"), args.augmentation)
    transform = transform_cls(
        resize = args.resize,
    )

    dataset = CustomDataset(img_paths, transform = transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    if args.ensemble:
        dirlist = []
        for p in os.listdir(model_dir):
            dirlist.append(os.path.join(model_dir,p))
    else:
        dirlist = [model_dir]

    print("Calculating inference results..") 

    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            vote = {
                "age":np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "mask" : np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "gender" : np.array([[0.,0.] for _ in range(images.shape[0])]),
                "multi" : np.array([[0.]*18 for _ in range(images.shape[0])])
                }

            for model_path in dirlist:
                json_data = read_json(os.path.join(model_path, 'config.json'))
                
                category = json_data['category']

                num_classes = get_numclass(category)

                model = load_model(model_path, num_classes, device,json_data).to(device) 
                model.eval()

                logit = model(images)
                logit = nn.functional.softmax(logit,dim=-1)

                vote[category] += logit.cpu().numpy()
            
            if category == "multi":
                pred = np.argmax(vote[category],axis=-1)
            else:
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
    parser.add_argument("--augmentation", type=str, default="BaseAugmentation" , help="select augmentation (default: BaseAugmentation)")
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

    inference(data_dir, model_dir, output_dir, args)
