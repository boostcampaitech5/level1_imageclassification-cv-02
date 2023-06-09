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
    """_summary_
    model_dir에 있는 모델을 args에 따라 data_dir에 있는 데이터 label을 예측한 뒤 예측값을 output_dir에 csv파일로 저장

    Args:
        data_dir (str): 라벨을 예측할 데이터가 있는 파일 경로
        model_dir (str) : 학습한 model을 불러올 파일 경로
        output_dir(str) : 예측한 데이터를 저장할 파일 경로
        args : inference argument
    """
    # -- settings
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

    # data transform
    dataset = CustomDataset(img_paths, transform = transform)

    # dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # -- ensemble
    if args.ensemble:
        dirlist = []
        for p in os.listdir(model_dir):
            dirlist.append(os.path.join(model_dir,p))
    else:
        dirlist = [model_dir]

    print("Calculating inference results..") 

    preds = []
    with torch.no_grad():
        # predict loop
        for idx, images in enumerate(loader):
            images = images.to(device)
            vote = {
                "age":np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "mask" : np.array([[0.,0.,0.] for _ in range(images.shape[0])]),
                "gender" : np.array([[0.,0.] for _ in range(images.shape[0])]),
                "multi" : np.array([[0.]*18 for _ in range(images.shape[0])])
                }

            # predict 수행
            for model_path in dirlist:
                json_data = read_json(os.path.join(model_path, 'config.json'))
                
                category = json_data['category']

                num_classes = get_numclass(category)

                model = load_model(model_path, num_classes, device,json_data).to(device) 
                model.eval()

                logit = model(images)
                logit = nn.functional.softmax(logit,dim=-1)

                vote[category] += logit.cpu().numpy()
            
            # 한번에 18개의 label로 나누는 경우와, 카테고리를 나누어서 구한 뒤 통합하는 경우
            if category == "multi":
                pred = np.argmax(vote[category],axis=-1)
            else:
                age = np.argmax(vote['age'],axis=-1)
                mask = np.argmax(vote['mask'],axis=-1)
                gender = np.argmax(vote['gender'],axis=-1)

                pred = mask * 6 + gender * 3 + age

            preds.extend(list(pred))

    # 예측 label 저장
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
