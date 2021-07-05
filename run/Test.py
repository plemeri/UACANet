import torch
from torch.nn import modules
import yaml
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.dataloader import *
from utils.utils import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xnet.yaml')
    return parser.parse_args()

def test(opt):
    model = eval(opt.Model.name)(opt.Model)
    model.load_state_dict(torch.load(opt.Test.pth_path))
    model.cuda()
    model.eval()    

    print('#' * 20, 'Test prep done, start testing', '#' * 20)

    for dataset in tqdm.tqdm(opt.Test.datasets, desc='Total TestSet', total=len(opt.Test.datasets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
        data_path = os.path.join(opt.Test.gt_path, dataset)
        save_path = os.path.join(opt.Test.out_path, dataset)

        os.makedirs(save_path, exist_ok=True)
        image_root = os.path.join(data_path, 'images')
        gt_root = os.path.join(data_path, 'masks')
        test_dataset = PolypDataset(image_root, gt_root, opt.Test)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=1,
                                      num_workers=opt.Test.num_workers,
                                      pin_memory=opt.Test.pin_memory)
        # test_loader = test_dataset(image_root, gt_root, opt.Test)

        for i, sample in tqdm.tqdm(enumerate(test_loader), desc=dataset + ' - Test', total=len(test_loader), position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}'):
            image = sample['image']
            original_size = sample['original_size']
            name = sample['name']
            
            image = image.cuda()
            original_size = (int(original_size[0]), int(original_size[1]))

            if opt.Test.rot_aug is True:
                outs = []
                for k in range(4):
                    outs.append(model(image.rot90(k=k, dims=(2, 3)))['pred'].rot90(k=4-k, dims=(2, 3)))
                out = torch.cat(outs, dim=0)
                out = out.mean(dim=0, keepdims=True)
            else:
                out = model(image)['pred']

            if 'boundary' in sample.keys():
                crop_size = sample['crop_size']
                crop_size = (int(crop_size[0]), int(crop_size[1]))
                lb, rb, tb, bb = sample['boundary']

                out = F.interpolate(out, crop_size, mode='bilinear', align_corners=True)
                out = F.pad(out, (lb, original_size[1] - rb, tb, original_size[0] - bb), value=out.min().data, mode='constant')

            else:
                out = F.interpolate(out, original_size, mode='bilinear', align_corners=True)

                # print('recover boundary', original_size, crop_size, lb, rb, tb, bb)

            out = out.data.sigmoid().cpu().numpy().squeeze()
            out = (out - out.min()) / (out.max() - out.min() + 1e-8)
            # out = cv2.resize(out, original_size)

            Image.fromarray(((out > 0.5) * 255).astype(np.uint8)).save(os.path.join(save_path, name[0]))

    print('#' * 20, 'Test done', '#' * 20)

if __name__ == "__main__":
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    test(opt)
