import torch
import os
import argparse
import tqdm
import sys

import torch.nn.functional as F
import numpy as np

from PIL import Image
from torch.nn import modules

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from utils.utils import *
from utils.dataloader import *
from lib import *

def test(opt, args):
    model = eval(opt.Model.name)(channels=opt.Model.channels,
                                output_stride=opt.Model.output_stride,
                                pretrained=opt.Model.pretrained)
    model.load_state_dict(torch.load(os.path.join(
        opt.Test.Checkpoint.checkpoint_dir, 'latest.pth')), strict=True)
    model.cuda()
    model.eval()

    if args.verbose is True:
        testsets = tqdm.tqdm(opt.Test.Dataset.testsets, desc='Total TestSet', total=len(
            opt.Test.Dataset.testsets), position=0, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        testsets = opt.Test.Dataset.testsets

    for testset in testsets:
        data_path = os.path.join(opt.Test.Dataset.root, testset)
        save_path = os.path.join(opt.Test.Checkpoint.checkpoint_dir, testset)

        os.makedirs(save_path, exist_ok=True)

        test_dataset = eval(opt.Test.Dataset.type)(root=data_path, transform_list=opt.Test.Dataset.transform_list)

        test_loader = data.DataLoader(dataset=test_dataset,
                                        batch_size=1,
                                        num_workers=opt.Test.Dataloader.num_workers,
                                        pin_memory=opt.Test.Dataloader.pin_memory)

        if args.verbose is True:
            samples = tqdm.tqdm(test_loader, desc=testset + ' - Test', total=len(test_loader),
                                position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
        else:
            samples = test_loader

        for sample in samples:
            sample = to_cuda(sample)
            out = model(sample)
            out['pred'] = F.interpolate(
                out['pred'], sample['shape'], mode='bilinear', align_corners=True)

            out['pred'] = out['pred'].data.cpu()
            out['pred'] = torch.sigmoid(out['pred'])
            out['pred'] = out['pred'].numpy().squeeze()
            out['pred'] = (out['pred'] - out['pred'].min()) / \
                (out['pred'].max() - out['pred'].min() + 1e-8)
            Image.fromarray(((out['pred'] > .5) * 255).astype(np.uint8)
                            ).save(os.path.join(save_path, sample['name'][0]))


if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    test(opt, args)
