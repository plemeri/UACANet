import os
import torch
import argparse
import yaml
import tqdm
import sys

from torch.autograd import Variable
from easydict import EasyDict as ed

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.dataloader import *
from utils.utils import *

def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/xomcnet.yaml')
    return parser.parse_args()

def train(opt):
    model = eval(opt.Model.name)(opt.Model).cuda()
    
    image_root = os.path.join(opt.Train.train_path, 'images')
    gt_root = os.path.join(opt.Train.train_path, 'masks')

    train_dataset = PolypDataset(image_root, gt_root, opt.Train)
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=opt.Train.batchsize,
                                  shuffle=opt.Train.shuffle,
                                  num_workers=opt.Train.num_workers,
                                  pin_memory=opt.Train.pin_memory)

    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.Train.lr)
    scheduler = PolyLr(optimizer, gamma=opt.Train.gamma,
                        minimum_lr=opt.Train.min_learning_rate,
                        max_iteration=len(train_loader) * opt.Train.epoch,
                        warmup_iteration=opt.Train.warmup_iteration)
    model.train()

    print('#' * 20, 'Train prep done, start training', '#' * 20)

    for epoch in tqdm.tqdm(range(1, opt.Train.epoch + 1), desc='Epoch', total=opt.Train.epoch, position=0, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}'):
        pbar = tqdm.tqdm(enumerate(train_loader, start=1), desc='Iter', total=len(train_loader), position=1, leave=False, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')
        for i, sample in pbar:
            optimizer.zero_grad()
            images, gts = sample['image'], sample['gt']
            images = images.cuda()
            gts = gts.cuda()
            out = model(images, gts)
            out['loss'].backward()
            clip_gradient(optimizer, opt.Train.clip)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix({'loss': out['loss'].item()})

        os.makedirs(opt.Train.train_save, exist_ok=True)
        if epoch % opt.Train.checkpoint_epoch == 0:
            torch.save(model.state_dict(), os.path.join(opt.Train.train_save, 'latest.pth'))

    print('#' * 20, 'Train done', '#' * 20)

if __name__ == '__main__':
    args = _args()
    opt = ed(yaml.load(open(args.config), yaml.FullLoader))
    train(opt)
