from __future__ import print_function, absolute_import
import os.path as osp
import os
import json
import sys
import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import time
import visdom
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils.myDataSet import myDataSet
from utils.transform_test_image import get_test_attrs
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.resnet import ResNet, resnet50
from test import compute_dist
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

vis = visdom.Visdom(env='ZSL-ms-without relu')
vis.check_connection()

def rank(yn,comp):
    comp_l = comp  + get_delta(yn)
    return torch.sum(comp_l>comp[yn])
    
def get_l(r):
    return torch.sum( torch.tensor( [1/i for i in range(1,r+1)] ) )

def ale_loss(yns,comps):
    summ = torch.tensor(0)
    for i in range(comps.shape[0]):
        comp = comps[i]
        yn = yns[i]
        
        r = rank(yn,comp)
        lr = get_l(r)
        lr_over_r = lr / r
        comp_2 = comp+get_delta(yn)-comp[xn]
        comp_3 = comp2[comp2>0]
        comp_4 = lr_over_r*comp_3
        summ = summ + torch.sum(comp_4)
    return summ

def get_data(args):

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = True),
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=args.gpu)

    val_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = False),
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=args.gpu)
    
    return train_loader, val_loader


def test(test_loader, test_cls_list, test_attrs, model_dir):
    model = resnet50(pretrained=False, cut_at_pooling=False, num_features=1024, norm=False, dropout=0, num_classes=30)
    model = nn.DataParallel(model).cuda()

    checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    feat, name = [], []

    for i, d in enumerate(test_loader):
        imgs, fnames, _ = d
        inputs = Variable(imgs)
        _, outputs = model(inputs)
        # outputs = F.sigmoid(outputs)
        feat.append(outputs.data.cpu().numpy())
        name.extend(fnames)
    feat = np.vstack(feat)
    # name = name.hstack(name)
    dist = compute_dist(feat, test_attrs, 'cosine')
    result = []
    for i, v in enumerate(dist):
        max = v.max()
        v = list(v)
        index = v.index(max)
        result.append((int(name[i][:3]), test_cls_list[index]))
    n = 0
    for pre, tar in result:
        if pre == tar:
            n = n+1
    print('the acc is {}/{}'.format(n, len(result)))


def main(args):
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    train_loader, test_loader = get_data(args)
    
    model = ALE()
    
    print(model)
    
    model = nn.DataParallel(model).cuda()
    
    # Optimizer
    if hasattr(model.module, 'base'):
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    def adjust_lr(epoch):
        if epoch in [80]:
            lr = 0.1 * args.lr
            print('=====> adjust lr to {}'.format(lr))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()

        loss = AverageMeter()
        iteration  = 935 * epoch
        # print(iteration)

        for i,d in enumerate(train_loader):
            iteration += 1

            img_embeds, class_embeds, metas = d

            optimizer.zero_grad()

            comps = model(img_embeds)
            mse_loss = nn.L1Loss(size_average=False)(outputs, attr_targets)
            mse_loss = mse_loss/args.batch_size
            loss.update(mse_loss.data[0], img_embeds.size(0))
            
            mse_loss.backward()
            optimizer.step()

            vis.line(X=torch.ones((1,)) * iteration,
                     Y=torch.Tensor((loss.avg,)),
                     win='reid softmax loss of network',
                     update='append' if iteration > 0 else None,
                     opts=dict(xlabel='iteration', title='Loss', legend=['Loss'])
                     )

            if (i + 1) % args.print_freq == 0:
                print('Epoch: [{}][{}/{}]\t Loss {:.6f} ({:.6f})\t'
                      .format(epoch, i + 1, len(train_loader),
                              loss.val, loss.avg))

            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_top1': 0,
            }, False, fpath=osp.join(args.model_dir, 'checkpoint.pth.tar'))
        test(test_loader, test_cls_list, test_attrs, args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)

    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50')

    parser.add_argument('--features', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/Dataset')
    parser.add_argument('--model-dir', type=str, metavar='PATH', default='/home/stage/yuan/ZSL/script/model')

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())

