from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
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
from utils.myData import myDataSet
from utils.transform_test_image import get_test_attrs
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.ale import ALE
from test import compute_dist
from zsl_eval.evaluate import evaluate
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    opts = dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=opts
    ), opts


def update_vis_plot(iteration, zz, zzz, window1,  update_type,
                    opts,epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([zz, zzz, zz + zzz]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type,
        opts=opts
    )
    



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_delta(yn, class_num=50):
    delta = torch.zeros(class_num)
    delta[yn]=1
    return delta

def rank(yn,comp):
    comp_l = comp  + get_delta(yn)
    #print("comp_l: ",comp_l.grad_fn)
    mygte = comp_l[comp_l>=comp[yn]] 
    mygte = mygte / mygte
    #mygte = comp_l>=comp[yn]
    #print("mygte: ",mygte.grad_fn)
    return torch.sum(mygte).int()
    
def get_l(r):
    return torch.sum( torch.tensor( [1/i for i in range(1,r+1)] ) )    
    
def ale_loss(yns,comps):
    summ = torch.tensor(0)
    for i in range(comps.shape[0]):
        comp = comps[i]
        yn = yns[i]
        r = rank(yn,comp)
        #print("RR: ",r)
        lr = get_l(r)
        #print("LR: ",lr)
        #print(r.grad_fn)
        lr_over_r = lr / r
        #print("Over: ",lr_over_r)
        #print(lr_over_r.shape)
        #print(comp.shape)
        comp_2 = comp+get_delta(yn)-comp[yn]
        #print(comp_2.shape)
        comp_3 = comp_2[comp_2>0]
        #print(comp_3.shape)
        comp_4 = lr_over_r*comp_3
        #print(comp_4.shape)
        summ = summ + torch.sum(comp_4)
        #print(summ.shape)


    return summ

def get_data(args):

    # Data loading code
    train_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = True, features=args.features),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = False, features=args.features),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader


def top1_acc(gts,comps):
    preds = comps.max(1)[1]
    return torch.sum(gts==preds).item() / comps.size(0)

def top5_acc(gts,comps):
    preds = torch.topk(comps,5)[1]
    retval = sum([ 1 for (i,x) in enumerate(gts) if x.item() in preds[i] ])
    return retval / comps.size(0)

def test(test_loader,args, model_dir=None, model=None):
    
    #checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint.pth.tar'))
    #model.module.load_state_dict(checkpoint['state_dict'])
    
    if args.gpu:
        model = model.cuda()
    model.eval()
    loss = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    with torch.no_grad():
        for i, d in enumerate(test_loader):
            img_embeds, class_embeds, metas = d
            if args.gpu:
                img_embeds = img_embeds.cuda()
            comps = model(img_embeds)
            if args.gpu:
                comps = comps.cpu()
            mse_loss = ale_loss(metas["class"],comps)
            mse_loss = mse_loss/args.batch_size
            loss.update(mse_loss.item(), img_embeds.size(0))

            acctop1 = top1_acc(metas["class"],comps)
            acctop5 = top5_acc(metas["class"],comps)
            acc1.update(acctop1,img_embeds.size(0))
            acc5.update(acctop5,img_embeds.size(0))
        
    print('Test Acc1: {:.3f} Acc5: {:.3f} Loss: {:.4f}'.format( 
    acc1.avg,acc5.avg,loss.avg
    ))
    print()
    
    return acc1.avg, acc5.avg, loss.avg

def eval_func( inp,embedding_matrix,model_dir):
    """
    input set X, [n_samples, d_features]
    ground-truth output embeddings (or attributes) per class, S, [n_classes, d_attributes]
    
    retval:
        [n_samples, n_classes] (i guess so?)
    """
    model_path = osp.join(model_dir, 'checkpoint.pth.tar')
    #embedding_matrix = torch.from_numpy(embedding_matrix).cuda()
    embedding_matrix = embedding_matrix.cuda()
    model = ALE(embedding_matrix,img_embed_size = inp.shape[1] , word_embed_size= embedding_matrix.shape[1] , gpu=True, dropout=0)
    model = model.cuda()
    with HiddenPrints():
        checkpoint = load_checkpoint(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    inp = torch.from_numpy(inp).cuda()
    retval = model(inp)
    
    retval = retval.cpu().detach().numpy()
    
    return retval
    
    

def main(args):
    vis = visdom.Visdom(env=args.model_dir)
    vis.check_connection()
    args.model_dir = osp.join("./log",args.model_dir)
    print(args.model_dir)
    if args.test:
        evaluate(eval_func,args.dset_path, args.model_path)
        exit()
    """
    vis_title = 'Blablablab'
    vis_legend = ['one Loss', 'two Loss', 'Total Loss']
    iter_plot, opts = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
    update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                    iter_plot,  'append', opts)
    """
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    train_loader, test_loader = get_data(args)
    embedding_matrix = train_loader.dataset.get_embedding_matrix().float()
    if args.gpu:
        embedding_matrix = embedding_matrix.cuda()
        
    model = ALE(embedding_matrix,img_embed_size = train_loader.dataset.get_image_embed_size() , word_embed_size=train_loader.dataset.get_word_embed_size() , gpu=args.gpu, dropout=args.dropout)
    
    print(model)
    
    if args.gpu:
        #model = nn.DataParallel(model).cuda()
        model = model.cuda()
        print("is_cuda: ",next(model.parameters()).is_cuda)
        print("device: ",next(model.parameters()).device)
    param_groups = model.parameters()
    
    """
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    """
    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                weight_decay=args.weight_decay,
                                amsgrad=True)

    def adjust_lr(epoch):
        if epoch in [80]:
            lr = 0.1 * args.lr
            print('=====> adjust lr to {}'.format(lr))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)
    best_zsl_acc = 0
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()

        loss = AverageMeter()
        acc1 = AverageMeter()
        acc5 = AverageMeter()
        iteration  = 935 * epoch
        
        for i,d in enumerate(train_loader):
            iteration += 1
    
            img_embeds, class_embeds, metas = d
            if args.gpu:
                img_embeds = img_embeds.cuda()
        
            img_embeds.requires_grad_()
            

            optimizer.zero_grad()
            comps = model(img_embeds)
            if args.gpu:
                comps = comps.cpu()
            mse_loss = ale_loss(metas["class"],comps)
            mse_loss = mse_loss/args.batch_size
            loss.update(mse_loss.item(), img_embeds.size(0))
            
            mse_loss.backward()
            optimizer.step()
            
            acc1_train = top1_acc(metas["class"],comps)
            acc5_train = top5_acc(metas["class"],comps)
            acc1.update(acc1_train,img_embeds.size(0))
            acc5.update(acc5_train,img_embeds.size(0))


        if (epoch + 1) % args.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t Loss  {:.6f}\t Acc1 {:.3f}\t Acc5 {:.3f}\t'
                  .format(epoch, i + 1, len(train_loader),
                         loss.avg, acc1.avg, acc5.avg))
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1
        }, False, fpath=osp.join(args.model_dir, 'checkpoint.pth.tar'))
        
        acc1_val,acc5_val,loss_val = test(test_loader,args, args.model_dir,model=model)
        zsl_acc, zsl_acc_seen, zsl_acc_unseen = evaluate(eval_func,args.dset_path, args.model_dir, embedding_matrix)
        print("------")
        
        ##### PLOTS
        #Loss
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((loss.avg,)),
                 win='loss',
                 update='append' if epoch > 0 else None,
                 name="train",
                 opts=dict(xlabel='Epoch', title='Loss', legend=['train','val'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((loss_val,)),
                 win='loss',
                 update='append' if epoch > 0 else None,
                 name="val",
                 opts=dict(xlabel='Epoch', title='Loss', legend=['train','val'])
                 )
        #acc1
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((acc1.avg,)),
                 win='acc1',
                 update='append' if epoch > 0 else None,
                 name="train",
                 opts=dict(xlabel='Epoch', title='Acc1', legend=['train','val'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((acc1_val,)),
                 win='acc1',
                 update='append' if epoch > 0 else None,
                 name="val",
                 opts=dict(xlabel='Epoch', title='Acc1', legend=['train','val'])
                 )
        #acc5
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((acc5.avg,)),
                 win='acc5',
                 update='append' if epoch > 0 else None,
                 name="train",
                 opts=dict(xlabel='Epoch', title='Acc5', legend=['train','val'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((acc5_val,)),
                 win='acc5',
                 update='append' if epoch > 0 else None,
                 name="val",
                 opts=dict(xlabel='Epoch', title='Acc5', legend=['train','val'])
                 )
        #testing
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_acc,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="zsl_acc",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_acc_seen,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="seen",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_acc_unseen,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="unseen",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen'])
                 )
        ##########

        
        if zsl_acc > best_zsl_acc:
            best_zsl_acc = zsl_acc
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'zsl_acc': zsl_acc,
            }, False, fpath=osp.join(args.model_dir, 'checkpoint_best.pth.tar'))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=4)


    # model
    
    parser.add_argument('--features', type=str, default=None)
    parser.add_argument('--dset-path', type=str, default=None)

    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")

    parser.add_argument('--print-freq', type=int, default=1)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--model-dir', type=str, metavar='PATH', default='./model')

    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    
    parser.add_argument('--gpu',  dest='gpu', action='store_true',
                        help='Enable gpu and cuda')
    
    parser.add_argument('--test',  dest='test', action='store_true',
                        help="test the results on bulent's script")
    
    
    main(parser.parse_args())

