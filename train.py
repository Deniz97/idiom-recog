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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable
from utils.myData import myDataSet
from utils.rnnData import rnnData
from utils.transform_test_image import get_test_attrs
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.ale import ALE
from test import compute_dist
from eval_zsl.evaluate import evaluate
from visdomsave import vis
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_delta(yn, class_num=50):
    delta = torch.ones(class_num)
    delta[yn]=0
    return delta

def rank(yn,comp):
    comp_l = comp  + get_delta(yn,class_num=comp.shape[0])
    #print("comp_l: ",comp_l.grad_fn)
    mygte = comp_l[comp_l>=comp[yn]] 
    mygte = mygte / mygte
    #mygte = comp_l>=comp[yn]
    #print("mygte: ",mygte.grad_fn)
    return torch.sum(mygte).int()
    
def get_l(r):
    return torch.sum( torch.tensor( [1/i for i in range(1,r+1)] ) )    
    
def ale_loss(comps,yns):
    summ = torch.tensor(0)
    for i in range(comps.shape[0]):
        comp = comps[i]
        yn = yns[i]
        r = rank(yn,comp)
        lr = get_l(r)
        lr_over_r = lr / r
        comp_2 = comp+get_delta(yn,class_num = comp.shape[0])-comp[yn]
        comp_3 = comp_2[comp_2>=0]
        comp_4 = lr_over_r*comp_3
        summ = summ + torch.sum(comp_4)
    return summ/comps.shape[0]


def get_data(args):
    train_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = True, db=args.dset),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = False, db=args.dset, a = train_loader.dataset.a, b = train_loader.dataset.b ),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def get_data_rnn(args):
    train_loader = torch.utils.data.DataLoader(
            rnnData(),
            batch_size=args.rnn_batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            rnnData(idioms=train_loader.dataset.idioms, keys=train_loader.dataset.valid_keys,
                    words=train_loader.dataset.words ),
            batch_size=args.rnn_batch_size, shuffle=False,
            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader

def top1_acc(gts,comps):
    preds = comps.max(1)[1]
    acc= torch.sum(gts==preds).item() / comps.size(0)
    return acc

def top5_acc(gts,comps):
    preds = torch.topk(comps,5 if 5<=comps.shape[1] else comps.shape[1])[1]
    retval = sum([ 1 for (i,x) in enumerate(gts) if x.item() in preds[i] ])
    return retval / comps.size(0)


def rnn_test(val_loader,args, model=None, criterion = None):
    
    #checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint.pth.tar'))
    #model.module.load_state_dict(checkpoint['state_dict'])
    
    
    model = model.cuda()
    model.eval()
    loss = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(val_loader):
            img_embeds, class_embeds, metas = d
            img_embeds = img_embeds.cuda()
            comps, _ = model(img_embeds)
            comps = comps.cpu()
            loss_value = criterion(comps,metas["class"])
            loss_value = loss_value/ img_embeds.size(0)
            loss.update(loss_value.item(), img_embeds.size(0))

            acctop1 = top1_acc(metas["class"],comps)
            acctop5 = top5_acc(metas["class"],comps)
            acc1.update(acctop1,img_embeds.size(0))
            acc5.update(acctop5,img_embeds.size(0))
        
    print('Test Acc1: {:.3f} Acc5: {:.3f} Loss: {:.4f}'.format( 
    acc1.avg,acc5.avg,loss.avg
    ))
    print()
    
    return acc1.avg, acc5.avg, loss.avg


def test(val_loader,args, em=None,model=None, criterion = None):
    
    #checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint.pth.tar'))
    #model.module.load_state_dict(checkpoint['state_dict'])
    model.set_embedding(em)
    if args.gpu:
        model = model.cuda()
    model.eval()
    loss = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()

    with torch.no_grad():

        for i, d in enumerate(val_loader):
            inps, targets, keys, lengths = d
            inps = inps.cuda()

            #packed the sequence
            packed_inputs = pack_padded_sequence(inps, lengths, batch_first=True)
            ###

            packed_outs = model(packed_inputs)
            
            outs, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_outs, batch_first=True, padding_value=0)
            ##check this
            outs = outs[:,lengths,:]
            loss_value = criterion(outs, targets )
            loss.update(loss_value.item(), inps.size(0))
        
    print('Loss: {:.4f}'.format(loss.avg))
    
    return loss.avg

def eval_func( inp,embedding_matrix, model):
    """
    input set X, [n_samples, d_features]
    ground-truth output embeddings (or attributes) per class, S, [n_classes, d_attributes]
    
    retval:
        [n_samples, n_classes] (i guess so?)
    """

    model_path = osp.join(model_dir, 'checkpoint.pth.tar')

    embedding_matrix = torch.from_numpy(embedding_matrix).float().cuda()
    model.set_embedding(embedding_matrix)
    model = model.cuda()
    #with HiddenPrints():
    #    checkpoint = load_checkpoint(model_path)
    #    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    inp = torch.from_numpy(inp).cuda()
    retval = model(inp)

    retval = retval.cpu().detach().numpy()
    
    return retval
    
def train_rnn(args,vis):

    train_loader, val_loader = get_data_rnn(args)
    
    if args.att=="rnn":
        model = torch.nn.RNN(300,300,1)
    elif args.att=="lst":
        model = torch.nn.LSTM(300,300,1)
    elif args.att=="gru":
        pass

    
    print(model)
    
    model = nn.DataParallel(model).cuda()
    print("is_cuda_rnn: ",next(model.parameters()).is_cuda)
    print("device_rnn: ",next(model.parameters()).device)
    param_groups = model.parameters()
    
    if args.cost == "MSE":
        criterion = torch.nn.MSELoss()
    elif args.cost == "COS":
        coss_loss =torch.nn.CosineEmbeddingLoss() #margin can be added
        criterion = lambda x,y:coss_loss(x,y,torch.ones((x.shape[0])))
    else:
        assert False, "Unknown cost function"
    """
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    """
    optimizer = torch.optim.Adam(param_groups, lr=args.rnn_lr,
                                weight_decay=args.rnn_wd,
                                amsgrad=True)

    def adjust_lr(epoch):
        if epoch in [5,10,15]:

            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                print('=====> adjust lr to {}'.format(g['lr']))
    
    best_loss = 0
    
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()

        loss = AverageMeter()
        
        for i,d in enumerate(train_loader):
    
            inps, targets, keys, lengths = d
            inps = inps.cuda()

            #packed the sequence
            packed_inputs = pack_padded_sequence(inps, lengths, batch_first=True)

            ###

            optimizer.zero_grad()
            packed_outs, _ = model(packed_inputs)
            
            outs, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_outs, batch_first=True, padding_value=0)
            ##check this
            outs = outs[:,lengths,:]
            loss_value = criterion(outs, targets )
            loss.update(loss_value.item(), inps.size(0))
                        
            loss_value.backward()
            optimizer.step()


        if (epoch + 1) % args.print_freq == 0:
            print('Epoch: [{}][{}/{}]\t Loss  {:.6f}\t'
                  .format(epoch, i + 1, len(train_loader),
                         loss.avg))
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1
        }, False, fpath=osp.join(args.model_dir, 'rnn_checkpoint.pth.tar'))
        
        
        test_loss_val = rnn_test(val_loader,args,model=model, criterion=criterion)
        #zsl_acc, zsl_acc_seen, zsl_acc_unseen = evaluate(eval_func,args.dset_path, args.model_dir, train_loader.dataset.get_embedding_matrix_all(), model=model)
        #zsl_harmonic = 2*( zsl_acc_seen * zsl_acc_unseen ) / ( zsl_acc_seen + zsl_acc_unseen )
        
        #print("Harmonic: %.3f" % zsl_harmonic)
        #print("------")
        
        ##### PLOTS
        #Loss
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((loss.avg,)),
                 win='rnnloss',
                 update='append' if epoch > 0 else None,
                 name="train",
                 opts=dict(xlabel='Epoch', title='Loss', legend=['train','val'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((test_loss_val,)),
                 win='rnnloss',
                 update='append' if epoch > 0 else None,
                 name="val",
                 opts=dict(xlabel='Epoch', title='Loss', legend=['train','val'])
                 )
        ##########

        
        if best_loss > test_loss_val:
            best_loss = test_loss_val
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'harmonic': best_loss,
            }, False, fpath=osp.join(args.model_dir, 'rnn_checkpoint_best.pth.tar'))

    return model, train_loader

def get_em(args,loader,model, words_dict, idioms_dict,mode="train"): #do for all clasees
    if args.att=="label":
        return loader.dataset.get_embedding_matrix()

    dataset = loader.dataset
    labels = dataset.get_labels(mode)

    em = torch.zeros((len(labels),300)).double()

    for i,w in enumerate(labels):
        words = w.split("-")
        words_embeds = torch.zeros((1,len(words),300))
        if args.mode=="all" or ( len(words)>1 and ((not args.mode=="available") or (not w in idioms_dict))):
            if args.att in ["rnn","lstm","gru"]:
                em[i], _ = model(words_embeds)
            elif args.att == "avg":
                for word in words:
                    em[i] += torch.tensor(words_dict[word]).double()
                em[i] /= len(words)
            elif args.att=="fisher":
                #TODO
                pass
        else:
            if "-" in w:
                em[i] = torch.tensor(idioms_dict[words[0]]).double()
            else:
                em[i] = torch.tensor(words_dict[words[0]]).double()


    return em.float()


def main(args):
    vis = visdom.Visdom(env=args.model_dir)
    vis.check_connection()
    args.model_dir = osp.join("./log",args.model_dir)
    print(args.model_dir)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    
    rnn_model = None
    if args.att in ["rnn","lstm","gru"]:
        rnn_model, rnn_loader = train_rnn(args,vis)
    else:
        rnn_loader,_ = get_data_rnn(args)

    train_loader, val_loader = get_data(args)


    train_embedding_matrix = get_em(args,train_loader,words_dict = rnn_loader.dataset.words,idioms_dict=rnn_loader.dataset.idioms, model=rnn_model,mode="train").cuda()
    val_embedding_matrix = get_em(args,val_loader,words_dict = rnn_loader.dataset.words,idioms_dict=rnn_loader.dataset.idioms,model=rnn_model, mode="val").cuda()
    all_embedding_matrix = get_em(args,val_loader,words_dict = rnn_loader.dataset.words,idioms_dict=rnn_loader.dataset.idioms,model=rnn_model,mode="all").cuda()
    unseen_embedding_matrix = get_em(args,val_loader,words_dict = rnn_loader.dataset.words,idioms_dict=rnn_loader.dataset.idioms,model=rnn_model,mode="unseen").cuda()

    
    model = ALE(train_embedding_matrix,img_embed_size = train_loader.dataset.get_image_embed_size(), gpu=True, dropout=args.dropout)
    
    print(model)
    
    model = nn.DataParallel(model).cuda()
    print("is_cuda: ",next(model.parameters()).is_cuda)
    print("device: ",next(model.parameters()).device)
    param_groups = model.parameters()
    
    if args.cost == "ALE":
        criterion = ale_loss
    elif args.cost == "CEL":
        print("Using cross-entrophy loss")
        criterion =torch.nn.CrossEntropyLoss()
    else:
        assert False, "Unknown cost function"
    """
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    """
    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                weight_decay=args.wd,
                                amsgrad=True)

    def adjust_lr(epoch):
        if epoch in [5,10,15]:

            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                print('=====> adjust lr to {}'.format(g['lr']))
    
    best_harmonic = 0
    
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.train()
        model.set_embedding(train_loader.dataset.get_embedding_matrix())

        loss = AverageMeter()
        acc1 = AverageMeter()
        acc5 = AverageMeter()
        
        for i,d in enumerate(train_loader):
    
            img_embeds, class_embeds, metas = d
            if args.gpu:
                img_embeds = img_embeds.cuda()

            optimizer.zero_grad()
            comps = model(img_embeds)
            if args.gpu:
                comps = comps.cpu()
            loss_value = criterion(comps, metas["class"])
            loss.update(loss_value.item(), img_embeds.size(0))
                        
            loss_value.backward()
            
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
        
        model.set_embedding(val_loader.dataset.get_embedding_matrix())
        
        acc1_val,acc5_val,loss_val = test(val_loader,args,em=val_embedding_matrix, model=model, criterion=criterion)
        zsl_acc, zsl_acc_seen, zsl_acc_unseen = evaluate(eval_func,args.dset, all_embedding_matrix, unseen_embedding_matrix, model=model)
        zsl_harmonic = 2*( zsl_acc_seen * zsl_acc_unseen ) / ( zsl_acc_seen + zsl_acc_unseen )
        
        print("Harmonic: %.3f" % zsl_harmonic)
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
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen','harmonic'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_acc_seen,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="seen",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen','harmonic'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_acc_unseen,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="unseen",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen','harmonic'])
                 )
        vis.line(X=torch.ones((1,)) * epoch,
                 Y=torch.Tensor((zsl_harmonic,)),
                 win='test',
                 update='append' if epoch > 0 else None,
                 name="harmonic",
                 opts=dict(xlabel='Epoch', title='testing', legend=['zsl_acc','seen','unseen', 'harmonic' ])
                 )
        ##########

        
        if zsl_harmonic > best_harmonic:
            best_harmonic = zsl_harmonic
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'harmonic': best_harmonic,
            }, False, fpath=osp.join(args.model_dir, 'checkpoint_best.pth.tar'))

        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('--rnn-batch-size', type=int, default=64)


    # model
    
    parser.add_argument('--features', type=str, default=None)
    parser.add_argument('--dset', type=str, default="AWA2")

    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rnn-lr', type=float, default=0.01)

    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--rnn-wd', type=float, default=1e-4)

    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')

    parser.add_argument('--epochs', type=int, default=20)

    # misc
    working_dir = osp.dirname(osp.abspath(__file__))

    parser.add_argument('--model-dir', type=str, metavar='PATH', default='./model')
    parser.add_argument('--att', type=str, metavar='PATH', default='rnn') #label,rnn, lstm, gru,avg,fisher,hocanın formülleri
    parser.add_argument('--mode',type=str, metavar='PATH',default=None)
    
    parser.add_argument('--gpu',  dest='gpu', action='store_false',
                        help='Enable gpu and cuda')
    
    
    parser.add_argument('--cost', type=str, metavar='PATH', default='ALE')
    parser.add_argument('--rnn-cost', type=str, metavar='PATH', default='MSE')
    
    
    

    
    
    try:
        args = parser.parse_args()
        main(args)
        vis.create_log(args.model_dir)
    except KeyboardInterrupt:
        print("Saving and Exiting...")
        vis.create_log(args.model_dir)
        exit()

