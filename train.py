from __future__ import print_function, absolute_import
import os.path as osp
import os
import sys
import json
import sys
import argparse
import random
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
from sklearn import cluster
from warp import WARPLoss
from utils.myData import myDataSet
from utils.rnnData import rnnData
from utils.transform_test_image import get_test_attrs
from utils.utils import AverageMeter, save_checkpoint, load_checkpoint
from model.ale import ALE
from eval_zsl.evaluate import evaluate
from visdomsave import vis
from model.affine import Affine
from model.fc import FC
from progress.bar import Bar
import pickle as pc
from argparse import Namespace

perclass_accs_global = {}

def draw_vis(vis,title,name,epoch,value,legend):
    if vis is None:
        return
    vis.line(X=torch.ones((1,)) * epoch,
             Y=torch.Tensor((value,)),
             win = title,
             update='append' if epoch > 0 else None,
             name=name,
             opts=dict(xlabel='Epoch', title = title , legend= legend ))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_delta(yn, class_num):
    delta = torch.ones(class_num) 
    delta[yn]=0
    return delta.cuda()

def rank(yn,comp):
    comp = comp.detach()
    comp_l = comp  + get_delta(yn,class_num=comp.shape[0])
    mygte = torch.ones(comp.shape)[comp_l>=comp[yn]]
    return int(torch.sum(mygte).item())
    
def get_l(r):
    return sum([1/i for i in range(1,r+1)])
    
def ale_loss(comps,yns):
    summ = torch.tensor(0).cuda()
    total_offenders = 0
    for i in range(comps.shape[0]):
        comp = comps[i]
        #print(comp)
        yn = yns[i]
        #print(comp[yn])
        #print("-------")
        r = rank(yn,comp)
        lr = get_l(r)
        lr_over_r = lr / r
        comp_2 = comp+get_delta(yn,class_num = comp.shape[0])-comp[yn]
        #comp_3 = comp_2[comp_2>=0]
        comp_3 = torch.nn.functional.threshold(comp_2, 0, 0)
        comp_4 = lr_over_r*comp_3
        #comp_4 = comp_3
        summ = summ + torch.sum(comp_4)
    return summ/comps.shape[0]

def get_data(args, is_in):
    train_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = "train", db=args.dset, is_in = is_in),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = "val", db=args.dset, a = train_loader.dataset.a, b = train_loader.dataset.b, is_in = is_in ),
            batch_size=1024, shuffle=False,
            num_workers=4, pin_memory=True)
    seen_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = "seen", db=args.dset, a = train_loader.dataset.a, b = train_loader.dataset.b, is_in = is_in ),
            batch_size=1024, shuffle=False,
            num_workers=4, pin_memory=True)
    unseen_loader = torch.utils.data.DataLoader(
            myDataSet(is_train = "unseen", db=args.dset, a = train_loader.dataset.a, b = train_loader.dataset.b, is_in = is_in ),
            batch_size=1024, shuffle=False,
            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, seen_loader, unseen_loader

def get_data_rnn(args):
    train_loader = torch.utils.data.DataLoader(
            rnnData( args=args),
            batch_size=args.rnn_batch_size, shuffle=True, pin_memory = True,
            num_workers=4)

    val_loader = torch.utils.data.DataLoader(
            rnnData( keys=train_loader.dataset.valid_keys,
                    embeddings=train_loader.dataset.embeddings,  args=args ),
            batch_size=args.rnn_batch_size, shuffle=False, pin_memory = True,
            num_workers=4)
    
    return train_loader, val_loader

def top1_acc(gts,comps):
    preds = comps.max(1)[1].cuda()
    #print("Top1 Acc(batch): %d/%d" %(gts[gts==preds].shape[0], gts.shape[0]))
    #print("------")
    acc= torch.sum(gts==preds).item()
    acc = acc / comps.size(0)
    return acc

def top1_acc_perclass(gts,comps,res):
    acc = torch.zeros((1)).float().cuda()
    #print("-------")
    #print("comps: ",comps.shape)
    #print("olds res: ",res.sum())
    preds = comps.max(1)[1].cuda()
    for clas in gts.unique():
        idx = gts==clas
        #print("gtx[idx]: ",gts[idx].shape)
        if torch.sum(idx) == 0:
            continue
        #print("Non zero for %d, sum: %d" % ( clas, torch.sum(idx) ) )
        #print("true preds: ",gts[idx] == preds[idx])
        res[clas] += ( torch.sum(gts[idx] == preds[idx])  ).int().item()
        #print("acc: ",acc)
    #print(gts.shape)
    #print("New res: ",res.sum())

def calc_perclass(res,counts,strm):
    
    labels, counts = counts
    #print("Res sum: %d, Counts sum: %d" % (res.sum(),counts.sum()))
    for i,r in enumerate(res):
        if res[i] > 0:
            pass
            #print("res %d - counts %d :: " % (res[i],counts[i]), end="")
        if counts[i] ==0 and res[i] > 0:
            print("WEIRD")
        if counts[i] != 0:
            res[i] = res[i] / counts[i]


    ###LOG
    perclass_accs_global[strm] = {}
    for i,l in enumerate(labels):
        if counts[i] != 0:
            perclass_accs_global[strm][l] = res[i].item()

        
    ##

    #print("\nOut of %d" % counts[counts!=0].sum() )
    #print()
    #print("-----------")
    #print(res[res>1].shape)
    #print(res.shape)
    #print(res[res!=0].shape)
    res = res.sum()
    kk = counts!=0
    cc = counts[kk].shape[0]
    res = res / cc

    #print("counts: ",counts[counts!=0].shape)
    return res

def top5_acc(gts,comps):
    preds = torch.topk(comps,5 if 5<=comps.shape[1] else comps.shape[1])[1]
    retval = sum([ 1 for (i,x) in enumerate(gts) if x.item() in preds[i] ])
    return retval / comps.size(0)

def test(val_loader,args, em=None, model=None, criterion = None, strm="Validation"):
    #checkpoint = load_checkpoint(osp.join(model_dir, 'checkpoint.pth.tar'))
    #model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.set_embedding(em)
    loss = AverageMeter()
    acc1 = AverageMeter()
    acc5 = AverageMeter()
    bar_val = Bar(strm, max=len(val_loader))
    perclass_accuracies = torch.zeros((em.shape[0])).cuda()
    with torch.no_grad():
        for i, d in enumerate(val_loader):
            img_embeds, metas = d
            img_embeds = img_embeds.cuda()
            comps = model(img_embeds)
            classes = metas["class"].cuda()
            loss_value = criterion(comps, classes)
            loss.update(loss_value.item(), img_embeds.size(0))

            acctop1 = top1_acc(classes,comps)
            acctop5 = top5_acc(classes,comps)
            top1_acc_perclass(classes,comps, perclass_accuracies)
            acc1.update(acctop1,img_embeds.size(0))
            acc5.update(acctop5,img_embeds.size(0))
            bar_val.suffix = 'Epoch: [{}/{}]\t Loss  {:.6f}\t Acc1 {:.3f}\t Acc5 {:.3f}\t'.format( (i + 1), len(val_loader),
                    loss.avg, acc1.avg, acc5.avg )
            bar_val.next()
        bar_val.finish() 
        
    accpc = calc_perclass(perclass_accuracies,val_loader.dataset.valid_sample_per_class,strm).item()
    print(strm+" acc_pc: %f" % accpc)
    
    return acc1.avg, acc5.avg, accpc, loss.avg



def pack_seq(inps,targets,keys,lengths):
    lengths, sort_order = lengths.sort(descending=True)
    inps = inps[sort_order,...]
    targets = targets[sort_order,...]
    keys = np.asarray(keys)
    keys = keys[sort_order,...]
    
    packed_inputs = pack_padded_sequence(inps, lengths, batch_first=True)

    return packed_inputs, targets, keys

def unpack_seq(packed_outs):
    outs, out_lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_outs, batch_first=True, padding_value=0)
    outs = outs.cpu()
    out_lengths = out_lengths.cpu() - 1
    outs = outs[torch.arange(outs.size(0)), out_lengths]
    #outs = torch.nn.functional.normalize(outs,p=1,dim=1)*2-1
    return outs

def lstm2_pre(inps,lengths):

    if inps.shape[1] == 1:
        hidden = torch.zeros((1,inps.shape[0],300)).cuda()
        lengths = lengths - 1
        return inps, hidden, lengths
    else:
        hidden = torch.zeros((1,inps.shape[0],300)).cuda()
        for i,inp in enumerate(inps):
            if lengths[i] == 1:
                continue
            else:
                inps[i] = torch.cat ( ( inps[i][1:,...].cuda(), torch.zeros((1,300)).cuda() ), 0  ).cuda()
                hidden[0][i] = inps[i][:1,...].cuda()
                lengths[i] = lengths[i] - 1
        return inps, hidden, lengths

def rnn_test(val_loader,args,model=None, criterion = None,epoch=0):

    model = model.cuda()
    model.eval()
    loss = AverageMeter()

    with torch.no_grad():
        bar = Bar("Validation", max=len(val_loader))
        for i, d in enumerate(val_loader):
            inps, targets, keys, lengths = d
            inps = inps.cuda()
            if args.att == "lstm2":
                inps, hidden, lengths = lstm2_pre(inps,lengths)
            packed_inputs, targets, keys = pack_seq(inps,targets,keys, lengths)
            if args.att == "lstm2":
                outs, _ = model(packed_inputs, ( hidden, torch.zeros(hidden.shape).cuda()))
            else:
                outs, _ = model(packed_inputs)
            if args.att in ["rnn","lstm","gru","lstm2"]:
                outs = unpack_seq(outs)    
            outs = outs.cpu()
            loss_value = criterion(outs, targets )
            loss.update(loss_value.item(), inps.size(0))
            bar.suffix = 'Epoch: [{}][{}/{}] \t\t Loss  {:.6f}'.format(epoch, i + 1, len(val_loader),loss.avg)
            bar.next()
        bar.finish()
    
    return loss.avg

def eval_func( inp,embedding_matrix, model):
    """
    input set X, [n_samples, d_features]
    ground-truth output embeddings (or attributes) per class, S, [n_classes, d_attributes]
    
    retval:
        [n_samples, n_classes] (i guess so?)
    """
    embedding_matrix = embedding_matrix.float().cuda()
    model.set_embedding(embedding_matrix)
    model = model.cuda()
    model.eval()
    inp = torch.from_numpy(inp).cuda()
    retval = model(inp)

    retval = retval.cpu().detach().numpy()
    
    return retval


def train_rnn_tick(train_loader,args,model,optimizer,criterion,epoch):
    train_loader.dataset.epoch = epoch
    loss = AverageMeter()
    bar = Bar('Training', max=len(train_loader))
    for i,d in enumerate(train_loader):
    
        inps, targets, keys, lengths = d
        inps = inps.cuda()
        
        if args.att == "lstm2":
            inps, hidden, lengths = lstm2_pre(inps,lengths)
            packed_inputs, targets, keys = pack_seq(inps,targets,keys, lengths)
            optimizer.zero_grad()
            outs, _ = model(packed_inputs,(hidden,torch.zeros(hidden.shape).cuda()) )
        else:
            packed_inputs, targets, keys = pack_seq(inps,targets,keys, lengths)
            optimizer.zero_grad()
            outs, _ = model(packed_inputs)

        if args.att in ["rnn","lstm","gru","lstm2"]:
            outs = unpack_seq(outs)
        outs = outs.cpu()
        loss_value = criterion(outs, targets )
        loss.update(loss_value.item(), inps.size(0))
                    
        loss_value.backward()
        optimizer.step()
        bar.suffix = 'Epoch: [{}][{}/{}]\t Loss  {:.6f}\t'.format(epoch, i + 1, len(train_loader),loss.avg)
        bar.next()
    bar.finish()

def train_rnn(args,vis):
    print(str(args))
    train_loader, val_loader = get_data_rnn(args)
    print("Done loading data")
    if args.att=="rnn":
        model = torch.nn.RNN(300,300,1)
    elif args.att in ["lstm","lstm2"]:
        model = torch.nn.LSTM(300,300,1)
    elif args.att=="gru":
        model = torch.nn.GRU(300,300,1)
    elif args.att == "affine":
        model = Affine(word_embed_size = 300)
    elif args.att == "fc":
        model = FC(word_embed_size = 300)
    elif args.att == "fcb":
        model = FC(word_embed_size = 300, bias=True)

    model = model.cuda()
    print(model)
    
    #model = nn.DataParallel(model).cuda()
    print("is_cuda_rnn: ",next(model.parameters()).is_cuda)
    print("device_rnn: ",next(model.parameters()).device)
    param_groups = model.parameters()
    
    if args.rnn_cost == "MSE":
        criterion = torch.nn.MSELoss()
    elif args.rnn_cost == "COS":
        coss_loss =torch.nn.CosineEmbeddingLoss() #margin can be added
        criterion = lambda x,y:coss_loss(x,y,torch.ones((x.shape[0])))
    else:
        assert False, "Unknown rnn cost function"
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
        if epoch!= 0 and epoch in [args.rnn_epochs//3,2*args.rnn_epochs//3]:

            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                print('=====> adjust lr to {}'.format(g['lr']))
    
    best_loss = 100
    best_epoch = -1
    print("starting training")
    for epoch in range(0, args.rnn_epochs):
        adjust_lr(epoch)
        model.train()
        

        train_rnn_tick(train_loader,args,model,optimizer,criterion,epoch)
        
        
        test_loss_val = rnn_test(val_loader,args,model=model, criterion=criterion,epoch=epoch)
        
        ##### PLOTS
        #Loss
        if vis is not None:
            vis.line(X=torch.ones((1,)) * epoch,
                     Y=torch.Tensor((loss.avg,)),
                     win='rnnloss',
                     update='append' if epoch > 0 else None,
                     name="rnntrain",
                     opts=dict(xlabel='Epoch', title='rnnLoss', legend=['rnntrain','rnnval'])
                     )
            vis.line(X=torch.ones((1,)) * epoch,
                     Y=torch.Tensor((test_loss_val,)),
                     win='rnnloss',
                     update='append' if epoch > 0 else None,
                     name="rnnval",
                     opts=dict(xlabel='Epoch', title='rnnLoss', legend=['rnntrain','rnnval'])
                    )
        ##########

        
        if vis is None and best_loss - best_loss/100 > test_loss_val:
            best_loss = test_loss_val
            best_epoch = epoch
            with open("rnn_results.pc","rb") as filem:
                rnn_results = pc.load(filem)
            key = args.att+"_"+args.rnn_cost
            if key not in rnn_results or rnn_results[key]["best_loss"] - rnn_results[key]["best_loss"]/100 > test_loss_val:
                print("FOUND NEW BEST: ",key)
                with open("what_changed.txt","w") as filem:
                    filem.write("Found new best: %s\n" % key)
                rnn_results[key] = {}
                rnn_results[key]["args"] = args
                rnn_results[key]["best_loss"] = test_loss_val
                rnn_results[key]["best_epoch"] = epoch
                with open("rnn_results.pc","wb") as filem:
                    pc.dump(rnn_results,filem)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'loss': best_loss,
                }, False, fpath=osp.join("./models/", key+'_checkpoint_best.pth.tar'))
                with open("./models/"+key+'_checkpoint_best.txt',"w") as filem:
                    filem.write(str(args))

    if vis is None:
        return model, train_loader, best_loss, best_epoch

def get_em(args,loader, rnn_loader,model,fv,mode="train"): #do for all clasees
    
    dataset = loader.dataset
    labels = dataset.get_labels(mode)
    if args.att=="random":
        retval = torch.randn(len(labels),300, requires_grad=True)
        return retval

    if args.att=="label":
        return loader.dataset.get_embedding_matrix()

    if args.att == "fisher":
        em = torch.zeros((len(labels),300*args.kmeansk)).float()
    else:
        em = torch.zeros((len(labels),300)).float()
    if args.att in ["rnn","lstm","gru","lstm2"]:
        em = em.cuda()
    if model is not None and mode=="train":
        model.train()
    if model is not None and not mode=="train":
        model.eval()
    for i,w in enumerate(labels):
        words = w.split("-")
        word_embeds = torch.zeros((1,len(words),300))
        for idx,word in enumerate(words):
            word_embeds[0][idx] = rnn_loader.dataset.get_embedding(word)
        if args.mode != "nall" and ( len(words)>1 or args.att == "fisher" or args.mode=="all" ):
            if args.att in ["rnn","lstm","gru"]:
                inp = word_embeds.permute(1,0,2).contiguous().cuda()
                temp, _ = model(inp)
                em[i] = temp[-1,0,:]
            if args.att == "lstm2":
                inps,hidden, _ = lstm2_pre(word_embeds.cuda(),torch.zeros((10)))
                inps = inps.permute(1,0,2).contiguous().cuda()
                
                temp, _ = model(inps,(hidden.cuda(),torch.zeros(hidden.shape).cuda()))
                em[i] = temp[-1,0,:]
            elif args.att == "avg":
                for j,word in enumerate(words):
                    em[i] += word_embeds[0][j]
                em[i] /= len(words)
            elif args.att=="fisher":
                em[i] = fv.get_fv(word_embeds[0].numpy())
            elif args.att in ["affine","fc"]:
                temp, _ = model( torch.nn.utils.rnn.pack_sequence( [word_embeds[0].cuda()] ))
                em[i] = temp[0].cpu()
        else:
            if "-" in w:
                em[i] = rnn_loader.dataset.get_embedding(w)
            else:
                em[i] = word_embeds[0][0]
    if fv is not None:
        fv.print_stats()
    if not args.joint:
        em = em.detach()
    retval = torch.zeros(em.shape).float().cuda()
    for i,e in enumerate(em):
        retval[i] = em[i] / torch.norm(em[i],p=2)
    return retval

class FisherVector:
    def __init__(self,words,kmeansk, vis=None):
        self.words = np.asarray(words)
        self.kmeans_ = cluster.MiniBatchKMeans(n_clusters=kmeansk,verbose=0)
        self.kmeansk = kmeansk
        self.vis = vis
        self.stats = [0]* ( kmeansk + 1)

    def train(self):
        self.kmeans_.fit(self.words)
        print("Done k-means, centers: ")
        print(self.kmeans_.cluster_centers_.shape)

    def get_fv(self,embeddings):
        retval = torch.zeros((self.kmeansk*300))
        counter = [0] * self.kmeansk
        nb = 0
        for i in embeddings:
            c = self.kmeans_.predict(i.reshape(1,-1))[0]
            retval[c:c+300] += torch.tensor(i)[0]
            counter[c] += 1
        for i,c in enumerate(counter):
            if c!= 0:
                retval[i:i+100] /= c
                nb+=1
        self.stats[nb] += 1


        return retval
    def print_stats(self):
        print("Fisher Stats: ",self.stats)
        if self.vis is not None:
            vis.text(self.stats,"fvStats")

def main(args,vis,):
    best_seen = -1
    best_harmonic = -1
    best_epoch = -1
    best_unseen = -1
    print(str(args))
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # cudnn.benchmark = True
    if vis is not None:
        vis.text(str(args),win="args")
    rnn_model = None
    fv = None
    best_harmonic = 0
    torch.autograd.set_detect_anomaly(True)
    if args.att in ["rnn","lstm","gru","affine","fc","lstm2"]:
        #rnn_model, rnn_loader, best_loss, best_epoch = train_rnn(args,vis)
        rnn_loader,_ = get_data_rnn(args)
        if args.att=="rnn":
            rnn_model = torch.nn.RNN(300,300,1)
        elif args.att in ["lstm","lstm2"]:
            rnn_model = torch.nn.LSTM(300,300,1)
        elif args.att=="gru":
            rnn_model = torch.nn.GRU(300,300,1)
        elif args.att == "affine":
            rnn_model = Affine(word_embed_size = 300)
        elif args.att == "fc":
            rnn_model = FC(word_embed_size = 300)
        elif args.att == "fcb":
            rnn_model = FC(word_embed_size = 300, bias=True)

        rnn_model = rnn_model.cuda()
        checkpoint = load_checkpoint(osp.join("./models/", args.att+"_"+args.rnn_cost+'_checkpoint_best.pth.tar'))
        rnn_model.load_state_dict(checkpoint['state_dict'])
        rnn_model = rnn_model.cuda()
    else:
        rnn_loader,_ = get_data_rnn(args)
        if args.att=="fisher":
            fv = FisherVector(rnn_loader.dataset.get_all_embeddings(30000),args.kmeansk)
            fv.train()
    print("Rnn Model: ")
    print(rnn_model)

    train_loader, val_loader, seen_loader, unseen_loader = get_data(args,rnn_loader.dataset.is_in)
    print("Got data")
    train_embedding_matrix = get_em(args,train_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="train").cuda()
    val_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv, mode="val").cuda()
    all_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="all").cuda()
    unseen_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="unseen").cuda()
    if not args.joint:
        if rnn_model is not None:
            rnn_model = rnn_model.cpu()
        rnn_loader = None
        rnn_model = None
    print("got embeddings")
    
    model = ALE(train_embedding_matrix,img_embed_size = train_loader.dataset.get_image_embed_size(), dropout=args.dropout, batch_size = args.batch_size)
    model = model.cuda()
    if args.joint:
        checkpoint = load_checkpoint( osp.join("./models/", str(False)+"_"+args.att+"_"+args.cost+'_checkpoint_best.pth.tar') )
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()


    
    #model = nn.DataParallel(model).cuda()
    param_groups = model.parameters()
    
    if args.cost == "ALE":
        criterion = ale_loss
    elif args.cost == "CEL":
        print("Using cross-entrophy loss")
        criterion =torch.nn.CrossEntropyLoss().cuda()
    elif args.cost == "WARP":
        criterion = WARPLoss()
    else:
        assert False, "Unknown cost function"

    if args.joint:
        if args.rnn_cost == "MSE":
            rnn_criterion = torch.nn.MSELoss()
        elif args.rnn_cost == "COS":
            coss_loss =torch.nn.CosineEmbeddingLoss() #margin can be added
            rnn_criterion = lambda x,y:coss_loss(x,y,torch.ones((x.shape[0])))
        else:
            assert False, "Unknown rnn cost function"
    """
    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    """
    optimizer = torch.optim.Adam(param_groups, lr=args.lr,
                                weight_decay=args.wd,
                                amsgrad=True)
    if args.joint and args.att in ["rnn","lstm","gru","affine","fc","lstm2"]:
        rnn_optimizer = torch.optim.Adam(rnn_model.parameters() if args.att != "random" else [train_embedding_matrix], lr=args.joint_lr,
                                weight_decay=args.joint_lr,
                                amsgrad=True)
    def adjust_lr(epoch):
        if epoch!= 0 and  epoch in [args.epochs//3,2*args.epochs//3]:

            for g in optimizer.param_groups:
                g['lr'] *= 0.1
                print('=====> adjust lr to {}'.format(g['lr']))
            if args.joint and args.att in ["rnn","lstm","gru","affine","fc","lstm2"]:
                for g in rnn_optimizer.param_groups:
                    g['lr'] *= 0.1
                    print('=====> adjust lr to {}'.format(g['lr']))


    best_val_pc = -1
    bar = Bar('Training', max=len(train_loader))
    for epoch in range(0, args.epochs):
        adjust_lr(epoch)
        model.set_embedding(train_embedding_matrix)
        model.train()
        perclass_accuracies = torch.zeros((train_embedding_matrix.shape[0])).cuda()
        if rnn_model is not None:
            rnn_model.train()
        loss = AverageMeter()
        acc1 = AverageMeter()
        acc5 = AverageMeter()
        

        for i,d in enumerate(train_loader):
            img_embeds, metas = d
            img_embeds = img_embeds.cuda()
            optimizer.zero_grad()
            if args.joint and args.att in ["rnn","lstm","gru","affine","fc","lstm2","fcb"]:
                rnn_optimizer.zero_grad()
            comps = model(img_embeds)
            classes = metas["class"].cuda()
            loss_value = criterion(comps, classes)
            loss_value.backward()
            loss.update(loss_value.item(), img_embeds.size(0))
            optimizer.step()

            if args.joint and args.att in ["rnn","lstm","gru","affine","fc","lstm2","random","fcb"]:
                rnn_optimizer.step()
                train_embedding_matrix = get_em(args,train_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="train").cuda()
                model.set_embedding(train_embedding_matrix)

            acc1_train = top1_acc(classes,comps)
            acc5_train = top5_acc(classes,comps)
            top1_acc_perclass(classes,comps, perclass_accuracies)
            acc1.update(acc1_train,img_embeds.size(0))
            acc5.update(acc5_train,img_embeds.size(0))
            # plot progress
            bar.suffix = 'Epoch: [{}][{}/{}]\t {}\t Loss  {:.6f}\t Acc1 {:.3f}\t Acc5 {:.3f}\t'.format(epoch, (i + 1), len(train_loader), args.att,
                    loss.avg, acc1.avg, acc5.avg )
            bar.next()
        bar.finish()
        if args.joint and args.att in ["rnn","lstm","gru","affine","fc","lstm2"] and args.att!="random" :
            train_rnn_tick(rnn_loader,args,model,rnn_optimizer,rnn_criterion,epoch)
            rnn_model.eval()
            val_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv, mode="val").cuda()
            all_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="all").cuda()
            unseen_embedding_matrix = get_em(args,val_loader,rnn_loader = rnn_loader, model=rnn_model,fv=fv,mode="unseen").cuda()
        accpc = calc_perclass(perclass_accuracies,train_loader.dataset.train_sample_per_class,"training")
        print("Train Accpc: %f" % accpc)
        
            
        
        acc1_val,acc5_val, accpc_val, loss_val = test(val_loader,args,em=val_embedding_matrix, model=model, criterion=criterion)
        if vis is not None:
            zsl_acc, zsl_acc_seen, zsl_acc_unseen = evaluate(args,eval_func,args.dset, all_embedding_matrix, unseen_embedding_matrix, model=model)
            zsl_harmonic = 2*( zsl_acc_seen * zsl_acc_unseen ) / ( zsl_acc_seen + zsl_acc_unseen )
            print("Harmonic: %.6f" % zsl_harmonic)
            print("------")      
            ##### PLOTS
            #Loss
            draw_vis(vis=vis,title="Loss",name="train",epoch=epoch,value=loss.avg,legend=['train','val'])
            draw_vis(vis=vis,title="Loss",name="val",epoch=epoch,value=loss_val,legend=['train','val'])
            #acc1
            draw_vis(vis=vis,title="Acc1",name="train",epoch=epoch,value=acc1.avg,legend=['train','val'])
            draw_vis(vis=vis,title="Acc1",name="val",epoch=epoch,value=acc1_val,legend=['train','val'])
            #acc5
            draw_vis(vis=vis,title="Acc5",name="train",epoch=epoch,value=acc5.avg,legend=['train','val'])
            draw_vis(vis=vis,title="Acc5",name="val",epoch=epoch,value=acc5_val,legend=['train','val'])
            #accperclass
            draw_vis(vis=vis,title="Accpc",name="train",epoch=epoch,value=accpc,legend=['train','val'])
            draw_vis(vis=vis,title="Accpc",name="val",epoch=epoch,value=accpc_val,legend=['train','val'])
            #testing
            draw_vis(vis=vis,title="Testing",name="zsl_acc",epoch=epoch,value=zsl_acc,legend=['zsl_acc','seen','unseen','harmonic'])
            draw_vis(vis=vis,title="Testing",name="seen",epoch=epoch,value=zsl_acc_seen,legend=['zsl_acc','seen','unseen','harmonic'])
            draw_vis(vis=vis,title="Testing",name="unseen",epoch=epoch,value=zsl_acc_unseen,legend=['zsl_acc','seen','unseen','harmonic'])
            draw_vis(vis=vis,title="Testing",name="harmonic",epoch=epoch,value=zsl_harmonic,legend=['zsl_acc','seen','unseen','harmonic'])
            ##########
        key =  str(args.joint)+"_"+args.att+"_"+args.cost
        with open(key+"pclog.txt","w") as filem:
                    filem.write(str(perclass_accs_global))
        if  vis is None and accpc_val > best_val_pc + best_val_pc/100 :
            #zsl_acc, zsl_acc_seen, zsl_acc_unseen = evaluate(args,eval_func,args.dset, all_embedding_matrix, unseen_embedding_matrix, model=model)
            #zsl_harmonic = 2*( zsl_acc_seen * zsl_acc_unseen ) / ( zsl_acc_seen + zsl_acc_unseen )
            #print("Harmonic: %.6f" % zsl_harmonic)
            with open("ale_results.pc","rb") as filem:
                ale_results = pc.load(filem)
            key =  str(args.joint)+"_"+args.att+"_"+args.cost
            
            if key not in ale_results or ale_results[key]["best_valpc"] + ale_results[key]["best_valpc"]/100 < accpc_val:
                best_val_pc = accpc_val
                _, _, accpc_seen, _ = test(seen_loader,args,em=val_embedding_matrix, model=model, criterion=criterion,strm="Seen")
                _, _, accpc_unseen, _ = test(unseen_loader,args,em=val_embedding_matrix, model=model, criterion=criterion, strm="Unseen")
                test_harmonic = 2*( accpc_seen * accpc_unseen ) / ( accpc_seen + accpc_unseen )
                print("Test Harmonic: %.6f" % test_harmonic)
                print("------")
                best_harmonic = test_harmonic
                best_seen = accpc_seen
                best_unseen = accpc_unseen
                best_epoch = epoch
                print("FOUND NEW BEST: ",key)
                with open("what_changed.txt","w") as filem:
                    filem.write("Found new best: %s\n" % key)
                with open(key+"pclog.txt","w") as filem:
                    filem.write(str(perclass_accs_global))
                ale_results[key] = {}
                ale_results[key]["args"] = str(args)
                ale_results[key]["best_valpc"] = accpc_val
                ale_results[key]["best_epoch"] = best_epoch
                ale_results[key]["best_harmonic"] = best_harmonic
                ale_results[key]["best_seen"] = best_seen
                ale_results[key]["best_unseen"] = best_unseen
                with open("ale_results.pc","wb") as filem:
                    pc.dump(ale_results,filem)
                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                }, False, fpath=osp.join("./models/", key+'_checkpoint_best.pth.tar'))
                with open("./models/"+key+'_checkpoint_best.txt',"w") as filem:
                    filem.write(str(args))
        print("------")

    if vis is None:
        return best_val_pc, best_harmonic, best_seen, best_unseen, best_epoch


def log_results(log_string,args, is_rnn = False):
    if is_rnn:
        with open("rnn_logs.txt","a") as filem:
            filem.write(log_string+"\n")
            filem.write("Args: %s\n" % str(args))
            filem.write("\n------\n")
    else:
        with open("logs.txt","a") as filem:
            filem.write(log_string+"\n")
            filem.write("Args: %s\n" % str(args))
            filem.write("\n------\n")

def randomize_params(args):
    args.lr = random.choice([1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4])
    args.rnn_lr = random.choice([1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4])
    args.joint_lr = random.choice([1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4])
    args.wd = random.choice([5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5])
    args.rnn_wd = random.choice([5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5])
    args.joint_wd = random.choice([5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5])
    args.batch_size = random.choice([64,256,1024]) 
    args.rnn_batch_size = random.choice([64,256,1024]) 
    args.cost = random.choice(["ALE","CEL"])
    args.rnn_cost = random.choice(["MSE","COS"])
    args.rnn_count = random.choice([10000,50000,100000])
    args.rnn_count_word = random.choice([args.rnn_count*3//16, args.rnn_count*3//8,args.rnn_count*3//4])
    args.curiculum = random.choice( ["mixed"]) #["mixed","curriculum"])
    args.kmeansk = random.choice([1,2,3,4])
    #args.joint = random.choice([True,False])
    #args.e2e = random.choice([True,False])


def run_experiment(args,vis):
    
    randomize_params(args)
    if args.rnn_only:
        _,_,loss,epoch = train_rnn(args,vis)
        log_string = "%s, loss: %f, epoch:%d" % (args.att,loss,epoch)
        print(log_string)
        log_results(log_string,args,True)
    else:
        pc,h,s,u,epoch = main(args,vis)
        if args.att == "fisher":
            log_string = "%s(%d) without joint: %f, %f, %f, %f at epoch %d" % (args.att,args.kmeansk,pc,s,u,h,epoch)
        else:
            log_string = "%s (best rnn) without joint: %f, %f, %f, %f at epoch %d" % (args.att,pc,s,u,h,epoch)
        print(log_string)
        log_results(log_string,args)

def run_experiments(args,vis):
    possible =  ["rnn","gru","lstm","lstm2","affine","fc","fcb" ] #fisher, avg, gru
    if args.rnn_only:
        possible = ["rnn","gru","lstm","lstm2","affine","fc", "fcb"]
        #possible = ["fcb"]
    if args.joint:
        possible = ["rnn","gru","lstm","lstm2","affine","fc", "fcb" ] #fisher

    random.shuffle(possible)
    while True :
        for att_type in possible: 
            args.att = att_type
            run_experiment(args,vis)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ZSL")

    # dat
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--rnn-batch-size', type=int, default=64)


    # model
    parser.add_argument('--dset', type=str, default="SUN")

    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--rnn-dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--rnn-lr', type=float, default=0.01)
    parser.add_argument('--joint-lr', type=float, default=0.01)

    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--rnn-wd', type=float, default=1e-4)
    parser.add_argument('--joint-wd', type=float, default=1e-4)

    # training configs
    parser.add_argument('--resume', type=str, default=None, metavar='PATH')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--kmeansk', type=int, default=3)
    parser.add_argument('--rnn-epochs', type=int, default=30)
    parser.add_argument('--rnn-count', type=int, default=50)
    parser.add_argument('--rnn-count-word', type=int, default=50)

    parser.add_argument('--att', type=str, metavar='PATH', default='avg') #label,rnn, lstm, gru,avg,fisher,hocanın formülleri
    parser.add_argument('--mode',type=str, metavar='PATH',default="all")
    
    parser.add_argument('--cost', type=str, metavar='PATH', default='ALE')
    parser.add_argument('--rnn-cost', type=str, metavar='PATH', default='COS')
    parser.add_argument('--gpu', type=str, metavar='PATH', default='1')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--crazy', action='store_true')
    parser.add_argument('--rnn-only', action='store_true')
    parser.add_argument('--curriculum', type=str, metavar='PATH', default='mixed')

    parser.add_argument('--draw-best', action='store_true')
    try:
        args = parser.parse_args()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if args.crazy:
            run_experiments(args,None)
        elif args.rnn_only:
            _,_,loss,epoch = train_rnn(args,None)
            
            log_string = "%s, loss: %f, epoch:%d" % (args.att,loss,epoch)
            print(log_string)
            log_results(log_string,args,True)
        elif args.draw_best:
            with open("ale_results.pc","rb") as filem:
                ale_results = pc.load(filem)
            for k in ale_results:
                argss = eval(ale_results[k]["args"])
                key = str(argss.joint)+"-"+argss.att+"-"+argss.cost
                vis = visdom.Visdom(env=key)
                vis.check_connection()
                argss.curriculum = args.curriculum
                main(argss,vis)
            with open("rnn_results.pc","rb") as filem:
                rnn_results = pc.load(filem)
            for k in rnn_results:
                if k == "foo":
                    continue
                argss = rnn_results[k]["args"]
                key = argss.att+"-"+argss.rnn_cost
                vis = visdom.Visdom(env=key)
                vis.check_connection()
                train_rnn(argss,vis)

        else:
            key = str(args.joint)+"-"+args.att+"-"+args.cost
            #assert False, "Take care"
            vis = visdom.Visdom(env=key+"-deneme")
            vis.check_connection()
            pc,h,s,u,epoch = main(args,vis)
            if args.att == "fisher":
                log_string = "%s(%d) without joint: %f, %f, %f, %f at epoch %d" % (args.att,args.kmeansk,pc,s,u,h,epoch)
            else:
                log_string = "%s without joint: %f, %f, %f, %f at epoch %d" % (args.att,pc,s,u,h,epoch)
            print(log_string)
            log_results(log_string,args,False)
            
    except KeyboardInterrupt:
        print("Saving and Exiting...")
        exit()

