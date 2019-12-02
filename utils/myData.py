from __future__ import print_function, absolute_import
import json
import torch.utils.data as data
import os.path as osp
from scipy.io import loadmat
import torch
import pickle as pc
import numpy as np
import random


class myDataSet(data.Dataset):
    def __init__(self, db="AWA2", is_train = "train", split = 1, a = None, b = None, is_in = None):
        self.is_train = is_train  # training set or test set
        self.split = split
        ps_path = "/slow_data/denizulug/GBU/xlsa17/data"
        #embeddings = set(embeddings)
        data_path = osp.join(ps_path,db.upper())
        a= loadmat(osp.join(data_path,"att_splits.mat")) if a is None else a
        b=loadmat(osp.join(data_path,"res101.mat")) if b is None else b
        self.a = a
        self.b = b
        self.only_existing = False #hand set
        with open(osp.join(data_path,"allclasses.txt"),"r") as filem:
            lines = filem.readlines()
            lines = [ x.rstrip().replace("+","-").replace("_","-") for x in lines]
            self.class_list = lines
            self.org_class_list = lines
            if self.only_existing:
                self.class_list = [ x for x in self.class_list if is_in(x) ]

        with open(osp.join(data_path,"trainclasses"+str(split)+".txt"),"r") as filem:
            lines = filem.readlines()
            lines = [ x.rstrip().replace("+","-").replace("_","-") for x in lines]
            self.train_class_list = lines
            if self.only_existing:
                self.train_class_list = [ x for x in self.train_class_list if is_in(x) ]
        with open(osp.join(data_path,"valclasses"+str(split)+".txt"),"r") as filem:
            lines = filem.readlines()
            lines = [ x.rstrip().replace("+","-").replace("_","-") for x in lines]
            self.val_class_list = lines
            if self.only_existing:
                self.val_class_list = [ x for x in self.val_class_list if is_in(x) ]
            #self.trainval_class_list = list(set(sorted(self.train_class_list + self.val_class_list)))
            self.trainval_class_list = self.val_class_list
        with open(osp.join(data_path,"testclasses.txt"),"r") as filem:
            lines = filem.readlines()
            lines = [ x.rstrip().replace("+","-").replace("_","-") for x in lines]
            self.unseen_class_list = sorted(lines)
            if self.only_existing:
                self.unseen_class_list = [ x for x in self.unseen_class_list if is_in(x) ]
        self.trainval_class_list = self.class_list
        print("loaded classes - dataset")
        #with open("scaler.pc","rb") as filem:
        #    scaler = pc.load(filem)
        #    self.scaler = scaler["scaler"]
        #    self.max = scaler["max"]
        self.feature_list = b["features"].T.astype(float)
        #self.norm = True
        #if self.norm:
        #    old = self.feature_list
        #    self.feature_list = self.scaler.transform(self.feature_list, copy=True)
        #    self.feature_list = self.feature_list.astype(float)
        #    self.feature_list /= self.max 

        self.feature_list = torch.from_numpy( self.feature_list ).float()
        #the actul string classname
        self.label_list = [ self.org_class_list[x[0]-1] for x in b["labels"]]
        #self.label_list = [ x for x in self.org_label_list if x in self.class_list ]
        #indexed class index
        self.image_paths = [ x[0][0].replace("/BS/xian/work/data/","/slow_data/denizulug/").replace("//","/") for x in b["image_files"]]
        self.class_att_table = torch.from_numpy(a["att"].T).float()

        self.train, self.valid  = [], []
        #self.train = [ x[0]-1 for x in a["train_loc"] ] if self.is_train else []
        #self.valid = [ x[0]-1 for x in a["val_loc"] ] if not self.is_train else []
        s_train_class_list = set(self.train_class_list)
        s_val_class_list = set(self.val_class_list)
        self.train = [] if not self.is_train == "train" else  [ x[0]-1 for x in a["trainval_loc"] if self.label_list[x[0]-1] in s_train_class_list ] 
        self.valid = [] if not self.is_train == "val" else [ x[0]-1 for x in a["trainval_loc"] if self.label_list[x[0]-1] in s_val_class_list ] 
        self.seen = [] if not self.is_train == "seen" else [ x[0]-1 for x in a["test_seen_loc"] ] 
        self.unseen = [] if not self.is_train == "unseen" else [ x[0]-1 for x in a["test_unseen_loc"] ] 
        print("loaded indexes - dataset")
        ###over fitting
        """
        self.train = [ x[0]-1 for x in a["trainval_loc"] if self.label_list[x[0]-1] in self.train_class_list ]
        self.train = self.train[ len(self.train)//5:]
        self.valid = self.train[ :len(self.train)//5]
        self.val_class_list = self.train_class_list
        """

        ###
        self._calc_sample_per_class()
    
    

    def _calc_sample_per_class(self):
        ##train
        if self.is_train == "train":
            retval = torch.zeros(len(self.train_class_list)).int()
            for i in self.train:
                retval[self.train_class_list.index(self.label_list[i])] += 1
            self.train_sample_per_class = (self.train_class_list, retval)
            #print(retval.shape)
            #print(retval[retval!=0].shape)

        ##val
        if self.is_train=="val":
            retval = torch.zeros(len(self.trainval_class_list)).int()
            for i in self.valid:
                retval[self.trainval_class_list.index(self.label_list[i])] += 1
            #print(retval.shape)
            #print(retval[retval!=0].shape)
            self.valid_sample_per_class = (self.trainval_class_list, retval)

        ## seen
        elif self.is_train=="seen":
            retval = torch.zeros(len(self.trainval_class_list)).int()
            for i in self.seen:
                retval[self.trainval_class_list.index(self.label_list[i])] += 1
            #print(retval.shape)
            #print(retval[retval!=0].shape)
            self.valid_sample_per_class = (self.trainval_class_list, retval)

        ##unseen
        elif self.is_train=="unseen":
            retval = torch.zeros(len(self.trainval_class_list)).int()
            for i in self.unseen:
                retval[self.trainval_class_list.index(self.label_list[i])] += 1
            #print(retval.shape)
            #print(retval[retval!=0].shape)
            self.valid_sample_per_class = (self.trainval_class_list, retval)
    
    def get_embedding_matrix(self):
        if self.is_train == "train":
            retval = torch.zeros((len(self.train_class_list),self.get_word_embed_size())).float()
            for i,em in enumerate(self.class_list):
                if em in self.train_class_list:
                    retval[self.train_class_list.index(em)] = self.class_att_table[i]
        else:
            retval = torch.zeros(( len(self.trainval_class_list) , self.get_word_embed_size() )).float()
            for i,cl in enumerate(self.class_list):
                if cl in self.trainval_class_list:
                    retval[self.trainval_class_list.index(cl)] = self.class_att_table[i]
            
        return retval
    
    def get_embedding_matrix_all(self):
        return self.class_att_table
        
    def get_image_embed_size(self):
        return self.feature_list.shape[1]
    def get_word_embed_size(self):
        return self.class_att_table.shape[1]
    def get_labels(self,mode):
        if mode=="all":
            return self.class_list
        elif mode=="train":
            return self.train_class_list
        elif mode=="val":
            return self.trainval_class_list
        elif mode=="unseen":
            return self.unseen_class_list
        else:
            assert False,"Unrecognized mode"

    def __getitem__(self, index):
        orgIndex = index
        if self.is_train == "train":
            index = self.train[index]
        elif self.is_train == "val":
            index = self.valid[index]
        elif self.is_train == "seen":
            index = self.seen[index]
        elif self.is_train == "unseen":
            index = self.unseen[index]
         
        img_embedding = self.feature_list[index]
        
        img_path = self.image_paths[index]
       
        
        # Save meta data
        meta = {'index': orgIndex, 'path' : img_path, "label" : self.label_list[index], "class" : self.train_class_list.index(self.label_list[index]) if self.is_train == "train" 
               else self.trainval_class_list.index(self.label_list[index]), "is_idiom" : 1 if len(self.label_list[index].split("-"))>1 else 0  }

        return img_embedding, meta

    def __len__(self):
        if self.is_train == "train":
            return len(self.train)
        if self.is_train == "val":
            return len(self.valid)
        if self.is_train == "seen":
            return len(self.seen)
        if self.is_train == "unseen":
            return len(self.unseen)
        
        
        
        
        
"""
antelope
bat
beaver
blue+whale
bobcat
buffalo
chihuahua
chimpanzee
collie
cow
dalmatian
deer
dolphin
elephant
fox
german+shepherd
giant+panda
giraffe
gorilla
grizzly+bear
hamster
hippopotamus
horse
humpback+whale
killer+whale
leopard
lion
mole
moose
mouse
otter
ox
persian+cat
pig
polar+bear
rabbit
raccoon
rat
rhinoceros
seal
sheep
siamese+cat
skunk
spider+monkey
squirrel
tiger
walrus
weasel
wolf
zebra

"""
