from __future__ import print_function, absolute_import
import json
import torch.utils.data as data
import os.path as osp
from scipy.io import loadmat
import random
import torch
import pickle as pc


class myDataSet(data.Dataset):
    def __init__(self, dset_name="awa2", is_train = True, split = 1 , features=None):
        self.is_train = is_train  # training set or test set
        self.split = split
        self.features = features
        ps_path = "/slow_data/denizulug/GBU/xlsa17/data"
        if dset_name=="awa2":
            data_path = osp.join(ps_path,"AWA2")
            
        a=loadmat(osp.join(data_path,"att_splits.mat"))
        b=loadmat(osp.join(data_path,"res101.mat"))
        with open(osp.join(data_path,"allclasses.txt"),"r") as filem:
                  lines = filem.readlines()
                  lines = [ x.rstrip() for x in lines]
                  self.class_list = lines
        self.feature_list = b["features"].T
        #the actul string classname
        self.label_list = [ self.class_list[x[0]-1] for x in b["labels"]]
        #0indexed class index
        self.class_indices = [ x[0]-1 for x in b["labels"] ]
        self.image_paths = [ x[0][0].replace("/BS/xian/work/data/","/slow_data/denizulug/").replace("//","/") for x in b["image_files"]]
        if self.features == None:
            self.class_att_table = a["att"].T
        else:
            with open(self.features,"rb") as filem:
                self.class_att_table = pc.load(filem)

        self.train, self.valid, self.test_seen, self.test_unseen = [], [], [], []
        self.train = [ x[0]-1 for x in a["train_loc"] ]
        self.valid = [ x[0]-1 for x in a["val_loc"] ]
        print("Done dataset init")
    
    def get_class_table(self,data_path):
        osp.join(data_path,"allclasses.txt")
        pass
    
    def _calc_mean(self):
        pass
    
    def get_embedding_matrix(self):
        return torch.from_numpy(self.class_att_table)
    def get_image_embed_size(self):
        return self.feature_list.shape[1]
    def get_word_embed_size(self):
        return self.class_att_table.shape[1]

    def __getitem__(self, index):
        orgIndex = index
        if self.is_train:
            index = self.train[index]
        else:
            index = self.valid[index]
         
        img_embedding = torch.from_numpy(self.feature_list[index]).float()
        try:
            class_embedding = torch.from_numpy( self.class_att_table[ self.class_indices[index]] ).float()
        except:
            print(index)
            print(self.class_indices[index])
            print(self.class_att_table[ self.class_indices[index]])
            print("--")
            exit()
        
        img_path = self.image_paths[index]
       
        
        # Save meta data
        meta = {'index': orgIndex, 'path' : img_path, "label" : self.label_list[index], "class" : self.class_indices[index]}

        
        return img_embedding, class_embedding, meta

    def __len__(self):
        if self.is_train:
            return len(self.train)
        else:
            return len(self.valid)
        
        
        
        
        
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