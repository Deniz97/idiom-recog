from __future__ import print_function, absolute_import
import json
import torch.utils.data as data
import os.path as osp
from scipy.io import loadmat
import random
import torch
import pickle as pc
import numpy as np
from random import shuffle

#NOTE
#FIX tvmonitor (in one of databases)
#FIX pottedplant
#FIX +s and _s to -s

class rnnData(data.Dataset):
    def __init__(self,idioms=None, keys=None, words=None, counts=50):
        self.idioms = idioms
        self.keys = keys
        self.words = words
        self.max_len = 9
        self.counts = counts
        if idioms is None:
            with open("/slow_data/denizulug/crawl_idioms"+str(counts)+"k","rb") as filem:
                idioms = pc.load(filem)
                self.idioms = idioms
                keys = list(idioms.keys())
                shuffle(keys)
                self.valid_keys = keys[:len(keys)*3//10]
                self.keys = keys[len(keys)*3//10:]

        if words is None:
            with open("/slow_data/denizulug/crawl_words.pc","rb") as filem:
                self.words = pc.load(filem)

        self.embed_size = self.get_embed_size()
    
    def get_embed_size(self):
        return len(self.idioms[self.keys[0]])

    def get_all_embeddings(self,count=None):
        retval = list(self.words.values())
        if count is None:
            count = self.counts*1000
        shuffle(retval)
        retval = retval[:count]
        for i,r in enumerate(retval):
            retval[i] = r / torch.norm(r,p=2)

        return retval

    def get_word_embedding(self,word):
        retval = torch.tensor(self.words[word])
        return retval/torch.norm( retval,p=2)
        #return torch.tensor(self.words[word])
    
    def get_idiom_embedding(self,word):
        retval = torch.tensor(self.idioms[word])
        retval= retval / torch.norm(retval,p=2)
        return retval
        #return torch.tensor(self.idioms[word])
    def is_in(self, token):
        return ( token in self.words) or (token in self.idioms)
        

    def __getitem__(self, index):
        
        key = self.keys[index]

        target = self.get_idiom_embedding(key).float()
        words = key.split("-")

        length = torch.tensor(len(words)).int()
        inp_seq = torch.zeros((self.max_len,self.embed_size))
        for i,w in enumerate(words):
            inp_seq[i] = self.get_word_embedding(w)


        return inp_seq, target, key, length

    def __len__(self):
        return len(self.keys)
        
        
        
        
        
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