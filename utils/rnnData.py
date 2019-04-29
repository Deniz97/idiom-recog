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


class rnnData(data.Dataset):
    def __init__(self,idioms=None, keys=None, words=None):
        self.idioms = idioms
        self.keys = keys
        self.words = words
        self.max_len = 9
        if idioms is None:
            with open("/slow_data/denizulug/crawl_idioms.pc","rb") as filem:
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

        print("Done rnn dataset init")
    
    def get_embed_size(self):
        return len(self.idioms[self.keys[0]])

    def __getitem__(self, index):
        
        key = self.keys[index]
        ttt = (key=="" or key==" ")

        target = torch.tensor(self.idioms[key]).float()
        words = key.split("-")

        length = torch.tensor(len(words)).int()
        inp_seq = torch.zeros((self.max_len,self.embed_size))
        for i,w in enumerate(words):
            inp_seq[i] = torch.tensor(self.words[w])


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