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

import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
    return data

class rnnData(data.Dataset):
    def __init__(self, keys=None, embeddings=None, args = None, is_train = False):
        self.is_train = is_train
        self.max_len = 7
        self.curriculum = args.curriculum
        self.epoch = 0
        if args.curriculum == "curriculum":
            assert False,"Curriculum learning not yet supported"
        
        if keys is None:
            with open("/slow_data/denizulug/wiki_idiom_keys_sorted.pc","rb") as filem:
                idioms = pc.load(filem)
                idiom_keys = idioms[:int(args.rnn_count)]
            with open("/slow_data/denizulug/wiki_word_keys_sorted.pc","rb") as filem:
                words = pc.load(filem)
                word_keys = words[:int(args.rnn_count_word)]
            
            self.valid_keys = idiom_keys[:len(idiom_keys)*5//20] 
            idiom_keys = idiom_keys[len(idiom_keys)*5//20:]
            self.keys = word_keys + idiom_keys
            shuffle(self.keys)
        else:
            self.keys = keys

            

            """
            with open("/slow_data/denizulug/crawl_idioms"+str(counts)+"k","rb") as filem:
                idioms = pc.load(filem)
                self.idioms = idioms
                keys = list(idioms.keys())
                shuffle(keys)
                self.valid_keys = keys[:len(keys)*3//10]
                self.keys = keys[len(keys)*3//10:]
            """
        if embeddings is None:
            with open("/slow_data/denizulug/wiki_embeddings.pc","rb") as filem:
                self.embeddings = pc.load(filem)
        else:
            self.embeddings = embeddings

        self.embed_size = self.get_embed_size()
    
    def get_embed_size(self):
        return len(self.embeddings[self.keys[0]])

    def get_all_word_embeddings(self,count=None):
        assert False,"Not yet implemented - get_word_embeddings"
        """
        if count is None:
            count = self.counts
        retval = torch.zeros((count))
        for i,k in enumerate(self.words):
            if not "-" in k:
                retval[i] = self.words[k]

        retval = list(self.words.values())
        if count is None:
            count = self.counts*1000
        shuffle(retval)
        retval = retval[:count]
        for i,r in enumerate(retval):
            retval[i] = r / torch.norm(r,p=2)

        return retval
        """

    def get_embedding(self,word):
        retval = torch.tensor(self.embeddings[word])
        if retval.shape[0] == 0:
            print("WW4W: ",word)
        return retval/torch.norm( retval,p=2)
        #return torch.tensor(self.words[word])
    
    def is_in(self, token):
        return ( token in self.keys)
        

    def __getitem__(self, index):
        key = self.keys[index]

        target = self.get_embedding(key).float()
        words = key.split("-")

        length = torch.tensor(len(words)).int()
        inp_seq = torch.zeros((self.max_len,self.embed_size))
        for i,w in enumerate(words):
            inp_seq[i] = self.get_embedding(w)
            if inp_seq[i].shape[0] == 0:
                print("WWW ",words)
        if target.shape[0] == 0:
            print("2WWW ",words)

        if len(words)==0 :
            print("3WWWW: ",words)
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