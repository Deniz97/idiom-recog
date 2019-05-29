import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class FC(nn.Module):
    def __init__(self, word_embed_size=300,dropout=0, linearity=None, bias=False):
        super(FC, self).__init__()
  
        self.word_embed_size = word_embed_size

        self.dropout = None
        self.linearity = linearity
        
        self.W = nn.Linear(word_embed_size,word_embed_size,bias=bias)

        if linearity is not None:
            if linearity == "tanh":
                pass
            elif linearity == "relu":
                pass
            else:
                assert False, "Not yet implemented"

        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)

    
    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)

        data = x.data
        data = self.W(data)
        if self.linearity:
            data = self.linearity(data)

        batch_size = x.batch_sizes[0]
        retval = torch.zeros((batch_size,self.word_embed_size)).cuda()

        total = 0
        lengths = torch.zeros((x.batch_sizes[0])).cuda()
        for i in x.batch_sizes:
            retval[:i,...] += data[total:total+i,...]
            total = total+i
            #
            lengths[:i] += 1

        for i,l in enumerate(lengths):
            retval[i] /= l

        return retval, None


    