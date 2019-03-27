import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class ALE(nn.Module):
    def __init__(self, embedding_matrix, img_embed_size=100,word_embed_size=None,gpu=False,dropout=0):
        super(ALE, self).__init__()
        self.gpu = gpu
        self.embedding_matrix = embedding_matrix
        
        self.img_embed_size = img_embed_size
        if word_embed_size == None:
            self.word_embed_size = embedding_matrix.shape[1]
        else:
            self.word_embed_size = word_embed_size
        
        self.num_class = embedding_matrix.shape[0]
        self.softm = nn.Softmax(dim=1)
        self.dropout = None
        self.bilin = nn.Bilinear(img_embed_size,word_embed_size,1,bias=False)
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
                    
    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        batch_size = x.shape[0]
        retval = torch.zeros(batch_size,self.num_class)
        
        for i in range(self.num_class):
            retval[:,i] = self.bilin(x , self.embedding_matrix[i].expand(batch_size,self.word_embed_size)
                            .contiguous() ).squeeze()

        #retval = self.softm(retval)
        return retval
    """
    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        batch_size = x.shape[0]
        x_expand = torch.zeros(batch_size,self.num_class,self.img_embed_size)
        if self.gpu:
            x_expand = x_expand.cuda()
        for i in range(self.num_class):
            x_expand[:,i,:] = x
        #x_expand.contiguous()
        retval = self.bilin(x_expand, self.embedding_matrix
                            .expand(batch_size,self.num_class,self.word_embed_size)
                            .contiguous() )
        retval = retval.squeeze()
        retval = self.softm(retval)
        return retval
    """
    
    
    