import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class ALE(nn.Module):
    def __init__(self, embedding_matrix, img_embed_size=100, gpu=False,dropout=0):
        super(ALE, self).__init__()
        self.gpu = gpu
        self.embedding_matrix = embedding_matrix
        
        self.img_embed_size = img_embed_size
        self.word_embed_size = embedding_matrix.shape[1]
        
        self.num_class = embedding_matrix.shape[0]
        self.softm = nn.Softmax(dim=1)
        self.dropout = None
        self.bilin = nn.Bilinear(img_embed_size,self.word_embed_size,1,bias=False)
        if dropout>0:
            self.dropout = nn.Dropout(p=dropout)
        self.w = torch.randn(img_embed_size,self.word_embed_size)
    def set_embedding(self,em):
        self.embedding_matrix = em.float().cuda()
        self.word_embed_size = em.shape[1]
        self.num_class = em.shape[0]
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


    
    
    