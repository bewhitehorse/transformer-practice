import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math,copy,time
from torch.autograd import Variable
import matplotlib.pyplot as plt


# # Model Architecture

# ## Embedding

class Embedding(nn.Module):
    def __init__(self, voc_size, d_model):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(voc_size, d_model)
        self.d_model = d_model #8
    def forward(self, x):
        # print('src_after_embedding:{}'.format(self.embedding(x).shape))
        return self.embedding(x)*math.sqrt(self.d_model)


# ## Positional Encoding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model) #[max_len, d_model]
        pos = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float)*(-torch.log(torch.tensor(10000.0)))/d_model)
        pe[:,0::2] = torch.sin(pos*div_term)
        pe[:,1::2] = torch.cos(pos*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self,x):
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
        


# ## Multihead Attention

# ### Attention

def attention(query, key, value, mask = None, dropout = None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    scores = torch.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)

    attn = torch.matmul(scores, value)
    return attn


# ### Multihead Attention


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0 #8%2
        self.depth = d_model // num_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.depth).transpose(1,2)
        
    def forward(self, query, key, value, mask = None):
        # print('multiheadattention1:{}'.format(mask.shape))
        if mask is not None: 
          mask = mask.unsqueeze(1) # 没懂
        # print('multiheadattention2:{}'.format(mask.shape))
        # print('query_before_split:{}'.format(query.shape))

        # print('query_after_split:{}'.format(query.shape)) #4,2,10,4
        # print('batch_size.{}'.format(self.batch_size))
        # print('d_model:{}'.format(self.d_model))
        # print('num_heads:{}'.format(self.num_heads))

        query, key, value = [l(x) for l, x in zip(self.linears, (query, key, value))]
        # print('query2:{}'.format(query.shape)) 
        
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        # print('query_after_split:{}'.format(query.shape)) #4,2,10,4     
        attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        batch_size, _, seq_length, _ = attn.size()
        attn_output = attn.transpose(1,2).contiguous().view(batch_size, seq_length, self.d_model)

        return self.linears[-1](attn_output)
            
        


# ## PositionwiseFeedForward


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)        
        
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ## SubLayerConnection

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # print(sublayer(self.layernorm(x).shape))
        return x + self.dropout(sublayer(self.layernorm(x)))


# ## Encoder


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ### EncoderLayer

class EncoderLayer(nn.Module):
    def __init__(self,self_attn, feed_forward, d_model, dropout):

        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout),2)

        self.size = d_model #为什么要有？
    
    def forward(self,x, mask):
        # print('encoderlayer1:{}'.format(x.shape))
        x = self.sublayer[0](x, lambda x: self.self_attn(x,x,x,mask))
        # print('encoderlayer2:{}'.format(x.shape))
        return self.sublayer[1](x, self.feed_forward)


# ### Encoder

class Encoder(nn.Module):
    def __init__(self,layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.layernorm = nn.LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)


# ## Decoder

# ### DecoderLayer


class DecoderLayer(nn.Module):
    def __init__(self,self_attn, cross_attn, feed_forward, d_model, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout),3)

        self.size = d_model #为什么要有？
        
    def forward(self, x, memory, src_mask, tgt_mask):
        self_attn_x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        cross_attn_x = self.sublayer[1](self_attn_x, lambda self_attn_x: self.cross_attn(self_attn_x, memory, memory, src_mask))
        # print(cross_attn_x.shape)
        return self.sublayer[2](cross_attn_x, self.feed_forward)
        


# ### Decoder

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.layernorm = nn.LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.layernorm(x)


# ## Generator

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


# ## EncoderDecoder

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        
    def encode(self, src, src_mask):
        # print('encode_src:{}'.format(src.shape))
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, tgt, src, src_mask, tgt_mask):
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)
        


# ## Full Model

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(c(attn), c(ff), d_model, dropout), N),
        Decoder(DecoderLayer(c(attn), c(attn), c(ff), d_model, dropout), N),
        nn.Sequential(Embedding(src_vocab, d_model), c(position)),
        nn.Sequential(Embedding(tgt_vocab, d_model), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

