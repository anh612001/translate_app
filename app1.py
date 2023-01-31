BATCH_SIZE = 20
EPOCHS = 100

D_MODEL = 512
HEADS = 8
N = 6


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pandas as pd
import spacy
import torchtext
from torchtext import data
from torchtext.data import Field, BucketIterator, TabularDataset
from sklearn.model_selection import train_test_split
import math
import copy
import os
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
# from janome.tokenizer import Tokenizer

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'

import numpy as np
import pandas as pd

#path = "JaVi.csv"
#df = pd.read_csv(path, error_bad_lines=False, usecols=[0,1], header=0)

JA = spacy.load('ja_core_news_sm')
VI = spacy.load('vi_core_news_lg')

def tokenize_ja(sentence):
    return [tok.text for tok in JA.tokenizer(sentence)]

def tokenize_vi(sentence):
    return [tok.text for tok in VI.tokenizer(sentence)]

JA_TEXT = Field(tokenize=tokenize_ja) 
VI_TEXT = Field(tokenize=tokenize_vi, init_token='<sos>', eos_token='<eos>')

path = os.path.dirname(__file__)
my_model= path+'/transformer1-model.pt'

data_fields = [('Japanese', JA_TEXT), ('Vietnamese', VI_TEXT)]

train, val, test = TabularDataset.splits(path=path +'/',
                        train='train.csv', 
                        validation='val.csv',
                        test = 'test.csv',
                        format='csv',
              
                        fields = data_fields )

JA_TEXT.build_vocab(train, val) 
VI_TEXT.build_vocab(train, val)

train_iter = BucketIterator(
    train,
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.Vietnamese),
    shuffle=True
)

batch = next(iter(train_iter)) 


global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.Japanese))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.Vietnamese) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)

class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

train_iter = MyIterator(
    train,
    batch_size=1300,
    device=0,
    repeat=False,
    sort_key=lambda x: (len(x.Japanese), len(x.Vietnamese)),
    batch_size_fn=batch_size_fn,
    train=True,
    shuffle=True
)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        # print("inside embedder init")
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        # print("inside embedder forward")
        return self.embed(x)


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        # print("inside PositionalEncoder init")
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # print("inside PositionalEncoder forward")
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], \
        requires_grad=False).to(device)
        # print(x)
        return x

def create_masks(input_seq, target_seq):
    # print("inside create_masks")
    input_pad = JA_TEXT.vocab.stoi['<pad>']
    # creates mask with 0s wherever there is padding in the input
    input_msk = (input_seq != input_pad).unsqueeze(1)
    
    target_pad = VI_TEXT.vocab.stoi['<pad>']
    target_msk = (target_seq != target_pad).unsqueeze(1)
    size = target_seq.size(1) # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    nopeak_mask = torch.autograd.Variable(torch.from_numpy(nopeak_mask) == 0).to(device)
    target_msk = target_msk & nopeak_mask
    
    # print(input_msk)
    # print(target_msk)
    return input_msk, target_msk


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        # print("inside MultiHeadAttention __init__")
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        # print("inside MultiHeadAttention forward")
        bs = q.size(0)
        
        # perform linear operation and split into h heads
        
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        
        output = self.out(concat)
    
        # print(output)
        return output

def attention(q, k, v, d_k, mask=None, dropout=None):
    # print("inside attention")
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    # print(scores)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    # print(output)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        # print("inside FeedForward init")
        super().__init__() 
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) 
    def forward(self, x):
        # print("inside FeedForward forward")
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        # print(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        # print("inside Norm init")
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        # print("inside Norm forward")
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        # print(norm)
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        # print("inside EncoderLayer init")
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # print("inside EncoderLayer forward")
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        # print("inside EncoderLayer forward : "+x)
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        # print("inside EncoderLayer __init__")
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model)
        self.attn_2 = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model).to(device)

    def forward(self, x, e_outputs, src_mask, trg_mask):
            # print("inside EncoderLayer forward")
            x2 = self.norm_1(x)
            x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
            x2 = self.norm_2(x)
            x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs,
            src_mask))
            x2 = self.norm_3(x)
            x = x + self.dropout_3(self.ff(x2))
            # print(x)
            return x

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        # print("inside Encoder __init__")
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        # print("inside Encoder forward")
        x = self.embed(src)
        x = self.pe(x)
        for i in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads):
        # print("inside Decoder __init__")
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        # print("inside Decoder forward")
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

"""The main Transformer model code"""

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads):
        # print("inside Transformer __init__")
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads)
        self.decoder = Decoder(trg_vocab, d_model, N, heads)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        # print("inside Transformer forward")
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


src_vocab = len(JA_TEXT.vocab)
trg_vocab = len(VI_TEXT.vocab)
#print(len(JA_TEXT.vocab))

model = Transformer(src_vocab, trg_vocab, D_MODEL, N, HEADS)

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


input_pad = JA_TEXT.vocab.stoi['<pad>']
target_pad = VI_TEXT.vocab.stoi['<pad>']

MultiHeadAttention(HEADS, D_MODEL)

def train_model(model, epochs, print_every=50):
    
    model.train()
    
    start = datetime.now()
    temp = start
    
    total_loss = 0
    
    for epoch in range(epochs):
       
        for i, batch in enumerate(train_iter):
            src = batch.Japanese.transpose(0, 1)
            trg = batch.Vietnamese.transpose(0, 1)
            # the French sentence we input has all words except
            # the last, as it is using each word to predict the next
            
            trg_input = trg[:, :-1]
            
            # the words we are trying to predict
            
            targets = trg[:, 1:].contiguous().view(-1)
            
            # create function to make masks using mask code above
            
            src_mask, trg_mask = create_masks(src, trg_input)
            
            preds = model(src, trg_input, src_mask, trg_mask)
            
            optim.zero_grad()
            
            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)),
                targets,
                ignore_index=target_pad
            )
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            Loss_hist = []
            Perplexity_hist = []
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                perplexity  = math.exp(loss_avg)
                print("time = {}, epoch {}, iter = {}, loss = {}, perplexity = {}, {} per {} iters".format(
                    (datetime.now() - start) // 60,
                    epoch + 1,
                    i + 1,
                    loss_avg,
                    perplexity,
                    datetime.now() - temp,
                    print_every
                ))
                if i+1 == 450:
                    Loss_hist.append(loss_avg)
                    Perplexity_hist.append(perplexity)
                total_loss = 0
                temp = datetime.now()
    
        print()

model.load_state_dict(torch.load(my_model))

def translate(model, src, max_len=80, custom_sentence=False):
    
    model.eval()
    
    if custom_sentence == True:
        src = tokenize_ja(src)
        src = torch.autograd.Variable(torch.LongTensor([[JA_TEXT.vocab.stoi[tok] for tok in src]])).to(device)
    
    src_mask = (src != input_pad).unsqueeze(-2)
    e_outputs = model.encoder(src, src_mask)

    outputs = torch.zeros(max_len).type_as(src.data)
    outputs[0] = torch.LongTensor([VI_TEXT.vocab.stoi['<sos>']])
    
    for i in range(1, max_len):    
        trg_mask = np.triu(np.ones((1, i, i)), k=1).astype('uint8')
        trg_mask = torch.autograd.Variable(torch.from_numpy(trg_mask) == 0).to(device)

        out = model.out(
            model.decoder(
                outputs[:i].unsqueeze(0),
                e_outputs,
                src_mask,
                trg_mask
            )
        )
        out = F.softmax(out, dim=-1)
        val, ix = out[:, -1].data.topk(1)

        outputs[i] = ix[0][0]
        if ix[0][0] == VI_TEXT.vocab.stoi['<eos>']:
            break

    return ' '.join([VI_TEXT.vocab.itos[ix] for ix in outputs[:i]])

import streamlit as st

st.title("Translate Japanese to Vietnamese")
input=st.text_input("Write")

result=st.button('Translate')
if result:
  prediction=translate(model, input, custom_sentence=True).replace('<sos>','')
  st.write(prediction)

#trigger = st.button('Translate', on_click=predict)

#print(translate(model, '私は眠らなければなりません', custom_sentence=True))


#print(translate(model, '明日帰ったら電話します。', custom_sentence=True))

