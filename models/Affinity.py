import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.LSTM_EncDec import ConvDecoder,LinearDecoder


DEVICE = torch.device('cuda:0') 

# class Affplus(nn.Module):
#     def __init__(self, x_len):
#         super(Affplus, self).__init__()
#         self.softplus = F.softplus
#         self.alpha = torch.randn(x_len, requires_grad=True)
#         self.params = nn.Parameter(self.alpha)
    
#     def forward(self,x):
#         return self.alpha * self.softplus(x) + (1-self.alpha) * x

# class Affinitycell(nn.Module):
#     def __init__(self, x_len, h_len):
#         super(Affinitycell, self).__init__()
#         self.h_len = h_len
#         self.affplus = Affplus(self.h_len)
#         self.b = nn.Parameter(torch.Tensor(self.h_len))
#         self.W = nn.Parameter(torch.Tensor(self.h_len, x_len+h_len))
#         nn.init.kaiming_normal_(self.W)
        
    
    # def forward(self, x, h):
    #     # print(self.W @ torch.cat((h,x),dim=-1)+self.b)
    #     h_new = self.affplus(self.W @ torch.cat((h,x),dim=-1)+self.b)
    #     # print(self.W.shape)
    #     # print(torch.cat((h,x),dim=-1).shape)
    #     # print(self.b.shape)
    #     print(self.W)
    #     # print(torch.cat((h,x),dim=-1))
    #     # print(self.b)
    #     # h_new = torch.tanh(self.W @ torch.cat((h,x),dim=-1)+self.b)
    #     # print(h_new)
    #     return h_new

    # class AffinityLayer(nn.Module):
    # def __init__(self, x_len, h_len):
    #     super(AffinityLayer, self).__init__()
    #     self.affcell = Affinitycell(x_len, h_len)
    #     self.h_len = h_len
    
    # def forward(self, x):
    #     # x [N, C]
    #     # print(x.shape)
    #     y = torch.Tensor(self.h_len,1)
    #     h = torch.zeros(self.h_len)
    #     # print(h.shape)
    #     for i in x:
    #         h = self.affcell(i,h)
    #         y = torch.cat((y,torch.unsqueeze(h, 1)),dim=1)
    #     y = y[:,1:]
    #     return y.transpose(0,1)


def Softplus(x):
    return torch.log(1+torch.exp(x))

class Affplus(nn.Module):
    def __init__(self, x_len):
        super(Affplus, self).__init__()
        self.softplus = Softplus
        self.alpha = torch.randn(x_len, requires_grad=True).to(DEVICE)
        self.params = nn.Parameter(self.alpha)
        # nn.init.
    
    def forward(self,x):
        return self.alpha * self.softplus(x) + (1-self.alpha) * x
    

class Affcell(nn.Module):
    def __init__(self, x_len, h_len, dropout=0.05):
        super(Affcell, self).__init__()
        self.h_len = h_len
        self.affplus = Affplus(self.h_len)
        self.b = nn.Parameter(torch.Tensor(self.h_len).to(DEVICE))
        self.W = nn.Parameter(torch.Tensor(self.h_len, x_len+h_len).to(DEVICE))
        self.dropout = nn.Dropout(dropout)
        # self.norm= nn.BatchNorm1d(self.h_len)
        self.norm= nn.LayerNorm(self.h_len)
        nn.init.kaiming_normal_(self.W)
        
    
    def forward(self, x, h):
        b,_ = x.shape
        # W [p, q]
        # x [b, c]
        # h [b, q-c]
        # h_tilde = (self.W @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b
        h_tilde = self.affplus((self.W @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b)
        h_tilde = self.dropout(h_tilde)
        # h_tilde = F.relu((self.W @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b)
        return h_tilde
    
class AdaAffcell(nn.Module):
    def __init__(self, x_len, h_len, dropout=0.05):
        super(AdaAffcell, self).__init__()
        self.h_len = h_len
        self.affplus = Affplus(self.h_len)
        self.b_a = nn.Parameter(torch.Tensor(self.h_len))
        self.b_g = nn.Parameter(torch.Tensor(self.h_len))
        self.W_a = nn.Parameter(torch.Tensor(self.h_len, x_len+h_len))
        self.W_g = nn.Parameter(torch.Tensor(self.h_len, x_len+h_len))
        self.sigmoid = torch.sigmoid
        # self.norm= nn.BatchNorm1d(self.h_len)
        self.norm= nn.LayerNorm(self.h_len)
        nn.init.kaiming_normal_(self.W_a)
        nn.init.kaiming_normal_(self.W_g)
        self.dropout = nn.Dropout(dropout)
        
    
    def forward(self, x, h):
        b,_ = x.shape
        # W [p, q]
        # x [b, c]
        # h [b, q-c]
        # h_tilde = (self.W @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b
        alpha = self.sigmoid((self.W_g @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b_g)
        x = (self.W_a @ torch.cat((h,x),dim=-1).transpose(0,-1)).transpose(0,-1)+self.b_a
        # h_tilde = alpha*Softplus(x) +(1-alpha)*x
        h_tilde = alpha*torch.tanh(x) +(1-alpha)*x
        h_tilde = self.dropout(h_tilde)
        return self.norm(h_tilde)


class AffinityLayer(nn.Module):
    def __init__(self, in_len, out_len, batch_first = True):
        super(AffinityLayer, self).__init__()
        self.affcell = AdaAffcell(in_len, out_len)
        self.out_len = out_len
        self.batch_first = batch_first
    
    def forward(self, X):
        if X.dim() == 3 and self.batch_first:
            # X [B, N, C]
            b, n, _ = X.shape
            # print(b,n)
            y = torch.Tensor(b, 1, self.out_len).to(DEVICE)
            h = torch.zeros(b, self.out_len).to(DEVICE)
            for i in range(n):
                x = X[:, i, :]
                # print(x.shape)
                h = self.affcell(x, h)
                # print(h.shape)
                y = torch.cat((y, torch.unsqueeze(h, 1)), dim=1)
            y = y[:,1:,:]
        return y



class Model(nn.Module):
    """
    Vanilla Affinity Network
    with O(L) complexity
    Paper link: 
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.task_name = configs.task_name
        self.pred_len = configs.pred_len


        # How to implement for RNN models?

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.aff = AffinityLayer(in_len = configs.enc_in, 
                                out_len = configs.d_model)
            self.aff2 = AffinityLayer(in_len =  configs.d_model, 
                                out_len = configs.d_model)
            self.aff3 = AffinityLayer(in_len =  configs.d_model, 
                                out_len = configs.d_model)
            self.projection = nn.Linear(configs.d_model,configs.c_out, bias=True)
            self.encdec = ConvDecoder(configs.seq_len, self.pred_len, configs.c_out)
            # self.encdec = LinearDecoder(configs.seq_len, self.pred_len)


        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc):
        # Embedding
        x_out = self.aff(x_enc)
        x_out = self.aff2(x_out)
        x_out = self.aff3(x_out)
        dec_out = self.projection(x_out)
        return dec_out


    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            dec_out = self.encdec(dec_out)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        # if self.task_name == 'imputation':
        #     dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'anomaly_detection':
        #     dec_out = self.anomaly_detection(x_enc)
        #     return dec_out  # [B, L, D]
        # if self.task_name == 'classification':
        #     dec_out = self.classification(x_enc, x_mark_enc)
        #     return dec_out  # [B, N]
        return None
