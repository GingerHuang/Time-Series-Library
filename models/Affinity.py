import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.LSTM_EncDec import ConvDecoder,LinearDecoder




class Affplus(nn.Module):
    def __init__(self, x_len, alpha=None):
        super(Affplus, self).__init__()
        self.softplus = F.softplus
        self.alpha = alpha or torch.randn(x_len, requires_grad=True)
        self.params = nn.Parameter(self.alpha)
    
    def forward(self,x):
        return self.alpha * self.softplus(x) + (1-self.alpha) * x

class Affinitylayer(nn.Module):
    def __init__(self, x_len, h_len):
        super(Affinitylayer, self).__init__()
        self.h_len = h_len
        self.affplus = Affplus(self.h_len)
        self.W = nn.Parameter(torch.Tensor(h_len, x_len+h_len))
        self.b= nn.parameter(torch.Tensor(h_len))
    
    def forward(self, x, h=None):
        h = h or torch.zeros(self.h_len)
        h_new = self.affplus(self.W @ torch.cat((h,x),dim=-1) ,self.b)
        return h_new

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
            self.lstm = nn.LSTM(input_size = configs.enc_in, 
                                hidden_size = configs.d_model, 
                                num_layers = configs.hidden_layers, 
                                bias=True, batch_first=True, 
                                dropout=0.05,
                                bidirectional = False)
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
        x_out,_ = self.lstm(x_enc)
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
