import torch
import torch.nn as nn
import torch.nn.functional as F

# class DecoderLayer(nn.Module):
#     def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
#                  dropout=0.1, activation="relu"):
#         super(DecoderLayer, self).__init__()
#         d_ff = d_ff or 4 * d_model
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.norm3 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
#         x = x + self.dropout(self.self_attention(
#             x, x, x,
#             attn_mask=x_mask,
#             tau=tau, delta=None
#         )[0])
#         x = self.norm1(x)

#         x = x + self.dropout(self.cross_attention(
#             x, cross, cross,
#             attn_mask=cross_mask,
#             tau=tau, delta=delta
#         )[0])

#         y = x = self.norm2(x)
#         y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
#         y = self.dropout(self.conv2(y).transpose(-1, 1))

#         return self.norm3(x + y)


class ConvDecoder(nn.Module):
    def __init__(self, y_len, pred_len, c_out, d_ff=None, dropout=0.1, activation="relu"):
        super(ConvDecoder, self).__init__()
        d_ff = d_ff or 4 * y_len
        self.conv1 = nn.Conv1d(in_channels=y_len, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=pred_len, kernel_size=1)
        self.norm1 = nn.LayerNorm(c_out)
        self.norm2 = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        # print(x.shape)
        x = x + self.dropout(x)
        y = self.norm1(x)
        # print(y.shape, x.shape)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        return self.norm2(y)
    

class LinearDecoder(nn.Module):
    def __init__(self, y_len, pred_len,  dropout=0.1):
        super(LinearDecoder, self).__init__()
        self.projection = nn.Linear(y_len, pred_len)
        self.norm = nn.LayerNorm(pred_len)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

    def forward(self, x):
        y = self.projection(x.transpose(-1, -2))
        y = self.norm(y)
        y = self.dropout(y.transpose(-1, -2))
        return y