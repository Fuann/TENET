import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_
from torch.nn import CrossEntropyLoss

def scaled_dot_product_attention(q, k, v, mask=None, attention_mask=None):
    # calculate attention
    matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / np.sqrt(dk)

    if mask is not None:
        nd, ns = scaled_attention_logits.size(-2), scaled_attention_logits.size(-1)
        scaled_attention_logits += mask[ns - nd : ns, :ns] * -1e4

    if attention_mask is not None:
        # Apply the attention mask
        scaled_attention_logits = scaled_attention_logits + attention_mask

    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model_size = d_model_size

        self.depth = int(d_model_size / self.num_heads)

        self.Wq = torch.nn.Linear(d_model_size, d_model_size)
        self.Wk = torch.nn.Linear(d_model_size, d_model_size)
        self.Wv = torch.nn.Linear(d_model_size, d_model_size)
        self.dense = torch.nn.Linear(d_model_size, d_model_size)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.Wq.weight, gain=1 / math.sqrt(2))
        xavier_uniform_(self.Wk.weight, gain=1 / math.sqrt(2))
        xavier_uniform_(self.Wv.weight, gain=1 / math.sqrt(2))

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.permute([0, 2, 1, 3])

    def forward(self, v, k, q, mask=None, attention_mask=None):
        batch_size = q.shape[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_into_heads(q, batch_size)
        k = self.split_into_heads(k, batch_size)
        v = self.split_into_heads(v, batch_size)

        output = scaled_dot_product_attention(q, k, v, mask, attention_mask)
        scaled_attention = output[0].permute([0, 2, 1, 3])
        attn = output[1]
        original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        output = self.dense(original_size_attention)

        return output, attn


if __name__ == '__main__':
    # B x N x T
    x = torch.rand(4, 256, 30)
    x = x.permute(0, 2, 1)
    nnet = nn.MultiheadAttention(256, 8)
    x, _ = nnet(x, x, x)
    print(x.shape)
