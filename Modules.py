import math, random

import torch
from torch import nn, Tensor
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None, method='default'):
        batch_size, n_heads, seq_len, d_model = q.shape
        if method=='broadcast':
            q, k, v = q.permute(1, 2, 0, 3), k.permute(1, 2, 0, 3), v.permute(1, 2, 0, 3)
            attn = torch.matmul(q / self.temperature, k)
            if mask is not None:
                attn = attn.masked_fill(mask == 0, -1e9)
            attn = self.dropout(F.softmax(attn, dim=-1))
            output = torch.matmul(attn, v)
            output, attn = output.permute(2, 0, 1, 3), attn.permute(2, 0, 1, 3)
        elif method=='full':
            q, k, v = q.view(-1, d_model), k.view(-1, d_model), v.view(-1, d_model)
        
        attn = torch.matmul(q / self.temperature, k.transpose(-2, -1)) # batch_size, n_heads, query_seq_len, key_seq_len
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v).view(batch_size, n_heads, seq_len, d_model)

        return output, attn
    
class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, method='default'):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, sz_t, len_k, len_v = q.size(0), q.size(1), k.size(0), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_t, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_t, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=mask, method=method)    

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous()[:,:len_q].view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
class EpisodicMemory(nn.Module):

    def __init__(self, memory_size, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.k = torch.zeros(size=(memory_size, 1, d_model), device=device)
        self.v = torch.zeros(size=(memory_size, 1, d_model), device=device)
        self.mask = torch.zeros(size=(memory_size,), dtype=bool, device=device)
        # self.f = torch.zeros(size=(memory_size, d_hid), device=device, requires_grad=False)
        # self.i = torch.zeros(size=(memory_size, d_hid), device=device, requires_grad=False)
        self.memory_size = memory_size

        self.cleanup_ctr = 0

        self.slf_attn = MultiHeadAttention(1, d_model, d_k, d_v, dropout=dropout)

        self.verbose = False

    def __len__(self):
        return (self.mask==True).sum()

    def add_seqs(self, k, v, n=None):
        batch_size, seq_len, _ = k.shape
        flat_size = batch_size * seq_len
        k, v = k.view(flat_size, 1, -1), v.view(flat_size, 1, -1)
        if n == None: n = flat_size
        if len(self) + n > self.memory_size: n = self.memory_size - len(self)
        idxs_to_add = torch.randperm(flat_size, device=device)[:n]
        add_pos = torch.arange(self.memory_size, device=device)[~self.mask][:n]
            
        self.k.data[add_pos, :1] += k[idxs_to_add]
        self.v.data[add_pos, :1] += v[idxs_to_add]

        self.mask.data[add_pos] = True

        if n > 0 and self.verbose: print(f'Added {n} sequences to memory. Memory table has {len(self)}/{self.memory_size} sequences left.')

    # def update_freq(self, f, i):
    #     seq_len = i.shape[1]
    #     f = f.sum(dim=1)

    #     with torch.no_grad():
    #         f_add = torch.zeros_like(self.f)
    #         f_add[:, :seq_len] += f.sum(dim=0).T

    #         i_add = torch.zeros_like(self.i)
    #         i_add[:, :seq_len] += (f[:, :seq_len] * i[...,None].repeat(1, 1, self.memory_size)).sum(dim=0).T

    #         self.f = self.f + f_add
    #         self.i = self.i + i_add

    def cleanup(self, mode='random', every=10, n=128):
        self.cleanup_ctr += 1
        if (len(self) != self.memory_size) or (self.cleanup_ctr % every != 0): 
            return

        n_before = len(self)

        # avg_loss_gain = torch.where(self.f!=0, self.i/self.f, self.f)
        botk = 64

        # seqs_to_drop = torch.topk(avg_loss_gain.sum(dim=-1), largest=False, k=botk)[1]

        if mode=='random': seqs_to_drop = torch.tensor(random.sample(range(self.memory_size), n))
        
        self.v.data[seqs_to_drop] = 0
        self.k.data[seqs_to_drop] = 0
        self.mask.data[seqs_to_drop] = False

        if n_before - len(self) > 0  and self.verbose: 
            print(f'Removed {n_before - len(self)} sequences from memory. Memory table has {len(self)}/{self.memory_size} sequences left.')

    def retrieve_seqs(self, q):
        mem_output, mem_attn = self.slf_attn(q, self.k, self.v, method='full')
        return mem_output, mem_attn