import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from Modules import PositionalEncoding, EpisodicMemory

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        enc_output = self.transformer_encoder(src, src_mask)
        output = self.linear(enc_output)
        return output
    

class EpisodicMemoryTransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, memory_size: int = 2048):
        super().__init__()
        self.model_type = 'EpisodicMemoryTransformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(2*d_model, ntoken)

        self.episodic_memory = EpisodicMemory(memory_size, d_model, d_k=d_hid, d_v=d_hid)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, trg: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            trg: Tensor, shape ``[seq_len, batch_size]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        enc_output = self.transformer_encoder(src, None)
        mem_output, _ = self.episodic_memory.retrieve_seqs(enc_output)

        if trg is not None:
            trg = self.pos_encoder(self.embedding(trg) * math.sqrt(self.d_model))
            trg_output = self.transformer_encoder(trg, None)
            self.episodic_memory.add_seqs(enc_output, trg_output)
            self.episodic_memory.cleanup(mode='random', every=10, n=1024)
        
        enc_output = torch.concatenate((enc_output, mem_output), dim=-1)
        output = self.linear(enc_output)
        return output