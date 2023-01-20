import torch
from torch import nn
import numpy as np


class MDM(nn.Module):
    def __init__(self, njoints, nfeats, latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 activation="gelu", **kwargs):

        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.input_feats = self.njoints * self.nfeats

        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.normalize_output = kwargs.get('normalize_encoder_output', False)

        self.input_processing = InputProcessing(self.input_feats, self.latent_dim)
        self.pos_encoding = PositionalEncoding(self.latent_dim, )
        self.time_embedding = TimestepEmbedding(self.latent_dim)
        self.output_processing = OutputProcessing(self.njoints, self.nfeats, self.input_feats, self.latent_dim)

        trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=self.num_heads,
                                                       dim_feedforward=self.ff_size, dropout= self.dropout,
                                                       activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(trans_encoder_layer, num_layers=self.num_layers)

    def forward(self, x, time_steps):
        ts_pe = self.pos_encoding.pe[time_steps].permute(1, 0, 2)
        ts_emb = self.time_embedding(ts_pe)
        x = self.input_processing(x)
        x = torch.cat((ts_emb, x), dim=0)
        x = self.pos_encoding(x)
        output = self.seqTransEncoder(x)[1:]
        output = self.output_processing(output)
        return output


class InputProcessing(nn.Module):
    """
    linearly projects input into the transformer dimension
    """
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.poseEmbedding = nn.Linear(input_feats, latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)
        return self.poseEmbedding(x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding.
    Used for: positional encoding of frames in the motion sequence,
              time step embedding
    #TODO max_sequence_length should be >= max time steps
    """
    def __init__(self, latent_dim, max_sequence_len=1000, dropp=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropp)
        pe = torch.zeros(max_sequence_len, latent_dim)
        pos = torch.arange(0, max_sequence_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, latent_dim, 2, dtype=torch.float) * (-np.log(max_sequence_len) / latent_dim))
        pe[:, 0::2] = torch.sin(pos * denom)
        pe[:, 1::2] = torch.cos(pos * denom)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedding(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        return self.time_embed(x)


class OutputProcessing(nn.Module):
    """
    projects output back into the original motion dimension
    """
    def __init__(self, njoints, nfeats, input_feats, latent_dim):
        super().__init__()
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseDeEmbedding = nn.Linear(latent_dim, input_feats)

    def forward(self, x):
        nframes, bs, d = x.shape
        x = self.poseDeEmbedding(x)
        x = x.reshape(nframes, bs, self.njoints, self.nfeats).permute(1, 2, 3, 0)
        return x
