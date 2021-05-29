"""
Set of context encoders.
"""
from itertools import combinations
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

from models.utils import *

class AttentionContextEncoder(nn.Module):
    def __init__(self, domain, args):
        super(AttentionContextEncoder, self).__init__()

        self.args = args
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent(), int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.binary_encoder = torch.nn.Embedding(domain.num_bin() * 2, args.nembed_ctx)

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.ent_rnn = nn.GRU(
            input_size=args.nembed_ctx,
            hidden_size=args.nembed_ctx,
            batch_first=True,
            bias=True)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        # init
        init_cont([self.property_encoder, self.relation_encoder], args.init_range)
        self.binary_encoder.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.ent_rnn, args.init_range)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def forward(self, entities, binary_features):
        """
            - Input
                entities: shape (bsz, num_ent, num_frames, dim_ent)
                binary_features: shape (bsz, num_ent, num_frames, domain.num_bin)
            - Output
                entity_embeddings: shape (bsz, num_ent, args.nembed_ctx)
        """
        bsz = entities.size(0)
        num_ent = entities.size(1)
        num_frames = entities.size(2)
        dim_ent = entities.size(3)
        num_bin = binary_features.size(3)

        property_embeddings = self.property_encoder(entities)

        ent_rel_pairs = []
        for i in range(num_ent):
            rel_pairs = []
            for j in range(num_ent):
                if i == j:
                    continue
                dist = torch.sqrt((entities[:,i,:,0] - entities[:,j,:,0])**2 + (entities[:,i,:,1] - entities[:,j,:,1])**2)
                rel_pairs.append((torch.cat([entities[:,i,:,:] - entities[:,j,:,:], dist.unsqueeze(2)], 2).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        relation_embeddings = self.relation_encoder(ent_rel_pairs).sum(2)
        entity_embeddings = torch.cat([property_embeddings, relation_embeddings], 3)
        entity_embeddings = entity_embeddings.view(bsz * num_ent, num_frames, self.args.nembed_ctx) # convert to (batch, seq_len, input_size) format

        binary_feature_embeddings = self.binary_encoder(binary_features)
        binary_feature_embeddings = torch.sum(binary_feature_embeddings, 3)
        binary_feature_embeddings = binary_feature_embeddings.view(bsz * num_ent, num_frames, self.args.nembed_ctx)

        entity_embeddings = entity_embeddings + binary_feature_embeddings

        entity_embeddings = self.dropout(entity_embeddings)

        init_ctx_h = self._zero(1, bsz * num_ent, self.args.nembed_ctx)

        self.ent_rnn.flatten_parameters()
        _, entity_embeddings = self.ent_rnn(entity_embeddings, init_ctx_h)

        entity_embeddings = entity_embeddings.squeeze(0).view(bsz, num_ent, self.args.nembed_ctx)

        entity_embeddings = self.dropout(entity_embeddings)

        return entity_embeddings # shape (bsz, num_ent, args.nembed_ctx)


class MlpContextEncoder(nn.Module):
    """A module that encodes dialogues context using an MLP."""
    def __init__(self, domain, args):
        super(MlpContextEncoder, self).__init__()
        self.args = args
        self.ent_embed = nn.Linear(domain.dim_ent(), args.nembed_ctx)
        self.bin_embed = nn.Embedding(domain.num_bin() * 2, args.nembed_ctx)

        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(args.dropout)

        self.ent_rnn = nn.GRU(
            input_size=args.nembed_ctx,
            hidden_size=args.nembed_ctx,
            batch_first=True,
            bias=True)
        
        # init
        init_cont([self.ent_embed], args.init_range)
        self.bin_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.ent_rnn, args.init_range)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def forward(self, entities, binary_features):
        """
            - Input
                entities: shape (bsz, num_ent, num_frames, dim_ent)
                binary_features: shape (bsz, num_ent, num_frames, domain.num_bin)
            - Output
                entity_embeddings: shape (bsz, num_ent, args.nembed_ctx)
        """
        bsz = entities.size(0)
        num_ent = entities.size(1)
        num_frames = entities.size(2)
        dim_ent = entities.size(3)
        num_bin = binary_features.size(3)

        entity_embeddings = self.ent_embed(entities)
        entity_embeddings = entity_embeddings.view(bsz * num_ent, num_frames, self.args.nembed_ctx) # convert to (batch, seq_len, input_size) format

        binary_feature_embeddings = self.bin_embed(binary_features)
        binary_feature_embeddings = torch.sum(binary_feature_embeddings, 3)
        binary_feature_embeddings = binary_feature_embeddings.view(bsz * num_ent, num_frames, self.args.nembed_ctx)

        entity_embeddings = entity_embeddings + binary_feature_embeddings

        entity_embeddings = self.dropout(entity_embeddings)

        init_ctx_h = self._zero(1, bsz * num_ent, self.args.nembed_ctx)

        self.ent_rnn.flatten_parameters()
        _, entity_embeddings = self.ent_rnn(entity_embeddings, init_ctx_h)

        entity_embeddings = entity_embeddings.squeeze(0).view(bsz, num_ent, self.args.nembed_ctx)

        entity_embeddings = self.dropout(entity_embeddings)

        return entity_embeddings # shape (bsz, num_ent, args.nembed_ctx)

class StaticContextEncoder(nn.Module):
    def __init__(self, domain, args):
        super(StaticContextEncoder, self).__init__()

        self.num_ent = domain.num_ent()
        self.dim_ent = domain.dim_ent()

        self.property_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent(), int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.relation_encoder = nn.Sequential(
            torch.nn.Linear(domain.dim_ent() + 1, int(args.nembed_ctx / 2)),
            nn.Tanh(),
            nn.Dropout(args.dropout)
        )

        self.dropout = nn.Dropout(args.dropout)

        init_cont([self.property_encoder, self.relation_encoder], args.init_range)

    def forward(self, ctx):
        ctx_t = ctx.transpose(0, 1)
        ents = ctx_t.view(ctx_t.size(0), self.num_ent, self.dim_ent)
        prop_emb = self.property_encoder(ents)
        ent_rel_pairs = []
        for i in range(self.num_ent):
            rel_pairs = []
            for j in range(self.num_ent):
                if i == j:
                    continue
                dist = torch.sqrt((ents[:,i,0] - ents[:,j,0])**2 + (ents[:,i,1] - ents[:,j,1])**2)
                rel_pairs.append((torch.cat([ents[:,i,:] - ents[:,j,:], dist.unsqueeze(1)], 1).unsqueeze(1)))
            ent_rel_pairs.append(torch.cat(rel_pairs, 1).unsqueeze(1))
        ent_rel_pairs = torch.cat(ent_rel_pairs, 1)
        rel_emb = self.relation_encoder(ent_rel_pairs).sum(2)
        out = torch.cat([prop_emb, rel_emb], 2)
        return out