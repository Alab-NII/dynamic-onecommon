import sys
import re
import time
import copy
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable
import torch.nn.functional as F

import data
from models.utils import *
import models
from domain import get_domain
from engines.hierarchical_reference_engine import HierarchicalReferenceEngine
from models.ctx_encoder import *


class HierarchicalReferenceModel(nn.Module):
    corpus_ty = data.ReferenceSentenceCorpus
    engine_ty = HierarchicalReferenceEngine

    def __init__(self, word_dict, args):
        super(HierarchicalReferenceModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.num_ent = domain.num_ent()

        # define modules:
        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        ctx_encoder_ty = models.get_ctx_encoder_type(args.ctx_encoder_type)
        self.ctx_encoder = ctx_encoder_ty(domain, args)

        self.encoder = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True,
            batch_first=True)

        self.memory = nn.GRUCell(
            input_size=args.nhid_lang,
            hidden_size=args.nhid_lang,
            bias=True)

        self.decoder_reader = nn.GRU(
            input_size=args.nembed_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.decoder_writer = nn.GRUCell(
            input_size=args.nembed_ctx + args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.hid2output = nn.Sequential(
            nn.Linear(args.nhid_lang, args.nembed_word),
            nn.Dropout(args.dropout))

        self.lang_attn = nn.Sequential(
            nn.Linear(args.nembed_ctx + args.nhid_lang, args.nhid_attn),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(args.nhid_attn, args.nhid_attn),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            torch.nn.Linear(args.nhid_attn, 1))

        if args.share_attn:
            self.sel_attn = self.lang_attn
            self.ref_attn = self.lang_attn
        else:
            self.sel_attn = nn.Sequential(
                nn.Linear(args.nembed_ctx + args.nhid_lang, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, 1))

            self.ref_attn = nn.Sequential(
                nn.Linear(args.nembed_ctx + args.nhid_lang, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, 1))

        # tie the weights between reader and writer
        self.decoder_writer.weight_ih = self.decoder_reader.weight_ih_l0
        self.decoder_writer.weight_hh = self.decoder_reader.weight_hh_l0
        self.decoder_writer.bias_ih = self.decoder_reader.bias_ih_l0
        self.decoder_writer.bias_hh = self.decoder_reader.bias_hh_l0

        self.dropout = nn.Dropout(args.dropout)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.encoder, args.init_range)
        init_rnn_cell(self.memory, args.init_range)
        init_rnn(self.decoder_reader, args.init_range)
        init_cont(self.lang_attn, args.init_range)
        if not args.share_attn:
            init_cont(self.sel_attn, args.init_range)
            init_cont(self.ref_attn, args.init_range)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder_reader.flatten_parameters()

    def embed_sentence(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def encode_sentence(self, inpt_emb, hid_idx, enc_init_h=None):
        bsz = inpt_emb.size(0)

        if enc_init_h is None:
            enc_init_h = self._zero(bsz, self.args.nhid_lang)

        enc_init_h = enc_init_h.unsqueeze(0)

        enc_hs, _ = self.encoder(inpt_emb, enc_init_h)

        expand_hid_idx = hid_idx.unsqueeze(1).unsqueeze(2).expand(bsz, 1, enc_hs.size(2))
        enc_last_h = torch.gather(enc_hs, 1, expand_hid_idx)

        return enc_hs, enc_last_h.squeeze(1)

    def decode_sentence(self, ctx_h, inpt_emb, mem_h):
        bsz = ctx_h.size(0)
        utterance_len = inpt_emb.size(1)
        num_ent = ctx_h.size(1)

        # compute attention for decoder
        expand_mem_h = mem_h.unsqueeze(1).expand(bsz, num_ent, self.args.nhid_lang)
        lang_logit = self.lang_attn(torch.cat([ctx_h, expand_mem_h], 2))
        lang_prob = F.softmax(lang_logit, dim=1).expand(bsz, num_ent, self.args.nembed_ctx)
        attended_ctx_h = torch.sum(torch.mul(ctx_h, lang_prob), 1)

        # concat attended_ctx_h and inpt_emb
        attended_ctx_h = attended_ctx_h.unsqueeze(1).expand(bsz, utterance_len, self.args.nembed_ctx)
        concatenated_inpt_emb = torch.cat([attended_ctx_h, inpt_emb], 2)

        # run decoder
        concatenated_inpt_emb = concatenated_inpt_emb.transpose(0, 1).contiguous()
        mem_h = mem_h.unsqueeze(0)
        dec_hs, _ = self.decoder_reader(concatenated_inpt_emb, mem_h)

        return dec_hs.transpose(0, 1).contiguous()

    def unembed_sentence(self, dec_hs):
        # convert to the embed space
        out_emb = self.hid2output(dec_hs.view(-1, dec_hs.size(2)))
        # unembed
        out = F.linear(out_emb, self.word_embed.weight)
        return out

    def forward_encoder(self, inpt, hid_idx, enc_init_h=None):
        # embed
        inpt_emb = self.embed_sentence(inpt)

        # encode sentence and turn
        enc_hs, enc_last_h = self.encode_sentence(inpt_emb, hid_idx, enc_init_h)

        return enc_hs, enc_last_h

    def forward_decoder(self, ctx_h, inpt, mem_h):
        # embed
        inpt_emb = self.embed_sentence(inpt)
        # run decoder
        dec_hs = self.decode_sentence(ctx_h, inpt_emb, mem_h)
        # unembed
        out = self.unembed_sentence(dec_hs)
        return out

    def reference_resolution(self, ctx_h, enc_hs, ref_inpts):
        if ref_inpts is None:
            return None

        bsz = ctx_h.size(0)
        num_ent = ctx_h.size(1)
        num_markables = ref_inpts.size(1)

        # reshape
        ref_inpts = ref_inpts.view(bsz, 3 * num_markables)
        ref_inpts = ref_inpts.unsqueeze(2).expand(bsz, 3 * num_markables, self.args.nhid_lang)

        # gather indices
        ref_inpts = torch.gather(enc_hs, 1, ref_inpts)

        # reshape
        ref_inpts = ref_inpts.view(bsz, num_markables, 3, self.args.nhid_lang)

        # take sum
        ref_inpts = torch.sum(ref_inpts, 2)
        
        expand_ref_inpts = ref_inpts.unsqueeze(2).expand(bsz, num_markables, num_ent, self.args.nhid_lang)
        expand_ctx_h = ctx_h.unsqueeze(1).expand(bsz, num_markables, num_ent, self.args.nembed_ctx)

        ref_logits = self.ref_attn(torch.cat([expand_ctx_h, expand_ref_inpts], 3))

        return ref_logits.squeeze(3) 

    def selection(self, ctx_h, mem_h):
        bsz = ctx_h.size(0)
        num_ent = ctx_h.size(1)

        exapnd_mem_h = mem_h.unsqueeze(1).expand(bsz, num_ent, self.args.nhid_lang)
        sel_logit = self.sel_attn(torch.cat([ctx_h, exapnd_mem_h], 2))

        return sel_logit.squeeze(2)
    
    def forward(self, ctx, inpts, ref_inpts, hid_idxs, sel_idx):
        ctx_h = self.ctx_encoder(ctx) # shape: (bsz, num_ent, args.nembed_ctx)

        bsz = ctx_h.size(0)
        num_utterances = len(inpts)

        mem_h = self._zero(bsz, self.args.nhid_lang)
        outs = []

        for ui in range(num_utterances):
            # predict utterance
            out = self.forward_decoder(ctx_h, inpts[ui], mem_h)
            outs.append(out)

            # encode utterance
            enc_hs, enc_last_h = self.forward_encoder(inpts[ui], hid_idxs[ui])

            # encode utterance for reference resolution
            next_ref_enc_hs, _ = self.forward_encoder(inpts[ui], hid_idxs[ui], mem_h)
            if ui == 0:
                ref_enc_hs = next_ref_enc_hs
            else:
                ref_enc_hs = torch.cat([ref_enc_hs, next_ref_enc_hs], 1)

            # run through memory
            mem_h = self.memory(enc_last_h, mem_h)

        # predict referents
        ref_outs = self.reference_resolution(ctx_h, ref_enc_hs, ref_inpts)

        # predict target
        sel_out = self.selection(ctx_h, mem_h)

        return outs, ref_outs, sel_out


    def read(self, ctx_h, inpt, mem_h, prefix_token='THEM:'):
        bsz = 1

        # Add a 'THEM:' token to the start of the message
        if prefix_token:
            prefix = self.word2var(prefix_token).unsqueeze(0)
            inpt = torch.cat([prefix, inpt], dim=1)

        inpt_emb = self.embed_sentence(inpt)

        # initialize encoder state
        enc_init_h = self._zero(bsz, self.args.nhid_lang)
        enc_init_h = enc_init_h.unsqueeze(0)

        # run encoder
        enc_hs, enc_last_h = self.encoder(inpt_emb, enc_init_h)

        # run memory
        enc_last_h = enc_last_h.squeeze(1)
        mem_h = self.memory(enc_last_h, mem_h)

        return mem_h

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def write(self, ctx_h, mem_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):
        bsz = 1
        num_ent = ctx_h.size(1)

        # autoregress starting from start_token
        inpt = self.word2var(start_token)

        outs = [inpt.unsqueeze(0)]
        logprobs = []
        lang_hs = []

        # compute attention for decoder
        expand_mem_h = mem_h.unsqueeze(1).expand(bsz, num_ent, self.args.nhid_lang)
        lang_logit = self.lang_attn(torch.cat([ctx_h, expand_mem_h], 2))
        lang_prob = F.softmax(lang_logit, dim=1).expand(bsz, num_ent, self.args.nembed_ctx)
        attended_ctx_h = torch.sum(torch.mul(ctx_h, lang_prob), 1)

        # initialize decoder state
        dec_h = mem_h

        for _ in range(max_words):
            # embed
            inpt_emb = self.embed_sentence(inpt)

            # concat attended_ctx_h and inpt_emb
            concatenated_inpt_emb = torch.cat([attended_ctx_h, inpt_emb], 1)

            # run decoder
            dec_h = self.decoder_writer(concatenated_inpt_emb, dec_h)

            if self.word_dict.get_word(inpt.item()) in stop_tokens:
                break

            # convert to the embed space
            out_emb = self.hid2output(dec_h)

            # unembed
            out = F.linear(out_emb, self.word_embed.weight)

            scores = out.div(temperature)
            scores = scores.sub(scores.max().item()).squeeze(0)

            mask = Variable(self.special_token_mask)
            scores = scores.add(mask)

            prob = F.softmax(scores, dim=0)
            logprob = F.log_softmax(scores, dim=0)
            inpt = prob.multinomial(1).detach()
            outs.append(inpt.unsqueeze(1))
            logprob = logprob.gather(0, inpt)
            logprobs.append(logprob)

        outs = torch.cat(outs, 1)

        # read the output utterance
        mem_h = self.read(ctx_h, outs, mem_h, prefix_token=None)

        return outs, logprobs, mem_h

