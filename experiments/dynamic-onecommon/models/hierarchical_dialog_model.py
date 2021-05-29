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
from engines.hierarchical_dialog_engine import HierarchicalDialogEngine
from models.ctx_encoder import *


class HierarchicalDialogModel(nn.Module):
    corpus_ty = data.SentenceCorpus
    engine_ty = HierarchicalDialogEngine

    def __init__(self, word_dict, args):
        super(HierarchicalDialogModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.num_selectable = domain.num_selectable()

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
            bias=True,
            batch_first=True)

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
        else:
            self.sel_attn = nn.Sequential(
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

    def initialize_from(self, pretrained_model):
        self.word_embed = copy.deepcopy(pretrained_model.word_embed)

        self.ctx_encoder.property_encoder = copy.deepcopy(pretrained_model.ctx_encoder.property_encoder)
        self.ctx_encoder.relation_encoder = copy.deepcopy(pretrained_model.ctx_encoder.relation_encoder)

        self.encoder.weight_ih_l0 = pretrained_model.encoder.weight_ih_l0
        self.encoder.weight_hh_l0 = pretrained_model.encoder.weight_hh_l0
        self.encoder.bias_ih_l0 = pretrained_model.encoder.bias_ih_l0
        self.encoder.bias_hh_l0 = pretrained_model.encoder.bias_hh_l0

        self.memory.weight_ih = pretrained_model.memory.weight_ih
        self.memory.weight_hh = pretrained_model.memory.weight_hh
        self.memory.bias_ih = pretrained_model.memory.bias_ih
        self.memory.bias_hh = pretrained_model.memory.bias_hh

        self.decoder_reader.weight_ih_l0 = pretrained_model.decoder_reader.weight_ih_l0
        self.decoder_reader.weight_hh_l0 = pretrained_model.decoder_reader.weight_hh_l0
        self.decoder_reader.bias_ih_l0 = pretrained_model.decoder_reader.bias_ih_l0
        self.decoder_reader.bias_hh_l0 = pretrained_model.decoder_reader.bias_hh_l0
        self.decoder_writer.weight_ih = self.decoder_reader.weight_ih_l0
        self.decoder_writer.weight_hh = self.decoder_reader.weight_hh_l0
        self.decoder_writer.bias_ih = self.decoder_reader.bias_ih_l0
        self.decoder_writer.bias_hh = self.decoder_reader.bias_hh_l0

        self.hid2output = copy.deepcopy(pretrained_model.hid2output)
        self.lang_attn = copy.deepcopy(pretrained_model.lang_attn)
        if self.args.share_attn:
            self.sel_attn = self.lang_attn
        else:
            self.sel_attn = copy.deepcopy(pretrained_model.sel_attn)

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

    def encode_sentence(self, uttr_emb, hid_idx, enc_init_h=None):
        bsz = uttr_emb.size(0)

        if enc_init_h is None:
            enc_init_h = self._zero(bsz, self.args.nhid_lang)

        enc_init_h = enc_init_h.unsqueeze(0)

        enc_hs, _ = self.encoder(uttr_emb, enc_init_h)

        expand_hid_idx = hid_idx.unsqueeze(1).unsqueeze(2).expand(bsz, 1, enc_hs.size(2))
        enc_last_h = torch.gather(enc_hs, 1, expand_hid_idx)

        return enc_hs, enc_last_h.squeeze(1)

    def decode_sentence(self, entity_embeddings, uttr_emb, mem_h):
        bsz = entity_embeddings.size(0)
        utterance_len = uttr_emb.size(1)
        num_ent = entity_embeddings.size(1)

        # compute attention for decoder
        expand_mem_h = mem_h.unsqueeze(1).expand(bsz, num_ent, self.args.nhid_lang)
        lang_logit = self.lang_attn(torch.cat([entity_embeddings, expand_mem_h], 2))
        lang_prob = F.softmax(lang_logit, dim=1).expand(bsz, num_ent, self.args.nembed_ctx)
        attended_entity_embeddings = torch.sum(torch.mul(entity_embeddings, lang_prob), 1)

        # concat attended_entity_embeddings and uttr_emb
        attended_entity_embeddings = attended_entity_embeddings.unsqueeze(1).expand(bsz, utterance_len, self.args.nembed_ctx)
        concatenated_uttr_emb = torch.cat([attended_entity_embeddings, uttr_emb], 2)

        # run decoder
        mem_h = mem_h.unsqueeze(0)
        dec_hs, _ = self.decoder_reader(concatenated_uttr_emb, mem_h)

        return dec_hs

    def unembed_sentence(self, dec_hs):
        # convert to the embed space
        out_emb = self.hid2output(dec_hs)
        # unembed
        out = F.linear(out_emb, self.word_embed.weight)
        return out

    def forward_encoder(self, utterance, hid_idx, enc_init_h=None):
        # embed
        uttr_emb = self.embed_sentence(utterance)

        # encode sentence and turn
        enc_hs, enc_last_h = self.encode_sentence(uttr_emb, hid_idx, enc_init_h)

        return enc_hs, enc_last_h

    def forward_decoder(self, entity_embeddings, utterance, mem_h):
        # embed
        uttr_emb = self.embed_sentence(utterance)
        # run decoder
        dec_hs = self.decode_sentence(entity_embeddings, uttr_emb, mem_h)
        # unembed
        out = self.unembed_sentence(dec_hs)
        return out

    def selection(self, selectable_entity_embeddings, mem_h):
        num_selectable = selectable_entity_embeddings.size(1)

        mem_h_expand = mem_h.unsqueeze(1).expand(1, num_selectable, self.args.nhid_lang)

        selectable_entity_embeddings = torch.cat([selectable_entity_embeddings, mem_h_expand], 2)

        select_logits = self.sel_attn(selectable_entity_embeddings)

        return select_logits.squeeze(2)
    
    def forward(self, batch_input):
        """
            - Input: batch_input
                turn
                    1. all_entities: shape (bsz, max_ent_each_turn, num_frames, dim_ent)
                    2. all_entity_binary_features: (bsz, max_ent_each_turn, num_frames, num_bin)
                    3. selectable_entities: (bsz, num_selectable, num_frames, dim_ent)
                    4. selectable_entity_binary_features: (bsz, num_selectable, num_frames, num_bin)
                    5. utterances: num_utterances * (bsz, utterance_length)

            - Output: batch_output
                turn
                    1. token_logits: (bsz, utterances_length, len(self.word_dict))
                    2. select_logits: (bsz, num_selectable)
        """
        num_turns = len(batch_input)
        bsz = batch_input[0]["all_entities"].size(0)

        batch_output = []

        # initialize mem_h
        mem_h = self._zero(bsz, self.args.nhid_lang)

        for turn in range(num_turns):
            turn_input = batch_input[turn]
            num_utterances = len(turn_input["utterances"])
            num_selectable = turn_input["selectable_entities"].size(1)
            max_ent_each_turn = turn_input["all_entities"].size(1)

            turn_output = {}

            selectable_entity_embeddings = self.ctx_encoder(turn_input["selectable_entities"], turn_input["selectable_entity_binary_features"]) # shape (bsz, num_ent, args.nembed_ctx)
            all_entity_embeddings = self.ctx_encoder(turn_input["all_entities"], turn_input["all_entity_binary_features"]) # shape (bsz, num_ent, args.nembed_ctx)

            uttr_token_logits = []
            for ui in range(num_utterances):
                # forward decoder
                token_logits = self.forward_decoder(all_entity_embeddings, turn_input["utterances"][ui], mem_h)
                uttr_token_logits.append(token_logits)

                # encode utterance
                utterance = turn_input["utterances"][ui]
                hidden_idxs = turn_input["hidden_idxs"][ui]
                enc_hs, enc_last_h = self.forward_encoder(utterance, hidden_idxs)

                # run through memory
                mem_h = self.memory(enc_last_h, mem_h)

            turn_output["token_logits"] = torch.cat(uttr_token_logits, dim=1)

            # predict target
            mem_h_expand = mem_h.unsqueeze(1).expand(bsz, num_selectable, self.args.nhid_lang)
            selectable_entity_embeddings = torch.cat([selectable_entity_embeddings, mem_h_expand], 2)
            select_logits = self.sel_attn(selectable_entity_embeddings)
            turn_output["select_logits"] = select_logits.squeeze(2)
            
            batch_output.append(turn_output)
            
        return batch_output


    def read(self, all_entity_embeddings, input_utterance, mem_h, prefix_token='THEM:'):
        bsz = 1

        # Add a 'THEM:' token to the start of the message
        if prefix_token:
            prefix = self.word2var(prefix_token).unsqueeze(0)
            input_utterance = torch.cat([prefix, input_utterance], dim=1)

        inpt_emb = self.embed_sentence(input_utterance)

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

    def write(self, all_entity_embeddings, mem_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):
        bsz = 1
        max_ent_each_turn = all_entity_embeddings.size(1)

        # autoregress starting from start_token
        current_token = self.word2var(start_token)

        output_utterance = [current_token.unsqueeze(0)]
        output_logprobs = []
        output_lang_hs = []

        # compute attention for decoder
        expand_mem_h = mem_h.unsqueeze(1).expand(bsz, max_ent_each_turn, self.args.nhid_lang)
        lang_logit = self.lang_attn(torch.cat([all_entity_embeddings, expand_mem_h], 2))
        lang_prob = F.softmax(lang_logit, dim=1).expand(bsz, max_ent_each_turn, self.args.nembed_ctx)
        attended_entity_embeddings = torch.sum(torch.mul(all_entity_embeddings, lang_prob), 1)

        # initialize decoder state
        dec_h = mem_h

        for _ in range(max_words):
            # embed
            uttr_emb = self.embed_sentence(current_token)

             # concat attended_entity_embeddings and uttr_emb
            concatenated_uttr_emb = torch.cat([attended_entity_embeddings, uttr_emb], 1)

            # run decoder
            dec_h = self.decoder_writer(concatenated_uttr_emb, dec_h)

            if self.word_dict.get_word(current_token.item()) in stop_tokens:
                break

            # convert to the embed space
            out_emb = self.hid2output(dec_h)

            # unembed
            token_logits = F.linear(out_emb, self.word_embed.weight)
            token_scores = token_logits.div(temperature)
            token_scores = token_scores.sub(token_scores.max().item()).squeeze(0)

            mask = Variable(self.special_token_mask)
            token_scores = token_scores.add(mask)

            token_probs = F.softmax(token_scores, dim=0)
            token_logprobs = F.log_softmax(token_scores, dim=0)
            current_token = token_probs.multinomial(1).detach()
            output_utterance.append(current_token.unsqueeze(0))
            token_logprob = token_logprobs.gather(0, current_token)
            output_logprobs.append(token_logprob)

        output_utterance = torch.cat(output_utterance, 1)

        # read the output utterance
        mem_h = self.read(all_entity_embeddings, output_utterance, mem_h, prefix_token=None)

        return output_utterance.squeeze(0).unsqueeze(1), output_logprobs, mem_h
