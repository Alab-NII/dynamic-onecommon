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
from engines.rnn_dialog_engine import RnnDialogEngine
from models.ctx_encoder import *


class RnnDialogModel(nn.Module):
    corpus_ty = data.WordCorpus
    engine_ty = RnnDialogEngine

    def __init__(self, word_dict, args):
        super(RnnDialogModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.num_selectable = domain.num_selectable()

        # define modules:
        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        ctx_encoder_ty = models.get_ctx_encoder_type(args.ctx_encoder_type)
        self.ctx_encoder = ctx_encoder_ty(domain, args)

        self.reader = nn.GRU(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True,
            batch_first=True)

        self.writer = nn.GRUCell(
            input_size=args.nembed_word,
            hidden_size=args.nhid_lang,
            bias=True)

        self.hid2output = nn.Sequential(
            nn.Linear(args.nembed_ctx + args.nhid_lang, args.nembed_word),
            nn.Tanh(),
            nn.Dropout(args.dropout),
            nn.Linear(args.nembed_word, args.nembed_word),
            nn.Tanh(),
            nn.Dropout(args.dropout))

        self.lang_attn = nn.Sequential(
            torch.nn.Linear(args.nembed_ctx + args.nhid_lang, args.nhid_attn),
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
                torch.nn.Linear(args.nembed_ctx + args.nhid_lang, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, args.nhid_attn),
                nn.Tanh(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, 1))

        # tie the weights between reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.dropout = nn.Dropout(args.dropout)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        init_rnn(self.reader, args.init_range)
        init_cont(self.hid2output, args.init_range)
        init_cont(self.sel_attn, args.init_range)
        if not args.share_attn:
            init_cont(self.lang_attn, args.init_range)

    def initialize_from(self, pretrained_model):
        self.ctx_encoder.property_encoder = copy.deepcopy(pretrained_model.ctx_encoder.property_encoder)
        self.ctx_encoder.relation_encoder = copy.deepcopy(pretrained_model.ctx_encoder.relation_encoder)

        self.word_embed = copy.deepcopy(pretrained_model.word_embed)
        self.reader.weight_ih_l0 = pretrained_model.reader.weight_ih_l0
        self.reader.weight_hh_l0 = pretrained_model.reader.weight_hh_l0
        self.reader.bias_ih_l0 = pretrained_model.reader.bias_ih_l0
        self.reader.bias_hh_l0 = pretrained_model.reader.bias_hh_l0
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

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
        self.reader.flatten_parameters()

    def embed_dialogue(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def selection(self, selectable_entity_embeddings, lang_h):
        num_selectable = selectable_entity_embeddings.size(1)

        lang_h_expand = lang_h.unsqueeze(1).expand(1, num_selectable, self.args.nhid_lang)

        selectable_entity_embeddings = torch.cat([selectable_entity_embeddings, lang_h_expand], 2)

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
                    5. utterances: (bsz, utterances_length)

            - Output: batch_output
                turn
                    1. token_logits: (bsz, utterances_length, len(self.word_dict))
                    2. select_logits: (bsz, num_selectable)
        """
        num_turns = len(batch_input)
        bsz = batch_input[0]["all_entities"].size(0)

        batch_output = []

        lang_h = self._zero(1, bsz, self.args.nhid_lang)

        for turn in range(num_turns):
            turn_input = batch_input[turn]
            utterances_length = turn_input["utterances"].size(1)
            num_selectable = turn_input["selectable_entities"].size(1)
            max_ent_each_turn = turn_input["all_entities"].size(1)

            turn_output = {}            

            selectable_entity_embeddings = self.ctx_encoder(turn_input["selectable_entities"], turn_input["selectable_entity_binary_features"]) # shape (bsz, num_ent, args.nembed_ctx)
            all_entity_embeddings = self.ctx_encoder(turn_input["all_entities"], turn_input["all_entity_binary_features"]) # shape (bsz, num_ent, args.nembed_ctx)
            utterances_embeddings = self.embed_dialogue(turn_input["utterances"]) # (bsz, utterances_length, args.nembed_word)

            """
                output_embeddings: shape (bsz, utterances_length, args.nhid_lang)
                lang_h: (1, bsz, args.nhid_lang)
            """
            self.reader.flatten_parameters()
            output_embeddings, lang_h = self.reader(utterances_embeddings, lang_h)

            expanded_all_entity_embeddings = all_entity_embeddings.unsqueeze(1).expand(bsz, utterances_length, max_ent_each_turn, self.args.nembed_ctx)
            expanded_output_embeddings = output_embeddings.unsqueeze(2).expand(bsz, utterances_length, max_ent_each_turn, self.args.nhid_lang)
            lang_logit = self.lang_attn(torch.cat([expanded_all_entity_embeddings, expanded_output_embeddings], 3))
            lang_prob = F.softmax(lang_logit, dim=2).expand(bsz, utterances_length, max_ent_each_turn, self.args.nembed_ctx)
            attended_all_entity_embeddings = torch.sum(torch.mul(expanded_all_entity_embeddings, lang_prob), 2)
            token_logits = self.hid2output(torch.cat([attended_all_entity_embeddings, output_embeddings], 2))
            token_logits = F.linear(token_logits, self.word_embed.weight)
            turn_output["token_logits"] = token_logits

            # use selection idxs to gather last lang_hs
            last_lang_hs = []
            for bi in range(bsz):
                last_lang_hs.append(output_embeddings[bi][turn_input["selection_idxs"][bi]].unsqueeze(0))
            lang_h = torch.cat(last_lang_hs, 0).unsqueeze(0)

            lang_h_expand = lang_h.squeeze(0).unsqueeze(1).expand(bsz, num_selectable, self.args.nhid_lang)
            selectable_entity_embeddings = torch.cat([selectable_entity_embeddings, lang_h_expand], 2)
            select_logits = self.sel_attn(selectable_entity_embeddings)
            turn_output["select_logits"] = select_logits.squeeze(2)

            batch_output.append(turn_output)

        return batch_output

    def read(self, all_entity_embeddings, input_utterance, lang_h, prefix_token='THEM:'):
        # Add a 'THEM:' token to the start of the message
        prefix = self.word2var(prefix_token).unsqueeze(0)
        input_utterance = torch.cat([prefix, input_utterance], 1)

        utterance_embeddings = self.embed_dialogue(input_utterance)

        lang_hs, lang_h = self.reader(utterance_embeddings, lang_h)

        return lang_hs, lang_h

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def write(self, all_entity_embeddings, lang_h, max_words, temperature,
        start_token='YOU:', stop_tokens=data.STOP_TOKENS):
        max_ent_each_turn = all_entity_embeddings.size(1)

        summed_all_entity_embeddings = torch.sum(all_entity_embeddings, 1).squeeze(1)

        # autoregress starting from start_token
        current_token = self.word2var(start_token)

        output_utterance = [current_token.unsqueeze(0)]
        output_logprobs = []
        output_lang_hs = []
        for _ in range(max_words):
            # embed
            utterance_embeddings = self.embed_dialogue(current_token)

            lang_h = self.writer(utterance_embeddings, lang_h)
            output_lang_hs.append(lang_h)

            if self.word_dict.get_word(current_token.item()) in stop_tokens:
                break
            
            # compute language output            
            expanded_lang_h = lang_h.unsqueeze(1).expand(1, max_ent_each_turn, self.args.nhid_lang)
            lang_logit = self.lang_attn(torch.cat([all_entity_embeddings, expanded_lang_h], 2))
            lang_prob = F.softmax(lang_logit, dim=1).expand(1, max_ent_each_turn, self.args.nhid_lang)
            attended_all_entity_embeddings = torch.sum(torch.mul(all_entity_embeddings, lang_prob), 1)
            token_logits = self.hid2output(torch.cat([attended_all_entity_embeddings, lang_h], 1))
            token_logits = F.linear(token_logits, self.word_embed.weight)

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

        output_utterance = torch.cat(output_utterance, 0)

        return output_utterance, output_logprobs, lang_h, torch.cat(output_lang_hs, 0)
