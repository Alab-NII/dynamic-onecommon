import sys
from collections import defaultdict
import pdb

import numpy as np
import torch
from torch import optim, autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils
from dialog import DialogLogger
import domain
from engines import Criterion
import math
from collections import Counter

from nltk.parse import CoreNLPParser, CoreNLPDependencyParser



class Agent(object):
    """ Agent's interface. """
    def feed_context(self, ctx):
        pass

    def read(self, inpt):
        pass

    def write(self):
        pass

    def choose(self):
        pass

    def update(self, agree, reward, choice):
        pass

    def get_attention(self):
        return None


class RnnAgent(Agent):
    def __init__(self, model, args, binary_dict, name='Alice', train=False):
        super(RnnAgent, self).__init__()
        self.model = model
        self.args = args
        self.train_args = model.args
        self.binary_dict = binary_dict
        self.name = name
        self.human = False
        self.domain = domain.get_domain(args.domain)
        self.train = train
        if train:
            self.model.train()
            self.opt = optim.RMSprop(
            self.model.parameters(),
            lr=args.rl_lr,
            momentum=self.args.momentum)
            self.all_rewards = []
            self.t = 0
        else:
            self.model.eval()

    def _encode(self, inpt, dictionary):
        encoded = torch.Tensor(dictionary.w2i(inpt)).long().unsqueeze(0)
        return encoded

    def _decode(self, out, dictionary):
        return dictionary.i2w(out.data.squeeze(1).cpu())

    def feed_context(self, context, turn=0):
        # shape (bsz, num_ent, num_frames, dim_ent)
        all_entities = [
                            [
                                [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                for frame_id in range(len(entity["xs"]))
                            ]
                            for entity in context[turn]["all_entities"]
                        ]

        # shape (bsz, num_ent, num_frames, domain.num_bin)
        all_entity_binary_features = [
                                            [
                                               [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                for frame_id in range(len(entity["visible"]))
                                            ]
                                            for entity in context[turn]["all_entities"]
                                      ]

        # shape: turn -> (bsz, num_selectable, num_frames, dim_ent)
        selectable_entities = [
                                    [
                                        [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                        for frame_id in range(len(entity["xs"]))
                                    ]
                                    for entity in context[turn]["selectable_entities"]
                               ]

        # shape: turn -> (bsz, num_selectable, num_frames, num_bin)
        selectable_entity_binary_features = [
                                                [
                                                    [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                    for frame_id in range(len(entity["visible"]))
                                                ]
                                                for entity in context[turn]["selectable_entities"]
                                            ]

        self.selectable_entity_ids = context[turn]["selectable_entity_ids"]
        self.all_entity_ids = context[turn]["all_entity_ids"]
        num_frames = len(all_entity_binary_features[0])

        if turn > 0:
            if self.args.refresh_each_turn:
                num_frames = 11
                all_entities = [entity[-num_frames:] for entity in all_entities]
                all_entity_binary_features = [entity[-num_frames:] for entity in all_entity_binary_features]
                selectable_entities = [entity[-num_frames:] for entity in selectable_entities]
                selectable_entity_binary_features = [entity[-num_frames:] for entity in selectable_entity_binary_features]
            elif "previous_selected" not in self.train_args.abl_features:
                # update previous_selected in all_entity_binary_features
                for entity_idx, entity_id in enumerate(self.all_entity_ids):
                    if entity_id == self.previous_selected:
                        for frame_idx in range(num_frames):
                            all_entity_binary_features[entity_idx][frame_idx][3] = self.binary_dict.get_idx('<previous_selected>')

                # update previous_selected in all_entity_binary_features
                for entity_idx, entity_id in enumerate(self.selectable_entity_ids):
                    if entity_id == self.previous_selected:
                        for frame_idx in range(num_frames):
                            selectable_entity_binary_features[entity_idx][frame_idx][3] = self.binary_dict.get_idx('<previous_selected>')

        all_entities = torch.Tensor(all_entities).float().unsqueeze(0)
        all_entity_binary_features = torch.Tensor(all_entity_binary_features).long().unsqueeze(0)
        selectable_entities = torch.Tensor(selectable_entities).float().unsqueeze(0)
        selectable_entity_binary_features = torch.Tensor(selectable_entity_binary_features).long().unsqueeze(0)

        self.all_entity_embeddings = self.model.ctx_encoder(all_entities, all_entity_binary_features)

        self.selectable_entity_embeddings = self.model.ctx_encoder(selectable_entities, selectable_entity_binary_features)

        # if init
        if turn == 0 or self.args.refresh_each_turn:
            self.lang_h = self.model._zero(1, self.model.args.nhid_lang)

    def feed_partner_context(self, partner_context):
        pass

    def read(self, input_utterance):
        input_utterance = self._encode(input_utterance, self.model.word_dict)
        lang_hs, self.lang_h = self.model.read(self.all_entity_embeddings, input_utterance, self.lang_h.unsqueeze(0))
        self.lang_h = self.lang_h.squeeze(0)

    def write(self, max_words=100):
        output_utterance, output_logprobs, self.lang_h, output_lang_hs = self.model.write(self.all_entity_embeddings, self.lang_h, 
                                                            max_words, self.args.temperature)

        # remove 'YOU:'
        output_utterance = output_utterance.narrow(0, 1, output_utterance.size(0) - 1)
        return self._decode(output_utterance, self.model.word_dict)

    def _choose(self, sample=False):
        select_logits = self.model.selection(self.selectable_entity_embeddings, self.lang_h)

        prob = F.softmax(select_logits, dim=1)
        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=1).gather(1, idx)
        else:
            _, idx = prob.max(1, keepdim=True)
            logprob = None

        self.previous_selected = self.selectable_entity_ids[idx.item()]

        # Pick only your choice
        return self.previous_selected, idx.item(), prob.gather(1, idx), logprob

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _, _ = self._choose(sample=False)
        else:
            choice, _, _, logprob = self._choose(sample=True)
            self.logprobs.append(logprob)

        return choice

    def update(self, agree, reward, choice=None):
        if not self.train:
            return

        self.t += 1
        if len(self.logprobs) == 0:
            return

        self.all_rewards.append(reward)

        r = (reward - np.mean(self.all_rewards)) / max(1e-4, np.std(self.all_rewards))
        g = Variable(torch.zeros(1, 1).fill_(r))
        rewards = []
        for _ in self.logprobs:
            rewards.insert(0, g)
            g = g * self.args.gamma

        loss = 0
        for lp, r in zip(self.logprobs, rewards):
            loss -= lp * r

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.rl_clip)
        if self.args.visual and self.t % 10 == 0:
            self.model_plot.update(self.t)
            self.agree_plot.update('agree', self.t, int(agree))
            self.reward_plot.update('reward', self.t, reward)
            self.reward_plot.update('partner_reward', self.t, partner_reward)
            self.agree_reward_plot.update('reward', self.t, reward_agree)
            self.agree_reward_plot.update('partner_reward', self.t, partner_reward_agree)
            self.loss_plot.update('loss', self.t, loss.data[0][0])

        self.opt.step()


class HierarchicalAgent(RnnAgent):
    def __init__(self, model, args, binary_dict, name='Alice', train=False):
        super(HierarchicalAgent, self).__init__(model, args, binary_dict, name=name, train=train)

    def feed_context(self, context, turn=0):
        # shape (bsz, num_ent, num_frames, dim_ent)
        all_entities = [
                            [
                                [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                for frame_id in range(len(entity["xs"]))
                            ]
                            for entity in context[turn]["all_entities"]
                        ]

        # shape (bsz, num_ent, num_frames, domain.num_bin)
        all_entity_binary_features = [
                                            [
                                               [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                for frame_id in range(len(entity["visible"]))
                                            ]
                                            for entity in context[turn]["all_entities"]
                                      ]

        # shape: turn -> (bsz, num_selectable, num_frames, dim_ent)
        selectable_entities = [
                                    [
                                        [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                        for frame_id in range(len(entity["xs"]))
                                    ]
                                    for entity in context[turn]["selectable_entities"]
                               ]

        # shape: turn -> (bsz, num_selectable, num_frames, num_bin)
        selectable_entity_binary_features = [
                                                [
                                                    [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                    for frame_id in range(len(entity["visible"]))
                                                ]
                                                for entity in context[turn]["selectable_entities"]
                                            ]

        self.selectable_entity_ids = context[turn]["selectable_entity_ids"]
        self.all_entity_ids = context[turn]["all_entity_ids"]
        num_frames = len(all_entity_binary_features[0])

        if turn > 0:
            if self.args.refresh_each_turn:
                num_frames = 11
                all_entities = [entity[-num_frames:] for entity in all_entities]
                all_entity_binary_features = [entity[-num_frames:] for entity in all_entity_binary_features]
                selectable_entities = [entity[-num_frames:] for entity in selectable_entities]
                selectable_entity_binary_features = [entity[-num_frames:] for entity in selectable_entity_binary_features]
            elif "previous_selected" not in self.train_args.abl_features:
                # update previous_selected in all_entity_binary_features
                for entity_idx, entity_id in enumerate(self.all_entity_ids):
                    if entity_id == self.previous_selected:
                        for frame_idx in range(num_frames):
                            all_entity_binary_features[entity_idx][frame_idx][3] = self.binary_dict.get_idx('<previous_selected>')

                # update previous_selected in all_entity_binary_features
                for entity_idx, entity_id in enumerate(self.selectable_entity_ids):
                    if entity_id == self.previous_selected:
                        for frame_idx in range(num_frames):
                            selectable_entity_binary_features[entity_idx][frame_idx][3] = self.binary_dict.get_idx('<previous_selected>')

        all_entities = torch.Tensor(all_entities).float().unsqueeze(0)
        all_entity_binary_features = torch.Tensor(all_entity_binary_features).long().unsqueeze(0)
        selectable_entities = torch.Tensor(selectable_entities).float().unsqueeze(0)
        selectable_entity_binary_features = torch.Tensor(selectable_entity_binary_features).long().unsqueeze(0)

        self.all_entity_embeddings = self.model.ctx_encoder(all_entities, all_entity_binary_features)

        self.selectable_entity_embeddings = self.model.ctx_encoder(selectable_entities, selectable_entity_binary_features)

        # if init
        if turn == 0 or self.args.refresh_each_turn:
            self.mem_h = self.model._zero(1, self.model.args.nhid_lang)

    def read(self, input_utterance):
        input_utterance = self._encode(input_utterance, self.model.word_dict)
        self.mem_h = self.model.read(self.all_entity_embeddings, input_utterance, self.mem_h)

    def write(self, max_words=100):
        output_utterance, output_logprobs, self.mem_h = self.model.write(self.all_entity_embeddings, self.mem_h, 
                                                            max_words, self.args.temperature)

        # remove 'YOU:'
        output_utterance = output_utterance.narrow(0, 1, output_utterance.size(0) - 1)
        return self._decode(output_utterance, self.model.word_dict)

    def _choose(self, sample=False):
        select_logits = self.model.selection(self.selectable_entity_embeddings, self.mem_h)

        prob = F.softmax(select_logits, dim=1)
        if sample:
            idx = prob.multinomial(1).detach()
            logprob = F.log_softmax(choice_logit, dim=1).gather(1, idx)
        else:
            _, idx = prob.max(1, keepdim=True)
            logprob = None

        self.previous_selected = self.selectable_entity_ids[idx.item()]

        # Pick only your choice
        return self.previous_selected, idx.item(), prob.gather(1, idx), logprob

    def choose(self):
        if self.args.eps < np.random.rand():
            choice, _, _, _ = self._choose(sample=False)
        else:
            choice, _, _, logprob = self._choose(sample=True)
            self.logprobs.append(logprob)

        return choice