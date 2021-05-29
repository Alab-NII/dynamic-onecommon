import argparse
import random
import time
import itertools
import sys
import copy
import re
import os
import shutil

import pdb

from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.nn.modules.loss import _Loss
import numpy as np

from engines import EngineBase, Criterion
import utils


class RnnDialogEngine(EngineBase):
    def __init__(self, model, args, device=None, verbose=False):
        super(RnnDialogEngine, self).__init__(model, args, verbose=verbose)

    def _forward(self, batch):
        num_turns = len(batch["input"])

        batch["output"] = self.model.forward(batch["input"])

        # compute language loss
        vocab_size = batch["output"][0]["token_logits"].size(2)
        batch_token_logits = torch.cat([batch["output"][turn]["token_logits"] for turn in range(num_turns)], 1)
        batch_token_logits = batch_token_logits.view(-1, vocab_size)
        batch_target_utterances = torch.cat([batch["target"][turn]["utterances"] for turn in range(num_turns)], 1)
        batch_target_utterances = batch_target_utterances.view(-1)
        lang_loss = self.crit(batch_token_logits, batch_target_utterances)

        # compute selection loss
        num_selectable_ent = batch["output"][0]["select_logits"].size(1)
        batch_select_logits = torch.cat([batch["output"][turn]["select_logits"].unsqueeze(1) for turn in range(num_turns)], 1)
        batch_select_logits = batch_select_logits.view(-1, num_selectable_ent)
        batch_target_selected = torch.cat([batch["target"][turn]["selected"].unsqueeze(1) for turn in range(num_turns)], 1)
        batch_target_selected = batch_target_selected.view(-1)
        sel_loss = self.sel_crit(batch_select_logits, batch_target_selected)
        sel_correct = (batch_select_logits.max(dim=1)[1] == batch_target_selected).sum().item()
        sel_predict = batch_target_selected.size(0)

        return lang_loss, sel_loss, sel_correct, sel_predict

    def train_batch(self, batch):
        lang_loss, sel_loss, sel_correct, sel_predict = self._forward(batch)

        # default
        loss = self.args.lang_weight * lang_loss + self.args.sel_weight * sel_loss

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.opt.step()

        return lang_loss.item(), sel_loss.item(), sel_correct, sel_predict

    def valid_batch(self, batch):
        with torch.no_grad():
            lang_loss, sel_loss, sel_correct, sel_predict = self._forward(batch)

        return lang_loss.item(), sel_loss.item(), sel_correct, sel_predict

    def test_batch(self, batch):
        with torch.no_grad():
            lang_loss, sel_loss, sel_correct, sel_predict = self._forward(batch)

        return lang_loss.item(), sel_loss.item(), sel_correct, sel_predict

    def train_pass(self, trainset, trainset_stats):
        '''
        basic implementation of one training pass
        '''
        self.model.train()

        total_lang_loss, total_select_loss, total_select_correct, total_select_predict = 0, 0, 0, 0
        start_time = time.time()

        for batch in trainset:
            lang_loss, sel_loss, sel_correct, sel_predict = self.train_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select_predict += sel_predict

        total_lang_loss /= len(trainset)
        total_select_loss /= len(trainset)
        time_elapsed = time.time() - start_time
        return total_lang_loss, total_select_loss, total_select_correct / total_select_predict, time_elapsed

    def valid_pass(self, validset, validset_stats):
        '''
        basic implementation of one validation pass
        '''
        self.model.eval()

        total_lang_loss, total_select_loss, total_select_correct, total_select_predict = 0, 0, 0, 0
        for batch in validset:
            lang_loss, sel_loss, sel_correct, sel_predict = self.valid_batch(batch)
            total_lang_loss += lang_loss
            total_select_loss += sel_loss
            total_select_correct += sel_correct
            total_select_predict += sel_predict

        total_lang_loss /= len(validset)
        total_select_loss /= len(validset)
        return total_lang_loss, total_select_loss, total_select_correct / total_select_predict

    def iter(self, epoch, lr, traindata, validdata):
        trainset, trainset_stats = traindata
        validset, validset_stats = validdata

        train_lang_loss, train_select_loss,  train_select_accuracy, train_time = self.train_pass(trainset, trainset_stats)
        valid_lang_loss, valid_select_loss, valid_select_accuracy = self.valid_pass(validset, validset_stats)

        if self.verbose:
            print('| epoch %03d | trainlangloss(scaled) %.6f | trainlangppl %.6f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_lang_loss * self.args.lang_weight, np.exp(train_lang_loss), train_time, lr))
            print('| epoch %03d | trainselectloss(scaled) %.6f | trainselectaccuracy %.4f | s/epoch %.2f | lr %0.8f' % (
                epoch, train_select_loss * self.args.sel_weight, train_select_accuracy, train_time, lr))
            print('| epoch %03d | validlangloss %.6f | validlangppl %.8f' % (
                epoch, valid_lang_loss, np.exp(valid_lang_loss)))
            print('| epoch %03d | validselectloss %.6f | validselectaccuracy %.4f' % (
                epoch, valid_select_loss, valid_select_accuracy))

        if self.args.tensorboard_log:
            info = {'Train_Lang_Loss': train_lang_loss,
                'Train_Select_Loss': train_select_loss,
                'Valid_Lang_Loss': valid_lang_loss,
                'Valid_Select_Loss': valid_select_loss,
                'Valid_Select_Accuracy': valid_select_accuracy}
            for tag, value in info.items():
                self.logger.scalar_summary(tag, value, epoch)

            for tag, value in self.model.named_parameters():
                if value.grad is None:
                    continue
                tag = tag.replace('.', '/')
                self.logger.histo_summary(tag, value.data.cpu().numpy(), epoch)
                self.logger.histo_summary(
                    tag + '/grad', value.grad.data.cpu().numpy(), epoch)

        return valid_lang_loss, valid_select_loss

    def combine_loss(self, lang_loss, select_loss):
        return lang_loss * int(self.args.lang_weight > 0) + select_loss * int(self.args.sel_weight > 0)

    def train(self, corpus):
        best_model, best_combined_valid_loss = copy.deepcopy(self.model), 1e100
        validdata = corpus.valid_dataset(self.args.bsz)
        
        for epoch in range(1, self.args.max_epoch + 1):
            traindata = corpus.train_dataset(self.args.bsz)
            valid_lang_loss, valid_select_loss = self.iter(epoch, self.args.lr, traindata, validdata)

            combined_valid_loss = self.combine_loss(valid_lang_loss, valid_select_loss)
            if combined_valid_loss < best_combined_valid_loss:
                print("update best model: validlangloss %.8f | validselectloss %.8f" % 
                    (valid_lang_loss, valid_select_loss))
                best_combined_valid_loss = combined_valid_loss
                best_model = copy.deepcopy(self.model)
                best_model.flatten_parameters()

        return best_combined_valid_loss, best_model

