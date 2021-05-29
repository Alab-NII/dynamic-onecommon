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
    def __init__(self, model, args, train=False, verbose=False):
        super(RnnDialogEngine, self).__init__(model, args, train=train, verbose=verbose)

    def _forward(self, batch):
        batch["output"] = self.model.forward(batch["input"])

        num_turns = len(batch["input"])

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

        return lang_loss, sel_loss, sel_correct, sel_predict, batch["output"]
