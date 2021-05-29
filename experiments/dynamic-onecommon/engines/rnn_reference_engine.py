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


class RnnReferenceEngine(EngineBase):
    def __init__(self, model, args, train=False, verbose=False):
        super(RnnReferenceEngine, self).__init__(model, args, train, verbose)

    def _forward(self, batch):
        ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, _, _, _, sel_idx, _ = batch

        out, ref_out, sel_out = self.model.forward(ctx, inpt, ref_inpt, sel_idx)

        lang_loss = self.crit(out, tgt)

        if ref_inpt is not None:
            ref_tgt = ref_tgt.float()
            ref_loss = self.ref_crit(ref_out, ref_tgt)
            ref_correct = ((ref_out > 0).long() == ref_tgt.long()).sum().item()
            ref_total = ref_tgt.size(0) * ref_tgt.size(1) * ref_tgt.size(2)
        else:
            ref_loss = None
            ref_correct = 0
            ref_total = 0

        sel_loss = self.sel_crit(sel_out, sel_tgt)
        sel_correct = (sel_out.max(dim=1)[1] == sel_tgt).sum().item()
        sel_total = sel_out.size(0)

        return lang_loss, ref_loss, ref_correct, ref_total, sel_loss, sel_correct, sel_total, out, ref_out, sel_out

