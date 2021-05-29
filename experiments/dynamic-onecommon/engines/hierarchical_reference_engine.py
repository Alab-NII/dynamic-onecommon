import time
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

from engines import EngineBase, Criterion
import utils


class HierarchicalReferenceEngine(EngineBase):
    def __init__(self, model, args, train=False, verbose=False):
        super(HierarchicalReferenceEngine, self).__init__(model, args, train, verbose)

    def _forward(self, batch, normalize_loss=True):
        batch_ctx, batch_inpts, batch_tgts, batch_ref_inpts, batch_ref_tgts, batch_sel_tgt, batch_scenario_id, \
            batch_real_ids, batch_agent_idx, batch_chat_id, batch_sel_idx, batch_hid_idxs = batch

        batch_outs, batch_ref_outs, batch_sel_out = self.model.forward(batch_ctx, batch_inpts, batch_ref_inpts, batch_hid_idxs, \
                                    batch_sel_idx)

        # compute language loss
        batch_outs = torch.cat(batch_outs, 0)
        batch_tgts = torch.cat(batch_tgts, 0)
        batch_lang_loss = self.crit(batch_outs, batch_tgts)

        # compute reference loss
        if batch_ref_inpts is not None:
            batch_ref_tgts = batch_ref_tgts.float()
            batch_ref_loss = self.ref_crit(batch_ref_outs, batch_ref_tgts)
            batch_ref_correct = ((batch_ref_outs > 0).long() == batch_ref_tgts.long()).sum().item()
            batch_ref_total = batch_ref_tgts.size(0) * batch_ref_tgts.size(1) * batch_ref_tgts.size(2)
        else:
            batch_ref_loss = None
            batch_ref_correct = 0
            batch_ref_total = 0

        # compute selection loss
        batch_sel_loss = self.sel_crit(batch_sel_out, batch_sel_tgt)
        batch_sel_correct = (batch_sel_out.max(dim=1)[1] == batch_sel_tgt).sum().item()
        batch_sel_total = batch_sel_out.size(0)

        return batch_lang_loss, batch_ref_loss, batch_ref_correct, batch_ref_total, batch_sel_loss, \
               batch_sel_correct, batch_sel_total, batch_outs, batch_ref_outs, batch_sel_out
