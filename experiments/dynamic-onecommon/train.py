import argparse
import copy
import sys
import time
import random
import itertools
import re
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import models
import utils
from domain import get_domain

# simple fix for dataparallel that allows access to class attributes
class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def main():
    parser = argparse.ArgumentParser(description='Training script for Dynamic-OneCommon')

    # Dataset arguments
    parser.add_argument('--data', type=str, default='data/dynamic-onecommon',
        help='location of the data corpora')
    parser.add_argument('--domain', type=str, default='dynamic',
        help='domain for the dialogue')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')

    # Model arguments (hyperparameters)
    parser.add_argument('--model_type', type=str, default='rnn_model',
        help='type of model to use', choices=models.get_model_names())
    parser.add_argument('--ctx_encoder_type', type=str, default='mlp_encoder',
        help='type of context encoder to use', choices=models.get_ctx_encoder_names())
    parser.add_argument('--share_attn', action='store_true', default=False,
        help='share selection and language decoder')
    parser.add_argument('--nembed_word', type=int, default=128,
        help='size of word embeddings')
    parser.add_argument('--nhid_rel', type=int, default=128,
        help='size of the hidden state for the language module')
    parser.add_argument('--nembed_ctx', type=int, default=128,
        help='size of context embeddings')
    parser.add_argument('--nhid_lang', type=int, default=128,
        help='size of the hidden state for the language module')
    parser.add_argument('--nhid_strat', type=int, default=128,
        help='size of the hidden state for the strategy module')
    parser.add_argument('--nhid_attn', type=int, default=128,
        help='size of the hidden state for the attention module')

    # Training arguments (hyperparameters)
    parser.add_argument('--lr', type=float, default=0.001,
        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.5,
        help='gradient clipping')
    parser.add_argument('--dropout', type=float, default=0.5,
        help='dropout rate in embedding layer')
    parser.add_argument('--init_range', type=float, default=0.01,
        help='initialization range')
    parser.add_argument('--max_epoch', type=int, default=10,
        help='max number of epochs')
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--lang_weight', type=float, default=1.0,
        help='language loss weight')
    parser.add_argument('--sel_weight', type=float, default=1.0,
        help='selection loss weight')

    # Training arguments
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--for_multitask', action='store_true', default=False,
        help='for multitask training with OneCommon and Dynamic-OneCommon')
    parser.add_argument('--from_pretrained', action='store_true', default=False,
        help='initialize modules with pretrained parameters')
    parser.add_argument('--pretrained_model_file', type=str, default="rnn_ref_model_for_multitask",
        help='pretrained model file')
    parser.add_argument('--tensorboard_log', action='store_true', default=False,
        help='log training with tensorboard')

    # Ablation arguments
    parser.add_argument('--abl_features', nargs='*', default=[],
        help='ablate specified features')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=0,
        help='random seed')
    parser.add_argument('--repeat_train', action='store_true', default=False,
        help='repeat training 5 times with different random seeds')
    parser.add_argument('--model_dir', type=str,  default='saved_models',
        help='path to save the final model')
    parser.add_argument('--model_file', type=str,  default='tmp.th',
        help='path to save the final model')

    args = parser.parse_args()

    if args.repeat_train:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [args.seed]

    for seed in seeds:
        device = utils.use_cuda(args.cuda)
        utils.set_seed(seed)

        domain = get_domain(args.domain)
        model_ty = models.get_model_type(args.model_type)

        if args.from_pretrained:
            pretrained_model = utils.load_model(args.model_dir, args.pretrained_model_file + '_' + str(seed) + '.th')
            word_dict = pretrained_model.word_dict
        else:
            word_dict = None

        corpus = model_ty.corpus_ty(domain, args.data, train='train_{}.json'.format(seed), valid='valid_{}.json'.format(seed), test='test_{}.json'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True, word_dict=word_dict, abl_features=args.abl_features)

        model = model_ty(corpus.word_dict, args)

        if args.from_pretrained:
            model.initialize_from(pretrained_model)
            
        if args.cuda:
            model.cuda()
            #model = MyDataParallel(model)

        engine = model_ty.engine_ty(model, args, train=True, verbose=True)
        best_valid_loss, best_model = engine.train(corpus)

        utils.save_model(best_model, args.model_dir, args.model_file + '_' + str(seed) + '.th')

        del word_dict, corpus
        del model, engine
        del best_model, best_valid_loss
        if args.from_pretrained:
            del pretrained_model

if __name__ == '__main__':
    main()
