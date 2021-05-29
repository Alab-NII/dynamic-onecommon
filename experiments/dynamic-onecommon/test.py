"""
Performs evaluation of the model on the test dataset.
"""

import argparse
import copy
import json
import os
import pdb
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import data
import models
import utils
from engines import Criterion
from domain import get_domain

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns
import pandas as pd
sns.set(font_scale=1.15)

def read_json(path):
    try:
        return json.load(open(path))
    except:
        raise Exception('Error reading JSON from %s' % path)

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

def main():
    parser = argparse.ArgumentParser(description='Testing script for dynamic onecommon')

    # Model parameters
    parser.add_argument('--model_dir', type=str,  default='saved_models',
        help='path to save the final model')
    parser.add_argument('--model_file', type=str,  default='tmp',
        help='path to save the final model')

    # Testing parameters
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--seed', type=int, default=0,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--bleu_n', type=int, default=0,
        help='test ngram bleu')
    parser.add_argument('--refresh_each_turn', action='store_true', default=False,
        help='refresh each turn')

    # Misc parameters
    parser.add_argument('--output_target_selection', action='store_true', default=False,
        help='show errors')
    parser.add_argument('--show_errors', action='store_true', default=False,
        help='show errors')
    parser.add_argument('--repeat_test', action='store_true', default=False,
        help='repeat testing 5 times')

    test_args = parser.parse_args()

    if test_args.repeat_test:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [test_args.seed]

    repeat_select_accuracy = []
    repeat_first_turn_select_accuracy = []
    repeat_later_turn_select_accuracy = []
    repeat_same_target_select_accuracy = []
    repeat_change_target_select_accuracy = []

    for seed in seeds:
        model = utils.load_model(test_args.model_dir, test_args.model_file + '_' + str(seed) + '.th')
        train_args = model.args

        model_target_selection_annotation = {}

        domain = get_domain(train_args.domain)

        utils.set_seed(seed)

        device = utils.use_cuda(test_args.cuda)
        if test_args.cuda:
            model.cuda()
            model = torch.nn.DataParallel(model).module
        else:
            device = torch.device("cpu")
            model.to(device)
        model.eval()

        corpus = model.corpus_ty(domain, train_args.data, train='train_{}.json'.format(seed), valid='valid_{}.json'.format(seed), test='test_{}.json'.format(seed),
            freq_cutoff=train_args.unk_threshold, verbose=True, word_dict=model.word_dict, abl_features=train_args.abl_features)
        engine = model.engine_ty(model, test_args, train=False, verbose=True)

        crit = Criterion(model.word_dict)
        sel_crit = nn.CrossEntropyLoss()

        if test_args.bleu_n > 0:
            test_args.bsz = 1

        testset, testset_stats = corpus.test_dataset(test_args.bsz, device=device)
        test_lang_loss, test_select_loss, test_select_correct, test_select_total = 0, 0, 0, 0

        """
            Variables to keep track of the results for analysis
        """
        first_turn_correct = 0
        first_turn_total = 0
        later_turn_correct = 0
        later_turn_total = 0
        same_target_correct = 0
        same_target_total = 0
        change_target_correct = 0
        change_target_total = 0

        for batch in testset:
            num_turns = len(batch["input"])
            current_bsz = batch["input"][0]["all_entities"].size(0)

            if test_args.refresh_each_turn:
                batch["output"] = []
                for turn in range(num_turns):
                    turn_batch = {}
                    turn_batch["meta"] = batch["meta"]

                    num_frames = 11
                    turn_batch["input"] = []
                    turn_info = {}
                    turn_info["all_entities"] = batch["input"][turn]['all_entities'][:,:,-num_frames:,:]
                    turn_info["all_entity_binary_features"] = batch["input"][turn]['all_entity_binary_features'][:,:,-num_frames:,:]
                    turn_info["selectable_entities"] = batch["input"][turn]['selectable_entities'][:,:,-num_frames:,:]
                    turn_info["selectable_entity_binary_features"] = batch["input"][turn]['selectable_entity_binary_features'][:,:,-num_frames:,:]
                    turn_info["utterances"] = batch["input"][turn]["utterances"]
                    turn_info["selection_idxs"] = batch["input"][turn]["selection_idxs"]
                    if "hidden_idxs" in batch["input"][turn].keys():
                        turn_info["hidden_idxs"] = batch["input"][turn]["hidden_idxs"]
                    turn_batch["input"].append(turn_info)

                    turn_batch["target"] = [batch["target"][turn]]

                    with torch.no_grad():
                        turn_batch_output = model.forward(turn_batch["input"])
                        batch["output"] += turn_batch_output

                batch_lang_loss, batch_sel_loss, batch_sel_correct, batch_sel_total = 0, 0, 0, 0
                for turn in range(num_turns):
                    # flatten output and target
                    vocab_size = batch["output"][turn]["token_logits"].size(2)
                    batch["output"][turn]["token_logits"] = batch["output"][turn]["token_logits"].view(-1, vocab_size)
                    batch["target"][turn]['utterances'] = batch["target"][turn]['utterances'].flatten()

                    batch_lang_loss += crit(batch["output"][turn]["token_logits"], batch["target"][turn]['utterances']).item()
                    batch_sel_loss += sel_crit(batch["output"][turn]["select_logits"], batch["target"][turn]['selected']).item()
                    batch_sel_correct += (batch["output"][turn]["select_logits"].max(dim=1)[1] == batch["target"][turn]['selected']).sum().item()
                    batch_sel_total += batch["target"][turn]['selected'].size(0)
                
                # compute mean
                batch_lang_loss /= num_turns
                batch_sel_loss /= num_turns
            else:
                batch_lang_loss, batch_sel_loss, batch_sel_correct, batch_sel_total, batch["output"] = engine.test_batch(batch)
            
            # add to total statistics 
            test_lang_loss += batch_lang_loss
            test_select_loss += batch_sel_loss
            test_select_correct += batch_sel_correct
            test_select_total += batch_sel_total

            for turn in range(num_turns):
                if turn == 0:
                    first_turn_correct += (batch["output"][turn]["select_logits"].max(dim=1)[1] == batch["target"][turn]['selected']).sum().item()
                    first_turn_total += batch["target"][turn]['selected'].size(0)
                else:
                    later_turn_correct += (batch["output"][turn]["select_logits"].max(dim=1)[1] == batch["target"][turn]['selected']).sum().item()
                    later_turn_total += batch["target"][turn]['selected'].size(0)

                    for bi in range(current_bsz):
                        previous_selectable_entities = batch["meta"]["selectable_entity_ids"][turn - 1][bi]
                        previous_selected_entity_idx = batch["target"][turn - 1]["selected"][bi].item()
                        previous_selected_entity_id = previous_selectable_entities[previous_selected_entity_idx]

                        current_selectable_entities = batch["meta"]["selectable_entity_ids"][turn][bi]
                        current_selected_entity_idx = batch["target"][turn]["selected"][bi].item()
                        current_selected_entity_id = current_selectable_entities[current_selected_entity_idx]

                        select_correct = batch["output"][turn]["select_logits"][bi].max(dim=0)[1].item() == batch["target"][turn]['selected'][bi].item()

                        if previous_selected_entity_id == current_selected_entity_id:
                            if select_correct:
                                same_target_correct += 1
                            same_target_total += 1
                        else:
                            if select_correct:
                                change_target_correct += 1
                            change_target_total += 1

            if test_args.output_target_selection:
                for bi in range(current_bsz):
                    chat_id = batch['meta']['chat_ids'][bi]
                    agent_id = str(batch['meta']['agent_ids'][bi])

                    model_target_selection_annotation[chat_id] = {}
                    model_target_selection_annotation[chat_id][agent_id] = []

                    for turn in range(num_turns):
                        selectable_entity_ids = batch['meta']['selectable_entity_ids'][turn][bi]
                        selected_entity_idx = batch["output"][turn]["select_logits"][bi].max(dim=0)[1].item()
                        selected_entity_id = selectable_entity_ids[selected_entity_idx]
                        model_target_selection_annotation[chat_id][agent_id].append("agt_" + agent_id + "_" + selected_entity_id)

        test_lang_loss /= len(testset)
        test_select_loss /= len(testset)

        print('testlangloss %.8f | testlangppl %.8f' % (test_lang_loss, np.exp(test_lang_loss)))
        print('testselectloss %.8f | testselectaccuracy %.6f' % (test_select_loss, test_select_correct / test_select_total))

        print('first turn select accuracy: {:.6f} (total {})'.format(first_turn_correct / first_turn_total, first_turn_total))
        print('later turn select accuracy: {:.6f} (total {})'.format(later_turn_correct / later_turn_total, later_turn_total))
        print('same target select accuracy: {:.6f} (total {})'.format(same_target_correct / same_target_total, same_target_total))
        print('change target select accuracy: {:.6f} (total {})'.format(change_target_correct / change_target_total, change_target_total))

        repeat_select_accuracy.append(100.0 * (first_turn_correct + later_turn_correct) / (first_turn_total + later_turn_total))
        repeat_first_turn_select_accuracy.append(100.0 * first_turn_correct / first_turn_total)
        repeat_later_turn_select_accuracy.append(100.0 * later_turn_correct / later_turn_total)
        repeat_same_target_select_accuracy.append(100.0 * same_target_correct / same_target_total)
        repeat_change_target_select_accuracy.append(100.0 * change_target_correct / change_target_total)

        print("")

        del model, train_args
        del corpus, testset

    print('repeat select accuracy: {:.4f}% (std {:.4f}%)'.format(np.mean(repeat_select_accuracy), np.std(repeat_select_accuracy)))
    print('repeat first turn select accuracy: {:.4f}% (std {:.4f}%)'.format(np.mean(repeat_first_turn_select_accuracy), np.std(repeat_first_turn_select_accuracy)))
    print('repeat later turn select accuracy: {:.4f}% (std {:.4f}%)'.format(np.mean(repeat_later_turn_select_accuracy), np.std(repeat_later_turn_select_accuracy)))
    print('repeat same target select accuracy: {:.4f}% (std {:.4f}%)'.format(np.mean(repeat_same_target_select_accuracy), np.std(repeat_same_target_select_accuracy)))
    print('repeat change target select accuracy: {:.4f}% (std {:.4f}%)'.format(np.mean(repeat_change_target_select_accuracy), np.std(repeat_change_target_select_accuracy)))

    if test_args.output_target_selection:
        dump_json(model_target_selection_annotation, test_args.model_file + "_target_selection_annotation.json")

if __name__ == '__main__':
    main()