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

def reference_to_svg(kb, ref_out):
    svg = '''<svg id="svg" width="430" height="430"><circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>\n'''
    for i, obj in enumerate(kb):
        svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"{3}\" />\n".format(obj['x'], obj['y'],
                                                                                             obj['size'], obj['color'])
        if ref_out[i] == 1:
            svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"none\" stroke=\"{3}\" stroke-width=\"4\" stroke-dasharray=\"3,3\"\n/>".format(obj['x'], obj['y'],
                        obj['size'] + 4, "green")
    svg += '''</svg>'''
    return svg


def main():
    parser = argparse.ArgumentParser(description='Testing script for OneCommon')

    # Dataset arguments
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--domain', type=str, default='static',
        help='domain for the dialogue')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')

    # Model arguments
    parser.add_argument('--model_dir', type=str,  default='saved_models',
        help='path to save the final model')
    parser.add_argument('--model_file', type=str, required=True,
        help='trained model file')

    # Testing arguments
    parser.add_argument('--bsz', type=int, default=16,
        help='batch size')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--bleu_n', type=int, default=0,
        help='test ngram bleu')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='temperature')

    # Ablation arguments
    parser.add_argument('--abl_features', nargs='*', default=[],
        help='ablate specified features')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=0,
        help='random seed')
    parser.add_argument('--repeat_test', action='store_true', default=False,
        help='repeat testing 5 times with different random seeds')

    # For error analysis
    parser.add_argument('--transcript_file', type=str, default='final_transcripts.json',
        help='scenario file')
    parser.add_argument('--markable_file', type=str, default='markable_annotation.json',
        help='scenario file')
    parser.add_argument('--show_errors', action='store_true', default=False,
        help='show errors')

    args = parser.parse_args()

    if args.bleu_n > 0:
        # current support
        args.bsz = 1

    if args.repeat_test:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [args.seed]

    repeat_results = defaultdict(list)

    model_referent_annotation = {}

    for seed in seeds:
        device_id = utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        model = utils.load_model(args.model_dir, args.model_file + '_' + str(seed) + '.th')
        if args.cuda:
            model.cuda()
        else:
            device = torch.device("cpu")
            model.to(device)
        model.eval()

        corpus = model.corpus_ty(domain, args.data, train='train_reference_{}.txt'.format(seed), valid='valid_reference_{}.txt'.format(seed), test='test_reference_{}.txt'.format(seed), #test='selfplay_reference_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True, word_dict=model.word_dict)
        engine = model.engine_ty(model, args, train=False, verbose=True)

        with open(os.path.join(args.data, args.transcript_file), "r") as f:
            dialog_corpus = json.load(f)
        with open(os.path.join(args.data, args.markable_file), "r") as f:
            markable_annotation = json.load(f)
        with open(os.path.join(args.data, "aggregated_referent_annotation.json"), "r") as f:
            aggregated_referent_annotation = json.load(f)

        scenarios = {scenario['scenario_uuid']: scenario for scenario in dialog_corpus}

        testset, testset_stats = corpus.test_dataset(args.bsz)
        test_lang_loss, test_select_loss, test_reference_loss, test_select_correct, test_select_total, test_reference_correct, test_reference_total = 0, 0, 0, 0, 0, 0, 0

        """
            Variables to keep track of the results for analysis
        """

        # num_referents --> count, count correct
        total_num_markables = 0
        num_markables_counter = Counter()
        num_markables_correct = Counter()

        exact_match = 0
        exact_match_counter = Counter()

        # location of markable --> count, count correct, count exact match
        location_counter = Counter()
        location_correct = Counter()
        location_exact_match = Counter()

        # information to compute correlation between selection and reference score 
        select_correct = {}
        reference_correct = {}
        reference_total = {}

        # markable text --> count, count correct
        text_counter = Counter()
        text_correct = Counter()

        anaphora_list = ["it", "that", "thats", "this", "its", "they", "their", "itself", "them", "those", "it's"]
        total_anaphora = 0
        correct_anaphora = 0

        bleu_scores = []

        for batch in testset:
            batch_lang_loss, batch_ref_loss, batch_ref_correct, batch_ref_total, batch_sel_loss, batch_sel_correct, \
                batch_sel_total, batch_lang_outs, batch_ref_outs, batch_sel_out = engine.test_batch(batch)

            batch_ctx, batch_inpts, batch_tgts, batch_ref_inpts, batch_ref_tgts, batch_sel_tgt, batch_scenario_id, \
            batch_real_ids, batch_agent_idx, batch_chat_id, batch_sel_idx, batch_hid_idxs = batch

            if batch_ref_inpts is not None:
                bsz = batch_ref_inpts.size(0)
                num_markables = batch_ref_inpts.size(1)

                # compute more details of reference resolution
                for bi in range(bsz):
                    chat_id = batch_chat_id[bi]
                    agent_idx = batch_agent_idx[bi]
                    for mi in range(num_markables):
                        ref_results = ((batch_ref_outs[bi][mi] > 0).long() == batch_ref_tgts[bi][mi].long())

                        # add chat level details if not exists
                        if chat_id not in reference_correct:
                            reference_correct[chat_id] = ref_results.sum().item()
                        if chat_id not in reference_total:
                            assert ref_results.size(0) == 7
                            reference_total[chat_id] = ref_results.size(0)
                        if chat_id not in model_referent_annotation:
                            model_referent_annotation[chat_id] = {}

                        markables = []
                        # markables information from aggregated_referent_annotation
                        for markable in markable_annotation[chat_id]["markables"]:
                            markable_id = markable["markable_id"]
                            if markable_id in aggregated_referent_annotation[chat_id] and markable["speaker"] == agent_idx:
                                if "unidentifiable" in aggregated_referent_annotation[chat_id][markable_id] and aggregated_referent_annotation[chat_id][markable_id]["unidentifiable"]:
                                    continue
                                markables.append(markable)
                        assert len(markables) == batch_ref_tgts.size(1)

                        num_correct = ref_results.sum().item()
                        is_exact_match= ref_results.all().item()
                        num_gold_referents = batch_ref_tgts[bi][mi].sum().item()

                        """
                            Add information to variables
                        """
                        total_num_markables += 1
                        num_markables_counter[num_gold_referents] += 1
                        num_markables_correct[num_gold_referents] += num_correct

                        # compute exact match 
                        if is_exact_match:
                            exact_match += 1
                            exact_match_counter[num_gold_referents] += 1
                            text_correct[markables[mi]["text"].lower()] += 1

                        location_correct[mi] += num_correct
                        if is_exact_match:
                            location_exact_match[mi] += 1
                        location_counter[mi] += 1

                        text_counter[markables[mi]["text"].lower()] += 1

                        # test anaphora
                        if markables[mi]["text"].lower() in anaphora_list:
                            total_anaphora += 1
                            if is_exact_match:
                                correct_anaphora += 1

                        # keep track of model predictions for later visualization
                        chat = [chat for chat in dialog_corpus if chat['uuid'] == chat_id]
                        chat = chat[0]
                        if markables[mi]['markable_id'] not in model_referent_annotation[chat_id]:
                            model_referent_annotation[chat_id][markables[mi]['markable_id']] = {}
                            model_referent_annotation[chat_id][markables[mi]['markable_id']]['referents'] = []
                            model_referent_annotation[chat_id][markables[mi]['markable_id']]['ambiguous'] = False
                            model_referent_annotation[chat_id][markables[mi]['markable_id']]['unidentifiable'] = False
                            for ent, is_referent in zip(chat['scenario']['kbs'][agent_idx], (batch_ref_outs[bi][mi] > 0).long().tolist()):
                                if is_referent:
                                    model_referent_annotation[chat_id][markables[mi]['markable_id']]['referents'].append("agent_{}_{}".format(agent_idx, ent['id']))

            for bi in range(batch_sel_tgt.size(0)):
                chat_id = batch_chat_id[bi]
                is_correct = (batch_sel_out[bi].max(dim=0)[1].item() == batch_sel_tgt[bi].item())
                if is_correct:
                    select_correct[chat_id] = 1
                else:
                    select_correct[chat_id] = 0

            test_lang_loss += batch_lang_loss
            test_select_loss += batch_sel_loss
            if batch_ref_loss:
                test_reference_loss += batch_ref_loss
            test_select_correct += batch_sel_correct
            test_select_total += batch_sel_total
            test_reference_correct += batch_ref_correct
            test_reference_total += batch_ref_total

        # Main results
        test_lang_loss /= len(testset)
        test_select_loss /= len(testset)
        test_select_accuracy = test_select_correct / test_select_total
        test_reference_accuracy = test_reference_correct / test_reference_total
        print('testlangloss %.8f | testlangppl %.8f' % (test_lang_loss, np.exp(test_lang_loss)))
        print('testselectloss %.8f | testselectaccuracy %.6f' % (test_select_loss, test_select_accuracy))
        print('testreferenceloss %.8f | testreferenceaccuracy %.6f' % (test_reference_loss, test_reference_accuracy))
        print('reference_exact_match %.6f' % (exact_match / total_num_markables))
        for k in num_markables_counter.keys():
            print('{}: {:.4f} {:.4f} (out of {})'.format(k, num_markables_correct[k] / (num_markables_counter[k] * 7), exact_match_counter[k] / num_markables_counter[k], num_markables_counter[k]))
        print('test anaphora: {} (out of {})'.format(correct_anaphora / total_anaphora, total_anaphora))

        if args.bleu_n > 0:
            print('average bleu score {}'.format(np.mean(bleu_scores)))

        # reference/selection correlation
        reference_score = []
        selection_score = []
        for chat_id in reference_correct.keys():
            reference_score.append(reference_correct[chat_id] / reference_total[chat_id])
            selection_score.append(select_correct[chat_id])
        plt.xlabel('reference score', fontsize=14)
        plt.ylabel('selection score', fontsize=14)
        #ax = sns.scatterplot(x=reference_score, y=selection_score, size=0, legend=False)
        sns.regplot(x=reference_score, y=selection_score)
        #plt.axes().set_aspect('equal', 'datalim')
        plt.savefig('reference_selection_{}.png'.format(seed), dpi=300)
        plt.clf()
        reference_score = np.array(reference_score)
        selection_score = np.array(selection_score)
        print("reference selection correlation: {}".format(np.corrcoef(reference_score, selection_score)))

        # keep track of results for this run
        repeat_results["test_lang_loss"].append(test_lang_loss)
        repeat_results["test_select_loss"].append(test_select_loss)
        repeat_results["test_select_accuracy"].append(test_select_accuracy)
        repeat_results["test_reference_loss"].append(test_reference_loss)
        repeat_results["test_reference_accuracy"].append(test_reference_accuracy)
        repeat_results["correlation_score"].append(np.corrcoef(reference_score, selection_score)[0][1])
        repeat_results["num_markables_counter"].append(copy.copy(num_markables_counter))
        repeat_results["exact_match_counter"].append(copy.copy(exact_match_counter))
        repeat_results["num_markables_correct"].append(copy.copy(num_markables_correct))
        repeat_results["reference_exact_match"].append(exact_match / total_num_markables)
        repeat_results["test_perplexity"].append(np.exp(test_lang_loss))
        repeat_results["location_counter"].append(copy.copy(location_counter))
        repeat_results["location_correct"].append(copy.copy(location_correct))
        repeat_results["location_exact_match"].append(copy.copy(location_exact_match))


    print("=================================\n\n")
    print("repeat test lang loss %.8f" % np.mean(repeat_results["test_lang_loss"]))
    print("repeat test select loss %.8f" % np.mean(repeat_results["test_select_loss"]))
    print("repeat test select accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_select_accuracy"]), np.std(repeat_results["test_select_accuracy"])))
    print("repeat test reference loss %.8f" % np.mean(repeat_results["test_reference_loss"]))
    print("repeat test reference accuracy %.8f ( %.8f )" % (np.mean(repeat_results["test_reference_accuracy"]), np.std(repeat_results["test_reference_accuracy"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat correlation score %.8f ( %.8f )" % (np.mean(repeat_results["correlation_score"]), np.std(repeat_results["correlation_score"])))
    print("repeat reference exact match %.8f ( %.8f )" % (np.mean(repeat_results["reference_exact_match"]), np.std(repeat_results["reference_exact_match"])))
    print("repeat test perplexity %.8f ( %.8f )" % (np.mean(repeat_results["test_perplexity"]), np.std(repeat_results["test_perplexity"])))

    for k in num_markables_counter.keys():
        print("repeat accuracy and exact match:")
        num_markables = []
        exact_match = []
        exact_match_rate = []
        num_markables_correct = []
        for seed in range(len(seeds)):
            num_markables.append(repeat_results["num_markables_counter"][seed][k])
            exact_match.append(repeat_results["exact_match_counter"][seed][k])
            exact_match_rate.append(repeat_results["exact_match_counter"][seed][k] / repeat_results["num_markables_counter"][seed][k])
            num_markables_correct.append(repeat_results["num_markables_correct"][seed][k] / (repeat_results["num_markables_counter"][seed][k] * 7))
        print('{}: {:.5f} (std {}) {:.5f} (std {}) (count {})'.format(k, np.mean(num_markables_correct), np.std(num_markables_correct), np.mean(exact_match_rate), np.std(exact_match_rate), np.mean(num_markables)))

    dump_json(model_referent_annotation, "model_referent_annotation.json")

if __name__ == '__main__':
    main()