import argparse
import json
import os
import pdb
import re
import random

from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torch import autograd
import torch.nn as nn

from agent import *
import utils
from utils import ContextGenerator
from dialog import Dialog, DialogLogger
from models.rnn_dialog_model import RnnDialogModel
from models.hierarchical_dialog_model import HierarchicalDialogModel
from domain import get_domain

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.15)
base_color = sns.color_palette()[0]

def dump_json(file, path):
    try:
        with open(path, "w") as fout:
            json.dump(file, fout, indent=4, sort_keys=True)
    except:
        raise Exception('Error writing JSON to %s' % path)

class SelfPlay(object):
    def __init__(self, dialog, ctx_gen, args, logger=None):
        self.dialog = dialog
        self.ctx_gen = ctx_gen
        self.args = args
        self.logger = logger if logger else DialogLogger()

    def run(self):
        total_chat = 0
        total_successful_turns = 0
        for scenario_id, scenario in self.ctx_gen.iter():
            if self.args.adversarial_run:
                self.logger.dump('=' * 80)
                _, agree, adv_agree, _ = self.dialog.adversarial_run(scenario_id, scenario, self.logger)
                if agree:
                    total_chat += 1
                    total_successful_turns += int(adv_agree)
                    self.logger.dump('=' * 80)
                    self.logger.dump('')
                    if total_chat % 200 == 0:
                        self.logger.dump('%d: %s' % (total_chat, self.dialog.show_metrics()), forced=True)
                    if self.args.sample_chat and total_chat >= 200:
                        break
            else:
                self.logger.dump('=' * 80)
                _, successful_turns, _ = self.dialog.run(scenario_id, scenario, self.logger)
                total_chat += 1
                total_successful_turns += successful_turns
                self.logger.dump('=' * 80)
                self.logger.dump('')
                if total_chat % 200 == 0:
                    self.logger.dump('%d: %s' % (total_chat, self.dialog.show_metrics()), forced=True)
                if self.args.sample_chat and total_chat >= 200:
                    break
        if self.args.plot_metrics:
            self.dialog.plot_metrics()

        return total_successful_turns / total_chat

def get_agent_type(model):
    if isinstance(model, HierarchicalDialogModel):
        return HierarchicalAgent
    elif isinstance(model, (RnnDialogModel)):
        return RnnAgent
    else:
        assert False, 'unknown model type: %s' % (model)
        

def main():
    parser = argparse.ArgumentParser(description='Selfplay script for Dynamic-Onecommon')

    # Dataset arguments
    parser.add_argument('--data', type=str, default='data/dynamic-onecommon',
        help='location of the data corpus')
    parser.add_argument('--domain', type=str, default='dynamic',
        help='domain for the dialogue')
    parser.add_argument('--unk_threshold', type=int, default=10,
        help='minimum word frequency to be in dictionary')

    # Model arguments
    parser.add_argument('--model_dir', type=str,  default='saved_models',
        help='path to save the final model')
    parser.add_argument('--alice_model_file', type=str,
        help='Alice model file')
    parser.add_argument('--bob_model_file', type=str,
        help='Bob model file')
    parser.add_argument('--temperature', type=float, default=0.25,
        help='temperature')
    parser.add_argument('--pred_temperature', type=float, default=1.0,
        help='temperature')


    # Selfplay arguments
    parser.add_argument('--max_turns', type=int, default=20,
        help='maximum number of turns in a dialog')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--refresh_each_turn', action='store_true', default=False,
        help='refresh each turn')

    # Misc arguments
    parser.add_argument('--seed', type=int, default=0,
        help='random seed')
    parser.add_argument('--ref_text', type=str,
        help='file with the reference text')
    parser.add_argument('--log_file', type=str, default='selfplay.log',
        help='log dialogs to file')
    parser.add_argument('--verbose', action='store_true', default=False,
        help='print out converations')
    parser.add_argument('--repeat_selfplay', action='store_true', default=False,
        help='repeat selfplay')

    # For analysis
    parser.add_argument('--log_attention', action='store_true', default=False,
        help='log attention')
    parser.add_argument('--visual', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--eps', type=float, default=0.0,
        help='eps greedy')
    parser.add_argument('--validate', action='store_true', default=False,
        help='plot graphs')
    parser.add_argument('--plot_metrics', action='store_true', default=False,
        help='plot metrics')
    parser.add_argument('--sample_chat', action='store_true', default=False,
        help='sample 200 chats and output selfplay_transcripts.json')
    parser.add_argument('--adversarial_run', action='store_true', default=False,
        help='conduct adversarial evaluation')
    parser.add_argument('--plot_success', action='store_true', default=False,
        help='plot distribution of successful turns')


    args = parser.parse_args()

    if args.repeat_selfplay:
        seeds = [0, 1, 2, 3, 4]
    else:
        seeds = [args.seed]

    if args.sample_chat:
        seeds = [args.seed]

    repeat_results = defaultdict(list)

    for seed in seeds:
        utils.use_cuda(args.cuda)
        utils.set_seed(seed)

        with open(os.path.join(args.data, "selfplay.json"), "r") as f:
            scenarios = json.load(f)

        domain = get_domain(args.domain)

        logger = DialogLogger(verbose=args.verbose, log_file=args.log_file, scenarios=scenarios)

        alice_model = utils.load_model(args.model_dir, args.alice_model_file + '_' + str(seed) + '.th')
        bob_model = utils.load_model(args.model_dir, args.bob_model_file + '_' + str(seed) + '.th')

        ctx_gen = ContextGenerator(domain, scenarios, train_args=alice_model.args, selfplay_args=args)
        binary_dict = ctx_gen.get_binary_dict()

        alice_ty = get_agent_type(alice_model)
        alice = alice_ty(alice_model, args, binary_dict, name='Alice', train=False)
        bob_ty = get_agent_type(bob_model)
        bob = bob_ty(bob_model, args, binary_dict, name='Bob', train=False)

        dialog = Dialog([alice, bob], args)

        selfplay = SelfPlay(dialog, ctx_gen, args, logger)
        result = selfplay.run()

        print("Average successful turns: {:.2f}\n".format(result))

        if args.repeat_selfplay:
            if args.adversarial_run:
                repeat_results["first turn"].append(100.0 * dialog.metrics.metrics['first_turn'].value())
                repeat_results["first turn (shared 4)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_4'].value())
                repeat_results["first turn (shared 5)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_5'].value())
                repeat_results["first turn (shared 6)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_6'].value())
                repeat_results["adv. first turn"].append(100.0 * dialog.metrics.metrics['adv_first_turn'].value())
                repeat_results["adv. first turn (shared 4)"].append(100.0 * dialog.metrics.metrics['adv_first_turn_shared_4'].value())
                repeat_results["adv. first turn (shared 5)"].append(100.0 * dialog.metrics.metrics['adv_first_turn_shared_5'].value())
                repeat_results["adv. first turn (shared 6)"].append(100.0 * dialog.metrics.metrics['adv_first_turn_shared_6'].value())
                repeat_results["target change rate"].append(100.0 * dialog.metrics.metrics['target_change'].value())
                repeat_results["utterance increase rate"].append(100.0 * dialog.metrics.metrics['utterance_increase'].value())
                repeat_results["utterance decrease rate"].append(100.0 * dialog.metrics.metrics['utterance_decrease'].value())
            else:
                repeat_results["successful turns"].append(dialog.metrics.metrics['successful_turns'].value())
                repeat_results["first turn"].append(100.0 * dialog.metrics.metrics['first_turn'].value())
                repeat_results["first turn (shared 4)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_4'].value())
                repeat_results["first turn (shared 5)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_5'].value())
                repeat_results["first turn (shared 6)"].append(100.0 * dialog.metrics.metrics['first_turn_shared_6'].value())
                repeat_results["same target"].append(100.0 * dialog.metrics.metrics['same_target'].value())
                repeat_results["same target (shared 4)"].append(100.0 * dialog.metrics.metrics['same_target_shared_4'].value())
                repeat_results["same target (shared 5)"].append(100.0 * dialog.metrics.metrics['same_target_shared_5'].value())
                repeat_results["same target (shared 6)"].append(100.0 * dialog.metrics.metrics['same_target_shared_6'].value())
                repeat_results["change target"].append(100.0 * dialog.metrics.metrics['change_target'].value())
                repeat_results["change target (shared 4)"].append(100.0 * dialog.metrics.metrics['change_target_shared_4'].value())
                repeat_results["change target (shared 5)"].append(100.0 * dialog.metrics.metrics['change_target_shared_5'].value())
                repeat_results["change target (shared 6)"].append(100.0 * dialog.metrics.metrics['change_target_shared_6'].value())
                repeat_results["later turn"].append(100.0 * dialog.metrics.metrics['later_turn'].value())

        del scenarios
        del logger
        del alice_model, bob_model
        del alice, bob
        if args.sample_chat or args.plot_success:
            break
        else:
            del dialog, selfplay

    if args.repeat_selfplay:
        print("")
        for key in repeat_results.keys():
            print("repeat {}: {:.4f} (std {:.4f})".format(key, np.mean(repeat_results[key]), np.std(repeat_results[key])))

    if args.sample_chat:
        dump_json(dialog.selfplay_transcripts, "selfplay_transcripts.json")

    if args.plot_success:
        successful_turns = []
        for scenario_id in dialog.selfplay_transcripts.keys():
            successful_turns.append(dialog.selfplay_transcripts[scenario_id]["outcome"])
        sns.countplot(np.array(successful_turns), color=base_color)
        plt.xlabel('Score', fontsize=18)
        plt.ylabel('Count', fontsize=18)
        plt.tight_layout()
        plt.savefig("successful_turns.png")

if __name__ == '__main__':
    main()