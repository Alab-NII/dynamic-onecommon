# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Various helpers.
"""

import os
import random
import copy
import pdb

import torch
import numpy as np

from data import Dictionary

def backward_hook(grad):
    """Hook for backward pass."""
    print(grad)
    pdb.set_trace()
    return grad


def save_model(model, dir_name, file_name):
    """Serializes model to a file."""
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    if file_name != '':
        with open(os.path.join(dir_name, file_name), 'wb') as f:
            torch.save(model, f)


def load_model(dir_name, file_name):
    """Reads model from a file."""
    with open(os.path.join(dir_name, file_name), 'rb') as f:
        return torch.load(f)


def set_seed(seed):
    """Sets random seed everywhere."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def use_cuda(enabled, device_id=0):
    """Verifies if CUDA is available and sets default device to be device_id."""
    if not enabled:
        return None
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(device_id)
    return device_id

def prob_random():
    """Prints out the states of various RNGs."""
    print('random state: python %.3f torch %.3f numpy %.3f' % (
        random.random(), torch.rand(1)[0], np.random.rand()))

class ContextGenerator(object):
    """Dialogue context generator. Generates contexes from the file."""

    def __init__(self, domain, scenarios, train_args, selfplay_args):
        self.domain = domain
        self.binary_dict = Dictionary(init=False)
        self.binary_dict.word2idx = {"<visible>": 0, "<invisible>": 1, "<selectable>": 2, "<unselectable>": 3, 
            "<previous_selectable>": 4, "<previous_unselectable>": 5, "<previous_selected>": 6, "<previous_unselected>": 7}
        self.binary_dict.idx2word = ["<visible>", "<invisible>", "<selectable>", "<unselectable>", 
            "<previous_selectable>", "<previous_unselectable>", "<previous_selected>", "<previous_unselected>"]

        visible = self.binary_dict.get_idx('<visible>')
        invisible = self.binary_dict.get_idx('<invisible>')
        selectable = self.binary_dict.get_idx('<selectable>')
        unselectable = self.binary_dict.get_idx('<unselectable>')
        previous_selectable = self.binary_dict.get_idx('<previous_selectable>')
        previous_unselectable = self.binary_dict.get_idx('<previous_unselectable>')
        previous_selected = self.binary_dict.get_idx('<previous_selected>')
        previous_unselected = self.binary_dict.get_idx('<previous_unselected>')
        for scenario_id in scenarios:
            for agent_id in [0, 1]:
                for turn in range(len(scenarios[scenario_id]["agents"][agent_id])):
                    """
                        convert to model format:
                            - all_entities
                            - all_entity_ids
                            - selectable_entities
                            - selectable_entity_ids
                            - selected_entity
                    """
                    selectable_entity_ids = []
                    all_entity_ids = []

                    for ent_id in scenarios[scenario_id]["agents"][agent_id][turn]["context"]:
                        all_entity_ids.append(ent_id)
                        num_frames = len(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["xs"])

                        # expand color, size, selectable, previous_selectable, previous_selected
                        color = scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["color"]
                        scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["color"] = [color] * num_frames
                        size = scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["size"]
                        scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["size"] = [size] * num_frames
                        scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["visible"] = [visible if x else invisible for x in scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["visible"]]

                        if scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"]:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [selectable] * num_frames
                            selectable_entity_ids.append(ent_id)
                        else:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [unselectable] * num_frames
                        if scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"]:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_selectable] * num_frames
                        else:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_unselectable] * num_frames

                        # fill with previous_unselected
                        scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"] = [previous_unselected] * num_frames

                        # set previous_unselectable if refresh_each_turn
                        if selfplay_args.refresh_each_turn:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_unselectable] * num_frames

                        # ablation of entity attributes
                        if "location" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["xs"] = [0] * num_frames
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["ys"] = [0] * num_frames
                        if "color" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["color"] = [0] * num_frames
                        if "size" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["size"] = [0] * num_frames

                        # ablation of feature tokens
                        if "visible" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["visible"] = [visible] * num_frames
                        if "selectable" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [selectable] * num_frames
                        if "previous_selectable" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_selectable] * num_frames
                        if "previous_selected" in train_args.abl_features:
                            scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"] = [previous_selected] * num_frames

                        # ablation of dynamics
                        if "dynamics" in train_args.abl_features:
                            num_frames = 1
                            for key in scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id].keys():
                                scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id][key] = [scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id][key][-1]]

                    # create pad_entity
                    pad_entity = {"xs": [0] * num_frames, "ys": [0] * num_frames, "color": [0] * num_frames, "size": [0] * num_frames,
                                  "visible": [invisible] * num_frames, "selectable": [unselectable] * num_frames, "previous_selectable": [previous_unselectable] * num_frames, "previous_selected": [previous_unselected] * num_frames}

                    # create all_entities
                    scenarios[scenario_id]["agents"][agent_id][turn]["all_entities"] = []
                    for ent_id in scenarios[scenario_id]["agents"][agent_id][turn]["context"]:
                        scenarios[scenario_id]["agents"][agent_id][turn]["all_entities"].append(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id])

                    while(len(scenarios[scenario_id]["agents"][agent_id][turn]["all_entities"]) < self.domain.max_ent_each_turn()):
                        scenarios[scenario_id]["agents"][agent_id][turn]["all_entities"].append(copy.deepcopy(pad_entity))

                    scenarios[scenario_id]["agents"][agent_id][turn]["all_entity_ids"] = all_entity_ids                    

                    # create selectable_entities
                    scenarios[scenario_id]["agents"][agent_id][turn]["selectable_entities"] = []
                    np.random.shuffle(selectable_entity_ids)
                    scenarios[scenario_id]["agents"][agent_id][turn]["selectable_entity_ids"] = selectable_entity_ids
                    for ent_id in selectable_entity_ids:
                        scenarios[scenario_id]["agents"][agent_id][turn]["selectable_entities"].append(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id])
                    
        self.scenarios = scenarios

    def get_binary_dict(self):
        return self.binary_dict

    def sample(self):
        ctx_data = random.choice(self.ctxs)
        return ctx_data[0], ctx_data[1:3], ctx_data[3:]

    def iter(self, nepoch=1):
        scenario_ids = list(self.scenarios.keys())
        for e in range(nepoch):
            np.random.shuffle(scenario_ids)
            for scenario_id in scenario_ids:
                yield scenario_id, self.scenarios[scenario_id]



