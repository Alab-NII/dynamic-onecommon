import os
import random
import sys
import pdb
import copy
import re
import json
from collections import OrderedDict, defaultdict

import torch
import numpy as np

# special tokens
SPECIAL = [
    '<eos>',
    '<unk>',
    '<selection>',
    '<pad>',
]

# tokens that stops either a sentence or a conversation
STOP_TOKENS = [
    '<eos>',
    '<selection>',
]


def get_tag(tokens, tag):
    """Extracts the value inside the given tag."""
    return tokens[tokens.index('<' + tag + '>') + 1:tokens.index('</' + tag + '>')]


def to_float(tokens):
    return [float(token) for token in tokens.split()]


def read_lines(file_name):
    """Reads all the lines from the file."""
    assert os.path.exists(file_name), 'file does not exists %s' % file_name
    lines = []
    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


class Dictionary(object):
    """Maps words into indeces.

    It has forward and backward indexing.
    """

    def __init__(self, init=True):
        self.word2idx = OrderedDict()
        self.idx2word = []
        if init:
            # add special tokens if asked
            for i, k in enumerate(SPECIAL):
                self.word2idx[k] = i
                self.idx2word.append(k)

    def add_word(self, word):
        """Adds a new word, if the word is in the dictionary, just returns its index."""
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def i2w(self, idx):
        """Converts a list of indeces into words."""
        return [self.idx2word[i] for i in idx]

    def w2i(self, words):
        """Converts a list of words into indeces. Uses <unk> for the unknown words."""
        unk = self.word2idx.get('<unk>', None)
        return [self.word2idx.get(w, unk) for w in words]

    def get_idx(self, word):
        """Gets index for the word."""
        unk = self.word2idx.get('<unk>', None)
        return self.word2idx.get(word, unk)

    def get_word(self, idx):
        """Gets word by its index."""
        return self.idx2word[idx]

    def __len__(self):
        return len(self.idx2word)

    def read_tag(file_name, tag, freq_cutoff=-1, token_freqs=None, init_dict=True):
        """Extracts all the values inside the given tag.

        Applies frequency cuttoff if asked.
        """
        if token_freqs:
            token_freqs = token_freqs
        else:
            token_freqs = OrderedDict()
        with open(file_name, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                tokens = get_tag(tokens, tag)
                for token in tokens:
                    token_freqs[token] = token_freqs.get(token, 0) + 1
        dictionary = Dictionary(init=init_dict)
        token_freqs = sorted(token_freqs.items(),
                             key=lambda x: x[1], reverse=True)
        for token, freq in token_freqs:
            if freq > freq_cutoff * 2:
                dictionary.add_word(token)
        return dictionary

    def from_file(file_name, freq_cutoff, token_freqs=None):
        """Constructs a dictionary from the given file."""
        assert os.path.exists(file_name)
        word_dict = Dictionary.read_tag(
            file_name, 'dialogue', freq_cutoff=freq_cutoff, token_freqs=token_freqs)
        return word_dict

    def from_json_file(file_name):
        """Constructs a dictionary from the given json file."""
        assert os.path.exists(file_name)
        with open(file_name, "r") as f:
            dataset = json.load(f)

        token_freqs = OrderedDict()
        for chat_id in dataset:
            for agent_id in [0, 1]:
                max_turns = len(dataset[chat_id]["agents"][agent_id])
                for turn in range(max_turns):
                    for utterance in dataset[chat_id]["agents"][agent_id][turn]["utterances"]:
                        for token in utterance:
                            token_freqs[token] = token_freqs.get(token, 0) + 1
        return token_freqs


class WordCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=2, train='train.txt',
                 valid='valid.txt', test='test.txt', verbose=False, word_dict=None):
        self.verbose = verbose

        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff, token_freqs=token_freqs)
        else:
            self.word_dict = word_dict

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            output_idx = int(get_tag(tokens, 'output')[0])
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            dataset.append((input_vals, word_idxs, output_idx, scenario_id, real_ids, agent))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, output, scenario_ids, real_ids, agents = [], [], [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                output.append(dataset[j][2])
                scenario_ids.append(dataset[j][3])
                real_ids.append(dataset[j][4])
                agents.append(dataset[j][5])

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)

            # construct tensor for context
            ctx = torch.Tensor(inputs).float()
            data = torch.Tensor(words).long().transpose(0, 1).contiguous()
            # construct tensor for selection target
            sel_tgt = torch.Tensor(output).long()
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            batches.append((ctx, inpt, tgt, sel_tgt, scenario_ids, real_ids, agents))

        if shuffle:
            random.shuffle(batches)

        #print("{}: pad={:.2f}%".format(name, 100.0 * (stats['n'] - stats['nonpadn']) / stats['n']))

        return batches, stats

class SentenceCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=-1, train='train.json',
                 valid='valid.json', test='test.json', verbose=False, word_dict=None, abl_features=[]):
        self.verbose = verbose
        self.domain = domain
        self.abl_features = abl_features

        if word_dict is None:
            self.word_dict = Dictionary.from_json_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.binary_dict = Dictionary(init=False)
        self.binary_dict.word2idx = {"<visible>": 0, "<invisible>": 1, "<selectable>": 2, "<unselectable>": 3, 
            "<previous_selectable>": 4, "<previous_unselectable>": 5, "<previous_selected>": 6, "<previous_unselected>": 7}
        self.binary_dict.idx2word = ["<visible>", "<invisible>", "<selectable>", "<unselectable>", 
            "<previous_selectable>", "<previous_unselectable>", "<previous_selected>", "<previous_unselected>"]

        self.train = self.preprocess(os.path.join(path, train)) if train else []
        self.valid = self.preprocess(os.path.join(path, valid)) if valid else []
        self.test = self.preprocess(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def preprocess(self, file_name):
        """
            Preprocess the dataset to feed and train models. 

            Main variables:
                dict: inputs, outputs, targets

            For each turn,
                model inputs:
                - all_ent_input (num_frames, max_ent, dim_ent)
                - all_ent_binary (num_frames, max_ent, dim_bin)
                - dialog_tokens ()
                - selectable_ent_input (num_frames, selectable_ent, dim_ent)
                - selectable_ent_binary (num_frames, selectable_ent, dim_bin)
                
                targets:
                - selection

            later (in _split_into_batches), for each turn,
                - select_idxs ()
        """
        assert os.path.exists(file_name)
        with open(file_name, "r") as f:
            dataset = json.load(f)
        chat_ids = list(dataset.keys())
        np.random.shuffle(chat_ids)

        unk = self.word_dict.get_idx('<unk>')
        visible = self.binary_dict.get_idx('<visible>')
        invisible = self.binary_dict.get_idx('<invisible>')
        selectable = self.binary_dict.get_idx('<selectable>')
        unselectable = self.binary_dict.get_idx('<unselectable>')
        previous_selectable = self.binary_dict.get_idx('<previous_selectable>')
        previous_unselectable = self.binary_dict.get_idx('<previous_unselectable>')
        previous_selected = self.binary_dict.get_idx('<previous_selected>')
        previous_unselected = self.binary_dict.get_idx('<previous_unselected>')
        total_words, total_unks = 0, 0
        for chat_id in dataset:
            for agent_id in [0, 1]:
                for turn in range(len(dataset[chat_id]["agents"][agent_id])):
                    """
                        convert to model format:
                            - all_entities
                            - selectable_entities
                            - selectable_entity_ids
                            - selected_entity
                    """
                    selectable_entity_ids = []

                    for ent_id in dataset[chat_id]["agents"][agent_id][turn]["context"]:
                        num_frames = len(dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["xs"])

                        # expand color, size, selectable, previous_selectable, previous_selected
                        color = dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["color"]
                        dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["color"] = [color] * num_frames
                        size = dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["size"]
                        dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["size"] = [size] * num_frames

                        dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["visible"] = [visible if x else invisible for x in dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["visible"]]

                        if dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"]:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [selectable] * num_frames
                            selectable_entity_ids.append(ent_id)
                        else:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [unselectable] * num_frames
                        if dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"]:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_selectable] * num_frames
                        else:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_unselectable] * num_frames
                        if dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"]:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"] = [previous_selected] * num_frames
                        else:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"] = [previous_unselected] * num_frames

                        # ablation of entity attributes
                        if "location" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["xs"] = [0] * num_frames
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["ys"] = [0] * num_frames
                        if "color" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["color"] = [0] * num_frames
                        if "size" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["size"] = [0] * num_frames

                        # ablation of feature tokens
                        if "visible" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["visible"] = [visible] * num_frames
                        if "selectable" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["selectable"] = [selectable] * num_frames
                        if "previous_selectable" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selectable"] = [previous_selectable] * num_frames
                        if "previous_selected" in self.abl_features:
                            dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id]["previous_selected"] = [previous_selected] * num_frames

                        # ablation of dynamics
                        if "dynamics" in self.abl_features:
                            num_frames = 1
                            for key in dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id].keys():
                                dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id][key] = [dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id][key][-1]]

                    # convert words to indices
                    for ui, utterance in enumerate(dataset[chat_id]["agents"][agent_id][turn]["utterances"]):
                        dataset[chat_id]["agents"][agent_id][turn]["utterances"][ui] = self.word_dict.w2i(utterance)
                        # compute statistics
                        total_words += len(dataset[chat_id]["agents"][agent_id][turn]["utterances"][ui])
                        total_unks += np.count_nonzero([idx == unk for idx in dataset[chat_id]["agents"][agent_id][turn]["utterances"][ui]])

                    # create pad_entity
                    pad_entity = {"xs": [0] * num_frames, "ys": [0] * num_frames, "color": [0] * num_frames, "size": [0] * num_frames,
                                  "visible": [invisible] * num_frames, "selectable": [unselectable] * num_frames, "previous_selectable": [previous_unselectable] * num_frames, "previous_selected": [previous_unselected] * num_frames}

                    # create all_entities
                    dataset[chat_id]["agents"][agent_id][turn]["all_entities"] = []
                    for ent_id in dataset[chat_id]["agents"][agent_id][turn]["context"]:
                        dataset[chat_id]["agents"][agent_id][turn]["all_entities"].append(dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id])

                    while(len(dataset[chat_id]["agents"][agent_id][turn]["all_entities"]) < self.domain.max_ent_each_turn()):
                        dataset[chat_id]["agents"][agent_id][turn]["all_entities"].append(copy.deepcopy(pad_entity))

                    # create selectable_entities
                    dataset[chat_id]["agents"][agent_id][turn]["selectable_entities"] = []
                    np.random.shuffle(selectable_entity_ids)
                    dataset[chat_id]["agents"][agent_id][turn]["selectable_entity_ids"] = selectable_entity_ids
                    for ent_id in selectable_entity_ids:
                        dataset[chat_id]["agents"][agent_id][turn]["selectable_entities"].append(dataset[chat_id]["agents"][agent_id][turn]["context"][ent_id])
                    
                    dataset[chat_id]["agents"][agent_id][turn]["selected"] = selectable_entity_ids.index(dataset[chat_id]["agents"][agent_id][turn]["selection"])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total_words, total_unks, 100. * total_unks / total_words))

        return dataset

    def train_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.deepcopy(self.train), bsz, device=device, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.deepcopy(self.valid), bsz, device=device, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.deepcopy(self.test), bsz, device=device, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        chat_agent_ids = []
        for chat_id in dataset:
            for agent_id in [0, 1]:
                chat_agent_ids.append((chat_id, agent_id))

        if shuffle:
            np.random.shuffle(chat_agent_ids)

        # sort by (turn length, utterance length in turn 1, utterance length in turn 2, ...)
        chat_agent_ids.sort(key=lambda x: (
                                len(dataset[x[0]]["agents"][x[1]]), # turn length
                                len(dataset[x[0]]['agents'][x[1]][0]["utterances"]) if len(dataset[x[0]]["agents"][x[1]]) > 0 else 0, # utterance length in turn 1
                                len(dataset[x[0]]['agents'][x[1]][1]["utterances"]) if len(dataset[x[0]]["agents"][x[1]]) > 1 else 0, # utterance length in turn 2
                                len(dataset[x[0]]['agents'][x[1]][2]["utterances"]) if len(dataset[x[0]]["agents"][x[1]]) > 2 else 0, # utterance length in turn 3
                                len(dataset[x[0]]['agents'][x[1]][3]["utterances"]) if len(dataset[x[0]]["agents"][x[1]]) > 3 else 0, # utterance length in turn 4
                                len(dataset[x[0]]['agents'][x[1]][4]["utterances"]) if len(dataset[x[0]]["agents"][x[1]]) > 4 else 0, # utterance length in turn 5
                            ))

        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        i = 0
        while i < len(chat_agent_ids):
            chat_id, agent_id = chat_agent_ids[i]

            # same batch should have the same values for:
            num_turns = len(dataset[chat_id]["agents"][agent_id])
            turn2num_utterances = []
            for turn in range(5):
                turn2num_utterances.append(len(dataset[chat_id]['agents'][agent_id][turn]["utterances"]) if num_turns > turn else 0)

            batch = {}
            batch["input"] = [defaultdict(list) for _ in range(num_turns)] # turn-level info

            batch["target"] = [defaultdict(list) for _ in range(num_turns)] # turn-level info

            batch["meta"] = {}
            batch["meta"]["chat_ids"] = []
            batch["meta"]["agent_ids"] = []
            batch["meta"]["selectable_entity_ids"] = [[] for _ in range(num_turns)] # turn-level info

            for turn in range(num_turns):
                for ui in range(turn2num_utterances[turn]):
                    batch["input"][turn]["utterances"].append([])
                    batch["input"][turn]["hidden_idxs"].append([])
                    batch["target"][turn]["utterances"].append([])

            for _ in range(bsz):
                if i >= len(chat_agent_ids):
                    break

                chat_id, agent_id = chat_agent_ids[i]
                if len(dataset[chat_id]["agents"][agent_id]) != num_turns:
                    break

                same_num_utterances = True
                for turn in range(5):
                    num_utterances = len(dataset[chat_id]['agents'][agent_id][turn]["utterances"]) if num_turns > turn else 0
                    if num_utterances != turn2num_utterances[turn]:
                        same_num_utterances = False
                        break

                if not same_num_utterances:
                    break

                for turn in range(num_turns):
                    # shape: turn -> (bsz, max_ent_each_turn, num_frames, dim_ent)
                    batch["input"][turn]["all_entities"].append([
                                                                    [
                                                                        [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                                                        for frame_id in range(len(entity["xs"]))
                                                                    ]
                                                                    for entity in dataset[chat_id]["agents"][agent_id][turn]["all_entities"]
                                                                ])

                    # shape: turn -> (bsz, max_ent_each_turn, num_frames, num_bin)
                    batch["input"][turn]["all_entity_binary_features"].append([
                                                                                    [
                                                                                       [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                                                        for frame_id in range(len(entity["visible"]))
                                                                                    ]
                                                                                    for entity in dataset[chat_id]["agents"][agent_id][turn]["all_entities"]
                                                                              ])

                    # shape: turn -> (bsz, num_selectable, num_frames, dim_ent)
                    batch["input"][turn]["selectable_entities"].append([
                                                                            [
                                                                                [entity["xs"][frame_id], entity["ys"][frame_id], entity["color"][frame_id], entity["size"][frame_id]]
                                                                                for frame_id in range(len(entity["xs"]))
                                                                            ]
                                                                            for entity in dataset[chat_id]["agents"][agent_id][turn]["selectable_entities"]
                                                                       ])

                    # shape: turn -> (bsz, num_selectable, num_frames, num_bin)
                    batch["input"][turn]["selectable_entity_binary_features"].append([
                                                                                    [
                                                                                        [entity["visible"][frame_id], entity["selectable"][frame_id], entity["previous_selectable"][frame_id], entity["previous_selected"][frame_id]]
                                                                                        for frame_id in range(len(entity["visible"]))
                                                                                    ]
                                                                                    for entity in dataset[chat_id]["agents"][agent_id][turn]["selectable_entities"]
                                                                                ])

                    # shape: turn -> num_utterances * (bsz, num_utterances, utterance_length, 1)
                    for ui in range(turn2num_utterances[turn]):
                        batch["input"][turn]["utterances"][ui].append(dataset[chat_id]["agents"][agent_id][turn]["utterances"][ui])

                    # shape: turn -> num_utterances * (bsz, 1)
                    for ui in range(turn2num_utterances[turn]):
                        utterance = dataset[chat_id]["agents"][agent_id][turn]["utterances"][ui]
                        batch["input"][turn]["hidden_idxs"][ui].append(len(utterance) - 1)

                    # shape: turn -> (bsz, 1)
                    last_utterance = dataset[chat_id]["agents"][agent_id][turn]["utterances"][-1]
                    batch["input"][turn]["selection_idxs"].append(len(last_utterance) - 1)

                    # shape: turn -> (bsz, 1)
                    batch["target"][turn]["selected"].append(dataset[chat_id]["agents"][agent_id][turn]["selected"])

                    # shape: turn -> (bsz, num_selectable)
                    batch["meta"]["selectable_entity_ids"][turn].append(dataset[chat_id]["agents"][agent_id][turn]["selectable_entity_ids"])

                batch["meta"]["chat_ids"].append(chat_id)
                batch["meta"]["agent_ids"].append(agent_id)

                i += 1

            current_bsz = len(batch["meta"]["chat_ids"])

            # the longest utterance in each batch
            for turn in range(num_turns):
                for ui in range(turn2num_utterances[turn]):
                    max_utterance_len = max([len(utterance) for utterance in batch["input"][turn]["utterances"][ui]])

                    # pad all the utterances to match the longest utterances
                    for bi in range(current_bsz):
                        stats['n'] += max_utterance_len
                        stats['nonpadn'] += len(batch["input"][turn]["utterances"][ui][bi])
                        batch["input"][turn]["utterances"][ui][bi] += [pad] * (max_utterance_len - len(batch["input"][turn]["utterances"][ui][bi]))
                    
                    if ui + 1 < turn2num_utterances[turn]:
                        # add YOU:/THEM: as the last tokens in order to connect sentences
                        for bi in range(current_bsz):
                            batch["input"][turn]["utterances"][ui][bi].append(batch["input"][turn]["utterances"][ui + 1][bi][0])
                    else:
                        # add <pad> after <selection>
                        for bi in range(current_bsz):
                            batch["input"][turn]["utterances"][ui][bi].append(pad)

                    utterance_tensor = torch.Tensor(batch["input"][turn]["utterances"][ui]).long()
                    batch["input"][turn]["utterances"][ui] = utterance_tensor.narrow(1, 0, utterance_tensor.size(1) - 1)
                    batch["target"][turn]["utterances"][ui] = utterance_tensor.narrow(1, 1, utterance_tensor.size(1) - 1)

                    batch["input"][turn]["hidden_idxs"][ui] = torch.Tensor(batch["input"][turn]["hidden_idxs"][ui]).long()

                    if device is not None:
                        batch["input"][turn]["utterances"][ui].to(device)
                        batch["target"][turn]["utterances"][ui].to(device)
                        batch["input"][turn]["hidden_idxs"][ui].to(device)

                # convert to single turn-level tensor
                batch["target"][turn]["utterances"] = torch.cat(batch["target"][turn]["utterances"], dim=1)

                # convert to torch Tensors
                batch["input"][turn]["all_entities"] = torch.Tensor(batch["input"][turn]["all_entities"]).float()
                batch["input"][turn]["all_entity_binary_features"] = torch.Tensor(batch["input"][turn]["all_entity_binary_features"]).long()
                batch["input"][turn]["selectable_entities"] = torch.Tensor(batch["input"][turn]["selectable_entities"]).float()
                batch["input"][turn]["selectable_entity_binary_features"] = torch.Tensor(batch["input"][turn]["selectable_entity_binary_features"]).long()
                batch["input"][turn]["selection_idxs"] = torch.Tensor(batch["input"][turn]["selection_idxs"]).long()

                batch["target"][turn]["selected"] = torch.Tensor(batch["target"][turn]["selected"]).long()

                if device is not None:
                    batch["input"][turn]["all_entities"].to(device)
                    batch["input"][turn]["all_entity_binary_features"].to(device)
                    batch["input"][turn]["selectable_entities"].to(device)
                    batch["input"][turn]["selectable_entity_binary_features"].to(device)
                    batch["input"][turn]["selection_idxs"].to(device)
                    batch["target"][turn]["selected"].to(device)

            if num_turns > 0:
                batches.append(batch)

        if shuffle:
            np.random.shuffle(batches)

        return batches, stats

class ReferenceCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=-1, train='train_reference.txt',
                 valid='valid_reference.txt', test='test_reference.txt', verbose=False,
                 word_dict=None, seed=0, for_multitask=False):
        self.verbose = verbose

        if for_multitask:
            token_freqs = Dictionary.from_json_file("data/dynamic-onecommon/train_{}.json".format(seed))
        else:
            token_freqs = None

        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff, token_freqs=token_freqs)
        else:
            self.word_dict = word_dict

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            referent_idxs = [int(val) for val in get_tag(tokens, 'referents')]
            output_idx = int(get_tag(tokens, 'output')[0])
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            real_ids = get_tag(tokens, 'real_ids')
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            dataset.append((input_vals, word_idxs, referent_idxs, output_idx, scenario_id, real_ids, agent, chat_id))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.deepcopy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.deepcopy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.deepcopy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by markable length and pad
        dataset.sort(key=lambda x: len(x[2]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        i = 0
        while i < len(dataset):
            markable_length = len(dataset[i][2])

            ctxs, dials, refs, sels, scenario_ids, real_ids, agents, chat_ids, sel_idxs = [], [], [], [], [], [], [], [], []

            for _ in range(bsz):
                if i >= len(dataset) or len(dataset[i][2]) != markable_length:
                    break
                ctxs.append(dataset[i][0])
                dials.append(dataset[i][1])
                refs.append(dataset[i][2])
                sels.append(dataset[i][3])
                scenario_ids.append(dataset[i][4])
                real_ids.append(dataset[i][5])
                agents.append(dataset[i][6])
                chat_ids.append(dataset[i][7])
                sel_idxs.append(len(dataset[i][1]) - 1)
                i += 1

            # the longest dialogue in the batch
            max_len = max([len(dial) for dial in dials])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(dials)):
                stats['n'] += max_len
                stats['nonpadn'] += len(dials[j])
                # one additional pad
                dials[j] += [pad] * (max_len - len(dials[j]) + 1)

            # construct tensor for context (bsz, num_ent * dim_ent)
            ctx = torch.Tensor(ctxs).float()
            
            # dialog data (seq_len, bsz)
            data = torch.Tensor(dials).long().transpose(0, 1).contiguous()

            # construct tensor for reference target
            num_markables = int(markable_length / 10)

            ref_inpt = []
            ref_tgt = []
            for j in range(len(refs)):
                _ref_inpt = []
                _ref_tgt = []
                for k in range(num_markables):
                    _ref_inpt.append(refs[j][10 * k: 10 * k + 3])
                    _ref_tgt.append(refs[j][10 * k + 3: 10 * (k + 1)])
                ref_inpt.append(_ref_inpt)
                ref_tgt.append(_ref_tgt)

            if num_markables == 0:
                ref_inpt = None
                ref_tgt = None
            else:
                ref_inpt = torch.Tensor(ref_inpt).long()
                ref_tgt = torch.Tensor(ref_tgt).long()

            # construct tensor for selection target
            sel_tgt = torch.Tensor(sels).long()
            if device is not None:
                ctx = ctx.to(device)
                data = data.to(device)
                sel_tgt = sel_tgt.to(device)

            # construct tensor for input and target
            inpt = data.narrow(0, 0, data.size(0) - 1)
            tgt = data.narrow(0, 1, data.size(0) - 1).view(-1)

            sel_idxs = torch.Tensor(sel_idxs).long()

            batches.append((ctx, inpt, tgt, ref_inpt, ref_tgt, sel_tgt, scenario_ids, real_ids, agents,\
                            chat_ids, sel_idxs, None))

        if shuffle:
            random.shuffle(batches)

        #print("{}: pad={:.2f}%".format(name, 100.0 * (stats['n'] - stats['nonpadn']) / stats['n']))

        return batches, stats


class ReferenceSentenceCorpus(ReferenceCorpus):
    def _split_into_sentences(self, dataset, name="unknown"):
        '''
        splits dataset into sentences, e.g.:
            ['YOU:', 'hi', 'there', '<eos>', 'THEM:', 'lets', 'choose', 'one', '<selection>']
            --> [['YOU:', 'hi', 'there', '<eos>'], ['THEM:', 'lets', 'choose', 'one', '<selection>']]
        '''
        stops = [self.word_dict.get_idx(w) for w in ['YOU:', 'THEM:']]
        sent_dataset = []

        for input_vals, word_idxs, referent_idxs, output_idx, scenario_id, real_ids, agent, chat_id in dataset:
            sents, current = [], []
            for w in word_idxs:
                if w in stops:
                    if len(current) > 0:
                        sents.append(current)
                    current = []
                current.append(w)
            if len(current) > 0:
                sents.append(current)
            sent_dataset.append((input_vals, sents, referent_idxs, output_idx, scenario_id, real_ids, agent, chat_id))
        # Sort by number of sentences in a dialog
        sent_dataset.sort(key=lambda x: (len(x[1]), len(x[2])))

        return sent_dataset

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        '''
        returns batches
        each batch contains dialogues with same number of utterances and markables (so training can be parallelized)
        
        output
        ctx: input context, float tensor, (seq_len, bsz)
        inpts: input dialogues, list of tensors, dial_len * (seq_len, bsz)
        lens: sentence length, list of tensors, dial_len * (bsz)
        tgts: targets, list of tensors, dial_len * (seq_len * bsz)
        sel_tgt: selection targets, long tensor, (bsz)
        rev_idxs: ?, list of list of tensors, dial_len * seq_len * (bsz, 1)
        hid_idxs: ?, 
        '''
        if shuffle:
            np.random.shuffle(dataset)

        dataset = self._split_into_sentences(dataset, name=name)

        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0
        }

        i = 0
        while i < len(dataset):
            batch_num_utterances = len(dataset[i][1])
            batch_num_markables = int(len(dataset[i][2]) / 10)

            batch_ctx, batch_uttrs, batch_refs, batch_sel, batch_scenario_id, batch_real_ids, batch_agent_idx, batch_chat_id = [], [], [], [], [], [], [], []
            batch_hid_idxs, batch_sel_idx = [], []

            for _ in range(bsz):
                if i >= len(dataset) or len(dataset[i][1]) != batch_num_utterances or int(len(dataset[i][2]) / 10) != batch_num_markables:
                    break
                batch_ctx.append(dataset[i][0])
                batch_uttrs.append(dataset[i][1])
                batch_refs.append(dataset[i][2])
                batch_sel.append(dataset[i][3])
                batch_scenario_id.append(dataset[i][4])
                batch_real_ids.append(dataset[i][5])
                batch_agent_idx.append(dataset[i][6])
                batch_chat_id.append(dataset[i][7])
                i += 1

            current_batch_size = len(batch_ctx)

            # pad dialogues to parallelize computation
            batch_inpts, batch_tgts = [], []
            current_len = 0
            for ui in range(batch_num_utterances):

                batch_uttr = []
                batch_hid_idx = []

                for bi in range(current_batch_size):
                    batch_uttr.append(batch_uttrs[bi][ui])

                if ui + 1 < batch_num_utterances:
                    # add YOU:/THEM: as the last tokens in order to connect sentences
                    for bi in range(current_batch_size):
                        batch_uttr[bi].append(batch_uttrs[bi][ui + 1][0])
                else:
                    # add <pad> after <selection>
                    for bi in range(current_batch_size):
                        batch_uttr[bi].append(pad)

                max_len = max([len(uttr) for uttr in batch_uttr])

                for bi in range(current_batch_size):
                    stats['n'] += max_len
                    stats['nonpadn'] += len(batch_uttr[bi]) - 1
                    batch_hid_idx.append(len(batch_uttr[bi]) - 2)

                    # fix ref indices
                    for mi in range(batch_num_markables):
                        start = batch_refs[bi][10 * mi]
                        if start > current_len + len(batch_uttr[bi]) - 1:
                            batch_refs[bi][10 * mi] += max_len - len(batch_uttr[bi])
                            batch_refs[bi][10 * mi + 1] += max_len - len(batch_uttr[bi])
                            batch_refs[bi][10 * mi + 2] += max_len - len(batch_uttr[bi])

                    # pad sentences
                    batch_uttr[bi] += [pad] * (max_len - len(batch_uttr[bi]))

                batch_uttr = torch.Tensor(batch_uttr).long()
                batch_inpt = batch_uttr.narrow(1, 0, batch_uttr.size(1) - 1)
                batch_tgt = batch_uttr.narrow(1, 1, batch_uttr.size(1) - 1).reshape(-1)

                batch_hid_idx = torch.Tensor(batch_hid_idx).long()

                if device is not None:
                    batch_inpt.to(device)
                    batch_tgt.to(device)
                    batch_hid_idx.to(device)

                batch_inpts.append(batch_inpt)
                batch_tgts.append(batch_tgt)
                batch_hid_idxs.append(batch_hid_idx)

                current_len += max_len

            batch_ctx = torch.Tensor(batch_ctx).transpose(0, 1).contiguous()
            
            batch_sel_tgt = torch.Tensor(batch_sel).long().view(-1)

            if batch_num_markables == 0:
                batch_ref_inpts = None
                batch_ref_tgts = None
            else:
                batch_ref_inpts = []
                batch_ref_tgts = []
                for bi in range(current_batch_size):
                    ref_inpts = []
                    ref_tgts = []
                    for mi in range(batch_num_markables):
                        ref_inpts.append(batch_refs[bi][10 * mi: 10 * mi + 3])
                        ref_tgts.append(batch_refs[bi][10 * mi + 3: 10 * (mi + 1)])
                    batch_ref_inpts.append(ref_inpts)
                    batch_ref_tgts.append(ref_tgts)

                batch_ref_inpts = torch.Tensor(batch_ref_inpts).long()
                batch_ref_tgts = torch.Tensor(batch_ref_tgts).long()

                if device is not None:
                    batch_ref_inpts.to(device)
                    batch_ref_tgts.to(device)

            batch_sel_idx = batch_hid_idx[-1]

            batches.append((batch_ctx, batch_inpts, batch_tgts, batch_ref_inpts, batch_ref_tgts,
                            batch_sel_tgt, batch_scenario_id, batch_real_ids, batch_agent_idx, batch_chat_id,
                            batch_sel_idx, batch_hid_idxs))

        if shuffle:
            np.random.shuffle(batches)

        #print("{}: pad={:.2f}%".format(name, 100.0 * (stats['n'] - stats['nonpadn']) / stats['n']))

        return batches, stats


class MarkableCorpus(object):
    """An utility that stores the entire dataset.

    It has the train, valid and test datasets and corresponding dictionaries.
    """

    def __init__(self, domain, path, freq_cutoff=-1, train='train_markable.txt',
                 valid='valid_markable.txt', test='test_markable.txt', verbose=False, word_dict=None):
        self.verbose = verbose
        if word_dict is None:
            self.word_dict = Dictionary.from_file(
                os.path.join(path, train), freq_cutoff=freq_cutoff)
        else:
            self.word_dict = word_dict

        self.bio_dict = {"B":0, "I":1, "O":2, "<START>":3, "<STOP>":4, "<PAD>": 5}

        self.train = self.tokenize(os.path.join(path, train)) if train else []
        self.valid = self.tokenize(os.path.join(path, valid)) if valid else []
        self.test = self.tokenize(os.path.join(path, test)) if test else []

        # find out the output length from the train dataset
        self.output_length = max([len(x[1]) for x in self.train])

    def tokenize(self, file_name):
        """Tokenizes the file and produces a dataset."""
        lines = read_lines(file_name)
        random.shuffle(lines)

        unk = self.word_dict.get_idx('<unk>')
        dataset, total, unks = [], 0, 0
        for line in lines:
            tokens = line.split()
            input_vals = [float(val) for val in get_tag(tokens, 'input')]
            word_idxs = self.word_dict.w2i(get_tag(tokens, 'dialogue'))
            markable_idxs = [self.bio_dict[val] for val in get_tag(tokens, 'markables')]
            scenario_id = get_tag(tokens, 'scenario_id')[0]
            agent = int(get_tag(tokens, 'agent')[0])
            chat_id = get_tag(tokens, 'chat_id')[0]
            dataset.append((input_vals, word_idxs, markable_idxs, scenario_id, agent, chat_id))
            # compute statistics
            total += len(word_idxs)
            unks += np.count_nonzero([idx == unk for idx in word_idxs])

        if self.verbose:
            print('dataset %s, total %d, unks %s, ratio %0.2f%%' % (
                file_name, total, unks, 100. * unks / total))
        return dataset

    def train_dataset(self, bsz, shuffle=True):
        return self._split_into_batches(copy.copy(self.train), bsz, shuffle=shuffle, name="train")

    def valid_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.valid), bsz, shuffle=shuffle, name="valid")

    def test_dataset(self, bsz, shuffle=True, device=None):
        return self._split_into_batches(copy.copy(self.test), bsz, shuffle=shuffle, name="test")

    def _split_into_batches(self, dataset, bsz, shuffle=True, device=None, name="unknown"):
        """Splits given dataset into batches."""
        if shuffle:
            random.shuffle(dataset)

        # sort by dialog length and pad
        dataset.sort(key=lambda x: len(x[1]))
        pad = self.word_dict.get_idx('<pad>')

        batches = []
        stats = {
            'n': 0,
            'nonpadn': 0,
        }

        for i in range(0, len(dataset), bsz):
            inputs, words, markables, scenario_ids, agents, chat_ids = [], [], [], [], [], []
            for j in range(i, min(i + bsz, len(dataset))):
                inputs.append(dataset[j][0])
                words.append(dataset[j][1])
                markables.append(dataset[j][2])
                scenario_ids.append(dataset[j][3])
                agents.append(dataset[j][4])
                chat_ids.append(dataset[j][5])
                assert len(words) == len(markables)

            # the longest dialogue in the batch
            max_len = len(words[-1])

            # pad all the dialogues to match the longest dialogue
            for j in range(len(words)):
                stats['n'] += max_len
                stats['nonpadn'] += len(words[j])
                # one additional pad
                words[j] += [pad] * (max_len - len(words[j]) + 1)
                markables[j] += [self.bio_dict["<PAD>"]] * (max_len - len(markables[j]) + 1)

            # construct tensor for context
            ctx = torch.Tensor(inputs).float().squeeze()
            words = torch.Tensor(words).long().transpose(0, 1).contiguous().squeeze()
            markables = torch.Tensor(markables).long().transpose(0, 1).contiguous().squeeze()

            if device is not None:
                ctx = ctx.to(device)
                words = words.to(device)
                markables = markables.to(device)

            batches.append((ctx, words, markables, scenario_ids, agents, chat_ids))

        if shuffle:
            random.shuffle(batches)

        #print("{}: pad={:.2f}%".format(name, 100.0 * (stats['n'] - stats['nonpadn']) / stats['n']))

        return batches, stats