import argparse
from collections import defaultdict, Counter
from datetime import datetime
import json
import math
import os
import sys
import pdb
import re
import hashlib

from nltk import word_tokenize, pos_tag, bigrams

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(font_scale=1.15)
base_color = sns.color_palette()[0]

import plotly.express as px

from utils import *


def _check_finished(chat_data):
    chat_ids = list(chat_data.keys())
    for chat_id in chat_ids:
        max_turn = -1
        agent_select = {0: False, 1: False}
        for event in chat_data[chat_id]["events"]:
            turn = event['turn']
            if max_turn < turn:
                max_turn = turn
                agent_select = {0: False, 1: False}
            
            if turn == max_turn and event['action'] == 'select':
                agent_id = event['agent']
                agent_select[agent_id] = True

        finished = agent_select[0] and agent_select[1]
        if not finished:
            del chat_data[chat_id]
        elif int(chat_data[chat_id]["outcome"]) < 0:
            del chat_data[chat_id]


def basic_statistics(chat_data, scenarios, scenario_svgs):
    total_chats = 0
    outcomes = []
    total_turns = 0
    total_utterances = 0
    total_words = 0
    total_duration = 0

    for chat_id in chat_data:
        scenario_id = chat_data[chat_id]["scenario_id"]
        agent_ids = [agent_info["agent_id"] for agent_info in chat_data[chat_id]["agents_info"]]
        outcome = int(chat_data[chat_id]["outcome"])
        start_time = chat_data[chat_id]["time"]["start_time"]
        duration = chat_data[chat_id]["time"]["duration"]
        
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'message':
                msg = event['data']
                total_words += len([w.lower() for w in word_tokenize(msg)])
                total_utterances += 1                

        total_turns += min(outcome + 1, 5)
        total_chats += 1
        total_duration += duration
        outcomes.append(outcome)

    print("total chats: {}".format(total_chats))
    print("average outcome: {:.2f}".format(np.mean(outcomes)))
    print("average turns per chat: {:.2f}".format(total_turns / total_chats))
    print("average utterances per chat: {:.2f}".format(total_utterances / total_chats))
    print("average utterances per turn: {:.2f}".format(total_utterances / total_turns))
    print("average words per utterance: {:.2f}".format(total_words / total_utterances))
    print("average time per chat: {:.2f}".format(total_duration / total_chats / 60))
    print("average time per turn: {:.2f}".format(total_duration / total_turns / 60))

    with open(os.path.join("mturk", "transcripts.json"), "r") as f:
        all_chat_data = json.load(f)
    with open(os.path.join("mturk_2", "transcripts.json"), "r") as f:
        all_chat_data_2 = json.load(f)
        all_chat_data.update(all_chat_data_2)
    with open(os.path.join("mturk_3", "transcripts.json"), "r") as f:
        all_chat_data_3 = json.load(f)
        all_chat_data.update(all_chat_data_3)

    _check_finished(all_chat_data)
    print("\nacceptance rate: {:.2f} (total {})".format(100.0 * len(chat_data.keys()) / len(all_chat_data.keys()), len(all_chat_data.keys())))


def plot_success(chat_data):
    successful_turns = []
    for chat_id in chat_data:
        outcome = int(chat_data[chat_id]["outcome"])
        successful_turns.append(outcome)
    sns.countplot(np.array(successful_turns), color=base_color)
    plt.xlabel('Score', fontsize=18)
    plt.ylabel('Count', fontsize=18)
    plt.tight_layout()
    plt.savefig("successful_turns.png")

    success_counter = Counter() # num success at given turn
    fail_counter = Counter() # num fail at given turn
    for chat_id in chat_data:
        outcome = int(chat_data[chat_id]["outcome"])
        for turn in range(outcome):
            success_counter[turn] += 1
        if outcome < 5:
            fail_counter[outcome] += 1

    total_turns = 0
    for turn in range(5):
        print("Turn {}: {:.2f}%".format(turn + 1, 100.0 * success_counter[turn] / (success_counter[turn] + fail_counter[turn])))
        total_turns += success_counter[turn] + fail_counter[turn]
    print(total_turns)


def count_scenarios(chat_data, scenarios, scenario_svgs):
    scenario_counter = Counter()
    completed_scenario_counter = Counter()
    for chat_id in chat_data:
        assert chat_data[chat_id]["scenario_id"] in scenarios.keys()
        assert chat_data[chat_id]["scenario_id"] in scenario_svgs.keys()
        scenario_counter[chat_data[chat_id]["scenario_id"]] += 1
        if int(chat_data[chat_id]["outcome"]) > 0:
            completed_scenario_counter[chat_data[chat_id]["scenario_id"]] += 1
    print("all scenarios: {} unique (out of {})".format(len(scenario_counter), sum(scenario_counter.values())))
    print("completed scenarios: {} unique (out of {})".format(len(completed_scenario_counter), sum(completed_scenario_counter.values())))


def plot_selection(chat_data, scenarios):
    min_color = 53
    max_color = 203
    min_size = 7
    max_size = 11
    color_bin = 5
    color_range = 1 + int((max_color - min_color) / color_bin)
    size_range = max_size - min_size + 1

    total = np.zeros((color_range, size_range))
    selected = np.zeros((color_range, size_range))

    def _group_color(color):
        return int((color - min_color) / color_bin)

    def _group_size(size):
        return size - min_size

    for chat_id in chat_data:
        scenario_id = chat_data[chat_id]["scenario_id"]
        agent_ids = [agent_info["agent_id"] for agent_info in chat_data[chat_id]["agents_info"]]
        outcome = int(chat_data[chat_id]["outcome"])
        start_time = chat_data[chat_id]["time"]["start_time"]
        duration = chat_data[chat_id]["time"]["duration"]

        if scenario_id in scenarios:
            scenario = scenarios[scenario_id]
        else:
            assert False

        select_data = defaultdict(defaultdict)
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'select':
                agent_id = event['agent']
                ent_id = event['data']
                turn = event['turn']
                select_data[agent_id][turn] = ent_id

        for agent_id in select_data.keys():
            for turn in select_data[agent_id].keys():
                ent_id = str(select_data[agent_id][turn])
                size = _group_size(scenario["entities"][ent_id]["size"])
                color = _group_color(scenario["entities"][ent_id]["color"])
                selected[color][size] += 1

        agents = []
        entities = []
        for entity_id in scenario["entities"].keys():
            entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))
        max_turns = len(select_data[agent_id].keys())

        for agent_id in [0, 1]:
            agent = Agent.from_dict(scenario["agents"][agent_id])

            for turn in range(max_turns):
                selectable_ent_ids = []
                for i in range(len(entities)):
                    if distance(agent.loc_at_timestep(turn + 1), entities[i].loc_at_timestep(turn + 1)) < agent.r:
                        selectable_ent_ids.append(entities[i].id)
                assert len(selectable_ent_ids) == 7

                for ent_id in selectable_ent_ids:
                    size = _group_size(scenario["entities"][ent_id]["size"])
                    color = _group_color(scenario["entities"][ent_id]["color"])
                    total[color][size] += 1

    ax = sns.heatmap((selected / total), cmap=cm.Blues, yticklabels=3)
    plt.xlabel('size', fontsize=18)
    plt.ylabel('color', fontsize=18)
    xticklabels = [str(int(x.get_text()) + min_size) for x in ax.get_xticklabels()]
    ax.set_xticklabels(xticklabels)
    yticklabels = [str(int(y.get_text())*color_bin + min_color) for y in ax.get_yticklabels()]
    ax.set_yticklabels(yticklabels)
    plt.tight_layout()
    plt.savefig("selection_bias.png", dpi=400)
    plt.clf()


def vocab_analysis(chat_data):
    # Data from OneCommon
    onecommon_vocab = Counter()
    onecommon_pos2vocab = defaultdict(Counter)
    onecommon_pos = Counter()
    with open(os.path.join("onecommon", "final_transcripts.json"), "r") as f:
        onecommon_data = json.load(f)

    for chat in onecommon_data:
        chat_id = chat['uuid']
        events = chat['events']
        for event in events:
            if event['action'] == "message":
                msg = event['data']
                for w, p in pos_tag(word_tokenize(msg)):
                    onecommon_vocab[w.lower()] += 1
                    onecommon_pos2vocab[p][w.lower()] += 1
                    onecommon_pos[p] += 1

    onecommon_total_words = sum(onecommon_vocab.values())

    # Data from Dynamic OneCommon
    vocab = Counter()
    pos2vocab = defaultdict(Counter)
    pos = Counter()

    for chat_id in chat_data:
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'message':
                msg = event['data']
                for w, p in pos_tag(word_tokenize(msg)):
                    vocab[w.lower()] += 1
                    pos2vocab[p][w.lower()] += 1
                    pos[p] += 1

    total_words = sum(vocab.values())

    print("\nCommon Word Occupancy")
    onecommon_top_vocab = [w for w in onecommon_vocab.keys() if onecommon_vocab[w] >= 10]
    top_vocab = [w for w in vocab.keys() if vocab[w] >= 10]
    print("Dynamic: {:.4f}%".format(sum([vocab[w] for w in top_vocab]) / total_words))
    print("OneCommon: {:.4f}%".format(sum([onecommon_vocab[w] for w in onecommon_top_vocab]) / onecommon_total_words))

    print("\nPOS Frequency Rate")
    for p in pos.keys():
        if onecommon_pos[p] > 0:
            print("{}: {:.2f}".format(p, (pos[p] / total_words) / (onecommon_pos[p] / onecommon_total_words)))
        else:
            
            print("{}: NaN")

    print("\nPOS Variety Rate")
    for p in pos.keys():
        if len(onecommon_pos2vocab[p]) > 0:
            pos_variety = len([w for w in pos2vocab[p].keys() if vocab[w] >= 10])
            onecommon_pos_variety = len([w for w in onecommon_pos2vocab[p].keys() if vocab[w] >= 10])
            print("{}: {:.2f}".format(p, pos_variety / onecommon_pos_variety))
        else:
            print("{}: NaN")

    print("\nVocab Overlap")
    vocab_union = set(vocab.keys()).union(onecommon_vocab.keys())
    vocab_intersection = set(vocab.keys()).intersection(onecommon_vocab.keys())
    print("{:.2f}%".format(100.0 * len(vocab_intersection) / len(vocab_union)))
    common_vocab_union = set([w for w in vocab.keys() if vocab[w] >= 10]).union([w for w in onecommon_vocab.keys() if onecommon_vocab[w] >= 10])
    common_vocab_intersection = set([w for w in vocab.keys() if vocab[w] >= 10]).intersection([w for w in onecommon_vocab.keys() if onecommon_vocab[w] >= 10])
    print("{:.2f}% (common words only)".format(100.0 * len(common_vocab_intersection) / len(common_vocab_union)))


def utterance_analysis(chat_data):
    # Data from OneCommon
    onecommon_utterance_length = Counter()
    onecommon_total_utterances = 0
    onecommon_total_short_utterances = 0
    onecommon_total_words_in_long_utterances = 0
    with open(os.path.join("onecommon", "final_transcripts.json"), "r") as f:
        onecommon_data = json.load(f)

    for chat in onecommon_data:
        chat_id = chat['uuid']
        events = chat['events']
        for event in events:
            if event['action'] == "message":
                msg = event['data']
                tokens = word_tokenize(msg)
                onecommon_utterance_length[len(tokens)] += 1
                onecommon_total_utterances += 1
                if len(tokens) < 5:
                    onecommon_total_short_utterances += 1
                else:
                    onecommon_total_words_in_long_utterances += len(tokens)

    # Data from Dynamic OneCommon
    utterance_length = Counter()
    total_utterances = 0
    total_short_utterances = 0
    total_words_in_long_utterances = 0

    for chat_id in chat_data:
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'message':
                msg = event['data']
                tokens = word_tokenize(msg)
                utterance_length[len(tokens)] += 1
                total_utterances += 1
                if len(tokens) < 5:
                    total_short_utterances += 1
                else:
                    total_words_in_long_utterances += len(tokens)

    print("OneCommon short utterances rate: {:.4f}".format(onecommon_total_short_utterances / onecommon_total_utterances))
    print("Dynamic short utterances rate: {:.4f}".format(total_short_utterances / total_utterances))

    print("")

    print("OneCommon long utterances length: {:.2f}".format(onecommon_total_words_in_long_utterances / (onecommon_total_utterances - onecommon_total_short_utterances)))
    print("Dynamic long utterances length: {:.2f}".format(total_words_in_long_utterances / (total_utterances - total_short_utterances)))


def nuance_analysis(chat_data, nuance_dict):
    # Data from OneCommon
    onecommon_vocab = Counter()
    onecommon_total_turns = 0
    with open(os.path.join("onecommon", "final_transcripts.json"), "r") as f:
        onecommon_data = json.load(f)

    for chat in onecommon_data:
        chat_id = chat['uuid']
        events = chat['events']
        for event in events:
            if event['action'] == "message":
                msg = event['data']
                uni = [w.lower() for w in word_tokenize(msg)]
                bi = list((bigrams(uni)))
                onecommon_vocab.update(uni)
                onecommon_vocab.update(bi)
                onecommon_total_turns += 1

    onecommon_total_words = sum(onecommon_vocab.values())

    # Data from Dynamic OneCommon
    vocab = Counter()
    total_turns = 0
    for chat_id in chat_data:
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'message':
                msg = event['data']
                uni = [w.lower() for w in word_tokenize(msg)]
                bi = list((bigrams(uni)))
                vocab.update(uni)
                vocab.update(bi)
                total_turns += 1

    total_words = sum(vocab.values())

    print("OneCommon")
    for nuance_type in nuance_dict.keys():
        onecommon_count_type = 0
        for key_word in nuance_dict[nuance_type]:
            if type(key_word) == dict:
                # key word is a bigram
                onecommon_count_type += onecommon_vocab[tuple([key_word['0'], key_word['1']])]
            else:
                # key word is a unigram
                onecommon_count_type += onecommon_vocab[key_word]
        print("{}: {:.2f} per 100 utterances ({})".format(nuance_type, 100.0 * onecommon_count_type / onecommon_total_turns, len(nuance_dict[nuance_type])))

    print("Dynamic OneCommon")
    for nuance_type in nuance_dict.keys():
        count_type = 0
        for key_word in nuance_dict[nuance_type]:
            if type(key_word) == dict:
                # key word is a bigram
                count_type += vocab[tuple([key_word['0'], key_word['1']])]
            else:
                # key word is a unigram
                count_type += vocab[key_word]
        print("{}: {:.2f} per 100 utterances ({})".format(nuance_type, 100.0 * count_type / total_turns, len(nuance_dict[nuance_type])))


def grounding_analysis(chat_data, scenarios):
    first_turn_success = 0
    first_turn_total = 0
    first_turn_num_shared_success = Counter()
    first_turn_num_shared_total = Counter()
    first_turn_utterances = 0
    first_turn_words = 0

    same_target_success = 0
    same_target_total = 0
    same_target_utterances = 0
    same_target_words = 0
    same_target_num_shared_success = Counter()
    same_target_num_shared_total = Counter()

    change_target_success = 0
    change_target_total = 0
    change_target_utterances = 0
    change_target_words = 0
    change_target_num_shared_success = Counter()
    change_target_num_shared_total = Counter()

    for chat_id in chat_data:
        scenario_id = chat_data[chat_id]["scenario_id"]
        agent_ids = [agent_info["agent_id"] for agent_info in chat_data[chat_id]["agents_info"]]
        outcome = int(chat_data[chat_id]["outcome"])
        start_time = chat_data[chat_id]["time"]["start_time"]
        duration = chat_data[chat_id]["time"]["duration"]
        
        selections = []
        prev_turn = -1
        for event in chat_data[chat_id]["events"]:
            if event['action'] == 'select':
                turn = event['turn']
                ent_id = event['data']
                if prev_turn < turn:
                    prev_turn = turn
                    if turn < outcome: # success
                        selections.append(ent_id)

        if len(selections) < outcome:
            print(chat_id)
            continue

        scenario = scenarios[scenario_id]

        agents = []
        entities = []

        # load agents/entities
        for agent_id in [0, 1]:
            agents.append(Agent.from_dict(scenario["agents"][agent_id]))
        for entity_id in scenario["entities"].keys():
            entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))        

        # first turn analysis
        first_turn_total += 1
        common_ids = set([int(ent.id) for ent in agents[0].observe_at_timestep(entities, 1)]).intersection(set([int(ent.id) for ent in agents[1].observe_at_timestep(entities, 1)]))
        num_shared = len(common_ids)
        first_turn_num_shared_total[num_shared] += 1
        if outcome > 0:
            common_ids = set([int(ent.id) for ent in agents[0].observe_at_timestep(entities, 1)]).intersection(set([int(ent.id) for ent in agents[1].observe_at_timestep(entities, 1)]))
            first_turn_success += 1
            first_turn_num_shared_success[num_shared] += 1

        for event in chat_data[chat_id]["events"]:
            if event['turn'] == 0 and event['action'] == 'message':
                msg = event['data']
                first_turn_words += len([w.lower() for w in word_tokenize(msg)])
                first_turn_utterances += 1

        for turn in range(1, min(outcome + 1, 5)):
            common_ids = set([int(ent.id) for ent in agents[0].observe_at_timestep(entities, turn + 1)]).intersection(set([int(ent.id) for ent in agents[1].observe_at_timestep(entities, turn + 1)]))
            success = turn < outcome

            if selections[turn - 1] in common_ids:
                # same target analysis
                same_target_total += 1
                if success:
                    same_target_success += 1

                num_shared = len(common_ids)
                same_target_num_shared_total[num_shared] += 1
                if success:
                     same_target_num_shared_success[num_shared] += 1

                for event in chat_data[chat_id]["events"]:
                    if event['turn'] == turn and event['action'] == 'message':
                        msg = event['data']
                        same_target_words += len([w.lower() for w in word_tokenize(msg)])
                        same_target_utterances += 1
            else:
                # change target analysis
                change_target_total += 1
                if success:
                    change_target_success += 1

                num_shared = len(common_ids)
                change_target_num_shared_total[num_shared] += 1
                if success:
                     change_target_num_shared_success[num_shared] += 1

                for event in chat_data[chat_id]["events"]:
                    if event['turn'] == turn and event['action'] == 'message':
                        msg = event['data']
                        change_target_words += len([w.lower() for w in word_tokenize(msg)])
                        change_target_utterances += 1

    print("first turn success: {:.2f}% (total {})".format(100.0 * first_turn_success / first_turn_total, first_turn_total))
    for num_shared in sorted(first_turn_num_shared_total.keys()):
        print("\tnum shared {} success: {:.2f}% (total {})".format(num_shared, 100.0 * first_turn_num_shared_success[num_shared] / first_turn_num_shared_total[num_shared], first_turn_num_shared_total[num_shared]))
    print("Avg. utterances: {:.2f}".format(first_turn_utterances / first_turn_total))
    print("Avg. words per utterance: {:.2f}".format(first_turn_words / first_turn_utterances))
    print("")

    print("same target success: {:.2f}% (total {})".format(100.0 * same_target_success / same_target_total, same_target_total))
    for num_shared in sorted(same_target_num_shared_total.keys()):
        print("\tnum shared {} success: {:.2f}% (total {})".format(num_shared, 100.0 * same_target_num_shared_success[num_shared] / same_target_num_shared_total[num_shared], same_target_num_shared_total[num_shared]))
    print("Avg. utterances: {:.2f}".format(same_target_utterances / same_target_total))
    print("Avg. words per utterance: {:.2f}".format(same_target_words / same_target_utterances))
    print("")

    print("change target success: {:.2f}% (total {})".format(100.0 * change_target_success / change_target_total, change_target_total))
    for num_shared in sorted(change_target_num_shared_total.keys()):
        print("\tnum shared {} success: {:.2f}% (total {})".format(num_shared, 100.0 * change_target_num_shared_success[num_shared] / change_target_num_shared_total[num_shared], change_target_num_shared_total[num_shared]))
    print("Avg. utterances: {:.2f}".format(change_target_utterances / change_target_total))
    print("Avg. words per utterance: {:.2f}".format(change_target_words / change_target_utterances))
    print("")

    print("later turn success: {:.2f}% (total {})".format(100.0 * (same_target_success + change_target_success) / (same_target_total + change_target_total), same_target_total + change_target_total))


def worker_analysis(chat_data, num_top_workers=10, poor_worker_threshold=1.5):
    worker_count = Counter()
    worker_outcomes = defaultdict(list)
    poor_workers = set()

    for chat_id in chat_data:
        outcome = int(chat_data[chat_id]["outcome"])
        for agent_info in chat_data[chat_id]["agents_info"]:
            worker_id = agent_info["agent_id"]
            worker_count[worker_id] += 1
            worker_outcomes[worker_id].append(outcome)

    for worker_id, total in worker_count.most_common():
        if total >= 10:
            print("{}: {:.2f} (total {})".format(worker_id, np.mean(worker_outcomes[worker_id]), total))
        if np.mean(worker_outcomes[worker_id]) < poor_worker_threshold:
            poor_workers.add(worker_id)

    # sort by average outcome
    worker_outcomes = {k : v for k, v in sorted(worker_outcomes.items(), key=lambda item: -np.mean(item[1]))}
    # top workers
    top_worker_ids = []
    for worker_id, outcomes in sorted(worker_outcomes.items(), key=lambda item: -np.mean(item[1])):
        if len(outcomes) >= 10:
            top_worker_ids.append(worker_id)
        if len(top_worker_ids) >= num_top_workers:
            break
    top_worker_outcomes = []
    for worker_id in top_worker_ids:
        top_worker_outcomes += worker_outcomes[worker_id]

    # filter outcomes based on poor workers
    filtered_outcomes = []
    for chat_id in chat_data:
        outcome = int(chat_data[chat_id]["outcome"])
        is_valid = True
        for agent_info in chat_data[chat_id]["agents_info"]:
            worker_id = agent_info["agent_id"]
            if worker_id in poor_workers:
                is_valid = False
                break
        if is_valid:
            filtered_outcomes.append(outcome)

    print("")
    print("filtering threshold: {:.2f}".format(poor_worker_threshold))
    print("filtered outcome: {:.2f} (total {})".format(np.mean(filtered_outcomes), len(filtered_outcomes)))
    print("unique workers: {}".format(len(worker_count.keys())))
    print("ceiling performance (top {} workers): {:.2f}".format(num_top_workers, np.mean(top_worker_outcomes)))

def anonymize_workers():
    def _hash_worker_id(worker_id):
        return "H_" + hashlib.md5(worker_id.encode("utf-8")).hexdigest()

    for mturk_dir in ["mturk", "mturk_2", "mturk_3"]:
        for transcript_file in ["transcripts.json", "accepted-transcripts.json"]:
            with open(os.path.join(mturk_dir, transcript_file), "r") as f:
                chat_data = json.load(f)
            for chat_id in chat_data:
                for agent_idx, agent_info in enumerate(chat_data[chat_id]["agents_info"]):
                    worker_id = agent_info["agent_id"]
                    if worker_id.startswith("MT_"): # if not anonymized yet
                        chat_data[chat_id]["agents_info"][agent_idx]['agent_id'] = _hash_worker_id(worker_id)

                for event in chat_data[chat_id]["events"]:
                    if event['action'] == 'success':
                        worker_id = event['data']
                        if worker_id.startswith("MT_"): # if not anonymized yet
                            event['data'] = _hash_worker_id(worker_id)

            with open(os.path.join(mturk_dir, "anon-" + transcript_file), "w") as fout:
                json.dump(chat_data, fout, indent=4)


def output_text(chat_data, scenarios_1, scenarios_2, scenarios_3):
    with open('mturk/chat.txt', 'w') as mturk, open('mturk_2/chat_2.txt', 'w') as mturk_2, open('mturk_3/chat_3.txt', 'w') as mturk_3:
        for chat_id in chat_data:
            scenario_id = chat_data[chat_id]["scenario_id"]
            agent_ids = [agent_info["agent_id"] for agent_info in chat_data[chat_id]["agents_info"]]
            outcome = int(chat_data[chat_id]["outcome"])
            start_time = chat_data[chat_id]["time"]["start_time"]
            duration = chat_data[chat_id]["time"]["duration"]
            
            dialogue = "{} ({})\n".format(chat_id, scenario_id)
            current_turn = -1
            for event in chat_data[chat_id]["events"]:
                turn = event['turn']
                if current_turn < turn:
                    dialogue += "\tTurn {}\n".format(turn + 1)
                    current_turn = turn
                if event['action'] == 'message':
                    dialogue += "\t\t" + str(event['agent']) + ":" + event['data'] + "\n"
            dialogue += "\tOutcome: {}\n".format(outcome)
            dialogue += "\n"

            if scenario_id in scenarios_1:
                mturk.write(dialogue)
            elif scenario_id in scenarios_2:
                mturk_2.write(dialogue)
            elif scenario_id in scenarios_3:
                mturk_3.write(dialogue)
            else:
                assert False, "something is wrong"


def estimate_outcome(first_prob, later_prob, max_turns=5):
    estimated_outcome = 0
    for i in range(max_turns):
        estimated_outcome += first_prob * (later_prob ** i)
    return estimated_outcome

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript_file', type=str,
                        default="accepted-transcripts.json")
    parser.add_argument('--scenarios_file', type=str,
                        default="scenarios.json")
    parser.add_argument('--scenarios_2_file', type=str,
                        default="scenarios_2.json")
    parser.add_argument('--scenarios_3_file', type=str,
                        default="scenarios_3.json")
    parser.add_argument('--scenario_svgs_file', type=str,
                        default="scenario_svgs.json")
    parser.add_argument('--scenario_svgs_2_file', type=str,
                        default="scenario_svgs_2.json")
    parser.add_argument('--scenario_svgs_3_file', type=str,
                        default="scenario_svgs_3.json")
    parser.add_argument('--nuance_dict', type=str,
                        default="nuance_dict.json")

    # analyses to conduct
    parser.add_argument('--basic_statistics', action='store_true', default=False)
    parser.add_argument('--plot_success', action='store_true', default=False)
    parser.add_argument('--count_scenarios', action='store_true', default=False)
    parser.add_argument('--plot_selection', action='store_true', default=False)
    parser.add_argument('--vocab_analysis', action='store_true', default=False)
    parser.add_argument('--utterance_analysis', action='store_true', default=False)
    parser.add_argument('--nuance_analysis', action='store_true', default=False)
    parser.add_argument('--grounding_analysis', action='store_true', default=False)
    parser.add_argument('--worker_analysis', action='store_true', default=False)
    parser.add_argument('--anonymize_workers', action='store_true', default=False)
    parser.add_argument('--output_text', action='store_true', default=False)

    args = parser.parse_args()

    scenarios = {} # all scenarios

    with open(os.path.join("mturk", args.transcript_file), "r") as f:
        chat_data = json.load(f)
    with open(os.path.join("mturk_2", args.transcript_file), "r") as f:
        chat_data_2 = json.load(f)
        chat_data.update(chat_data_2)
    with open(os.path.join("mturk_3", args.transcript_file), "r") as f:
        chat_data_3 = json.load(f)
        chat_data.update(chat_data_3)
    with open(os.path.join("scenarios", args.scenarios_file), "r") as f:
        scenarios_1 = json.load(f)["scenarios"]
        scenarios.update(scenarios_1)
    with open(os.path.join("scenarios", args.scenarios_2_file), "r") as f:
        scenarios_2 = json.load(f)["scenarios"]
        scenarios.update(scenarios_2)
    with open(os.path.join("scenarios", args.scenarios_3_file), "r") as f:
        scenarios_3 = json.load(f)["scenarios"]
        scenarios.update(scenarios_3)
    with open(os.path.join("scenarios", args.scenario_svgs_file), "r") as f:
        scenario_svgs = json.load(f)
    with open(os.path.join("scenarios", args.scenario_svgs_2_file), "r") as f:
        scenario_svgs_2 = json.load(f)
        scenario_svgs.update(scenario_svgs_2)
    with open(os.path.join("scenarios", args.scenario_svgs_3_file), "r") as f:
        scenario_svgs_3 = json.load(f)
        scenario_svgs.update(scenario_svgs_3)
    with open(args.nuance_dict, "r") as f:
        nuance_dict = json.load(f)

    #remove erroneuous sample with outcome < 0
    del chat_data["C_7f85e958c86f40a9a45a469d28df9e2c"]

    if args.basic_statistics:
        basic_statistics(chat_data, scenarios, scenario_svgs)

    if args.plot_success:
        plot_success(chat_data)

    if args.count_scenarios:
        count_scenarios(chat_data, scenarios, scenario_svgs)

    if args.plot_selection:
        plot_selection(chat_data, scenarios)

    if args.vocab_analysis:
        vocab_analysis(chat_data)

    if args.utterance_analysis:
        utterance_analysis(chat_data)

    if args.nuance_analysis:
        nuance_analysis(chat_data, nuance_dict)

    if args.grounding_analysis:
        grounding_analysis(chat_data, scenarios)

    if args.worker_analysis:
        worker_analysis(chat_data)

    if args.anonymize_workers:
        anonymize_workers()

    if args.output_text:
        output_text(chat_data, scenarios_1, scenarios_2, scenarios_3)

