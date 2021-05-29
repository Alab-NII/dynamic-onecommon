import argparse
from collections import defaultdict, Counter
from datetime import datetime
import json
import math
import os
import sys
import pdb
import re

from itertools import combinations

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


def output_raw(chat_data, scenarios, sample_size=100):
    chat_ids = list(chat_data.keys())
    np.random.shuffle(chat_ids)
    sample_chat_ids = []
    target_selection_annotation = {}
    spatio_temporal_annotation = {}
    for chat_id in chat_ids:
        scenario_id = chat_data[chat_id]["scenario_id"]
        agent_ids = [agent_info["agent_id"] for agent_info in chat_data[chat_id]["agents_info"]]
        outcome = int(chat_data[chat_id]["outcome"])
        start_time = chat_data[chat_id]["time"]["start_time"]
        duration = chat_data[chat_id]["time"]["duration"]

        # skip samples with outcome < 2
        if outcome < 2:
            continue
        
        sample_agent_id = str(np.random.choice([0, 1]))

        target_selection_annotation[chat_id] = {}
        target_selection_annotation[chat_id][sample_agent_id] = [] # use agent 0

        spatio_temporal_annotation[chat_id] = {}

        sample_chat_ids.append(chat_id)
        if len(sample_chat_ids) >= sample_size:
            break

    if not os.path.exists("raw"):
        os.mkdir("raw")
    with open("raw/target_selection_annotation.json", "w") as fout:
        json.dump(target_selection_annotation, fout, indent=4, sort_keys=True)
    with open("raw/spatio_temporal_annotation.json", "w") as fout:
        json.dump(spatio_temporal_annotation, fout, indent=4, sort_keys=True)


def annotation_statistics(chat_data, scenarios, remove_practice_chat_ids=True):
    practice_chat_ids = ["C_046c759fd715445da35f43508b413e14", "C_098779c4ce474a438ed315e185383bbb", 
                         "C_1596a4eb70554927a82e1c972d329c97", "C_17bceff505034c6baa365ba97526c428",
                         "C_1db3aefa9c314a13994fd78c9227dff5", "C_278c5880cf35407a94195b4f3beb41b6",
                         "C_2a75c9c07f584690b79200e87482e8dc", "C_2b4acc9f52a04c6391bf4219b45907d1",
                         "C_305edb6ec99f416a8a17be18ae86d02e", "C_3bbdbef68ca842569808dfd9a88579f5"]
    annotators = ["admin", "annotator_B", "annotator_C"]

    print("=== Statistics for Target Selection ===\n")
    accuracies = []
    first_turn_accuracies = []
    same_target_accuracies = []
    change_target_accuracies = []
    later_turn_accuracies = []
    for annotator in annotators:
        with open(os.path.join("annotated", annotator + "_target_selection_annotation.json"), "r") as f:
            target_selection_annotation = json.load(f)
        first_turn_correct = 0
        same_target_correct = 0
        change_target_correct = 0
        later_turn_correct = 0
        first_turn_total = 0
        same_target_total = 0
        change_target_total = 0
        later_turn_total = 0

        for chat_id in target_selection_annotation.keys():
            if remove_practice_chat_ids and chat_id in practice_chat_ids:
                continue

            scenario_id = chat_data[chat_id]["scenario_id"]
            scenario = scenarios[scenario_id]

            agents = []
            entities = []
            for agent_id in [0, 1]:
                agents.append(Agent.from_dict(scenario["agents"][agent_id]))
            for entity_id in scenario["entities"].keys():
                entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))

            sample_agent_id = int(list(target_selection_annotation[chat_id].keys())[0])
            agent = agents[sample_agent_id]

            gold_selections = []
            for chat_event in chat_data[chat_id]['events']:
                if chat_event['action'] == 'select':
                    if chat_event['agent'] == sample_agent_id:
                        turn = chat_event['turn']
                        if len(gold_selections) <= turn:
                            gold_selections.append("agt_" + str(sample_agent_id) +"_ent_" + str(chat_event['data']))
                        else:
                            gold_selections[turn] = "agt_" + str(sample_agent_id) +"_ent_" + str(chat_event['data'])

            max_turns = len(gold_selections)
            assert max_turns == len(target_selection_annotation[chat_id][str(sample_agent_id)])
            for turn in range(max_turns):
                selectable_ent_ids = []
                for i in range(len(entities)):
                    if distance(agent.loc_at_timestep(turn + 1), entities[i].loc_at_timestep(turn + 1)) < agent.r:
                        selectable_ent_ids.append("agt_" + str(sample_agent_id) +"_ent_" + entities[i].id)
                assert len(selectable_ent_ids) == 7

                correct = (target_selection_annotation[chat_id][str(sample_agent_id)][turn] == gold_selections[turn])
                if turn == 0:
                    if correct:
                        first_turn_correct += 1
                    first_turn_total += 1
                else:
                    if correct:
                        later_turn_correct += 1
                    later_turn_total += 1
                    if gold_selections[turn - 1] in selectable_ent_ids:
                        if correct:
                            same_target_correct += 1
                        same_target_total += 1
                    else:
                        if correct:
                            change_target_correct += 1
                        change_target_total += 1

        accuracy = 100.0 * (first_turn_correct + same_target_correct + change_target_correct) / (first_turn_total + same_target_total + change_target_total)
        first_turn_accuracy = 100.0 * first_turn_correct / first_turn_total
        same_target_accuracy = 100.0 * same_target_correct / same_target_total
        change_target_accuracy = 100.0 * change_target_correct / change_target_total
        later_turn_accuracy = 100.0 * later_turn_correct / later_turn_total
        accuracies.append(accuracy)
        first_turn_accuracies.append(first_turn_accuracy)
        same_target_accuracies.append(same_target_accuracy)
        change_target_accuracies.append(change_target_accuracy)
        later_turn_accuracies.append(later_turn_accuracy)
        print("annotator: {}".format(annotator))
        print("first turn accuracy: {:.2f} ({})".format(first_turn_accuracy, first_turn_total))
        print("same target accuracy: {:.2f} ({})".format(same_target_accuracy, same_target_total))
        print("change target accuracy: {:.2f} ({})".format(change_target_accuracy, change_target_total))
        print("later turn accuracy: {:.2f} ({})".format(later_turn_accuracy, later_turn_total))
        print("")

    print("overall accuracies: {:.2f} (std. {:.2f})".format(np.mean(accuracies), np.std(accuracies)))
    print("overall first turn accuracy: {:.2f} (std. {:.2f})".format(np.mean(first_turn_accuracies), np.std(first_turn_accuracies)))
    print("overall same target accuracy: {:.2f} (std. {:.2f})".format(np.mean(same_target_accuracies), np.std(same_target_accuracies)))
    print("overall change target accuracy: {:.2f} (std. {:.2f})".format(np.mean(change_target_accuracies), np.std(change_target_accuracies)))
    print("overall later turn accuracy: {:.2f} (std. {:.2f})".format(np.mean(later_turn_accuracies), np.std(later_turn_accuracies)))

    print("\n=== Statistics for Spatio-temporal Expressions ===\n")
    with open(os.path.join("annotated", "admin_spatio_temporal_annotation.json"), "r") as f:
        spatio_temporal_annotation = json.load(f)

    previous = Counter()
    movement = Counter()
    current = Counter()
    total_utterances = 0
    for chat_id in spatio_temporal_annotation.keys():
        utterances = []
        for chat_event in chat_data[chat_id]['events']:
            if chat_event['action'] == 'message':
                utterances.append(chat_event['data'])

        for utterance_id in spatio_temporal_annotation[chat_id].keys():
            utterance = utterances[int(utterance_id.split("_")[-1])]
            if spatio_temporal_annotation[chat_id][utterance_id]["previous"]:
                previous[utterance] += 1
            if spatio_temporal_annotation[chat_id][utterance_id]["movement"]:
                movement[utterance] += 1
            if spatio_temporal_annotation[chat_id][utterance_id]["current"]:
                current[utterance] += 1

        total_utterances += len(utterances)

    total_previous = sum(previous.values())
    total_movement = sum(movement.values())
    total_current = sum(current.values())
    print("Previous: {:.2f}% (relative: {:.2f}%)".format(100.0 * total_previous / total_utterances, 100.0 * total_previous / (total_previous + total_movement + total_current)))
    print("Movement: {:.2f}% (relative: {:.2f}%)".format(100.0 * total_movement / total_utterances,100.0 * total_movement / (total_previous + total_movement + total_current)))
    print("Current: {:.2f}% (relative: {:.2f}%)".format(100.0 * total_current / total_utterances,100.0 * total_current / (total_previous + total_movement + total_current)))

def annotation_agreement(chat_data, scenarios, remove_practice_chat_ids=True):
    practice_chat_ids = ["C_046c759fd715445da35f43508b413e14", "C_098779c4ce474a438ed315e185383bbb", 
                         "C_1596a4eb70554927a82e1c972d329c97", "C_17bceff505034c6baa365ba97526c428",
                         "C_1db3aefa9c314a13994fd78c9227dff5", "C_278c5880cf35407a94195b4f3beb41b6",
                         "C_2a75c9c07f584690b79200e87482e8dc", "C_2b4acc9f52a04c6391bf4219b45907d1",
                         "C_305edb6ec99f416a8a17be18ae86d02e", "C_3bbdbef68ca842569808dfd9a88579f5"]
    annotators = ["admin", "annotator_B", "annotator_C"]
    reference_types = ["previous", "movement", "current"]

    spatio_temporal_annotations = {}
    for annotator in annotators:
        with open(os.path.join("annotated", annotator + "_spatio_temporal_annotation.json"), "r") as f:
            spatio_temporal_annotations[annotator] = json.load(f)

    agreed = {}
    positive = {}
    for reference_type in reference_types:
        agreed[reference_type] = 0
        positive[reference_type] = Counter()
    total_utterances = 0
    total_chats = 0

    for chat_id in spatio_temporal_annotations["annotator_B"].keys():
        if remove_practice_chat_ids and chat_id in practice_chat_ids:
                continue

        utterances = []
        for chat_event in chat_data[chat_id]['events']:
            if chat_event['action'] == 'message':
                utterances.append(chat_event['data'])

        for utterance_id in spatio_temporal_annotations["annotator_B"][chat_id].keys():
            utterance = utterances[int(utterance_id.split("_")[-1])]

            is_reference_type = Counter()
            is_movement = 0
            is_current = 0
            for annotator in annotators:
                for reference_type in reference_types:
                    if spatio_temporal_annotations[annotator][chat_id][utterance_id][reference_type]:
                        is_reference_type[reference_type] += 1
                        positive[reference_type][annotator] += 1         

            for reference_type in reference_types:
                if is_reference_type[reference_type] in [0, 3]:
                    agreed[reference_type] += 3
                else:
                    agreed[reference_type] += 1

        total_utterances += len(utterances)
        total_chats += 1

    print("total chats: {}".format(total_chats))

    for reference_type in reference_types:
        print("\n" + reference_type)
        observed_agreement = agreed[reference_type] / (total_utterances * 3)
        expected_pairwise_agreements = []
        for annotator_1, annotator_2 in combinations(annotators, 2):
            positive_rate_1 = positive[reference_type][annotator_1] / total_utterances
            positive_rate_2 = positive[reference_type][annotator_2] / total_utterances
            expected_pairwise_agreements.append(positive_rate_1 * positive_rate_2 + (1 - positive_rate_1) * (1 - positive_rate_2))
        expected_agreement = np.mean(expected_pairwise_agreements)
        cohens_kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

        print("observed agreement: {}".format(observed_agreement))
        print("expected agreement: {}".format(expected_agreement))
        print("Cohen's Kappa: {}".format(cohens_kappa))

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

    # analyses to conduct
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--sample_size', type=int, default=100)
    parser.add_argument('--output_raw', action='store_true', default=False)
    parser.add_argument('--annotation_statistics', action='store_true', default=False)
    parser.add_argument('--annotation_agreement', action='store_true', default=False)

    args = parser.parse_args()
    np.random.seed(args.seed)

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

    #check_finished(chat_data, scenarios, scenario_svgs)

    #remove erroneuous sample with outcome < 0
    del chat_data["C_7f85e958c86f40a9a45a469d28df9e2c"] 

    if args.output_raw:
        output_raw(chat_data, scenarios, sample_size=args.sample_size)

    if args.annotation_statistics:
        annotation_statistics(chat_data, scenarios)

    if args.annotation_agreement:
        annotation_agreement(chat_data, scenarios)
