import argparse
from collections import defaultdict, Counter
import copy
from datetime import datetime
import json
import math
import os
import sys
import pdb
import re

from nltk import word_tokenize, pos_tag, bigrams
import numpy as np
from tqdm import tqdm

from utils import *

def create_context(args, agent, entities, t, previous_selectable_ent_ids, previous_selected_ent_id):
    context = {}
    selectable_ent_ids = []
    expand_scale = 1 / agent.r

    """
        Agent moves
    """
    agent_move_ents = []
    if t > 0:
        candidate_ents = []
        for i in range(len(entities)):
            if distance(agent.loc_at_timestep(t), entities[i].loc_at_timestep(t)) < agent.r + args.delta_dist_range:
                candidate_ents.append(entities[i])

        for candidate_ent in candidate_ents:
            candidate_ent_id = "ent_" + str(candidate_ent.id)

            # final results to add
            xs = []
            ys = []
            visible = []
            color = (candidate_ent.color - args.base_color) / args.color_range
            size = (candidate_ent.size - args.base_size) / args.size_range
            previous_selectable = candidate_ent_id in previous_selectable_ent_ids
            previous_selected = (candidate_ent_id == previous_selected_ent_id)

            ent_x, ent_y = candidate_ent.loc_at_timestep(t)
            diff_x = agent.xs[t + 1] - agent.xs[t]
            diff_y = agent.ys[t + 1] - agent.ys[t]

            # initial condition
            if distance((ent_x, ent_y), (agent.xs[t], agent.ys[t])) < agent.r:
                visible.append(True)
                xs.append(expand_scale * (ent_x - agent.xs[t]))
                ys.append(expand_scale * (ent_y - agent.ys[t]))
            else:
                visible.append(False)
                xs.append(0)
                ys.append(0)

            frame_step = 1 / args.agt_move_frames
            for i in range(1, args.agt_move_frames):
                # if visible
                if distance((ent_x - diff_x * i * frame_step, ent_y - diff_y * i * frame_step), (agent.xs[t], agent.ys[t])) < agent.r:
                    visible.append(True)
                    xs.append(expand_scale * (ent_x - diff_x * i * frame_step - agent.xs[t]))
                    ys.append(expand_scale * (ent_y - diff_y * i * frame_step) - agent.ys[t])
                else:
                    visible.append(False)
                    xs.append(0)
                    ys.append(0)

            if distance((ent_x, ent_y), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
                visible.append(True)
                xs.append(expand_scale * (ent_x - agent.xs[t + 1]))
                ys.append(expand_scale * (ent_y - agent.ys[t + 1]))
            else:
                visible.append(False)
                xs.append(0)
                ys.append(0)

            if any(visible):
                context[candidate_ent_id] = {}
                context[candidate_ent_id]["xs"] = xs
                context[candidate_ent_id]["ys"] = ys
                context[candidate_ent_id]["visible"] = visible
                context[candidate_ent_id]["color"] = color
                context[candidate_ent_id]["size"] = size
                context[candidate_ent_id]["previous_selectable"] = previous_selectable
                context[candidate_ent_id]["previous_selected"] = previous_selected
                context[candidate_ent_id]["selectable"] = False
                agent_move_ents.append(candidate_ent_id)

    """
        Entity moves
    """
    candidate_ents = []
    for i in range(len(entities)):
        if distance(agent.loc_at_timestep(t + 1), entities[i].loc_at_timestep(t)) < agent.r + 2 * args.delta_dist_range:
            candidate_ents.append(entities[i])

    for candidate_ent in candidate_ents:
        candidate_ent_id = "ent_" + str(candidate_ent.id)

        # final results to add
        xs = []
        ys = []
        visible = []
        color = (candidate_ent.color - args.base_color) / args.color_range
        size = (candidate_ent.size - args.base_size) / args.size_range
        previous_selectable = candidate_ent_id in previous_selectable_ent_ids
        previous_selected = (candidate_ent_id == previous_selected_ent_id)

        xvals, yvals, dists = bezier_curve([(0,0), candidate_ent.moves[t][0], candidate_ent.moves[t][1]])
        ent_x, ent_y = candidate_ent.loc_at_timestep(t)
        moved_ent_x, moved_ent_y = candidate_ent.loc_at_timestep(t + 1)

        # remove first and last elements
        xvals = xvals[1:-1]
        yvals = yvals[1:-1]
        dist_step = dists[-1] / len(dists)
        visible = []
        # initial condition
        if distance((ent_x, ent_y), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
            visible.append(True)
            xs.append(expand_scale * (ent_x - agent.xs[t + 1]))
            ys.append(expand_scale * (ent_y - agent.ys[t + 1]))
        else:
            visible.append(False)
            xs.append(0)
            ys.append(0)

        frame_step = int(100 / args.ent_move_frames)
        for i in range(1, args.ent_move_frames):
            # if visible
            if distance((ent_x + xvals[i * frame_step], ent_y + yvals[i * frame_step]), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
                visible.append(True)
                xs.append(expand_scale * (ent_x + xvals[i * frame_step] - agent.xs[t + 1]))
                ys.append(expand_scale * (ent_y + yvals[i * frame_step] - agent.ys[t + 1]))
            else:
                visible.append(False)
                xs.append(0)
                ys.append(0)

        if distance((moved_ent_x, moved_ent_y), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
            visible.append(True)
            xs.append(expand_scale * (moved_ent_x - agent.xs[t + 1]))
            ys.append(expand_scale * (moved_ent_y - agent.ys[t + 1]))
            selectable = True
            selectable_ent_ids.append(candidate_ent_id)
        else:
            visible.append(False)
            xs.append(0)
            ys.append(0)
            selectable = False

        if any(visible):
            if t == 0:
                context[candidate_ent_id] = {}
                context[candidate_ent_id]["xs"] = xs
                context[candidate_ent_id]["ys"] = ys
                context[candidate_ent_id]["visible"] = visible
                context[candidate_ent_id]["color"] = color
                context[candidate_ent_id]["size"] = size
                context[candidate_ent_id]["previous_selectable"] = previous_selectable
                context[candidate_ent_id]["previous_selected"] = previous_selected
                context[candidate_ent_id]["selectable"] = selectable
            else:
                if candidate_ent_id not in context:
                    context[candidate_ent_id] = {}
                    context[candidate_ent_id]["xs"] = [0] * (args.agt_move_frames + 1)
                    context[candidate_ent_id]["ys"] = [0] * (args.agt_move_frames + 1)
                    context[candidate_ent_id]["visible"] = [False] * (args.agt_move_frames + 1)
                    context[candidate_ent_id]["color"] = color
                    context[candidate_ent_id]["size"] = size
                    context[candidate_ent_id]["previous_selectable"] = previous_selectable
                    context[candidate_ent_id]["previous_selected"] = previous_selected
                # remove first state
                context[candidate_ent_id]["xs"] += xs[1:]
                context[candidate_ent_id]["ys"] += ys[1:]
                context[candidate_ent_id]["visible"] += visible[1:]
                context[candidate_ent_id]["selectable"] = selectable
        elif candidate_ent_id in agent_move_ents:
            context[candidate_ent_id]["xs"] += xs[1:]
            context[candidate_ent_id]["ys"] += ys[1:]
            context[candidate_ent_id]["visible"] += visible[1:]
            context[candidate_ent_id]["selectable"] = selectable

    if len(context.keys()) > args.max_ent_each_turn:
        all_ent_ids = list(context.keys())
        valid_ent_ids = []

        ent_id2visible_span = Counter()
        for ent_id in all_ent_ids:
            if ent_id in selectable_ent_ids:
                valid_ent_ids.append(ent_id)
            else:
                ent_id2visible_span[ent_id] = sum(context[ent_id]["visible"])

        valid_ent_ids += [item[0] for item in ent_id2visible_span.most_common(args.max_ent_each_turn - len(selectable_ent_ids))]

        for ent_id in all_ent_ids:
            if ent_id not in valid_ent_ids:
                del context[ent_id]

    return context, selectable_ent_ids

def create_utterances_and_selection(args, chat_events, agent_id, current_turn):
    utterances = []
    selection = None

    prev_speaker = None
    for event in chat_events:
        if event['turn'] == current_turn:
            if event['action'] == 'message':
                if prev_speaker != event['agent']:
                    if event['agent'] == agent_id:
                        utterance = ["YOU:"]
                    else:
                        utterance = ["THEM:"]
                    utterance += [w.lower() for w in word_tokenize(event['data'])] + ["<eos>"] # todo: better tokenization
                    utterances.append(utterance)
                    prev_speaker = event['agent']
                else:
                    # same speaker again, remove last <eos>
                    utterances[-1].pop()
                    utterances[-1] += [w.lower() for w in word_tokenize(event['data'])] + ["<eos>"] # todo: better tokenization
            elif event['action'] == 'select':
                if event['agent'] == agent_id:
                    selection = 'ent_' + str(event['data'])

    if prev_speaker is None:
        utterances.append([np.random.choice(["YOU:", "THEM:"]), "<selection>"])
    elif prev_speaker == agent_id:
        utterances.append(["THEM:", "<selection>"])
    else:
        utterances.append(["YOU:", "<selection>"])

    return utterances, selection

def transform_to_model_format(args, chat_data, scenarios):
    # final results to return
    train = {}
    valid = {}
    test = {}

    high_quality_chat_ids = [] # chat_ids with outcome >= 2
    scenario_id2chat_id = defaultdict(list) # map scenario_id to chat_id
    for chat_id in chat_data:
        outcome = int(chat_data[chat_id]["outcome"])
        if outcome >= 2:
            high_quality_chat_ids.append(chat_id)
        scenario_id = chat_data[chat_id]["scenario_id"]
        scenario_id2chat_id[scenario_id].append(chat_id)

    np.random.shuffle(high_quality_chat_ids)

    # add valid and test chat_ids
    for chat_id in high_quality_chat_ids:
        scenario_id = chat_data[chat_id]["scenario_id"]
        if len(scenario_id2chat_id[scenario_id]) > 1:
            continue # do not use duplicated scenario chats

        if len(test.keys()) < args.test_size:
            test[chat_id] = {}
            test[chat_id]["scenario_id"] = scenario_id
            test[chat_id]["agents"] = []
        elif len(valid.keys()) < args.valid_size:
            valid[chat_id] = {}
            valid[chat_id]["scenario_id"] = scenario_id
            valid[chat_id]["agents"] = []
        else:
            break

    # add train chat_ids
    for chat_id in chat_data:
        if (chat_id not in test) and (chat_id not in valid):
            train[chat_id] = {}
            train[chat_id]["scenario_id"] = chat_data[chat_id]["scenario_id"]
            train[chat_id]["agents"] = []

    # add chat data in model format
    max_ent_in_turn = -1
    for chat_id in tqdm.tqdm(chat_data):
        if chat_id in test:
            dataset = test
        elif chat_id in valid:
            dataset = valid
        else:
            dataset = train

        scenario_id = chat_data[chat_id]["scenario_id"]
        scenario = scenarios[scenario_id]

        agents = []
        entities = []
        for agent_id in [0, 1]:
            agents.append(Agent.from_dict(scenario["agents"][agent_id]))
        for entity_id in scenario["entities"].keys():
            entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))

        outcome = int(chat_data[chat_id]["outcome"])
        max_turns = min(outcome + 1, 5)
        for agent_id in [0, 1]:
            agent_info = []
            previous_selectable_ent_ids = []
            previous_selected_ent_id = None
            for turn in range(max_turns):
                turn_info = {}
                turn_info["context"], selectable_ent_ids = create_context(args, agents[agent_id], entities, turn, previous_selectable_ent_ids, previous_selected_ent_id)
                turn_info["utterances"], turn_info["selection"] = create_utterances_and_selection(args, chat_data[chat_id]["events"], agent_id, turn)
                previous_selectable_ent_ids = selectable_ent_ids
                previous_selected_ent_id = turn_info["selection"]
                max_ent_in_turn = max(max_ent_in_turn, len(turn_info["context"].keys()))
                
                # sanity check (if fail, remove from dataset)
                if turn_info["selection"] is None:
                    break
                if turn_info["selection"] not in selectable_ent_ids:
                    break

                agent_info.append(turn_info)

            dataset[chat_id]["agents"].append(agent_info)

    print("maximum entities in single turn: {}".format(max_ent_in_turn))

    return train, valid, test

def transform_for_selplay(args, scenarios):
    # final results to return
    selfplay = {}

    scenario_ids = list(scenarios.keys())

    np.random.shuffle(scenario_ids)

    # add scenario data in model format
    max_ent_in_turn = -1
    for scenario_id in tqdm.tqdm(scenario_ids):
        selfplay[scenario_id] = {}
        selfplay[scenario_id]["agents"] = []

        scenario = scenarios[scenario_id]

        agents = []
        entities = []
        for agent_id in [0, 1]:
            agents.append(Agent.from_dict(scenario["agents"][agent_id]))
        for entity_id in scenario["entities"].keys():
            entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))

        max_turns = 5
        for agent_id in [0, 1]:
            agent_info = []
            previous_selectable_ent_ids = []
            previous_selected_ent_id = None
            for turn in range(max_turns):
                turn_info = {}
                turn_info["context"], selectable_ent_ids = create_context(args, agents[agent_id], entities, turn, previous_selectable_ent_ids, previous_selected_ent_id)
                previous_selectable_ent_ids = selectable_ent_ids
                max_ent_in_turn = max(max_ent_in_turn, len(turn_info["context"].keys()))
                agent_info.append(turn_info)

            selfplay[scenario_id]["agents"].append(agent_info)

    print("maximum entities in single turn: {}".format(max_ent_in_turn))

    return selfplay


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
    parser.add_argument('--scenarios_4_file', type=str,
                        default="scenarios_4.json")
    parser.add_argument('--scenario_svgs_file', type=str,
                        default="scenario_svgs.json")
    parser.add_argument('--scenario_svgs_2_file', type=str,
                        default="scenario_svgs_2.json")
    parser.add_argument('--scenario_svgs_3_file', type=str,
                        default="scenario_svgs_3.json")
    parser.add_argument('--nuance_dict', type=str,
                        default="nuance_dict.json")
    parser.add_argument('--agt_move_frames', type=int, default=5)
    parser.add_argument('--ent_move_frames', type=int, default=10)
    parser.add_argument('--max_ent_each_turn', type=int, default=20)
    parser.add_argument('--valid_size', type=int, default=500)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)

    # --- world parameters --- 
    parser.add_argument('--color_range', type=int, default=150,
        help='range of color')
    parser.add_argument('--size_range', type=int, default=5,
        help='range of size')
    parser.add_argument('--min_dist_ent', type=float, default=0.06,
        help='minimum distance between entities')
    parser.add_argument('--min_shared', type=int, default=4,
        help='minimum number of entities shared in both agents view')
    parser.add_argument('--max_shared', type=int, default=6,
        help='maximum number of entities shared in both agents view')
    parser.add_argument('--num_agt_view', type=int, default=7,
        help='number of entities in each agents view')
    parser.add_argument('--max_timesteps', type=int, default=5,
        help='maximum number of total timesteps')
    parser.add_argument('--num_scenarios', type=int, default=1000,
        help='number of total scenarios to generate')
    parser.add_argument('--max_dist_agt', type=float, default=0.25,
        help='minimum distance between agents')
    parser.add_argument('--delta_theta_range', type=float, default= 2 * math.pi / 3,
        help='range of change in angle')
    parser.add_argument('--delta_dist_range', type=float, default=0.25,
        help='range of change in distance')

    # --- for selplay --- 
    parser.add_argument('--for_selfplay', action='store_true', default=False,
        help='convert senarios for D-OCC selfplay')

    # --- for onecommon --- 
    parser.add_argument('--for_onecommon', action='store_true', default=False,
        help='convert senarios for OCC selfplay')

    # --- for repeat --- 
    parser.add_argument('--for_repeat', action='store_true', default=False,
        help='convert senarios for D-OCC repeat training/testing')

    args = parser.parse_args()

    # current support
    args.num_agents = 2
    args.agt_r = 0.25
    args.world_r = args.agt_r * 2.5
    args.base_color = 128
    args.base_size = 9
    args.agt_loc_r = args.agt_r * 1

    if args.for_selfplay:
        with open(os.path.join("scenarios", args.scenarios_4_file), "r") as f:
            scenarios = json.load(f)["scenarios"]

        selfplay = transform_for_selplay(args, scenarios)

        with open(os.path.join("model_format", "selfplay.json"), "w") as fout:
            json.dump(selfplay, fout, indent=4)

    elif args.for_onecommon:
        with open(os.path.join("model_format", "selfplay.json"), "r") as f:
            scenarios = json.load(f)

        docc_shared = defaultdict(list)
        num_shared_counter = Counter()

        turn = 0 # only consider first turn
        for scenario_id in scenarios.keys():
            agents_input_vals = []
            agents_real_ids = []

            for agent_id in [0, 1]:
                agent_input_vals = []
                agent_real_ids = []
                for ent_id in scenarios[scenario_id]["agents"][agent_id][turn]["context"]:
                    if scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]['selectable']:
                        agent_input_vals.append(str(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]['xs'][-1]))
                        agent_input_vals.append(str(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]['ys'][-1]))
                        agent_input_vals.append(str(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]['size']))
                        agent_input_vals.append(str(scenarios[scenario_id]["agents"][agent_id][turn]["context"][ent_id]['color']))
                        agent_real_ids.append(ent_id.split('_')[1])
                assert len(agent_input_vals) == 7 * 4
                assert len(agent_real_ids) == 7
                agents_input_vals.append(agent_input_vals)
                agents_real_ids.append(agent_real_ids)
            
            num_shared = len(set(agents_real_ids[0]).intersection(agents_real_ids[1]))
            assert num_shared in [4, 5, 6]

            docc_shared[num_shared].append(scenario_id)
            for agent_id in [0, 1]:
                docc_shared[num_shared].append(" ".join(agents_input_vals[agent_id]))
            for agent_id in [0, 1]:
                docc_shared[num_shared].append(" ".join(agents_real_ids[agent_id]))

            num_shared_counter[num_shared] += 1

        for num_shared in docc_shared.keys():
            print("shared_{}: {}".format(num_shared, num_shared_counter[num_shared]))
            with open(os.path.join("model_format", "docc_shared_{}.txt".format(num_shared)), "w") as fout:
                for line in docc_shared[num_shared]:
                    fout.write(line + "\n")

    elif args.for_repeat:
        seeds = [0, 1, 2, 3, 4]

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

        #remove erroneuous sample with outcome < 0
        del chat_data["C_7f85e958c86f40a9a45a469d28df9e2c"]

        for seed in seeds:
            np.random.seed(seed)

            train, valid, test = transform_to_model_format(args, chat_data, scenarios)

            with open(os.path.join("model_format", "train_{}.json".format(seed)), "w") as fout:
                json.dump(train, fout, indent=4)
            with open(os.path.join("model_format", "valid_{}.json".format(seed)), "w") as fout:
                json.dump(valid, fout, indent=4)
            with open(os.path.join("model_format", "test_{}.json".format(seed)), "w") as fout:
                json.dump(test, fout, indent=4)
    else:
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

        #remove erroneuous sample with outcome < 0
        del chat_data["C_7f85e958c86f40a9a45a469d28df9e2c"]

        train, valid, test = transform_to_model_format(args, chat_data, scenarios)

        with open(os.path.join("model_format", "train.json"), "w") as fout:
            json.dump(train, fout, indent=4)
        with open(os.path.join("model_format", "valid.json"), "w") as fout:
            json.dump(valid, fout, indent=4)
        with open(os.path.join("model_format", "test.json"), "w") as fout:
            json.dump(test, fout, indent=4)

