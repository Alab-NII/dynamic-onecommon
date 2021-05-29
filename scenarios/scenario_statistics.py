import argparse
import copy
import json
import math
import os
import random
import string
from functools import reduce
from collections import Counter
import bisect

import glob

import numpy as np
from matplotlib import pyplot as plt
import pdb
import math
import tqdm

from utils import *

from generate_scenarios import Agent, Entity

def unify_scenarios(args):
	scenarios = {}
	scenarios["scenarios"] = {}
	world_parameters = None
	for seed_file in glob.glob("data/scenario_seed_*.json"):
		seed_scenarios = json.load(open(seed_file))

		# check world parameters
		if world_parameters:
			for key in world_parameters.keys():
				if key == "seed":
					scenarios["world_parameters"]["seed"].append(seed_scenarios["world_parameters"]["seed"])
				if seed_scenarios["world_parameters"][key] != world_parameters[key]:
					print("mismatch in world parameters!")
					assert False
		else:
			scenarios["world_parameters"] = seed_scenarios["world_parameters"]
			scenarios["world_parameters"]["seed"] = [seed_scenarios["world_parameters"]["seed"]]

		for scenario_id in seed_scenarios["scenarios"].keys():
			if scenario_id not in scenarios["scenarios"]:
				scenarios["scenarios"][scenario_id] = seed_scenarios["scenarios"][scenario_id]
	
	scenarios["world_parameters"]["seed"] = sorted(scenarios["world_parameters"]["seed"])

	print("unified {} scenarios".format(len(scenarios["scenarios"])))
	with open("data/" + args.scenario_file, "w") as fout:
		json.dump(scenarios, fout, indent=4, sort_keys=True)

def compute_statistics(args):
	statistics = {}
	statistics["consecutive_overlap"] = Counter()
	statistics["common_life_span"] = Counter()
	statistics["size_bias"] = Counter()
	statistics["num_shared"] = Counter()

	scenarios = json.load(open("data/" + args.scenario_file))
	for scenario_id, scenario in scenarios["scenarios"].items():
		agents = []
		entities = []

		# load agents/entities
		for agent_id in [0, 1]:
			agents.append(Agent.from_dict(scenario["agents"][agent_id]))
		for entity_id in scenario["entities"].keys():
			entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))

		# compute consecutive overlap
		for t in range(1, args.max_timesteps + 1):
			for agent in agents:
				prev_observe_ids = set([ent.id for ent in agent.observe_at_timestep(entities, t - 1)])
				current_observe_ids = set([ent.id for ent in agent.observe_at_timestep(entities, t)])
				consecutive_overlap = len(prev_observe_ids.intersection(current_observe_ids))
				statistics["consecutive_overlap"][consecutive_overlap] += 1

		# compute life span
		common_life_span = {}
		for t in range(1, args.max_timesteps + 1):
			common_ids = set([ent.id for ent in agents[0].observe_at_timestep(entities, t)]).intersection(set([ent.id for ent in agents[1].observe_at_timestep(entities, t)]))
			statistics["num_shared"][len(common_ids)] += 1

			for common_id in common_ids:
				if common_id in common_life_span:
					common_life_span[common_id] += 1
				else:
					common_life_span[common_id] = 1
			existing_ids = list(common_life_span.keys())
			for ent_id in existing_ids:
				if ent_id not in common_ids:
					statistics["common_life_span"][common_life_span[ent_id]] += 1
					del common_life_span[ent_id]

		# compute size bias
		for entity_id, entity in scenario["entities"].items():
			statistics["size_bias"][entity['size']] += 1

	print("common life span:")
	print(statistics["common_life_span"])

	print("\nconsecutive overlap:")
	print(statistics["consecutive_overlap"])
	print("output consecutive_overlap.png")
	plt.bar(statistics["consecutive_overlap"].keys(), statistics["consecutive_overlap"].values())
	plt.savefig("consecutive_overlap.png")

	print("\nsize bias:")
	print(statistics["size_bias"])

	print("\nnum shared:")
	print(statistics["num_shared"])

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--unify_scenarios', action='store_true')
	parser.add_argument('--compute_statistics', action='store_true')
	parser.add_argument('--scenario_file', type=str, default="scenarios.json")
	parser.add_argument('--max_timesteps', type=int, default=5)
	args = parser.parse_args()

	if args.unify_scenarios:
		unify_scenarios(args)

	if args.compute_statistics:
		compute_statistics(args)



