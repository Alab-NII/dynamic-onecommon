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

import numpy as np
from matplotlib import pyplot as plt
import pdb
import math
import tqdm

from utils import *

class Agent():
	def __init__(self, init_x=None, init_y=None, r=None):
		self.xs = [init_x]
		self.ys = [init_y]
		self.r = r

	@classmethod
	def gen_new_agent(cls, args):
		x, y = sample_from_polar(args.agt_loc_r)
		return cls(x, y, args.agt_r)

	def observe_at_timestep(self, entities, t, before_ent_move=False):
		"""
			Return observable entities at timestep t.
			If before_ent_move=True, return observable entities before they move.
		"""
		observe_ents = []
		for i in range(len(entities)):
			if before_ent_move:
				if distance(self.loc_at_timestep(t), entities[i].loc_at_timestep(t - 1)) < self.r:
					observe_ents.append(entities[i])
			else:
				if distance(self.loc_at_timestep(t), entities[i].loc_at_timestep(t)) < self.r:
					observe_ents.append(entities[i])
		return observe_ents

	def gen_new_move(self, args):
		success = False
		for _ in range(1000):
			delta_x, delta_y = sample_from_polar(args.delta_dist_range)
			new_x = self.xs[-1] + delta_x
			new_y = self.ys[-1] + delta_y
			if distance((new_x, new_y)) < args.agt_loc_r:
				success = True
				break
		self.xs.append(new_x)
		self.ys.append(new_y)
		return new_x, new_y, success

	def reset_move(self):
		last_x = self.xs.pop()
		last_y = self.ys.pop()
		return last_x, last_y

	def loc_at_timestep(self, t):
		return (self.xs[t], self.ys[t])

	@classmethod
	def from_dict(cls, agent_dict):
		agent = Agent()
		agent.xs = agent_dict["xs"]
		agent.ys = agent_dict["ys"]
		agent.r = agent_dict["r"]
		return agent

	def to_dict(self):
		agent_dict = {}
		agent_dict["xs"] = self.xs
		agent_dict["ys"] = self.ys
		agent_dict["r"] = self.r
		return agent_dict

class Entity():
	def __init__(self, id, init_x=None, init_y=None, init_theta=None, color=None, size=None):
		self.id = str(id)
		self.xs = [init_x]
		self.ys = [init_y]
		self.thetas = [init_theta]
		self.moves = [] # each move is a tuple of control points (p_1, p_2)
		self.color = color
		self.size = size

	@classmethod
	def gen_new_entity(cls, id, args):
		x, y = sample_from_polar(args.world_r)
		min_color = args.base_color - args.color_range / 2
		max_color = args.base_color + args.color_range / 2
		color = int(round(np.random.uniform(min_color, max_color)))
		min_size = args.base_size - args.size_range / 2
		max_size = args.base_size + args.size_range / 2
		size = int(round(np.random.uniform(min_size, max_size)))
		theta = np.random.uniform(0, 2 * math.pi)
		return cls(id, x, y, theta, color, size)

	def gen_new_move(self, args):
		success = False
		for _ in range(1000):
			delta_dist_1 = np.random.uniform(0, args.delta_dist_range / 2)
			delta_theta = np.random.uniform(-args.delta_theta_range, args.delta_theta_range)
			delta_dist_2 = np.random.uniform(0, args.delta_dist_range)

			current_theta = self.thetas[-1]

			delta_x_1 = delta_dist_1 * math.cos(current_theta)
			delta_y_1 = delta_dist_1 * math.sin(current_theta)
			delta_x_2 = delta_dist_2 * math.cos(current_theta + delta_theta)
			delta_y_2 = delta_dist_2 * math.sin(current_theta + delta_theta)

			new_x = self.xs[-1] + delta_x_1 + delta_x_2
			new_y = self.ys[-1] + delta_y_1 + delta_y_2
			if distance((new_x, new_y)) < args.world_r:
				success = True
				break
		self.xs.append(new_x)
		self.ys.append(new_y)
		self.thetas.append(current_theta + delta_theta)
		self.moves.append(((delta_x_1, delta_y_1), (delta_x_1 + delta_x_2, delta_y_1 + delta_y_2)))

		return new_x, new_y, current_theta + delta_theta, ((delta_x_1, delta_y_1), (delta_x_1 + delta_x_2, delta_y_1 + delta_y_2)), success

	def reset_move(self):
		last_x = self.xs.pop()
		last_y = self.ys.pop()
		last_theta = self.thetas.pop()
		last_move = self.moves.pop()
		return last_x, last_y, last_theta, last_move

	def loc_at_timestep(self, t):
		return (self.xs[t], self.ys[t])

	@classmethod
	def from_dict(cls, entity_id, entity_dict):
		entity = cls(entity_id)
		entity.xs = entity_dict["xs"]
		entity.ys = entity_dict["ys"]
		entity.thetas = entity_dict["thetas"]
		entity.moves = entity_dict["moves"]
		entity.color = entity_dict["color"]
		entity.size = entity_dict["size"]
		return entity

	def to_dict(self):
		ent_dict = {}
		ent_dict["xs"] = self.xs
		ent_dict["ys"] = self.ys
		ent_dict["thetas"] = self.thetas
		ent_dict["moves"] = self.moves
		ent_dict["color"] = self.color
		ent_dict["size"] = self.size
		return ent_dict

def generate_scenarios(args):
	scenarios = {}

	# add generation parameters
	scenarios = add_generation_parameters(scenarios, args)

	scenarios["scenarios"] = {}
	rejected = 0
	accepted = 0
	total_num_ents = total_num_entities(args)

	while len(scenarios["scenarios"]) < args.num_scenarios:
		agents = []
		entities = []

		"""
			Generate initial state.
		"""
		agents.append(Agent(0, 0, args.agt_r))
		delta_x, delta_y = sample_from_polar(args.max_dist_agt)
		x, y = agents[0].loc_at_timestep(0)
		agents.append(Agent(x + delta_x, y + delta_y, args.agt_r))

		while len(entities) < total_num_ents:
			ent = Entity.gen_new_entity(str(len(entities)), args)
			redo = False
			# if new object is too close to the previous entities, redo
			for prev_ent in entities:
				if distance(prev_ent.loc_at_timestep(0), ent.loc_at_timestep(0)) < args.min_dist_ent:
					redo = True
					break
			if redo is False:
				# append object to the list
				entities.append(ent)

		"""
			Generate dynamic moves.
		"""
		for t in range(1, args.max_timesteps + 1):
			np.random.shuffle(entities)
			new_entities = []
			for entity in entities:
				accept = False
				for _ in range(1000):
					new_x, new_y, new_theta, new_move, success = entity.gen_new_move(args)
					if not success:
						break
					entity_dist_constraints = True
					for prev_entity in new_entities:
						new_ent_xvals, new_ent_yvals = bezier_curve_equal_dists([(0,0), new_move[0], new_move[1]])
						prev_ent_xvals, prev_ent_yvals = bezier_curve_equal_dists([(0,0), prev_entity.moves[t - 1][0], prev_entity.moves[t - 1][1]])
						new_ent_xvals += entity.xs[t - 1]
						new_ent_yvals += entity.ys[t - 1]
						prev_ent_xvals += prev_entity.xs[t - 1]
						prev_ent_yvals += prev_entity.ys[t - 1]
						ent_dists = np.sqrt((new_ent_xvals - prev_ent_xvals) ** 2 + (new_ent_yvals - prev_ent_yvals) ** 2)
						if distance(entity.loc_at_timestep(t), prev_entity.loc_at_timestep(t)) < args.min_dist_ent or \
							np.min(ent_dists) < args.min_dist_ent * 0.6: # hard-coded
							entity_dist_constraints = False
							break
					if entity_dist_constraints:
						accept = True
						break
					else:
						entity.reset_move()
				if not accept:
					break
				new_entities.append(copy.deepcopy(entity))
			if not accept:
				break

			accept = False
			for _ in range(1000):
				for agent in agents:
					_, _, success = agent.gen_new_move(args)
					if not success:
						break
				if not success:
					break

				if distance(agents[0].loc_at_timestep(t), agents[1].loc_at_timestep(t)) < args.max_dist_agt and \
				   len(agents[0].observe_at_timestep(new_entities, t)) == args.num_agt_view and \
				   len(agents[1].observe_at_timestep(new_entities, t)) == args.num_agt_view and \
				   len(set(agents[0].observe_at_timestep(new_entities, t)).intersection(set(agents[1].observe_at_timestep(new_entities, t)))) >= args.min_shared and \
				   len(set(agents[0].observe_at_timestep(new_entities, t)).intersection(set(agents[1].observe_at_timestep(new_entities, t)))) <= args.max_shared:
					if t == 1:
						accept = True
						entities = new_entities
						previous_common_ent_ids = set([ent.id for ent in set(agents[0].observe_at_timestep(new_entities, t)).intersection(set(agents[1].observe_at_timestep(new_entities, t)))])
						common_life_span = {}
						for ent_id in previous_common_ent_ids:
							common_life_span[ent_id] = 1
						break
					else:
						common_ent_ids = set([ent.id for ent in set(agents[0].observe_at_timestep(new_entities, t)).intersection(set(agents[1].observe_at_timestep(new_entities, t)))])
						common_life_span_error = False
						for ent_id in common_ent_ids:
							if ent_id in common_life_span.keys():
								if common_life_span[ent_id] + 1 > args.max_common_life_span:
									common_life_span_error = True
									break

						if len(previous_common_ent_ids.intersection(common_ent_ids)) <= args.max_common_consecutive_overlap and not common_life_span_error:
							accept = True
							entities = new_entities
							for ent_id in common_ent_ids:
								if ent_id in common_life_span.keys():
									common_life_span[ent_id] += 1
								else:
									common_life_span[ent_id] = 1

							for ent_id in previous_common_ent_ids:
								if ent_id not in common_ent_ids:
									del common_life_span[ent_id]

							previous_common_ent_ids = common_ent_ids
							break
						else:
							for agent in agents:
								agent.reset_move()
								continue	
				else:
					for agent in agents:
						agent.reset_move()
					continue
			if not accept:
				break

		if not accept:
			rejected += 1
		else:
			accepted += 1
			scenario = {}
			scenario["agents"] = []
			for agent in agents:
				scenario["agents"].append(agent.to_dict())
			scenario["entities"] = {}
			for entity in entities:
				scenario["entities"][entity.id] = entity.to_dict()

			scenario_id = generate_uuid('DS')
			scenarios["scenarios"][scenario_id] = scenario
		print("accepted: {}, acceptance rate: {:.2f}%".format(accepted, 100.0 * accepted / (accepted + rejected)))

	print("acceptance rate: {:.2f}%".format(100.0 * accepted / (accepted + rejected)))

	return scenarios

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
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
	parser.add_argument('--max_common_consecutive_overlap', type=int, default=3,
		help='maximum number of consecutive overlap of common entities')
	parser.add_argument('--max_common_life_span', type=int, default=3,
		help='maximum number of timesteps an entity can be in common')
	parser.add_argument('--seed', type=int, default=2020,
		help='seed')

	# --- save parameters ----
	parser.add_argument('--save_path', type=str, default='data',
		help='save generated context data')

	# --- misc parameters ----
	parser.add_argument('--tutorial', action='store_true',
		help='create scenarios for tutorial')


	args = parser.parse_args()

	np.random.seed(args.seed)

	# current support
	args.num_agents = 2
	args.agt_r = 0.25
	args.world_r = args.agt_r * 2.5
	args.base_color = 128
	args.base_size = 9
	args.agt_loc_r = args.agt_r * 1
	args.file_name = "scenario_seed_{}.json".format(args.seed)

	if args.tutorial:
		args.world_r = args.agt_r * 2.25
		args.num_scenarios = 15
		args.file_name = "tutorial_scenarios.json"

	# generate scenarios
	scenarios = generate_scenarios(args)

	# save scenarios
	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)
	with open(os.path.join(args.save_path, args.file_name), "w") as fout:
		json.dump(scenarios, fout, indent=4, sort_keys=True)

