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
from scipy.special import comb 
from matplotlib import pyplot as plt
import pdb
import math
import tqdm

def generate_uuid(prefix):
	return prefix + '_' + ''.join([np.random.choice(list(string.digits + string.ascii_letters)) for _ in range(16)])

def bernstein_poly(i, n, t):
	return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=101):
	nPoints = len(points)
	xPoints = np.array([p[0] for p in points])
	yPoints = np.array([p[1] for p in points])

	t = np.linspace(0.0, 1.0, nTimes)

	polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

	xvals = np.dot(xPoints, polynomial_array)
	yvals = np.dot(yPoints, polynomial_array)

	current_dist = 0
	dists = []
	for i in range(len(xvals) - 1):
		current_dist += distance((xvals[i + 1],yvals[i + 1]), (xvals[i],yvals[i]))
		dists.append(current_dist)

	return xvals[::-1], yvals[::-1], dists

def bezier_curve_equal_dists(points, nTimes=101, splits=10):
	nPoints = len(points)
	xPoints = np.array([p[0] for p in points])
	yPoints = np.array([p[1] for p in points])

	t = np.linspace(0.0, 1.0, nTimes)

	polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

	xvals = np.dot(xPoints, polynomial_array)
	yvals = np.dot(yPoints, polynomial_array)

	current_dist = 0
	dists = []
	for i in range(len(xvals) - 1):
		current_dist += distance((xvals[i + 1],yvals[i + 1]), (xvals[i],yvals[i]))
		dists.append(current_dist)

	total_dist = current_dist
	equal_dist_split_idx = 0
	equal_dist_split = total_dist / splits
	equal_dist_xvals = []
	equal_dist_yvals = []
	for si in range(splits):
		idx = bisect.bisect_left(dists, equal_dist_split * si)
		equal_dist_xvals.append(xvals[idx])
		equal_dist_yvals.append(yvals[idx])

	return np.array(equal_dist_xvals[::-1]), np.array(equal_dist_yvals[::-1])

def sample_from_polar(max_dist):
	while True:
		x = np.random.uniform(-max_dist, max_dist)
		y = np.random.uniform(-max_dist, max_dist)
		if distance((x, y)) < max_dist:
			break
	return x, y

def distance(loc_1, loc_2=(0, 0)):
	'''
		loc is a tuple of (x,y)
	'''
	x = []
	y = []
	for loc in [loc_1, loc_2]:
		if isinstance(loc, tuple):
			x.append(loc[0])
			y.append(loc[1])
		else:
			raise ValueError('Distance arguments should be agent, entity or tuple.')

	return np.sqrt((x[0] - x[1])**2 + (y[0] - y[1])**2)

"""
def compute_cos(vec_1, vec_2):
	len_vec_1 = distance(vec_1)
	len_vec_2 = distance(vec_2)
	inner_product = vec_1[0] * vec_2[0] + vec_1[1] * vec_2[1]
	return inner_product / len_vec_1 / len_vec_2
"""

def _probabilistic_round(x):
	return int(x) + int(np.random.uniform() < x - int(x))

def total_num_entities(args):
	'''
	make sure density of entities per area is fixed
	'''
	world_area = math.pi * (args.world_r ** 2)
	_density = args.num_agt_view / (math.pi * (args.agt_r ** 2))
	return _probabilistic_round(world_area * _density)

def add_generation_parameters(scenarios, args):
	scenarios["world_parameters"] = {}
	scenarios["world_parameters"]["color_range"] = args.color_range
	scenarios["world_parameters"]["size_range"] = args.size_range
	scenarios["world_parameters"]["min_dist_ent"] = args.min_dist_ent
	scenarios["world_parameters"]["min_shared"] = args.min_shared
	scenarios["world_parameters"]["max_shared"] = args.max_shared
	scenarios["world_parameters"]["num_agt_view"] = args.num_agt_view
	scenarios["world_parameters"]["max_timesteps"] = args.max_timesteps
	scenarios["world_parameters"]["num_scenarios"] = args.num_scenarios
	scenarios["world_parameters"]["max_dist_agt"] = args.max_dist_agt
	scenarios["world_parameters"]["delta_theta_range"] = args.delta_theta_range
	scenarios["world_parameters"]["delta_dist_range"] = args.delta_dist_range
	scenarios["world_parameters"]["max_common_consecutive_overlap"] = args.max_common_consecutive_overlap
	scenarios["world_parameters"]["max_common_life_span"] = args.max_common_life_span
	scenarios["world_parameters"]["num_agents"] = args.num_agents
	scenarios["world_parameters"]["agt_r"] = args.agt_r
	scenarios["world_parameters"]["world_r"] = args.world_r
	scenarios["world_parameters"]["base_color"] = args.base_color
	scenarios["world_parameters"]["base_size"] = args.base_size
	scenarios["world_parameters"]["agt_loc_r"] = args.agt_loc_r
	scenarios["world_parameters"]["seed"] = args.seed
	return scenarios


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