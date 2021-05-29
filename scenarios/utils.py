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

