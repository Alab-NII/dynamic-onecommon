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

from generate_scenarios import Agent, Entity

def agent_move_animation(agent_id, agent, entities, t, args, reverse=False, add_agent_id=False):
	"""
		Return SVG animation of agent movement at timestep t
	"""
	svg_margin = 15
	svg_r_margin = 5
	svg_radius = 200

	init_svg = []
	animation_svg = []
	end_svg = []

	expand_scale = svg_radius / agent.r

	candidate_ents = []
	keytimes = {}
	visibility = {}
	for i in range(len(entities)):
		if distance(agent.loc_at_timestep(t), entities[i].loc_at_timestep(t)) < agent.r + args.delta_dist_range:
			candidate_ents.append(entities[i])

	diff_x = agent.xs[t + 1] - agent.xs[t]
	diff_y = agent.ys[t + 1] - agent.ys[t]

	init_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	init_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	animation_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	animation_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	end_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	end_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	for candidate_ent in candidate_ents:
		keytimes[candidate_ent.id] = [str(0)]
		visibility[candidate_ent.id] = []

		# initial condition
		if not reverse:
			ent_x, ent_y = candidate_ent.loc_at_timestep(t)
			if distance((ent_x, ent_y), (agent.xs[t], agent.ys[t])) < agent.r:
				visibility[candidate_ent.id].append("visible")
				current_visibility = "visible"
			else:
				visibility[candidate_ent.id].append("hidden")
				current_visibility = "hidden"
		else:
			ent_x, ent_y = candidate_ent.loc_at_timestep(t)
			if distance((ent_x - diff_x, ent_y - diff_y), (agent.xs[t], agent.ys[t])) < agent.r:
				visibility[candidate_ent.id].append("visible")
				current_visibility = "visible"
			else:
				visibility[candidate_ent.id].append("hidden")
				current_visibility = "hidden"

		for keytime in np.linspace(0.01,0.99,99):
			if not reverse:
				# if visible
				if distance((ent_x - diff_x * keytime, ent_y - diff_y * keytime), (agent.xs[t], agent.ys[t])) < agent.r:
					if current_visibility == "hidden":
						keytimes[candidate_ent.id].append(str(keytime))
						visibility[candidate_ent.id].append("visible")
						current_visibility = "visible"
				else: # if not visible
					if current_visibility == "visible":
						keytimes[candidate_ent.id].append(str(keytime))
						visibility[candidate_ent.id].append("hidden")
						current_visibility = "hidden"
			else:
				if distance((ent_x - diff_x + diff_x * keytime, ent_y - diff_y + diff_y * keytime), (agent.xs[t], agent.ys[t])) < agent.r:
					if current_visibility == "hidden":
						keytimes[candidate_ent.id].append(str(keytime))
						visibility[candidate_ent.id].append("visible")
						current_visibility = "visible"
				else: # if not visible
					if current_visibility == "visible":
						keytimes[candidate_ent.id].append(str(keytime))
						visibility[candidate_ent.id].append("hidden")
						current_visibility = "hidden"

		# last condition
		keytimes[candidate_ent.id].append(str(1))
		if not reverse:
			if distance((ent_x - diff_x, ent_y - diff_y), (agent.xs[t], agent.ys[t])) < agent.r:
				visibility[candidate_ent.id].append("visible")
			else:
				visibility[candidate_ent.id].append("hidden")
		else:
			if distance((ent_x, ent_y), (agent.xs[t], agent.ys[t])) < agent.r:
				visibility[candidate_ent.id].append("visible")
			else:
				visibility[candidate_ent.id].append("hidden")

		# if candidate_ent is visible at any moment
		if "visible" in visibility[candidate_ent.id]:
			# if specified, add agent id to ent id
			if add_agent_id:
				candidate_ent_id = "agt_" + str(agent_id) + "_ent_" + str(candidate_ent.id)
			else:
				candidate_ent_id = "ent_" + str(candidate_ent.id)

			if not reverse:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t]))
				move_svg_x = int(expand_scale * -diff_x)
				move_svg_y = int(expand_scale * -diff_y)
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t + 1]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t + 1]))
			else:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t + 1]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t + 1]))
				move_svg_x = int(expand_scale * diff_x)
				move_svg_y = int(expand_scale * diff_y)
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t]))

			# add start svgs
			if visibility[candidate_ent.id][0] == "visible":
				init_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))
				animation_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))
			else:
				animation_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})" visibility="hidden"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))

			# add motion svgs
			animation_svg.append("""<animateMotion xlink:href="#{0}" dur="{dur_seconds}" begin="0s" fill="freeze" path="M0,0L{1},{2}" />""".format(
				candidate_ent_id, move_svg_x, move_svg_y, dur_seconds="{dur_seconds}"))
			if "hidden" in visibility[candidate_ent.id]:
				animation_svg.append("""<animate xlink:href="#{0}" attributeType="CSS" attributeName="visibility" from="{1}" to="{2}" values="{3}" keyTimes="{4}" dur="{dur_seconds}" fill="freeze"/>""".format(
					candidate_ent_id, visibility[candidate_ent.id][0], visibility[candidate_ent.id][-1], ";".join(visibility[candidate_ent.id]), ";".join(keytimes[candidate_ent.id]), dur_seconds="{dur_seconds}"))

			# add end svgs
			if visibility[candidate_ent.id][-1] == "visible":
				end_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
					candidate_ent_id, end_svg_x, end_svg_y, candidate_ent.size, candidate_ent.color))

	init_svg.append("</svg>")
	animation_svg.append("</svg>")
	end_svg.append("</svg>")

	init_svg = " ".join(init_svg)
	animation_svg = " ".join(animation_svg)
	end_svg = " ".join(end_svg)

	return init_svg, animation_svg, end_svg

def entity_move_animation(agent_id, agent, entities, t, args, reverse=False, add_agent_id=False):
	"""
		Return SVG animation of entity movements at timestep t
	"""
	svg_margin = 15
	svg_r_margin = 5
	svg_radius = 200

	init_svg = []
	animation_svg = []
	end_svg = []

	expand_scale = svg_radius / agent.r

	candidate_ents = []
	keytimes = {}
	visibility = {}
	for i in range(len(entities)):
		if distance(agent.loc_at_timestep(t + 1), entities[i].loc_at_timestep(t)) < agent.r + 2 * args.delta_dist_range:
			candidate_ents.append(entities[i])

	init_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	init_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	animation_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	animation_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	end_svg.append("""<svg width="{0}" height="{0}" id="{1}">""".format(2 * svg_margin + 2 * svg_radius, agent_id))
	end_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
	for candidate_ent in candidate_ents:
		if not reverse:
			xvals, yvals, dists = bezier_curve([(0,0), candidate_ent.moves[t][0], candidate_ent.moves[t][1]])
			ent_x, ent_y = candidate_ent.loc_at_timestep(t)
			moved_ent_x, moved_ent_y = candidate_ent.loc_at_timestep(t + 1)
		else:
			xvals, yvals, dists = bezier_curve([(0,0),
												(candidate_ent.moves[t][0][0] - candidate_ent.moves[t][1][0], candidate_ent.moves[t][0][1] - candidate_ent.moves[t][1][1]),
												(-candidate_ent.moves[t][1][0], -candidate_ent.moves[t][1][1])])
			ent_x, ent_y = candidate_ent.loc_at_timestep(t + 1)
			moved_ent_x, moved_ent_y = candidate_ent.loc_at_timestep(t)

		# remove first and last elements
		xvals = xvals[1:-1]
		yvals = yvals[1:-1]
		dist_step = dists[-1] / len(dists)
		keydists = [dist_step * i for i in range(1, 100)]
		keytimes[candidate_ent.id] = [str(0)]
		visibility[candidate_ent.id] = []
		# initial condition
		if distance((ent_x, ent_y), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
			visibility[candidate_ent.id].append("visible")
			current_visibility = "visible"
		else:
			visibility[candidate_ent.id].append("hidden")
			current_visibility = "hidden"

		for i in range(99):
			keytime = 0.01 * bisect.bisect_left(dists, keydists[i])

			# if visible
			if distance((ent_x + xvals[i], ent_y + yvals[i]), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
				if current_visibility == "hidden":
					keytimes[candidate_ent.id].append(str(keytime))
					visibility[candidate_ent.id].append("visible")
					current_visibility = "visible"
			else:
				if current_visibility == "visible":
					keytimes[candidate_ent.id].append(str(keytime))
					visibility[candidate_ent.id].append("hidden")
					current_visibility = "hidden"

		keytimes[candidate_ent.id].append(str(1))
		if distance((moved_ent_x, moved_ent_y), (agent.xs[t + 1], agent.ys[t + 1])) < agent.r:
			visibility[candidate_ent.id].append("visible")
		else:
			visibility[candidate_ent.id].append("hidden")

		if "visible" in visibility[candidate_ent.id]:
			# if specified, add agent id to ent id
			if add_agent_id:
				candidate_ent_id = "agt_" + str(agent_id) + "_ent_" + str(candidate_ent.id)
			else:
				candidate_ent_id = "ent_" + str(candidate_ent.id)

			if not reverse:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t + 1]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t + 1]))
				move_svg_x_1 = int(expand_scale * candidate_ent.moves[t][0][0])
				move_svg_y_1 = int(expand_scale * candidate_ent.moves[t][0][1])
				move_svg_x_2 = int(expand_scale * candidate_ent.moves[t][1][0])
				move_svg_y_2 = int(expand_scale * candidate_ent.moves[t][1][1])
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t + 1] - agent.xs[t + 1]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t + 1] - agent.ys[t + 1]))
			else:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t + 1] - agent.xs[t + 1]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t + 1] - agent.ys[t + 1]))
				move_svg_x_1 = int(expand_scale * (candidate_ent.moves[t][0][0] - candidate_ent.moves[t][1][0]))
				move_svg_y_1 = int(expand_scale * (candidate_ent.moves[t][0][1] - candidate_ent.moves[t][1][1]))
				move_svg_x_2 = int(expand_scale * -candidate_ent.moves[t][1][0])
				move_svg_y_2 = int(expand_scale * -candidate_ent.moves[t][1][1])
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (candidate_ent.xs[t] - agent.xs[t + 1]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (candidate_ent.ys[t] - agent.ys[t + 1]))

			if visibility[candidate_ent.id][0] == "visible":
				init_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))
				animation_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))
			else:
				animation_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})" visibility="hidden"/>""".format(
					candidate_ent_id, init_svg_x, init_svg_y, candidate_ent.size, candidate_ent.color))
			
			# add motion svgs
			animation_svg.append("""<animateMotion xlink:href="#{0}" dur="{dur_seconds}" begin="0s" fill="freeze" path="M 0, 0 Q {1} {2}, {3} {4}" />""".format(
				candidate_ent_id, move_svg_x_1, move_svg_y_1, move_svg_x_2, move_svg_y_2, dur_seconds="{dur_seconds}"))
			if "hidden" in visibility[candidate_ent.id]:
				animation_svg.append("""<animate xlink:href="#{0}" attributeType="CSS" attributeName="visibility" from="{1}" to="{2}" values="{3}" keyTimes="{4}" dur="{dur_seconds}" fill="freeze"/>""".format(
					candidate_ent_id, visibility[candidate_ent.id][0], visibility[candidate_ent.id][-1], ";".join(visibility[candidate_ent.id]), ";".join(keytimes[candidate_ent.id]), dur_seconds="{dur_seconds}"))
			
			if visibility[candidate_ent.id][-1] == "visible":
				end_svg.append("""<circle id="{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})" onclick="makeSelection(&quot;{5}&quot;)"/>""".format(
					candidate_ent_id, end_svg_x, end_svg_y, candidate_ent.size, candidate_ent.color, candidate_ent_id))
				
				end_svg.append("""<circle id="{0}_marker" visibility="hidden" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="green" stroke-width="3" stroke-dasharray="3,3"/>""".format(
					candidate_ent_id, end_svg_x, end_svg_y, candidate_ent.size + 4))

	init_svg.append("</svg>")
	animation_svg.append("</svg>")
	end_svg.append("</svg>")

	init_svg = " ".join(init_svg)
	animation_svg = " ".join(animation_svg)
	end_svg = " ".join(end_svg)

	return init_svg, animation_svg, end_svg

def world_agent_move_animation(agents, entities, t, args, reverse=False):
		"""
			Return SVG animation of agent movement at timestep t
		"""
		svg_margin = 20
		svg_r_margin = 10
		svg_radius = 300

		agt_r = 0.25
		world_r = agt_r * 2.5

		init_svg = []
		animation_svg = []
		end_svg = []

		expand_scale = svg_radius / world_r

		init_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		init_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
		animation_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		animation_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
		end_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		end_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))

		for ent in entities:
			svg_x = svg_margin + svg_radius + int(expand_scale * (ent.xs[t]))
			svg_y = svg_margin + svg_radius + int(expand_scale * (ent.ys[t]))
			
			init_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, svg_x, svg_y, round(ent.size * 2 / 3), ent.color))
			animation_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, svg_x, svg_y, round(ent.size * 2 / 3), ent.color))
			end_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, svg_x, svg_y, round(ent.size * 2 / 3), ent.color))

		for agent_id, agent in enumerate(agents):
			if not reverse:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (agent.xs[t]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (agent.ys[t]))
				move_svg_x = int(expand_scale * (agent.xs[t + 1] - agent.xs[t]))
				move_svg_y = int(expand_scale * (agent.ys[t + 1] - agent.ys[t]))
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (agent.xs[t + 1]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (agent.ys[t + 1]))
			else:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (agent.xs[t + 1]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (agent.ys[t + 1]))
				move_svg_x = int(expand_scale * (agent.xs[t] - agent.xs[t + 1]))
				move_svg_y = int(expand_scale * (agent.ys[t] - agent.ys[t + 1]))
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (agent.xs[t]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (agent.ys[t]))

			# add init svgs
			init_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, init_svg_x, init_svg_y, int(expand_scale * agent.r)))
			animation_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, init_svg_x, init_svg_y, int(expand_scale * agent.r)))

			# add motion svgs
			animation_svg.append("""<animateMotion xlink:href="#agent_{0}" dur="{dur_seconds}" begin="0s" fill="freeze" path="M0,0L{1},{2}" />""".format(
				agent_id, move_svg_x, move_svg_y, dur_seconds="{dur_seconds}"))
			
			# add end svgs
			end_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, end_svg_x, end_svg_y, int(expand_scale * agent.r)))

		init_svg.append("</svg>")
		animation_svg.append("</svg>")
		end_svg.append("</svg>")

		init_svg = " ".join(init_svg)
		animation_svg = " ".join(animation_svg)
		end_svg = " ".join(end_svg)

		return init_svg, animation_svg, end_svg

def world_entity_move_animation(agents, entities, t, args, reverse=False):
		"""
			Return SVG animation of agent movement at timestep t
		"""
		svg_margin = 20
		svg_r_margin = 10
		svg_radius = 300

		agt_r = 0.25
		world_r = agt_r * 2.5

		init_svg = []
		animation_svg = []
		end_svg = []

		expand_scale = svg_radius / world_r

		init_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		init_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
		animation_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		animation_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))
		end_svg.append("""<svg width="{0}" height="{0}">""".format(2 * svg_margin + 2 * svg_radius))
		end_svg.append("""<circle cx="{0}" cy="{0}" r="{1}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(svg_margin + svg_radius, svg_r_margin + svg_radius))

		for ent in entities:
			if not reverse:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (ent.xs[t]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (ent.ys[t]))
				move_svg_x_1 = int(expand_scale * ent.moves[t][0][0])
				move_svg_y_1 = int(expand_scale * ent.moves[t][0][1])
				move_svg_x_2 = int(expand_scale * ent.moves[t][1][0])
				move_svg_y_2 = int(expand_scale * ent.moves[t][1][1])
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (ent.xs[t + 1]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (ent.ys[t + 1]))
			else:
				init_svg_x = svg_margin + svg_radius + int(expand_scale * (ent.xs[t + 1]))
				init_svg_y = svg_margin + svg_radius + int(expand_scale * (ent.ys[t + 1]))
				move_svg_x_1 = int(expand_scale * (ent.moves[t][0][0] - ent.moves[t][1][0]))
				move_svg_y_1 = int(expand_scale * (ent.moves[t][0][1] - ent.moves[t][1][1]))
				move_svg_x_2 = int(expand_scale * -ent.moves[t][1][0])
				move_svg_y_2 = int(expand_scale * -ent.moves[t][1][1])
				end_svg_x = svg_margin + svg_radius + int(expand_scale * (ent.xs[t]))
				end_svg_y = svg_margin + svg_radius + int(expand_scale * (ent.ys[t]))
			
			# add init svgs
			init_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, init_svg_x, init_svg_y, round(ent.size * 2 / 3), ent.color))
			animation_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, init_svg_x, init_svg_y, round(ent.size * 2 / 3), ent.color))
		
			# add motion svgs
			animation_svg.append("""<animateMotion xlink:href="#ent_{0}" dur="{dur_seconds}" begin="0s" fill="freeze" path="M 0, 0 Q {1} {2}, {3} {4}" />""".format(
				ent.id, move_svg_x_1, move_svg_y_1, move_svg_x_2, move_svg_y_2, dur_seconds="{dur_seconds}"))
			
			# add end svgs
			end_svg.append("""<circle id="ent_{0}" cx="{1}" cy="{2}" r="{3}" fill="rgb({4},{4},{4})"/>""".format(
				ent.id, end_svg_x, end_svg_y, round(ent.size * 2 / 3), ent.color))

		for agent_id, agent in enumerate(agents):
			svg_x = svg_margin + svg_radius + int(expand_scale * (agent.xs[t + 1]))
			svg_y = svg_margin + svg_radius + int(expand_scale * (agent.ys[t + 1]))
			init_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, svg_x, svg_y, int(expand_scale * agent.r)))
			animation_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, svg_x, svg_y, int(expand_scale * agent.r)))
			end_svg.append("""<circle id="agent_{0}" cx="{1}" cy="{2}" r="{3}" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>""".format(
				agent_id, svg_x, svg_y, int(expand_scale * agent.r)))

		init_svg.append("</svg>")
		animation_svg.append("</svg>")
		end_svg.append("</svg>")

		init_svg = " ".join(init_svg)
		animation_svg = " ".join(animation_svg)
		end_svg = " ".join(end_svg)

		return init_svg, animation_svg, end_svg

def scenario_to_svg(args, scenarios):
	scenario_svgs = {}

	for scenario_id, scenario in scenarios["scenarios"].items():
		agents = []
		entities = []
		for agent_id in [0, 1]:
			agents.append(Agent.from_dict(scenario["agents"][agent_id]))
		for entity_id in scenario["entities"].keys():
			entities.append(Entity.from_dict(entity_id, scenario["entities"][entity_id]))

		scenario_svgs[scenario_id] = {}
		scenario_svgs[scenario_id]["agents"] = []
		for agent_id in [0, 1]:
			scenario_svgs[scenario_id]["agents"].append({})
			scenario_svgs[scenario_id]["agents"][agent_id]["static_svgs"] = []
			scenario_svgs[scenario_id]["agents"][agent_id]["animation_svgs"] = []
			scenario_svgs[scenario_id]["agents"][agent_id]["reverse_animation_svgs"] = []

		if args.add_world_svgs:
			scenario_svgs[scenario_id]["world"] = {}
			scenario_svgs[scenario_id]["world"]["static_svgs"] = []
			scenario_svgs[scenario_id]["world"]["animation_svgs"] = []
			scenario_svgs[scenario_id]["world"]["reverse_animation_svgs"] = []

		for t in range(args.max_timesteps):
			for agent_id, agent in enumerate(agents):
				if t == 0:
					# Skip agent move
					init_svg, animation_svg, end_svg = entity_move_animation(agent_id, agent, entities, t, args, reverse=False, add_agent_id=True)
					_, reverse_animation_svg, _ = entity_move_animation(agent_id, agent, entities, t, args, reverse=True, add_agent_id=True)
					scenario_svgs[scenario_id]["agents"][agent_id]["static_svgs"].append(init_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["reverse_animation_svgs"].append(reverse_animation_svg)
				else:
					_, animation_svg, end_svg = agent_move_animation(agent_id, agent, entities, t, args, reverse=False, add_agent_id=True)
					_, reverse_animation_svg, _ = agent_move_animation(agent_id, agent, entities, t, args, reverse=True, add_agent_id=True)
					scenario_svgs[scenario_id]["agents"][agent_id]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["reverse_animation_svgs"].append(reverse_animation_svg)

					_, animation_svg, end_svg = entity_move_animation(agent_id, agent, entities, t, args, reverse=False, add_agent_id=True)
					_, reverse_animation_svg, _ = entity_move_animation(agent_id, agent, entities, t, args, reverse=True, add_agent_id=True)
					scenario_svgs[scenario_id]["agents"][agent_id]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["agents"][agent_id]["reverse_animation_svgs"].append(reverse_animation_svg)

			if args.add_world_svgs:
				if t == 0:
					# Skip agent move
					init_svg, animation_svg, end_svg = world_entity_move_animation(agents, entities, t, args, reverse=False)
					_, reverse_animation_svg, _ = world_entity_move_animation(agents, entities, t, args, reverse=True)
					scenario_svgs[scenario_id]["world"]["static_svgs"].append(init_svg)
					scenario_svgs[scenario_id]["world"]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["world"]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["world"]["reverse_animation_svgs"].append(reverse_animation_svg)
				else:
					_, animation_svg, end_svg = world_agent_move_animation(agents, entities, t, args, reverse=False)
					_, reverse_animation_svg, _ = world_agent_move_animation(agents, entities, t, args, reverse=True)
					scenario_svgs[scenario_id]["world"]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["world"]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["world"]["reverse_animation_svgs"].append(reverse_animation_svg)

					_, animation_svg, end_svg = world_entity_move_animation(agents, entities, t, args, reverse=False)
					_, reverse_animation_svg, _ = world_entity_move_animation(agents, entities, t, args, reverse=True)
					scenario_svgs[scenario_id]["world"]["static_svgs"].append(end_svg)
					scenario_svgs[scenario_id]["world"]["animation_svgs"].append(animation_svg)
					scenario_svgs[scenario_id]["world"]["reverse_animation_svgs"].append(reverse_animation_svg)

	return scenario_svgs

def scenario_to_html(args, scenario_svgs):
	# Read in templates
	with open("template_svg.txt", "r") as fin:
		template_svg = fin.read()
	with open("template_world_svg.txt", "r") as fin:
		template_world_svg = fin.read()

	for scnario_id, scenario_svg in list(scenario_svgs.items())[:args.max_html_scenarios]:
		with open("data/html/{}_agent.html".format(scnario_id), "w") as fout:
			fout.write(template_svg.format(
				agent_0_move_1=scenario_svg["agents"][0]["animation_svgs"][0].format(dur_seconds="2s"),
				agent_1_move_1=scenario_svg["agents"][1]["animation_svgs"][0].format(dur_seconds="2s"),
				agent_0_move_2=scenario_svg["agents"][0]["animation_svgs"][1].format(dur_seconds="2s"),
				agent_1_move_2=scenario_svg["agents"][1]["animation_svgs"][1].format(dur_seconds="2s"),
				agent_0_move_3=scenario_svg["agents"][0]["animation_svgs"][2].format(dur_seconds="2s"),
				agent_1_move_3=scenario_svg["agents"][1]["animation_svgs"][2].format(dur_seconds="2s"),
				agent_0_move_4=scenario_svg["agents"][0]["animation_svgs"][3].format(dur_seconds="2s"),
				agent_1_move_4=scenario_svg["agents"][1]["animation_svgs"][3].format(dur_seconds="2s"),
				agent_0_move_5=scenario_svg["agents"][0]["animation_svgs"][4].format(dur_seconds="2s"),
				agent_1_move_5=scenario_svg["agents"][1]["animation_svgs"][4].format(dur_seconds="2s"),
				agent_0_move_6=scenario_svg["agents"][0]["animation_svgs"][5].format(dur_seconds="2s"),
				agent_1_move_6=scenario_svg["agents"][1]["animation_svgs"][5].format(dur_seconds="2s"),
				agent_0_move_7=scenario_svg["agents"][0]["animation_svgs"][6].format(dur_seconds="2s"),
				agent_1_move_7=scenario_svg["agents"][1]["animation_svgs"][6].format(dur_seconds="2s"),
				agent_0_move_8=scenario_svg["agents"][0]["animation_svgs"][7].format(dur_seconds="2s"),
				agent_1_move_8=scenario_svg["agents"][1]["animation_svgs"][7].format(dur_seconds="2s"),
				agent_0_move_9=scenario_svg["agents"][0]["animation_svgs"][8].format(dur_seconds="2s"),
				agent_1_move_9=scenario_svg["agents"][1]["animation_svgs"][8].format(dur_seconds="2s"),
				reverse_agent_0_move_1=scenario_svg["agents"][0]["reverse_animation_svgs"][0].format(dur_seconds="2s"),
				reverse_agent_1_move_1=scenario_svg["agents"][1]["reverse_animation_svgs"][0].format(dur_seconds="2s"),
				reverse_agent_0_move_2=scenario_svg["agents"][0]["reverse_animation_svgs"][1].format(dur_seconds="2s"),
				reverse_agent_1_move_2=scenario_svg["agents"][1]["reverse_animation_svgs"][1].format(dur_seconds="2s"),
				reverse_agent_0_move_3=scenario_svg["agents"][0]["reverse_animation_svgs"][2].format(dur_seconds="2s"),
				reverse_agent_1_move_3=scenario_svg["agents"][1]["reverse_animation_svgs"][2].format(dur_seconds="2s"),
				reverse_agent_0_move_4=scenario_svg["agents"][0]["reverse_animation_svgs"][3].format(dur_seconds="2s"),
				reverse_agent_1_move_4=scenario_svg["agents"][1]["reverse_animation_svgs"][3].format(dur_seconds="2s"),
				reverse_agent_0_move_5=scenario_svg["agents"][0]["reverse_animation_svgs"][4].format(dur_seconds="2s"),
				reverse_agent_1_move_5=scenario_svg["agents"][1]["reverse_animation_svgs"][4].format(dur_seconds="2s"),
				reverse_agent_0_move_6=scenario_svg["agents"][0]["reverse_animation_svgs"][5].format(dur_seconds="2s"),
				reverse_agent_1_move_6=scenario_svg["agents"][1]["reverse_animation_svgs"][5].format(dur_seconds="2s"),
				reverse_agent_0_move_7=scenario_svg["agents"][0]["reverse_animation_svgs"][6].format(dur_seconds="2s"),
				reverse_agent_1_move_7=scenario_svg["agents"][1]["reverse_animation_svgs"][6].format(dur_seconds="2s"),
				reverse_agent_0_move_8=scenario_svg["agents"][0]["reverse_animation_svgs"][7].format(dur_seconds="2s"),
				reverse_agent_1_move_8=scenario_svg["agents"][1]["reverse_animation_svgs"][7].format(dur_seconds="2s"),
				reverse_agent_0_move_9=scenario_svg["agents"][0]["reverse_animation_svgs"][8].format(dur_seconds="2s"),
				reverse_agent_1_move_9=scenario_svg["agents"][1]["reverse_animation_svgs"][8].format(dur_seconds="2s"),
				agent_0_init=scenario_svg["agents"][0]["static_svgs"][0],
				agent_1_init=scenario_svg["agents"][1]["static_svgs"][0]))

	for scnario_id, scenario_svg in list(scenario_svgs.items())[:args.max_html_scenarios]:
		with open("data/html/{}_world.html".format(scnario_id), "w") as fout:
			fout.write(template_world_svg.format(
				move_1=scenario_svg["world"]["animation_svgs"][0].format(dur_seconds="2s"),
				move_2=scenario_svg["world"]["animation_svgs"][1].format(dur_seconds="2s"),
				move_3=scenario_svg["world"]["animation_svgs"][2].format(dur_seconds="2s"),
				move_4=scenario_svg["world"]["animation_svgs"][3].format(dur_seconds="2s"),
				move_5=scenario_svg["world"]["animation_svgs"][4].format(dur_seconds="2s"),
				move_6=scenario_svg["world"]["animation_svgs"][5].format(dur_seconds="2s"),
				move_7=scenario_svg["world"]["animation_svgs"][6].format(dur_seconds="2s"),
				move_8=scenario_svg["world"]["animation_svgs"][7].format(dur_seconds="2s"),
				move_9=scenario_svg["world"]["animation_svgs"][8].format(dur_seconds="2s"),
				reverse_move_1=scenario_svg["world"]["reverse_animation_svgs"][0].format(dur_seconds="2s"),
				reverse_move_2=scenario_svg["world"]["reverse_animation_svgs"][1].format(dur_seconds="2s"),
				reverse_move_3=scenario_svg["world"]["reverse_animation_svgs"][2].format(dur_seconds="2s"),
				reverse_move_4=scenario_svg["world"]["reverse_animation_svgs"][3].format(dur_seconds="2s"),
				reverse_move_5=scenario_svg["world"]["reverse_animation_svgs"][4].format(dur_seconds="2s"),
				reverse_move_6=scenario_svg["world"]["reverse_animation_svgs"][5].format(dur_seconds="2s"),
				reverse_move_7=scenario_svg["world"]["reverse_animation_svgs"][6].format(dur_seconds="2s"),
				reverse_move_8=scenario_svg["world"]["reverse_animation_svgs"][7].format(dur_seconds="2s"),
				reverse_move_9=scenario_svg["world"]["reverse_animation_svgs"][8].format(dur_seconds="2s"),
				init=scenario_svg["world"]["static_svgs"][0]))

	return

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

	parser.add_argument('--input_file', type=str, default="scenarios.json")
	parser.add_argument('--output_file', type=str, default="scenario_svgs.json")
	parser.add_argument('--add_world_svgs', action='store_true')
	parser.add_argument('--in_html', action='store_true')
	parser.add_argument('--max_html_scenarios', type=int, default=20)
	args = parser.parse_args()

	if args.in_html:
		if os.path.exists("data/" + args.output_file):
			scenario_svgs = json.load(open("data/" + args.output_file))
		else:
			scenario_svgs = scenario_to_svg(args, scenarios)

		if not os.path.exists("data/html"):
			os.mkdir("data/html")

		scenario_to_html(args, scenario_svgs)
	else:
		scenarios = json.load(open("data/" + args.input_file))

		scenario_svgs = scenario_to_svg(args, scenarios)

		with open("data/" + args.output_file, "w") as fout:
			json.dump(scenario_svgs, fout, indent=4, sort_keys=True)

