from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup, Response
from flask import current_app as app

from functools import wraps
import json
import sqlite3
import pdb

import time

from collections import defaultdict
import operator

import numpy as np

from cocoa.web.views.utils import userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from main.db_reader import DatabaseReader

from web.main.backend import Backend
get_backend = Backend.get_backend

selfplay = Blueprint('selfplay', __name__)

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'sample' and password == 'sample'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@selfplay.route('/selfplay')
@requires_auth
def visualize():
    backend = get_backend()
    selfplay_transcripts = app.config['selfplay_transcripts']
    scenario_svgs = app.config['scenario_svgs']

    if not request.args.get('scenario_id'):
        outcomes = []
        scenario_ids = []
        for scenario_id in selfplay_transcripts.keys():
            outcome = int(selfplay_transcripts[scenario_id]["outcome"])
            if outcome >= 0:
                outcomes.append(outcome)
                scenario_ids.append(scenario_id)

        return render_template('simple_chat_list.html',
                                chat_ids = scenario_ids,
                                outcomes = outcomes,
                                avg_outcome = "{:.2f}".format(np.mean(outcomes)),
                                num_chats = len(scenario_ids),
                                base_url = request.url + '?scenario_id=',
                                )
    else:
        scenario_id = request.args.get('scenario_id')
        chat_info = selfplay_transcripts[scenario_id]
        agent_0 = chat_info['agents_info'][0]
        agent_1 = chat_info['agents_info'][1]
        chat_text = ""
        agent_0_selections = []
        agent_1_selections = []
        agent_0_utterances = []
        agent_1_utterances = []
        for chat_event in chat_info['events']:
            if len(agent_0_utterances) <= chat_event['turn']:
                agent_0_utterances.append("")
                agent_1_utterances.append("")

            if chat_event['action'] == 'message':
                chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'])
                if chat_event['agent'] == 0:
                    agent_0_utterances[-1] += "A: " + chat_event['data'] + "<br>"
                    #lines = 1 + len("A: " + chat_event['data']) // 40
                    agent_1_utterances[-1] += "<span style='opacity:0;'>" + "A: " + chat_event['data'] + "<br>" + "</span>"
                else:
                    agent_1_utterances[-1] += "B: " + chat_event['data'] + "<br>"
                    #lines = 1 + len("B: " + chat_event['data']) // 40
                    agent_0_utterances[-1] += "<span style='opacity:0;'>" + "B: " + chat_event['data'] + "<br>" + "</span>"

            elif chat_event['action'] == 'select':
                chat_text += "<{} selected {}>\n".format(chat_event['agent'], chat_event['data'])
                if chat_event['agent'] == 0:
                    if len(agent_0_selections) <= chat_event['turn']:
                        agent_0_selections.append("agt_0_ent_" + str(chat_event['data']))
                    else:
                        agent_0_selections[chat_event['turn']] = "agt_0_ent_" + str(chat_event['data'])
                    agent_0_utterances[-1] += "A: SELECT <span style='color:green'>green</span><br>"
                    agent_1_utterances[-1] += "<span style='opacity:0;'>" + "A: SELECT <span style='color:green'>green</span><br>" + "</span>"
                else:
                    if len(agent_1_selections) <= chat_event['turn']:
                        agent_1_selections.append("agt_1_ent_" + str(chat_event['data']))
                    else:
                        agent_1_selections[chat_event['turn']] = "agt_1_ent_" + str(chat_event['data'])
                    agent_1_utterances[-1] += "B: SELECT <span style='color:green'>green</span><br>"
                    agent_0_utterances[-1] += "<span style='opacity:0;'>" + "B: SELECT <span style='color:green'>green</span><br>" + "</span>"

        agent_0_svgs = scenario_svgs[scenario_id]["agents"][0]
        agent_1_svgs = scenario_svgs[scenario_id]["agents"][1]

        entity_move_dur_seconds = 3
        agent_move_dur_seconds = 1
        backward_entity_move_dur_seconds = 0
        backward_agent_move_dur_seconds = 0

        worker_ids = [agent_0, agent_1]

        return render_template('simple_visualize.html',
                                chat_id=scenario_id,
                                chat_text=chat_text,
                                agent_0_static_svgs=agent_0_svgs["static_svgs"],#markup_svgs,
                                agent_0_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_0_svgs["animation_svgs"])],
                                agent_0_fast_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_0_svgs["animation_svgs"])],
                                agent_0_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_0_svgs["reverse_animation_svgs"])],
                                agent_0_fast_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_0_svgs["reverse_animation_svgs"])],
                                agent_1_static_svgs=agent_1_svgs["static_svgs"],#markup_svgs,
                                agent_1_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_1_svgs["animation_svgs"])],
                                agent_1_fast_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_1_svgs["animation_svgs"])],
                                agent_1_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_1_svgs["reverse_animation_svgs"])],
                                agent_1_fast_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_1_svgs["reverse_animation_svgs"])],
                                agent_0_selections=agent_0_selections,
                                agent_1_selections=agent_1_selections,
                                agent_0_utterances=agent_0_utterances,
                                agent_1_utterances=agent_1_utterances,
                                entity_move_dur_seconds = entity_move_dur_seconds,
                                agent_move_dur_seconds = agent_move_dur_seconds,
                                backward_entity_move_dur_seconds = backward_entity_move_dur_seconds,
                                backward_agent_move_dur_seconds = backward_agent_move_dur_seconds
                                )

