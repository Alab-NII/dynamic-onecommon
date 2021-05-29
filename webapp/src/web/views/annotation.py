from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup, Response
from flask import current_app as app

from functools import wraps
import json
import sqlite3
import pdb

from collections import defaultdict
import operator

from cocoa.web.views.utils import userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from main.db_reader import DatabaseReader

from web.main.backend import Backend
get_backend = Backend.get_backend

annotation = Blueprint('annotation', __name__)

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

@annotation.route('/_select_target/', methods=['GET'])
def select_target():
    chat_id = request.args.get('chat_id')
    sample_agent_id = request.args.get('sample_agent_id')
    annotator_id = request.args.get('annotator_id')
    selection = request.args.get('selection')
    turn = int(request.args.get('turn'))

    if annotator_id == "admin":
        if len(app.config['admin_target_selection_annotation'][chat_id][sample_agent_id]) > turn:
            app.config['admin_target_selection_annotation'][chat_id][sample_agent_id][turn] = selection
        else:
            app.config['admin_target_selection_annotation'][chat_id][sample_agent_id].append(selection)
    else:
        return jsonify(success=False)
    return jsonify(success=True)

@annotation.route('/_select_previous/', methods=['GET'])
def select_previous():
    chat_id = request.args.get('chat_id')
    annotator_id = request.args.get('annotator_id')
    utterance_id = request.args.get('utterance_id')

    if annotator_id == "admin":
        if utterance_id not in app.config['admin_spatio_temporal_annotation'][chat_id]:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id] = {'previous': True, 'movement': False, 'current': False, 'none': False}
        elif app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['previous']:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['previous'] = False
            if not any(app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id].values()):
                del(app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id])
        else:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['previous'] = True
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none'] = False
    else:
        return jsonify(success=False)
    return jsonify(success=True)

@annotation.route('/_select_movement/', methods=['GET'])
def select_movement():
    chat_id = request.args.get('chat_id')
    annotator_id = request.args.get('annotator_id')
    utterance_id = request.args.get('utterance_id')

    if annotator_id == "admin":
        if utterance_id not in app.config['admin_spatio_temporal_annotation'][chat_id]:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id] = {'previous': False, 'movement': True, 'current': False, 'none': False}
        elif app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['movement']:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['movement'] = False
            if not any(app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id].values()):
                del app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]
        else:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['movement'] = True
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none'] = False
    else:
        return jsonify(success=False)
    return jsonify(success=True)

@annotation.route('/_select_current/', methods=['GET'])
def select_current():
    chat_id = request.args.get('chat_id')
    annotator_id = request.args.get('annotator_id')
    utterance_id = request.args.get('utterance_id')

    if annotator_id == "admin":
        if utterance_id not in app.config['admin_spatio_temporal_annotation'][chat_id]:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id] = {'previous': False, 'movement': False, 'current': True, 'none': False}
        elif app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['current']:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['current'] = False
            if not any(app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id].values()):
                del app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]
        else:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['current'] = True
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none'] = False
    else:
        return jsonify(success=False)
    return jsonify(success=True)

@annotation.route('/_select_none/', methods=['GET'])
def select_none():
    chat_id = request.args.get('chat_id')
    annotator_id = request.args.get('annotator_id')
    utterance_id = request.args.get('utterance_id')

    if annotator_id == "admin":
        if utterance_id not in app.config['admin_spatio_temporal_annotation'][chat_id]:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id] = {'previous': False, 'movement': False, 'current': False, 'none': True}
        elif app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none']:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none'] = False
            if not any(app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id].values()):
                del app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]
        else:
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['previous'] = False
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['movement'] = False
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['current'] = False
            app.config['admin_spatio_temporal_annotation'][chat_id][utterance_id]['none'] = True
    else:
        return jsonify(success=False)
    return jsonify(success=True)

@annotation.route('/annotation')
@requires_auth
def annotation_list():
    backend = get_backend()
    accepted_transcripts = app.config['accepted_transcripts']
    scenario_svgs = app.config['scenario_svgs']

    if not request.args.get('annotator_id') or request.args.get('annotator_id') not in ["admin", "model"]:
        return render_template('annotation_list.html',
                                chat_ids = [],
                                target_selection_urls = [],
                                target_selection_statuses = [],
                                target_selection_completed = 0,
                                target_selection_total = 0,
                                spatio_temporal_urls = [],
                                spatio_temporal_statuses = [],
                                spatio_temporal_completed = 0,
                                spatio_temporal_total = 0,
                                annotator_id = "Annotator ID not found",
                                num_chats = 0
                                )
    elif not request.args.get('annotation_type') or request.args.get('annotation_type') not in ["target_selection", "spatio_temporal"]:
        if request.args.get('annotator_id') == "admin":
            spatio_temporal_annotation = app.config['admin_spatio_temporal_annotation']
            target_selection_annotation = app.config['admin_target_selection_annotation']
        elif request.args.get('annotator_id') == "model":
            spatio_temporal_annotation = {}
            target_selection_annotation = app.config['model_target_selection_annotation']
        annotator_id = request.args.get('annotator_id')

        chat_ids = set(spatio_temporal_annotation.keys())
        chat_ids = chat_ids.union(target_selection_annotation.keys())
        chat_ids = sorted(list(chat_ids))

        target_selection_urls = []
        target_selection_statuses = []
        target_selection_completed = 0
        target_selection_total = 0
        spatio_temporal_urls = []
        spatio_temporal_statuses = []
        spatio_temporal_completed = 0
        spatio_temporal_total = 0

        for chat_id in chat_ids:
            # target selection
            if chat_id in target_selection_annotation.keys():
                sample_agent_id = list(target_selection_annotation[chat_id].keys())[0]
                target_selection_urls.append(app.config["base_url"] + '{}/annotation?annotation_type=target_selection&chat_id={}&annotator_id={}'.format(app.config["prefix"], chat_id, annotator_id))

                # check status
                selections = {}
                for event in accepted_transcripts[chat_id]["events"]:
                    turn = event['turn']
                    agent_id = event['agent']
                    if event['action'] == 'select':
                        if agent_id == int(sample_agent_id):
                            ent_id = event['data']
                            selections[turn] = ent_id

                if len(target_selection_annotation[chat_id][sample_agent_id]) == 0:
                    target_selection_statuses.append("")
                elif len(target_selection_annotation[chat_id][sample_agent_id]) < len(selections):
                    target_selection_statuses.append("WIP...")
                else:
                    target_selection_statuses.append("Completed!")
                    target_selection_completed += 1
                target_selection_total += 1
            else:
                target_selection_urls.append(None)
                target_selection_statuses.append(None)

            # spatio temporal
            if chat_id in spatio_temporal_annotation.keys():
                spatio_temporal_urls.append(app.config["base_url"] + '{}/annotation?annotation_type=spatio_temporal&chat_id={}&annotator_id={}'.format(app.config["prefix"], chat_id, annotator_id))

                # check status
                num_utterances = 0
                for event in accepted_transcripts[chat_id]["events"]:
                    turn = event['turn']
                    agent_id = event['agent']
                    if event['action'] == 'message':
                        num_utterances += 1

                if len(spatio_temporal_annotation[chat_id]) == 0:
                    spatio_temporal_statuses.append("")
                elif len(spatio_temporal_annotation[chat_id]) < num_utterances:
                    spatio_temporal_statuses.append("WIP...")
                else:
                    spatio_temporal_statuses.append("Completed!")
                    spatio_temporal_completed += 1
                spatio_temporal_total += 1
            else:
                spatio_temporal_urls.append(None)
                spatio_temporal_statuses.append(None)

        return render_template('annotation_list.html',
                                chat_ids = chat_ids,
                                target_selection_urls = target_selection_urls,
                                target_selection_statuses = target_selection_statuses,
                                target_selection_completed = target_selection_completed,
                                target_selection_total = target_selection_total,
                                spatio_temporal_urls = spatio_temporal_urls,
                                spatio_temporal_statuses = spatio_temporal_statuses,
                                spatio_temporal_completed = spatio_temporal_completed,
                                spatio_temporal_total = spatio_temporal_total,
                                annotator_id = annotator_id,
                                num_chats = len(chat_ids),
                                )

    elif request.args.get('annotation_type') == "target_selection":
        if request.args.get('annotator_id') == "admin":
            target_selection_annotation = app.config['admin_target_selection_annotation']
        elif request.args.get('annotator_id') == "model":
            target_selection_annotation = app.config['model_target_selection_annotation']
        chat_id = request.args.get('chat_id')
        annotator_id = request.args.get('annotator_id')
        sample_agent_id = list(target_selection_annotation[chat_id].keys())[0]
        annotated_selections = target_selection_annotation[chat_id][sample_agent_id]

        chat_info = accepted_transcripts[chat_id]
        chat_text = ""
        agent_0_utterances = []
        agent_1_utterances = []
        selections = {}
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
                agent_id = chat_event['agent']
                ent_id = chat_event['data']
                turn = chat_event['turn']
                if agent_id == int(sample_agent_id):
                    selections[turn] = ent_id

        scenario_id = chat_info['scenario_id']
        agent_0_svgs = scenario_svgs[scenario_id]["agents"][0]
        agent_1_svgs = scenario_svgs[scenario_id]["agents"][1]
        if int(sample_agent_id) == 0:
            agent_svgs = agent_0_svgs
            partner_svgs = agent_1_svgs
        else:
            agent_svgs = agent_1_svgs
            partner_svgs = agent_0_svgs

        max_turns = len(selections) - 1

        entity_move_dur_seconds = 3
        agent_move_dur_seconds = 1
        backward_entity_move_dur_seconds = 0
        backward_agent_move_dur_seconds = 0

        return render_template('target_selection.html',
                                chat_id=chat_id,
                                annotator_id=annotator_id,
                                sample_agent_id=int(sample_agent_id),
                                annotated_selections=annotated_selections,
                                chat_text=chat_text,
                                agent_static_svgs=agent_svgs["static_svgs"],#markup_svgs,
                                agent_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_svgs["animation_svgs"])],
                                agent_fast_forward_svgs=[svg.format(dur_seconds=str(entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_svgs["animation_svgs"])],
                                agent_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds) + "s") for i, svg in enumerate(agent_svgs["reverse_animation_svgs"])],
                                agent_fast_backward_svgs=[svg.format(dur_seconds=str(backward_entity_move_dur_seconds / 2) + "s") if i % 2 == 0 else svg.format(dur_seconds=str(backward_agent_move_dur_seconds / 2) + "s") for i, svg in enumerate(agent_svgs["reverse_animation_svgs"])],
                                agent_0_utterances=agent_0_utterances,
                                agent_1_utterances=agent_1_utterances,
                                entity_move_dur_seconds = entity_move_dur_seconds,
                                agent_move_dur_seconds = agent_move_dur_seconds,
                                backward_entity_move_dur_seconds = backward_entity_move_dur_seconds,
                                backward_agent_move_dur_seconds = backward_agent_move_dur_seconds,
                                current_turn = min(len(annotated_selections), max_turns),
                                max_turns = max_turns,
                                )

    elif request.args.get('annotation_type') == "spatio_temporal":
        if request.args.get('annotator_id') == "admin":
            spatio_temporal_annotation = app.config['admin_spatio_temporal_annotation']
        chat_id = request.args.get('chat_id')
        annotator_id = request.args.get('annotator_id')
        annotated_spatio_temoral = spatio_temporal_annotation[chat_id]

        chat_info = accepted_transcripts[chat_id]
        chat_text = ""
        agent_0_utterances = []
        agent_1_utterances = []
        utterance_id = 0
        selections = {}
        turn2utterance_ids = defaultdict(list)
        prev_utterance_agent = -1
        for chat_event in chat_info['events']:
            if len(agent_0_utterances) <= chat_event['turn']:
                agent_0_utterances.append("")
                agent_1_utterances.append("")
                prev_utterance_agent = -1

            if chat_event['action'] == 'message':
                chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'])
                utterance_id_in_text = "utterance_{}".format(utterance_id)

                if chat_event['agent'] == prev_utterance_agent:
                    agent_0_utterances[-1] += "<br>"
                    agent_1_utterances[-1] += "<br>"
                prev_utterance_agent = chat_event['agent']

                if chat_event['agent'] == 0:
                    agent_0_utterances[-1] += "A: " + chat_event['data'] + "<br>" + \
                        "<input type='checkbox' id='utterance_{0}_prev' onclick='selectPrevious(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_prev'>Previous</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_move' onclick='selectMovement(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_move'>Movement</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_curr' onclick='selectCurrent(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_curr'>Current</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_none' onclick='selectNone(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_none'>None</label>".format(utterance_id)
                    #lines = 1 + len("A: " + chat_event['data']) // 40
                    agent_1_utterances[-1] += "<span style='opacity:0;'>" + "A: " + chat_event['data'] + "<br>" + "</span>" + "<br>"
                else:
                    agent_1_utterances[-1] += "B: " + chat_event['data'] + "<br>" + \
                        "<input type='checkbox' id='utterance_{0}_prev' onclick='selectPrevious(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_prev'>Previous</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_move' onclick='selectMovement(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_move'>Movement</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_curr' onclick='selectCurrent(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_curr'>Current</label>&emsp;".format(utterance_id) + \
                        "<input type='checkbox' id='utterance_{0}_none' onclick='selectNone(&quot;utterance_{0}&quot;)'> <label for='utterance_{0}_none'>None</label>".format(utterance_id)
                    #lines = 1 + len("B: " + chat_event['data']) // 40
                    agent_0_utterances[-1] += "<span style='opacity:0;'>" + "B: " + chat_event['data'] + "<br>" + "</span>" + "<br>"
                turn = chat_event['turn']
                turn2utterance_ids[turn].append(utterance_id_in_text) # add turn2utterance_ids
                utterance_id += 1
            elif chat_event['action'] == 'select':
                agent_id = chat_event['agent']
                ent_id = chat_event['data']
                turn = chat_event['turn']
                selections[turn] = ent_id

        scenario_id = chat_info['scenario_id']
        agent_0_svgs = scenario_svgs[scenario_id]["agents"][0]
        agent_1_svgs = scenario_svgs[scenario_id]["agents"][1]

        max_turns = len(selections) - 1

        entity_move_dur_seconds = 3
        agent_move_dur_seconds = 1
        backward_entity_move_dur_seconds = 0
        backward_agent_move_dur_seconds = 0

        return render_template('spatio_temporal.html',
                                chat_id=chat_id,
                                annotator_id=annotator_id,
                                chat_text=chat_text,
                                annotated_spatio_temoral=annotated_spatio_temoral,
                                turn2utterance_ids=dict(turn2utterance_ids),
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
                                agent_0_utterances=agent_0_utterances,
                                agent_1_utterances=agent_1_utterances,
                                entity_move_dur_seconds = entity_move_dur_seconds,
                                agent_move_dur_seconds = agent_move_dur_seconds,
                                backward_entity_move_dur_seconds = backward_entity_move_dur_seconds,
                                backward_agent_move_dur_seconds = backward_agent_move_dur_seconds,
                                current_turn = max_turns,
                                max_turns = max_turns,
                                )
