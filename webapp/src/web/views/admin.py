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

admin = Blueprint('admin', __name__)

@admin.route('/_accept_chat/', methods=['GET'])
def accept_chat():
    chat_id = request.args.get('chat_id')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id,))    
    res = cursor.fetchone()
    if res:
        review_status = res[0]
    else:
        review_status = -1

    if int(review_status) >= 0:
        cursor.execute(
            ''' UPDATE chat_review
            SET accept=1
            where chat_id=?''', (chat_id,))
    else:
        cursor.execute(
            '''INSERT INTO chat_review VALUES (?,?)''', (chat_id, 1))
    conn.commit()
    return jsonify(success=True)

@admin.route('/_reject_chat/', methods=['GET'])
def reject_chat():
    chat_id = request.args.get('chat_id')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id,))    
    res = cursor.fetchone()
    if res:
        review_status = res[0]
    else:
        review_status = -1

    if int(review_status) >= 0:
        cursor.execute(
            ''' UPDATE chat_review
            SET accept=0
            where chat_id=?''', (chat_id,))
    else:
        cursor.execute(
            '''INSERT INTO chat_review VALUES (?,?)''', (chat_id, 0))
    conn.commit()
    return jsonify(success=True)

@admin.route('/_accept_worker/', methods=['GET'])
def accept_worker():
    chat_id = request.args.get('chat_id')
    worker_id = request.args.get('worker_id')
    review_message = request.args.get('review_message')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''SELECT accept, message FROM worker_review where chat_id=? and worker_id=?''', (chat_id, worker_id))    
    res = cursor.fetchone()
    if res:
        review_status = res[0]
    else:
        review_status = -1

    if review_status >= 0:
        cursor.execute(
            ''' UPDATE worker_review
            SET accept=1, message=?
            where chat_id=? and worker_id=?''', (review_message, chat_id, worker_id))  
    else:
        cursor.execute(
            '''INSERT INTO worker_review VALUES (?,?,?,?)''', (worker_id, chat_id, 1, review_message))
    conn.commit()
    return jsonify(success=True)

@admin.route('/_reject_worker/', methods=['GET'])
def reject_worker():
    chat_id = request.args.get('chat_id')
    worker_id = request.args.get('worker_id')
    review_message = request.args.get('review_message')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''SELECT accept, message FROM worker_review where chat_id=? and worker_id=?''', (chat_id, worker_id))    
    res = cursor.fetchone()
    if res:
        review_status = res[0]
    else:
        review_status = -1

    if review_status >= 0:
        cursor.execute(
            ''' UPDATE worker_review
            SET accept=0, message=?
            where chat_id=? and worker_id=?''', (review_message, chat_id, worker_id))
    else:
        cursor.execute(
            '''INSERT INTO worker_review VALUES (?,?,?,?)''', (worker_id, chat_id, 0, review_message))

    conn.commit()
    return jsonify(success=True)

@admin.route('/_disconnect_all_users/', methods=['GET'])
def disconnect_all():
    backend = get_backend()
    backend.disconnect_all_users()
    return jsonify(success=True)

@admin.route('/_zero_active/', methods=['GET'])
def zero_active():
    try:
        db_path = app.config['user_params']['db']['location']
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''UPDATE scenario SET active="[]"''')
        conn.commit()  
    except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return jsonify(success=False)
    return jsonify(success=True)

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

@admin.route('/admin')
@requires_auth
def visualize():
    backend = get_backend()
    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if not request.args.get('chat_id'):
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = [x[0] for x in cursor.fetchall()]
        cursor.execute('SELECT DISTINCT chat_id FROM chat_review')
        reviewed_ids = [x[0] for x in cursor.fetchall()]

        if 'finished' in request.args:
            # list all chat_ids which are finished
            ids = [chat_id for chat_id in ids if DatabaseReader.check_finished_info(cursor, chat_id)]
        elif 'unfinished' in request.args:
            # list all chat_ids which are not finished
            ids = [chat_id for chat_id in ids if (not DatabaseReader.check_finished_info(cursor, chat_id))]
        elif 'completed' in request.args:
            # list all chat_ids which are completed
            ids = [chat_id for chat_id in ids if DatabaseReader.check_completed_info(cursor, chat_id)]
        elif 'incompleted' in request.args:
            # list all chat_ids which are not completed
            ids = [chat_id for chat_id in ids if (not DatabaseReader.check_completed_info(cursor, chat_id))]
        elif 'reviewed' in request.args:
            # list all chat_ids which are not completed
            ids = [chat_id for chat_id in ids if chat_id in reviewed_ids]
        elif 'unreviewed' in request.args:
            # list all chat_ids which are not completed
            ids = [chat_id for chat_id in ids if chat_id not in reviewed_ids]
        elif 'accepted' in request.args:
            accepted_ids = []
            for chat_id in ids:
                cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id, ))
                res = cursor.fetchone()
                if res is not None and res[0] == 1:
                    accepted_ids.append(chat_id)
            ids = accepted_ids
        elif 'rejected' in request.args:
            rejected_ids = []
            for chat_id in ids:
                cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id, ))
                res = cursor.fetchone()
                if res is not None and res[0] == 0:
                    rejected_ids.append(chat_id)
            ids = rejected_ids
        elif 'all' in request.args:
            # list all chat_ids which are not completed
            ids = ids
        else:
            # list only chat_ids which are finished but not reviewed yet
            ids = [chat_id for chat_id in ids if chat_id not in reviewed_ids and DatabaseReader.check_finished_info(cursor, chat_id)]
            ids = ids[:100]

        outcomes = []
        dialogues = []
        num_utterances = []
        durations = []
        worker_ids = []
        review_statuses = []
        num_accepted = 0
        num_rejected = 0
        num_completed = 0
        num_incompleted = 0
        num_finished = 0
        num_unfinished = 0
        for chat_id in ids:
            outcome = DatabaseReader.get_chat_outcome(cursor, chat_id)
            outcomes.append(outcome)

            chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
            chat_text = ""
            num_utterance = 0
            for chat_event in chat_info['events']:
                if chat_event['action'] == 'message':
                    chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'])
                    num_utterance += 1
            dialogues.append(chat_text)
            num_utterances.append(num_utterance)

            durations.append(chat_info["time"]["duration"] / 60)
            worker_ids.append((chat_info["agents_info"][0]["agent_id"], chat_info["agents_info"][1]["agent_id"]))

            if chat_id in reviewed_ids:
                cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id, ))
                res = cursor.fetchone()
                if res is None:
                    review_statuses.append("unknown")
                elif res[0] == 1:
                    review_statuses.append("accepted")
                    num_accepted += 1
                elif res[0] == 0:
                    review_statuses.append("rejected")
                    num_rejected += 1
                else:
                    review_statuses.append("unknown")
            else:
                review_statuses.append("")

            completed = DatabaseReader.check_completed_info(cursor, chat_id)
            if completed:
                num_completed += 1
            else:
                num_incompleted += 1

            finished = DatabaseReader.check_finished_info(cursor, chat_id)
            if finished:
                num_finished += 1
            else:
                num_unfinished += 1

        # compute average
        if len(ids) > 0:            
            avg_outcome = np.mean(outcomes)
            avg_utterance = np.mean(num_utterances)
            avg_duration = np.mean(durations)
        else:
            avg_outcome = 0
            avg_utterance = 0
            avg_duration = 0

        # convert to string
        durations = ["{:.2f}".format(dur) for dur in durations]
        avg_duration = "{:.2f}".format(avg_duration)
        avg_outcome =  "{:.2f}".format(avg_outcome)
        avg_utterance =  "{:.2f}".format(avg_utterance)

        return render_template('chat_list.html',
                                chat_ids = ids,
                                num_chats = len(ids),
                                admin_base_url = app.config["base_url"] + '{}/admin?chat_id='.format(app.config["prefix"]),
                                worker_base_url = app.config["base_url"] + '{}/worker?worker_id='.format(app.config["prefix"]),
                                outcomes = outcomes,
                                dialogues=dialogues,
                                num_utterances=num_utterances,
                                durations=durations,
                                worker_ids=worker_ids,
                                review_statuses=review_statuses,
                                num_accepted=num_accepted,
                                num_rejected=num_rejected,
                                num_completed=num_completed,
                                num_incompleted=num_incompleted,
                                num_finished=num_finished,
                                num_unfinished=num_unfinished,
                                avg_outcome=avg_outcome,
                                avg_utterance=avg_utterance,
                                avg_duration=avg_duration)
    else:
        chat_id = request.args.get('chat_id')
        chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
        agent_0 = chat_info['agents_info'][0]['agent_id']
        agent_1 = chat_info['agents_info'][1]['agent_id']
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

        # check chat review status
        cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id, ))
        res = cursor.fetchone()
        if res:
            chat_review_status = res[0]
        else:
            chat_review_status = -1

        # check worker review status
        cursor.execute('''SELECT accept, message FROM worker_review where chat_id=? and worker_id=?''', (chat_id, agent_0))
        res = cursor.fetchone()
        if res:
            agent_0_review_status = res[0]
            agent_0_message = res[1]
        else:
            agent_0_review_status = -1
            agent_0_message = ""

        cursor.execute('''SELECT accept, message FROM worker_review where chat_id=? and worker_id=?''', (chat_id, agent_1))
        res = cursor.fetchone()
        if res:
            agent_1_review_status = res[0]
            agent_1_message = res[1]
        else:
            agent_1_review_status = -1
            agent_1_message = ""

        cursor.execute('''SELECT * FROM survey where chat_id=?''', (chat_id, ))
        res = cursor.fetchone()
        if res:
            survey = True
            cooperative = res[3]
            humanlike = res[4]
            comments = res[5]
        else:
            survey = False
            cooperative = None
            humanlike = None
            comments = None

        scenario_id = chat_info['scenario_id']
        agent_0_svgs = app.config['scenario_db'].scenarios_map[scenario_id].kbs[0].to_dict()
        agent_1_svgs = app.config['scenario_db'].scenarios_map[scenario_id].kbs[1].to_dict()

        entity_move_dur_seconds = 3
        agent_move_dur_seconds = 1
        backward_entity_move_dur_seconds = 0
        backward_agent_move_dur_seconds = 0

        worker_ids = [agent_0, agent_1]

        return render_template('visualize.html',
                                chat_id=chat_id,
                                worker_base_url = app.config["base_url"] + '{}/worker?worker_id='.format(app.config["prefix"]),
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
                                backward_agent_move_dur_seconds = backward_agent_move_dur_seconds,
                                chat_review_status=chat_review_status,
                                agent_0_review_status=agent_0_review_status,
                                agent_1_review_status=agent_1_review_status,
                                agent_0_message=agent_0_message,
                                agent_1_message=agent_1_message,
                                worker_ids=worker_ids,
                                survey=survey,
                                cooperative=cooperative,
                                humanlike=humanlike,
                                comments=comments
                                )

