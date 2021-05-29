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

worker = Blueprint('worker', __name__)

@worker.route('/_accept_worker/', methods=['GET'])
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

@worker.route('/_reject_worker/', methods=['GET'])
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


@worker.route('/worker')
@requires_auth
def worker_information():
    backend = get_backend()
    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if not request.args.get('worker_id'):
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = [x[0] for x in cursor.fetchall()]

        worker_list = {}

        for chat_id in ids:
            outcome = DatabaseReader.get_chat_outcome(cursor, chat_id)
            chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
            completed = DatabaseReader.check_completed_info(cursor, chat_id)
            finished = DatabaseReader.check_finished_info(cursor, chat_id)

            workers = [chat_info['agents_info'][0]['agent_id'], chat_info['agents_info'][1]['agent_id']]
            for worker in workers:
                if worker not in worker_list:
                    worker_list[worker] = {}
                    worker_list[worker]["completed"] = 0
                    worker_list[worker]["incompleted"] = 0
                    worker_list[worker]["finished"] = 0
                    worker_list[worker]["unfinished"] = 0
                    #worker_list[worker]["accepted"] = 0
                    #worker_list[worker]["rejected"] = 0
                    worker_list[worker]["outcome"] = []

                if completed:
                    worker_list[worker]["completed"] += 1
                else:
                    worker_list[worker]["incompleted"] += 1
                if finished:
                    worker_list[worker]["finished"] += 1
                    worker_list[worker]["outcome"].append(outcome)
                else:
                    worker_list[worker]["unfinished"] += 1

        for worker in worker_list.keys():
            if len(worker_list[worker]["outcome"]) > 0:
                worker_list[worker]["avg_outcome"] = np.mean(worker_list[worker]["outcome"])
            else:
                worker_list[worker]["avg_outcome"] = "-"

        # sort worker_list by number of completed dialogues
        worker_list = {k: v for k, v in sorted(worker_list.items(), key=lambda item: -item[1]["completed"])}

        return render_template('worker_list.html',
                                num_workers = len(worker_list),
                                worker_ids = list(worker_list.keys()),
                                worker_list = worker_list,
                                worker_base_url = app.config["base_url"] + '{}/worker?worker_id='.format(app.config["prefix"]))

    else:
        worker_id = request.args.get('worker_id')

        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = [x[0] for x in cursor.fetchall()]

        chat_ids = []
        outcomes = []
        len_utterances = []
        durations = []
        num_completed = 0
        num_incompleted = 0
        num_finished = 0
        num_unfinished = 0
        num_accepted = 0
        num_rejected = 0
        num_unreviewed = 0
        for chat_id in ids:
            chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
            if worker_id not in [chat_info['agents_info'][0]['agent_id'], chat_info['agents_info'][1]['agent_id']]:
                continue

            chat_ids.append(chat_id)

            outcome = DatabaseReader.get_chat_outcome(cursor, chat_id)
            outcomes.append(outcome)

            num_utterance = 0
            for chat_event in chat_info['events']:
                if chat_event['action'] == 'message':
                    num_utterance += 1
                    len_utterance = len(chat_event['data'].split())
                    len_utterances.append(len_utterance)

            durations.append(chat_info["time"]["duration"] / 60)

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

            if completed:
                # check chat review status
                cursor.execute('''SELECT accept FROM chat_review where chat_id=?''', (chat_id, ))
                res = cursor.fetchone()
                if res:
                    chat_review_status = res[0]
                else:
                    chat_review_status = -1

                # check worker review status
                cursor.execute('''SELECT accept, message FROM worker_review where chat_id=? and worker_id=?''', (chat_id, worker_id))
                res = cursor.fetchone()
                if res:
                    worker_review_status = res[0]
                else:
                    worker_review_status = -1

                if worker_review_status == 1:
                    num_accepted += 1
                elif worker_review_status == 0:
                    num_rejected += 1
                else:
                    if chat_review_status == 1:
                        num_accepted += 1
                    elif chat_review_status == 0:
                        num_rejected += 1
                    else:
                        num_unreviewed += 1

        # compute average
        if len(chat_ids) > 0:            
            avg_outcome = np.mean(outcomes)
            avg_duration = np.mean(durations)
            completed_rate = num_completed / (num_completed + num_incompleted)
            finished_rate = num_finished / (num_finished + num_unfinished)

            # convert to string
            avg_outcome =  "{:.2f}".format(avg_outcome)
            avg_duration =  "{:.2f}".format(avg_duration)
            completed_rate = "{:.1f}".format(100.0 * completed_rate)
            finished_rate = "{:.1f}".format(100.0 * finished_rate)
        else:
            avg_outcome = ""
            avg_duration = ""

        if len(len_utterances) > 0:
            avg_len_utterance = np.mean(len_utterances)
            avg_len_utterance = "{:.2f}".format(avg_len_utterance)
        else:
            avg_len_utterance = ""

        if num_accepted + num_rejected > 0:
            accepted_rate = num_accepted / (num_accepted + num_rejected)
            accepted_rate = "{:.1f}".format(100.0 * accepted_rate)
        else:
            accepted_rate = ""

        return render_template('worker_info.html',
                                worker_id=worker_id,
                                total_chats=len(chat_ids),
                                avg_outcome=avg_outcome,
                                avg_len_utterance=avg_len_utterance,
                                avg_duration=avg_duration,
                                completed_rate=completed_rate,
                                finished_rate=finished_rate,
                                accepted_rate=accepted_rate
                                )

