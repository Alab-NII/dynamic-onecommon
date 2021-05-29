import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit
from gevent.pywsgi import WSGIServer
import sys
import copy

from cocoa.core.scenario_db import add_scenario_arguments, ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import read_json
from cocoa.systems.human_system import HumanSystem
from cocoa.web.main.logger import WebLogger

from core.scenario import Scenario
from systems import get_system
from main.db_reader import DatabaseReader
from main.backend import DatabaseManager

from flask import g
from web.main.backend import Backend
from flask import Flask, current_app
from flask_socketio import SocketIO
socketio = SocketIO()

get_backend = Backend.get_backend

DB_FILE_NAME = 'chat_state.db'
LOG_FILE_NAME = 'log.out'
ERROR_LOG_FILE_NAME = 'error_log.out'
TRANSCRIPTS_DIR = 'transcripts'

def close_connection(exception):
    backend = getattr(g, '_backend', None)
    if backend is not None:
        backend.close()


def create_app(debug=False, templates_dir='templates', prefix='sample'):
    """Create an application."""

    app = Flask(__name__, template_folder=os.path.abspath(templates_dir), static_url_path='/{}/static'.format(prefix))
    app.debug = debug
    app.config['SECRET_KEY'] = 'gjr39dkjn344_!67#'
    app.config['PROPAGATE_EXCEPTIONS'] = True

    from web.views.action import action
    from web.views.admin import admin
    from web.views.worker import worker
    from web.views.annotation import annotation
    from web.views.tutorial import tutorial
    from web.views.main import main
    from web.views.dataset import dataset
    from web.views.selfplay import selfplay
    from cocoa.web.views.chat import chat
    app.register_blueprint(main, url_prefix='/' + prefix)
    app.register_blueprint(chat, url_prefix='/' + prefix)
    app.register_blueprint(action, url_prefix='/' + prefix)
    app.register_blueprint(admin, url_prefix='/' + prefix)
    app.register_blueprint(worker, url_prefix='/' + prefix)
    app.register_blueprint(annotation, url_prefix='/' + prefix)
    app.register_blueprint(tutorial, url_prefix='/' + prefix)
    app.register_blueprint(dataset, url_prefix='/' + prefix)
    app.register_blueprint(selfplay, url_prefix='/' + prefix)

    app.teardown_appcontext_funcs = [close_connection]

    socketio.init_app(app)
    return app

def add_website_arguments(parser):
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    parser.add_argument('--output', type=str,
                        default="web/output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help='Name of directory for storing website output (debug and error logs, chats, '
                             'and database). Defaults to a web_output/current_date, with the current date formatted as '
                             '%%Y-%%m-%%d". '
                             'If the provided directory exists, all data in it is overwritten unless the '
                             '--reuse parameter is provided.')
    parser.add_argument('--reuse', action='store_true', help='If provided, reuses the existing database file in the '
                                                             'output directory.')

def add_systems(args, config_dict):
    """
    Params:
        config_dict: A dictionary that maps the bot name to a dictionary containing configs for the bot. The
            dictionary should contain the bot type (key 'type') and. for bots that use an underlying model for generation,
            the path to the directory containing the parameters, vocab, etc. for the model.
    Returns:
        agents: A dict mapping from the bot name to the System object for that bot.
        pairing_probabilities: A dict mapping from the bot name to the probability that a user is paired with that
            bot. Also includes the pairing probability for humans (backend.Partner.Human)
    """

    total_probs = 0.0
    systems = {HumanSystem.name(): HumanSystem()}
    pairing_probabilities = {}
    timed = False if params['debug'] else True
    for (sys_name, info) in config_dict.items():
        if "active" not in info.keys():
            warnings.warn("active status not specified for bot %s - assuming that bot is inactive." % sys_name)
        if info["active"]:
            model_name = info["type"]
            try:
                model = get_system(name, args, schema=schema, timed=timed)
            except ValueError:
                warnings.warn(
                    'Unrecognized model type in {} for configuration '
                    '{}. Ignoring configuration.'.format(info, sys_name))
                continue
            systems[sys_name] = model
            if 'prob' in info.keys():
                prob = float(info['prob'])
                pairing_probabilities[sys_name] = prob
                total_probs += prob

    if total_probs > 1.0:
        raise ValueError("Probabilities for active bots can't exceed 1.0.")
    if len(pairing_probabilities.keys()) != 0 and len(pairing_probabilities.keys()) != len(systems.keys()):
        remaining_prob = (1.0-total_probs)/(len(systems.keys()) - len(pairing_probabilities.keys()))
    else:
        remaining_prob = 1.0 / len(systems.keys())
    inactive_bots = set()
    for system_name in systems.keys():
        if system_name not in pairing_probabilities.keys():
            if remaining_prob == 0.0:
                inactive_bots.add(system_name)
            else:
                pairing_probabilities[system_name] = remaining_prob

    for sys_name in inactive_bots:
        systems.pop(sys_name, None)

    return systems, pairing_probabilities

def cleanup(flask_app):
    db_path = flask_app.config['user_params']['db']['location']
    transcripts = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'transcripts.json')
    accepted_transcripts = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'accepted-transcripts.json')
    review_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'reviews.json')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], transcripts)
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], accepted_transcripts, accepted_only=True)
    if flask_app.config['user_params']['end_survey'] == 1:
        surveys_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'surveys.json')
        DatabaseReader.dump_surveys(cursor, surveys_path)
    DatabaseReader.dump_review(cursor, review_path)
    conn.close()

    # output annotation
    with open("data/admin_spatio_temporal_annotation.json", "w") as fout:
        json.dump(app.config['admin_spatio_temporal_annotation'], fout, indent=4, sort_keys=True)

    with open("data/admin_target_selection_annotation.json", "w") as fout:
        json.dump(app.config['admin_target_selection_annotation'], fout, indent=4, sort_keys=True)

def init(output_dir, reuse=False):
    db_file = os.path.join(output_dir, DB_FILE_NAME)
    log_file = os.path.join(output_dir, LOG_FILE_NAME + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"))
    error_log_file = os.path.join(output_dir, ERROR_LOG_FILE_NAME + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"))
    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_DIR)
    # TODO: don't remove everything
    if not reuse:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        db = DatabaseManager.init_database(db_file)

        if os.path.exists(transcripts_dir):
            shutil.rmtree(transcripts_dir)
        os.makedirs(transcripts_dir)
    else:
        db = DatabaseManager(db_file)

    return db, log_file, error_log_file, transcripts_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_website_arguments(parser)
    add_scenario_arguments(parser)
    args = parser.parse_args()

    if not args.reuse and os.path.exists(args.output):
        overwrite = input("[warning] overwriting data: Continue? [Y]:")
        if not overwrite == "Y":
            sys.exit()
    elif args.reuse and not os.path.exists(args.output):
        raise ValueError("output directory does not exist (can not be reused)")

    params_file = args.config
    with open(params_file) as fin:
        params = json.load(fin)

    db, log_file, error_log_file, transcripts_dir = init(args.output, args.reuse)
    error_log_file = open(error_log_file, 'w')

    WebLogger.initialize(log_file)
    params['db'] = {}
    params['db']['location'] = db.db_file
    params['logging'] = {}
    params['logging']['app_log'] = log_file
    params['logging']['chat_dir'] = transcripts_dir

    if 'task_title' not in params.keys():
        raise ValueError("Title of task should be specified in config file with the key 'task_title'")

    instructions = None
    if 'instructions' in params.keys():
        instructions_file = open(params['instructions'], 'r')
        instructions = "".join(instructions_file.readlines())
        instructions_file.close()
    else:
        raise ValueError("Location of file containing instructions for task should be specified in config with the key "
                         "'instructions")
 
    templates_dir = None
    if 'templates_dir' in params.keys():
        templates_dir = params['templates_dir']
    else:
        raise ValueError("Location of HTML templates should be specified in config with the key templates_dir")
    if not os.path.exists(templates_dir):
            raise ValueError("Specified HTML template location doesn't exist: %s" % templates_dir)

    app = create_app(debug=False, templates_dir=templates_dir, prefix=params['prefix'])
    app.config["base_url"] = "http://{}:{}/".format(args.host, args.port) # specify base URL

    schema_path = args.schema_path

    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)

    schema = Schema(schema_path)
    scenarios = read_json(args.scenarios_path)
    scenario_db = ScenarioDB.from_dict(schema, scenarios, Scenario)
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    if 'quit_after' not in params.keys():
        params['quit_after'] = params['status_params']['chat']['num_seconds'] + 500

    if 'skip_chat_enabled' not in params.keys():
        params['skip_chat_enabled'] = False

    if 'end_survey' not in params.keys() :
        params['end_survey'] = 0

    if 'debug' not in params:
        params['debug'] = False

    systems, pairing_probabilities = add_systems(args, params['models'])

    db.add_scenarios(scenario_db, systems, update=args.reuse)

    app.config['systems'] = systems
    app.config['sessions'] = defaultdict(None)
    app.config['pairing_probabilities'] = pairing_probabilities
    app.config['num_chats_per_scenario'] = params.get('num_chats_per_scenario', {k: 1 for k in systems})
    for k in systems:
        assert k in app.config['num_chats_per_scenario']
    app.config['schema'] = schema
    app.config['user_params'] = params
    app.config['controller_map'] = defaultdict(None)
    app.config['instructions'] = instructions
    app.config['task_title'] = params['task_title']
    app.config['prefix'] = params['prefix']
    app.config['skip_tutorial'] = params['skip_tutorial']

    """
        read additional files
    """

    with open("web/mturk/transcripts/accepted-transcripts.json", "r") as f:
        accepted_transcripts = json.load(f)
    with open("web/mturk_2/transcripts/accepted-transcripts.json", "r") as f:
        accepted_transcripts_2 = json.load(f)
        accepted_transcripts.update(accepted_transcripts_2)
    with open("web/mturk_3/transcripts/accepted-transcripts.json", "r") as f:
        accepted_transcripts_3 = json.load(f)
        accepted_transcripts.update(accepted_transcripts_3)
    app.config['accepted_transcripts'] = accepted_transcripts

    with open("data/scenario_svgs.json", "r") as f:
        scenario_svgs = json.load(f)
    with open("data/scenario_svgs_2.json", "r") as f:
        scenario_svgs_2 = json.load(f)
        scenario_svgs.update(scenario_svgs_2)
    with open("data/scenario_svgs_3.json", "r") as f:
        scenario_svgs_3 = json.load(f)
        scenario_svgs.update(scenario_svgs_3)
    with open("data/scenario_svgs_4.json", "r") as f:
        scenario_svgs_4 = json.load(f)
        scenario_svgs.update(scenario_svgs_4)
    app.config['scenario_svgs'] = scenario_svgs

    """
        additional annotation
    """
    with open("data/admin_spatio_temporal_annotation.json", "r") as f:
        spatio_temporal_annotation = json.load(f)
        app.config['admin_spatio_temporal_annotation'] = spatio_temporal_annotation

    with open("data/admin_target_selection_annotation.json", "r") as f:
        target_selection_annotation = json.load(f)
        app.config['admin_target_selection_annotation'] = target_selection_annotation

    with open("data/model_target_selection_annotation.json", "r") as f:
        target_selection_annotation = json.load(f)
        app.config['model_target_selection_annotation'] = target_selection_annotation

    with open("data/selfplay_transcripts.json", "r") as f:
        selfplay_transcripts = json.load(f)
        app.config['selfplay_transcripts'] = selfplay_transcripts

    if 'icon' not in params.keys():
        app.config['task_icon'] = 'handshake.jpg'
    else:
        app.config['task_icon'] = params['icon']

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # create table chat review if not exists
    if not cursor.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='chat_review' ''').fetchone():
        cursor.execute(
            '''CREATE TABLE chat_review (chat_id text, accept integer)'''
        )
        conn.commit()
        print("created new table chat review")

    # create table worker review if not exists
    if not cursor.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='worker_review' ''').fetchone():
        cursor.execute(
            '''CREATE TABLE worker_review (worker_id text, chat_id text, accept integer, message text)'''
        )
        conn.commit()
        print("created new table worker review")


    print("prefix: {}".format(app.config['prefix']))
    print("App setup complete")

    server = WSGIServer(('', args.port), app, log=WebLogger.get_logger(), error_log=error_log_file)
    atexit.register(cleanup, flask_app=app)
    server.serve_forever()
