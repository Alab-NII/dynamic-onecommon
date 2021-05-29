import time
import pdb
from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup
from flask import current_app as app

from cocoa.web.views.utils import generate_userid, userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from web.main.backend import Backend

import pdb

get_backend = Backend.get_backend

main = Blueprint('main', __name__)

def complete_url(url):
    for _key in dict(request.args).keys():
        if _key in ['uid', 'page']:
            continue
        _value = request.args.get(_key,default=None, type=None)
        if _value:
            url += '&{}={}'.format(_key, _value)
    return url

@main.route('/index', methods=['GET', 'POST'])
@main.route('/', methods=['GET', 'POST'])
def main_room():
    """Main room."""
    if request.args.get('assignmentId',default=None, type=None) == 'ASSIGNMENT_ID_NOT_AVAILABLE':
        return render_template('sample_chat.html',
                               title=app.config['task_title'],
                               instructions=Markup(app.config['instructions']),
                               static_svg="<svg width=\"430\" height=\"430\" id=\"1\"> <circle cx=\"215\" cy=\"215\" r=\"205\" fill=\"none\" stroke=\"black\" stroke-width=\"2\" stroke-dasharray=\"3,3\"/> <circle id=\"agt_1_ent_16\" cx=\"256\" cy=\"230\" r=\"8\" fill=\"rgb(150,150,150)\"/> <circle id=\"agt_1_ent_22\" cx=\"370\" cy=\"309\" r=\"10\" fill=\"rgb(181,181,181)\"/> <circle id=\"agt_1_ent_25\" cx=\"357\" cy=\"148\" r=\"11\" fill=\"rgb(116,116,116)\"/> <circle id=\"agt_1_ent_31\" cx=\"50\" cy=\"103\" r=\"8\" fill=\"rgb(110,110,110)\"/> <circle id=\"agt_1_ent_37\" cx=\"413\" cy=\"240\" r=\"10\" fill=\"rgb(148,148,148)\"/> <circle id=\"agt_1_ent_6\" cx=\"269\" cy=\"151\" r=\"10\" fill=\"rgb(190,190,190)\"/> <circle id=\"agt_1_ent_7\" cx=\"349\" cy=\"355\" r=\"11\" fill=\"rgb(104,104,104)\"/> <circle id=\"agt_1_ent_9\" cx=\"150\" cy=\"366\" r=\"7\" fill=\"rgb(180,180,180)\"/> </svg>")

    if not request.args.get('uid'):
        if request.args.get('workerId'):
            # link for Turkers
            prefix = "MT_"
            user_id = prefix + request.args.get('workerId')
        else:
            prefix = "U_"
            user_id = generate_userid("U_")

        url = complete_url(app.config["base_url"] + app.config["prefix"] + '/?{}={}'.format('uid', user_id))
        return redirect(url)

    backend = get_backend()

    user_id = userid()

    backend.create_user_if_not_exists(user_id)

    backend.initialize_status(user_id)

    # request args
    hitId = request.args.get('hitId', default=None, type=None)
    assignmentId = request.args.get('assignmentId', default=None, type=None)
    turkSubmitTo = request.args.get('turkSubmitTo', default=None, type=None)
    workerId = request.args.get('workerId', default=None, type=None)

    mturk = True if hitId else None

    enable_chat = backend.check_tutorial(user_id) or app.config['skip_tutorial']

    base_url = request.url.split('?')[0]

    tutorial_link = complete_url(app.config["base_url"] + app.config["prefix"] + '/tutorial?{}={}'.format('uid', user_id))

    chat_link = complete_url(app.config["base_url"] + app.config["prefix"] + '/chat?{}={}'.format('uid', user_id))

    return render_template('main.html',
                            title=app.config['task_title'],
                            enable_chat=enable_chat,
                            tutorial_link=tutorial_link,
                            chat_link=chat_link)


