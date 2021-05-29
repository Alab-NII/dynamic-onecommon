from flask import Blueprint, jsonify, request
from cocoa.web.views.utils import userid, format_message
from web.main.backend import get_backend
import pdb

action = Blueprint('action', __name__)

@action.route('/_select_option/', methods=['GET'])
def select_option():
    backend = get_backend()
    selection = request.args.get('selection')
    turn = request.args.get('turn')
    #if selection_id == -1:
    #    return
    selection_id = int(selection.split('_')[-1])

    selected_item = backend.select(userid(), selection_id, turn)

    #ordered_item = backend.schema.get_ordered_item(selected_item)
    displayed_message = format_message("You selected", True)
    return jsonify(message=displayed_message)
