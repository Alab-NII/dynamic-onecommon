import sqlite3
from datetime import datetime
import json

from cocoa.core.dataset import Example
from cocoa.core.event import Event
from cocoa.core.util import write_json

import pdb

class DatabaseReader(object):
    date_fmt = '%Y-%m-%d %H-%M-%S'

    @classmethod
    def convert_time_format(cls, time):
        if time is None:
            return time
        try:
            dt = datetime.strptime(time, cls.date_fmt)
            s = str((dt - datetime.fromtimestamp(0)).total_seconds())
            return s
        except (ValueError, TypeError):
            try:
                dt = datetime.fromtimestamp(float(time)) # make sure that time is a UNIX timestamp
                return time
            except (ValueError, TypeError):
                print('Unrecognized time format: %s' % time)

        return None

    @classmethod
    def process_event_data(cls, action, data):
        """Construct structured data from strings.

        Data can be some json dict string that needs to be loaded,
        e.g. "{'price': 10}".

        """
        if action == 'eval':
            data = json.loads(data)
        return data

    @classmethod
    def get_chat_outcome(cls, cursor, chat_id):
        """Get outcome of the chat specified by chat_id.

        Returns:
            {}

        """
        cursor.execute('SELECT outcome FROM chat WHERE chat_id=?', (chat_id,))
        outcome = cursor.fetchone()[0]
        try:
            outcome = int(outcome)
        except ValueError:
            outcome = -1
        return outcome

    @classmethod
    def get_chat_agents_info(cls, cursor, chat_id):
        """Get info of the two agents in the chat specified by chat_id.

        Returns:
            [agent0_info, agent1_info]
        """
        try:
            cursor.execute('SELECT agent_ids FROM chat WHERE chat_id=?', (chat_id,))
            agent_ids = cursor.fetchone()[0]
            agent_ids = json.loads(agent_ids)
            cursor.execute('SELECT agent_types FROM chat WHERE chat_id=?', (chat_id,))
            agent_types = cursor.fetchone()[0]
            agent_types = json.loads(agent_types)
            agents_info = [{"agent_id": agent_ids["0"], "agent_type": agent_types["0"]},
                           {"agent_id": agent_ids["1"], "agent_type": agent_types["1"]}]
        except sqlite3.OperationalError:
            agents_info = [{"agent_id": "unknown", "agent_type": HumanSystem.name()},
                           {"agent_id": "unknown", "agent_type": HumanSystem.name()}]
        return agents_info

    @classmethod
    def get_chat_events(cls, cursor, chat_id, include_meta=False):
        """Read all events in the chat specified by chat_id.

        Returns:
            [Event]

        """
        cursor.execute('SELECT * FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()

        chat_events = []
        agent_chat = {0: False, 1: False}
        for row in logged_events:
            # Compatible with older event structure
            agent, action, data, time, turn = [row[k] for k in ('agent', 'action', 'data', 'time', 'turn')]
            try:
                start_time = row['start_time']
            except IndexError:
                start_time = time
            try:
                metadata = json.loads(row['metadata'])
            except IndexError:
                metadata = None

            if not include_meta:
                if action == 'join' or action == 'leave' or action == 'typing':
                    continue
            if action == 'message' and len(data.strip()) == 0:
                continue

            data = cls.process_event_data(action, data)
            agent_chat[agent] = True
            time = cls.convert_time_format(time)
            start_time = cls.convert_time_format(start_time)
            event = Event(agent, action, data, time, turn, start_time, metadata)
            chat_events.append(event)

        return chat_events

    @classmethod
    def has_chat(cls, cursor, chat_id):
        """Check if a chat is in the DB.
        """
        cursor.execute('SELECT scenario_id, outcome FROM chat WHERE chat_id=?', (chat_id,))
        result = cursor.fetchone()
        if result is None:
            return False
        return True

    @classmethod
    def get_chat_scenario_id(cls, cursor, chat_id):
        cursor.execute('SELECT scenario_id FROM chat WHERE chat_id=?', (chat_id,))
        uuid = cursor.fetchone()[0]
        return uuid

    @classmethod
    def get_chat_example(cls, cursor, chat_id, scenario_db, include_meta=False):
        """Read a dialogue from the DB.

        Args:
            chat_id (str)
            scenario_db (ScenarioDB): map scenario ids to Scenario

        Returns:
            Example

        """
        if not cls.has_chat(cursor, chat_id):
            return None

        scenario_id = cls.get_chat_scenario_id(cursor, chat_id)
        agents_info = cls.get_chat_agents_info(cursor, chat_id)
        events = cls.get_chat_events(cursor, chat_id, include_meta)
        outcome = cls.get_chat_outcome(cursor, chat_id)
        time = cls.get_chat_time(cursor, chat_id, include_meta)

        return Example(scenario_id, agents_info, outcome, events, time)

    @classmethod
    def dump_chats(cls, cursor, scenario_db, json_path, uids=None, accepted_only=False):
        """Dump chat transcripts to a JSON file.

        Args:
            scenario_db (ScenarioDB): retrieve Scenario by logged uuid.
            json_path (str): output path.
            uids (list): if provided, only log chats from these users.

        """
        if uids is None:
            cursor.execute('SELECT DISTINCT chat_id FROM event')
            ids = cursor.fetchall()
        else:
            ids = []
            uids = [(x,) for x in uids]
            for uid in uids:
                cursor.execute('SELECT chat_id FROM mturk_task WHERE name=?', uid)
                ids_ = cursor.fetchall()
                ids.extend(ids_)

        if accepted_only:
            cursor.execute('SELECT DISTINCT chat_id FROM chat_review WHERE accept=1')
            ids = cursor.fetchall()
        else:
            cursor.execute('SELECT DISTINCT chat_id FROM chat_review')
            ids = cursor.fetchall()

        examples = {}
        for res in ids:
            chat_id = res[0]
            ex = cls.get_chat_example(cursor, chat_id, scenario_db)
            if ex is None:
                continue
            examples[chat_id] = ex.to_dict()

        write_json(examples, json_path)
        print(len(examples.keys()))

    @classmethod
    def get_chat_time(cls, cursor, chat_id, include_meta=False):
        events = cls.get_chat_events(cursor, chat_id, include_meta)

        time = {}
        time["start_time"] = float(events[0].time)
        time["duration"] = float(events[-1].time) - float(events[0].time)

        return time


