import json
from cocoa.core.util import write_json
from cocoa.web.main.db_reader import DatabaseReader as BaseDatabaseReader

class DatabaseReader(BaseDatabaseReader):
    @classmethod
    def process_event_data(cls, action, data):
        if action == 'select':
            data = json.loads(data)
        return data

    # TODO: move this to cocoa. factor survey questions
    @classmethod
    def dump_surveys(cls, cursor, json_path):
        questions = ['obligation', 'cooperative', 'comments']

        cursor.execute('''SELECT * FROM survey''')
        logged_surveys = cursor.fetchall()
        survey_data = {}
        agent_types = {}

        for survey in logged_surveys:
            # todo this is pretty lazy - support variable # of questions per task eventually..
            (userid, cid, _, q1, q2, comments) = survey
            responses = dict(zip(questions, [q1, q2, comments]))
            cursor.execute('''SELECT agent_types, agent_ids FROM chat WHERE chat_id=?''', (cid,))
            chat_result = cursor.fetchone()
            agents = json.loads(chat_result[0])
            agent_ids = json.loads(chat_result[1])
            agent_types[cid] = agents
            if cid not in survey_data.keys():
                survey_data[cid] = {0: {}, 1: {}}
            partner_idx = 0 if agent_ids['1'] == userid else 1
            survey_data[cid][partner_idx] = responses

        write_json([agent_types, survey_data], json_path)

    @classmethod
    def list_chats(cls, cursor, scenario_db, max_chats):
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = sorted(cursor.fetchall())
        examples = []

        def is_single_agent(chat):
            agent_event = {0: 0, 1: 0}
            for event in chat.events:
                agent_event[event.agent] += 1
            return agent_event[0] == 0 or agent_event[1] == 0

        for chat_id in ids[:max_chats]:
            ex = cls.get_chat_example(cursor, chat_id[0], scenario_db)
            if ex is None or is_single_agent(ex):
                continue
            examples.append(ex)

        return examples

    @classmethod
    def check_completed_info(cls, cursor, chat_id):
        cursor.execute('SELECT * FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()

        max_turn = -1
        agent_select = {0: False, 1: False}

        for row in logged_events:
            agent, action, time, data, turn = [row[k] for k in ('agent', 'action', 'time', 'data', 'turn')]

            if max_turn < turn:
                max_turn = turn
                agent_select = {0: False, 1: False}

            if turn == max_turn and action == 'select':
                agent_select[agent] = True

        completed = agent_select[0] and agent_select[1] and max_turn >= 1

        return completed

    @classmethod
    def check_finished_info(cls, cursor, chat_id):
        cursor.execute('SELECT * FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
        logged_events = cursor.fetchall()

        max_turn = -1
        agent_select = {0: False, 1: False}

        for row in logged_events:
            agent, action, time, data, turn = [row[k] for k in ('agent', 'action', 'time', 'data', 'turn')]

            if max_turn < turn:
                max_turn = turn
                agent_select = {0: False, 1: False}

            if turn == max_turn and action == 'select':
                agent_select[agent] = True

        finished = agent_select[0] and agent_select[1]

        return finished

    @classmethod
    def dump_review(cls, cursor, json_path):
        review_info = {}

        # add chat review
        cursor.execute('SELECT chat_id, accept FROM chat_review')
        for chat_id, accept in cursor.fetchall():
            if chat_id not in review_info:
                review_info[chat_id] = {}
            outcome = int(DatabaseReader.get_chat_outcome(cursor, chat_id))
            review_info[chat_id]['accept'] = accept
            review_info[chat_id]['outcome'] = outcome

        # add worker review
        cursor.execute('SELECT worker_id, chat_id, accept, message FROM worker_review')
        for worker_id, chat_id, accept, message in cursor.fetchall():
            if chat_id not in review_info:
                review_info[chat_id] = {}
            review_info[chat_id][worker_id] = {}
            review_info[chat_id][worker_id]['accept'] = accept
            review_info[chat_id][worker_id]['message'] = message

        write_json(review_info, json_path)


