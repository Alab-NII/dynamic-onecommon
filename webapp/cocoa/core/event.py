
class Event(object):
    """
    An atomic event of a dialogue, which could be someone talking or making a selection.

    Params:
    agent: The index of the agent triggering the event
    time: Time at which event occurred
    action: The action this event corresponds to ('select', 'message', ..)
    data: Any data that is part of the event
    start_time: The time at which the event action was started (e.g. the time at which an agent starting typing a
    message to send)
    """

    decorative_events = ('join', 'leave', 'typing', 'eval', 'success')

    def __init__(self, agent, action, data, time, turn, start_time=None, metadata=None):
        self.agent = agent
        self.data = data
        self.time = time
        self.action = action
        self.turn = turn
        self.start_time = start_time
        self.metadata = metadata

    @staticmethod
    def from_dict(raw):
        return Event(raw['agent'], raw['time'], raw['action'], raw['data'], raw['turn'], start_time=raw.get('start_time'), metadata=raw.get('metadata'))

    def to_dict(self):
        return {'agent': self.agent, 'time': self.time, 'action': self.action, 'data': self.data, 'turn': self.turn,
                'start_time': self.start_time, 'metadata': self.metadata}

    @classmethod
    def MessageEvent(cls, agent, data, time, turn, start_time=None, metadata=None):
        return cls(agent, 'message', data, time, turn, start_time=start_time, metadata=metadata)

    @classmethod
    def JoinEvent(cls, agent, userid, time, turn):
        return cls(agent, 'join', userid, time, turn)

    @classmethod
    def LeaveEvent(cls, agent, userid, time, turn):
        return cls(agent, 'leave', userid, time, turn)

    @classmethod
    def TypingEvent(cls, agent, data, time, turn):
        return cls(agent, 'typing', data, time, turn)

    @classmethod
    def SuccessEvent(cls, agent, userid, time, turn):
        return cls(agent, 'success', userid, time, turn)

    @classmethod
    def EvalEvent(cls, agent, data, time, turn):
        return cls(agent, 'eval', data, time, turn)

    @staticmethod
    def gather_eval(events):
        event_dict = {e.time: e for e in events if e.action != 'eval'}
        for e in events:
            if e.action == 'eval':
                event_dict[e.time].tags = [k for k, v in e.data['labels'].iteritems() if v != 0]
            else:
                event_dict[e.time].tags = []
        events_with_eval = [v for k, v in sorted(event_dict.iteritems(), key=lambda x: x[0])]
        return events_with_eval
