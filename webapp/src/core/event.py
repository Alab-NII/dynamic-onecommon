from cocoa.core.event import Event as BaseEvent

class Event(BaseEvent):
    @classmethod
    def SelectionEvent(cls, agent, data, time, turn):
        return cls(agent, 'select', data, time, turn)
