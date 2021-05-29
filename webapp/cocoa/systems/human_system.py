from cocoa.systems.system import System
from cocoa.sessions.human_session import HumanSession


class HumanSystem(System):
    def __init__(self):
        super(HumanSystem, self).__init__()

    @classmethod
    def name(cls):
        return 'human'

    def new_session(self, scenario, my_index):
        return HumanSession(my_index)
