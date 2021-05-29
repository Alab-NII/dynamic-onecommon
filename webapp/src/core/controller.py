from cocoa.core.controller import Controller as BaseController
import pdb

class Controller(BaseController):
    def __init__(self, scenario, sessions, chat_id=None):
        super(Controller, self).__init__(scenario, sessions, chat_id, allow_cross_talk=True)
        self.turn = 0
        self.selections = {}
        self.selections[self.turn] = [None, None]

    def event_callback(self, event):
        if event.action == 'select':
            self.selections[self.turn][event.agent] = event.data

    """
    def get_outcome(self):
        if (self.selections[self.turn]["0"] is not None) and (self.selections[self.turn]["1"] is not None) and \
            int(self.selections[self.turn]["0"]) == int(self.selections[self.turn]["1"]):
            reward = 1
        else:
            reward = 0
        return {'reward': reward}
    """

    def game_over(self):
        return not self.inactive() and self.turn >= 5

    def turn_over_and_success(self):
        if self.selections[self.turn][0] is not None and self.selections[self.turn][1] is not None:
            if self.selections[self.turn][0] == self.selections[self.turn][1]:
                turn_success = True
            else:
                turn_success = False
            self.turn += 1
            self.selections[self.turn] = [None, None]
            return True, turn_success, self.turn
        else:
            return False, False, self.turn