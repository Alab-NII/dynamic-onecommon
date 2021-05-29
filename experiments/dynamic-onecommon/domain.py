# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
A collection of the implemented negotiation domain.
"""

import re
import pdb

def get_domain(name):
    """Creates domain by name."""
    if name == 'static':
        return StaticDomain()
    elif name == 'dynamic':
        return DynamicDomain()
    raise()

class Domain(object):
    """Domain interface."""
    def selection_length(self):
        """The length of the selection output."""
        pass

    def input_length(self):
        """The length of the context/input."""
        pass

    def generate_choices(self, ctx):
        """Generates all the possible valid choices based on the given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def parse_context(self, ctx):
        """Parses a given context.

        ctx: a list of strings that represents a context for the negotiation.
        """
        pass

    def score(self, context, choice):
        """Scores the dialogue.

        context: the input of the dialogue.
        choice: the generated choice by an agent.
        """
        pass

    def parse_choice(self, choice):
        """Parses the generated choice.

        choice: a list of strings like 'itemX=Y'
        """
        pass

    def parse_human_choice(self, inpt, choice):
        """Parses human choices. It has extra validation that parse_choice.

        inpt: the context of the dialogue.
        choice: the generated choice by a human
        """
        pass

    def score_choices(self, choices):
        """Scores choices.

        choices: agents choices.
        """
        pass

class StaticDomain(Domain):
    def selection_length(self):
        return 1

    def input_length(self):
        return 28

    def num_ent(self):
        return 7

    def dim_ent(self):
        return 4

    def generate_choices(self, inpt):
        cnts, _ = self.parse_context(inpt)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                right_choice = ['item%d=%d' % (i, n - c) for i, (n, c) in enumerate(zip(cnts, choice))]
                return [left_choice + right_choice]
            choices = []
            for c in range(cnts[idx] + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()
            return choices
        choices = gen(cnts)
        choices.append(['<no_agreement>'] * self.selection_length())
        choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def parse_context(self, ctx):
        cnts = [int(n) for n in ctx[0::2]]
        vals = [int(v) for v in ctx[1::2]]
        return cnts, vals

    def score(self, context, choice):
        assert len(choice) == (self.selection_length())
        choice = choice[0:len(choice) // 2]
        if choice[0] == '<no_agreement>':
            return 0
        _, vals = self.parse_context(context)
        score = 0
        for i, (c, v) in enumerate(zip(choice, vals)):
            idx, cnt = self.parse_choice(c)
            # Verify that the item idx is correct
            assert idx == i
            score += cnt * v
        return score

    def parse_choice(self, choice):
        match = self.item_pattern.match(choice)
        assert match is not None, 'choice %s' % choice
        # Returns item idx and it's count
        return (int(match.groups()[0]), int(match.groups()[1]))

    def parse_human_choice(self, inpt, output):
        choice = int(output)
        assert choice >= 0 and choice < self.num_ent()
        return choice

    def _to_int(self, x):
        try:
            return int(x)
        except:
            return 0

    def score_choices(self, choices):
        agree = (choices[0] == choices[1])
        scores = [int(agree), int(agree)]
        return agree, scores


class DynamicDomain(Domain):
    """Instance of the Dynamic-OneCommon domain."""

    def selection_length(self):
        return 1

    def input_length(self):
        return 28

    def num_selectable(self):
        return 7

    def dim_ent(self):
        return 4

    def num_bin(self):
        return 4

    def max_ent_each_turn(self):
        return 20

    def generate_choices(self, inpt):
        cnts, _ = self.parse_context(inpt)

        def gen(cnts, idx=0, choice=[]):
            if idx >= len(cnts):
                left_choice = ['item%d=%d' % (i, c) for i, c in enumerate(choice)]
                right_choice = ['item%d=%d' % (i, n - c) for i, (n, c) in enumerate(zip(cnts, choice))]
                return [left_choice + right_choice]
            choices = []
            for c in range(cnts[idx] + 1):
                choice.append(c)
                choices += gen(cnts, idx + 1, choice)
                choice.pop()
            return choices
        choices = gen(cnts)
        choices.append(['<no_agreement>'] * self.selection_length())
        choices.append(['<disconnect>'] * self.selection_length())
        return choices

    def parse_context(self, ctx):
        cnts = [int(n) for n in ctx[0::2]]
        vals = [int(v) for v in ctx[1::2]]
        return cnts, vals

    def score(self, context, choice):
        assert len(choice) == (self.selection_length())
        choice = choice[0:len(choice) // 2]
        if choice[0] == '<no_agreement>':
            return 0
        _, vals = self.parse_context(context)
        score = 0
        for i, (c, v) in enumerate(zip(choice, vals)):
            idx, cnt = self.parse_choice(c)
            # Verify that the item idx is correct
            assert idx == i
            score += cnt * v
        return score

    def parse_choice(self, choice):
        match = self.item_pattern.match(choice)
        assert match is not None, 'choice %s' % choice
        # Returns item idx and it's count
        return (int(match.groups()[0]), int(match.groups()[1]))

    def parse_human_choice(self, inpt, output):
        choice = int(output)
        assert choice >= 0 and choice < self.num_ent()
        return choice

    def _to_int(self, x):
        try:
            return int(x)
        except:
            return 0

    def score_choices(self, choices):
        agree = (choices[0] == choices[1])
        scores = [int(agree), int(agree)]
        return agree, scores
