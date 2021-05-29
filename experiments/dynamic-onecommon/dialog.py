import sys
import pdb

import numpy as np

import torch

from metric import MetricsContainer
import data
import utils
import domain

class DialogLogger(object):
    def __init__(self, verbose=False, log_file=None, append=False, scenarios=None):
        self.logs = []
        if verbose:
            self.logs.append(sys.stderr)
        if log_file:
            flags = 'a' if append else 'w'
            self.logs.append(open(log_file, flags))
        
        self.scenarios = scenarios

    def _dump(self, s, forced=False):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s):
        self._dump('{0: <5} : {1}'.format(name, s))

    def _scenario_to_svg(self, scenario, choice=None):
        svg_list = []
        for agent in [0,1]:
            svg = "<svg width=\"{0}\" height=\"{0}\" id=\"{1}\">".format(430, "agent_" + str(agent))
            svg += '''<circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/>'''
            for obj in scenario['kbs'][agent]:
                svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"{3}\"/>".format(obj['x'], obj['y'], 
                    obj['size'], obj['color'])
                if choice and choice[agent] == obj['id']:
                    if agent == 0:
                        agent_color = "red"
                    else:
                        agent_color = "blue"
                    svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"none\" stroke=\"{3}\" stroke-width=\"4\" stroke-dasharray=\"3,3\" />".format(obj['x'], obj['y'],
                        obj['size'] + 4, agent_color)
            svg += "</svg>"
            svg_list.append(svg)
        return svg_list

    def _attention_to_svg(self, scenario, agent, attention=None):
        svg = '''<svg id="svg" width="430" height="430"><circle cx="215" cy="215" r="205" fill="none" stroke="black" stroke-width="2" stroke-dasharray="3,3"/> '''
        for obj, attention_weight in zip(scenario['kbs'][agent], attention):
            svg += "<circle cx=\"{0}\" cy=\"{1}\" r=\"{2}\" fill=\"rgb(255,{3},{3})\" />".format(obj['x'], obj['y'],
                                                                                                 obj['size'], int((1 - attention_weight) * 255))
        svg += '''</svg>'''
        return svg

    def dump_sent(self, name, sent):
        self._dump_with_name(name, ' '.join(sent))

    def dump_attention(self, agent_name, agent_id, scenario_id, attention):
        svg = self._attention_to_svg(self.scenarios[scenario_id], agent_id, attention)
        self._dump_with_name('%s_attention' % agent_name, svg)

    def dump_scenario_id(self, scenario_id):
        self._dump_with_name('scenario_id', scenario_id + "\n")

    def dump_result(self, turn, agree):
        if agree:
            result = "Success!\n"
        else:
            result = "Fail\n"
        self._dump_with_name('Turn {}'.format(turn), result)

    def dump_overall_result(self, successful_turns):
        self._dump('Successful turns: {}'.format(successful_turns))

    def dump(self, s, forced=False):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = ' '.join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = ' '.join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += ' ' + self.name2choice[name]
                        self.name2example[name] += ' ' + self.name2choice[other_name]
                        self._dump(self.name2example[name])


class Dialog(object):
    def __init__(self, agents, args):
        # For now we only suppport dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()
        self.selfplay_transcripts = {}

    def _register_metrics(self):
        self.metrics.register_average('dialog_len')
        self.metrics.register_average('sent_len')
        if self.args.adversarial_run:
            self.metrics.register_percentage('first_turn')
            self.metrics.register_percentage('first_turn_shared_4')
            self.metrics.register_percentage('first_turn_shared_5')
            self.metrics.register_percentage('first_turn_shared_6')
            self.metrics.register_percentage('adv_first_turn')
            self.metrics.register_percentage('adv_first_turn_shared_4')
            self.metrics.register_percentage('adv_first_turn_shared_5')
            self.metrics.register_percentage('adv_first_turn_shared_6')
            self.metrics.register_percentage('target_change')
            self.metrics.register_percentage('utterance_decrease')
            self.metrics.register_percentage('utterance_increase')
        else:
            self.metrics.register_average('successful_turns')
            self.metrics.register_percentage('first_turn')
            self.metrics.register_percentage('first_turn_shared_4')
            self.metrics.register_percentage('first_turn_shared_5')
            self.metrics.register_percentage('first_turn_shared_6')
            self.metrics.register_percentage('same_target')
            self.metrics.register_percentage('same_target_shared_4')
            self.metrics.register_percentage('same_target_shared_5')
            self.metrics.register_percentage('same_target_shared_6')
            self.metrics.register_percentage('change_target')
            self.metrics.register_percentage('change_target_shared_4')
            self.metrics.register_percentage('change_target_shared_5')
            self.metrics.register_percentage('change_target_shared_6')
            self.metrics.register_percentage('later_turn')
        self.metrics.register_time('time')
        for agent in self.agents:
            self.metrics.register_percentage('%s_make_sel' % agent.name)

    def _is_selection(self, out):
        return '<selection>' in out

    def show_metrics(self):
        return ' '.join(['%s=%s' % (k, v) for k, v in self.metrics.dict().items()])

    def plot_metrics(self):
        self.metrics.plot()

    def run(self, scenario_id, scenario, logger, max_words=5000):
        for agent, agent_id in zip(self.agents, [0, 1]):
            #agent.feed_context(scenario["agents"][agent_id], turn=0)
            #agent.real_ids = real_ids
            agent.agent_id = agent_id

        # Choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        speaker = []
        self.metrics.reset()

        words_left = max_words
        length = 0
        expired = False

        # init logging
        logger.dump_scenario_id(scenario_id)
        self.selfplay_transcripts[scenario_id] = {}
        self.selfplay_transcripts[scenario_id]["agents_info"] = []
        for agent_id in [0, 1]:
            self.selfplay_transcripts[scenario_id]["agents_info"].append(self.agents[agent_id].train_args.model_file)
        self.selfplay_transcripts[scenario_id]["events"] = []

        current_turn = 0
        successful_turns = 0
        prev_selection = None
        while current_turn < 5:
            # feed context
            for agent, agent_id in zip(self.agents, [0, 1]):
                agent.feed_context(scenario["agents"][agent_id], turn=current_turn)

            same_target = True
            if current_turn > 0:
                for agent_id in [0, 1]:
                    if prev_selection not in scenario["agents"][agent_id][current_turn]['selectable_entity_ids']:
                        same_target = False
                        break

            num_shared = len(set(scenario["agents"][0][current_turn]['selectable_entity_ids']).intersection(set(scenario["agents"][1][current_turn]['selectable_entity_ids'])))

            while True:
                utterance = writer.write(max_words=words_left)
                words_left -= len(utterance)
                length += len(utterance)

                self.metrics.record('sent_len', len(utterance))

                conv.append(utterance)
                speaker.append(writer.agent_id)
                reader.read(utterance)
                if not writer.human:
                    logger.dump_sent(writer.name, utterance)

                if self._is_selection(utterance):
                    self.metrics.record('%s_make_sel' % writer.name, 1)
                    self.metrics.record('%s_make_sel' % reader.name, 0)
                    break
                else:
                    msg_event = {"action": "message", "agent": writer.agent_id, "data": " ".join(utterance[:-1]), "turn": current_turn}
                    self.selfplay_transcripts[scenario_id]["events"].append(msg_event)

                if words_left <= 1:
                    expired = True
                    break

                writer, reader = reader, writer

            choices = []
            for agent in self.agents:
                choice = agent.choose()
                choices.append(choice)
                select_event = {"action": "select", "agent": agent.agent_id, "data": int(choice.split('_')[1]), "turn": current_turn}
                self.selfplay_transcripts[scenario_id]["events"].append(select_event)

            agree, rewards = self.domain.score_choices(choices)
            if expired:
                agree = False

            if current_turn == 0:
                self.metrics.record('first_turn', int(agree))
                self.metrics.record('first_turn_shared_{}'.format(num_shared), int(agree))
            else:
                self.metrics.record('later_turn', int(agree))
                if same_target:
                    self.metrics.record('same_target', int(agree))
                    self.metrics.record('same_target_shared_{}'.format(num_shared), int(agree))
                else:
                    self.metrics.record('change_target', int(agree))
                    self.metrics.record('change_target_shared_{}'.format(num_shared), int(agree))

            logger.dump_result(current_turn + 1, agree)

            if agree:
                current_turn += 1
                successful_turns += 1
                prev_selection = choices[0]
            else:
                break

        logger.dump('-' * 80)
        logger.dump_overall_result(successful_turns)

        self.metrics.record('time')
        self.metrics.record('dialog_len', len(conv))
        self.metrics.record('successful_turns', successful_turns)
        self.selfplay_transcripts[scenario_id]["outcome"] = successful_turns

        logger.dump('-' * 80)
        logger.dump(self.show_metrics())
        logger.dump('-' * 80)

        return conv, successful_turns, rewards
    
    def adversarial_run(self, scenario_id, scenario, logger, max_words=5000):
        for agent, agent_id in zip(self.agents, [0, 1]):
            agent.agent_id = agent_id

        # Choose who goes first by random
        if np.random.rand() < 0.5:
            writer, reader = self.agents
        else:
            reader, writer = self.agents

        conv = []
        speaker = []
        #self.metrics.reset()

        words_left = max_words
        length = 0
        expired = False

        # init logging
        #logger.dump_scenario_id(scenario_id)

        current_turn = 0
        successful_turns = 0
        prev_selection = None

        # feed context
        for agent, agent_id in zip(self.agents, [0, 1]):
            agent.feed_context(scenario["agents"][agent_id], turn=current_turn)

        num_shared = len(set(scenario["agents"][0][current_turn]['selectable_entity_ids']).intersection(set(scenario["agents"][1][current_turn]['selectable_entity_ids'])))
        shared_ent_ids = set(scenario["agents"][0][current_turn]['selectable_entity_ids']).intersection(set(scenario["agents"][1][current_turn]['selectable_entity_ids']))
        non_shared_ent_ids = set(scenario["agents"][0][current_turn]['selectable_entity_ids']).difference(set(scenario["agents"][1][current_turn]['selectable_entity_ids']))

        while True:
            utterance = writer.write(max_words=words_left)
            words_left -= len(utterance)
            length += len(utterance)

            #self.metrics.record('sent_len', len(utterance))

            conv.append(utterance)
            speaker.append(writer.agent_id)
            reader.read(utterance)
            #if not writer.human:
                #logger.dump_sent(writer.name, utterance)

            if self._is_selection(utterance):
                #self.metrics.record('%s_make_sel' % writer.name, 1)
                #self.metrics.record('%s_make_sel' % reader.name, 0)
                break

            if words_left <= 1:
                expired = True
                break

            writer, reader = reader, writer

        choices = []
        for agent in self.agents:
            choice = agent.choose()
            choices.append(choice)

        agree, rewards = self.domain.score_choices(choices)
        if expired:
            agree = False

        self.metrics.record('first_turn', int(agree))
        self.metrics.record('first_turn_shared_{}'.format(num_shared), int(agree))

        if agree:
            # Choose who goes first by random
            if np.random.rand() < 0.5:
                writer, reader = self.agents
            else:
                reader, writer = self.agents

            adv_conv = []
            adv_speaker = []
            self.metrics.reset()

            words_left = max_words
            length = 0
            expired = False

            # init logging
            logger.dump_scenario_id(scenario_id + "_adv")

            current_turn = 0
            successful_turns = 0
            prev_selection = None

            selected_ent_id = choices[0]
            swap_ent_id = np.random.choice(list(non_shared_ent_ids))

            _selected_ent_idx = scenario["agents"][0][current_turn]['all_entity_ids'].index(selected_ent_id)
            selected_ent_color = scenario["agents"][0][current_turn]['all_entities'][_selected_ent_idx]['color'][0]
            selected_ent_size = scenario["agents"][0][current_turn]['all_entities'][_selected_ent_idx]['size'][0]
            num_frames = len(scenario["agents"][0][current_turn]['all_entities'][_selected_ent_idx]['color'])
            _swap_ent_idx = scenario["agents"][0][current_turn]['all_entity_ids'].index(swap_ent_id)
            swap_ent_color = scenario["agents"][0][current_turn]['all_entities'][_swap_ent_idx]['color'][0]
            swap_ent_size = scenario["agents"][0][current_turn]['all_entities'][_swap_ent_idx]['size'][0]

            for agent, agent_id in zip(self.agents, [0, 1]):
                # swap all_entities
                if selected_ent_id in scenario["agents"][agent_id][current_turn]['all_entity_ids']:
                    _selected_ent_idx = scenario["agents"][agent_id][current_turn]['all_entity_ids'].index(selected_ent_id)
                    scenario["agents"][agent_id][current_turn]['all_entities'][_selected_ent_idx]['color'] = [swap_ent_color] * num_frames
                    scenario["agents"][agent_id][current_turn]['all_entities'][_selected_ent_idx]['size'] = [swap_ent_size] * num_frames
                if swap_ent_id in scenario["agents"][agent_id][current_turn]['all_entity_ids']:
                    _swap_ent_idx = scenario["agents"][agent_id][current_turn]['all_entity_ids'].index(swap_ent_id)
                    scenario["agents"][agent_id][current_turn]['all_entities'][_swap_ent_idx]['color'] = [selected_ent_color] * num_frames
                    scenario["agents"][agent_id][current_turn]['all_entities'][_swap_ent_idx]['size'] = [selected_ent_size] * num_frames

                # swap all_selected_entities
                if selected_ent_id in scenario["agents"][agent_id][current_turn]['selectable_entity_ids']:
                    _selected_ent_idx = scenario["agents"][agent_id][current_turn]['selectable_entity_ids'].index(selected_ent_id)
                    scenario["agents"][agent_id][current_turn]['selectable_entities'][_selected_ent_idx]['color'] = [swap_ent_color] * num_frames
                    scenario["agents"][agent_id][current_turn]['selectable_entities'][_selected_ent_idx]['size'] = [swap_ent_size] * num_frames
                if swap_ent_id in scenario["agents"][agent_id][current_turn]['selectable_entity_ids']:
                    _swap_ent_idx = scenario["agents"][agent_id][current_turn]['selectable_entity_ids'].index(swap_ent_id)
                    scenario["agents"][agent_id][current_turn]['selectable_entities'][_swap_ent_idx]['color'] = [selected_ent_color] * num_frames
                    scenario["agents"][agent_id][current_turn]['selectable_entities'][_swap_ent_idx]['size'] = [selected_ent_size] * num_frames

                # feed context
                agent.feed_context(scenario["agents"][agent_id], turn=current_turn)

            while True:
                utterance = writer.write(max_words=words_left)
                words_left -= len(utterance)
                length += len(utterance)

                self.metrics.record('sent_len', len(utterance))

                adv_conv.append(utterance)
                adv_speaker.append(writer.agent_id)
                reader.read(utterance)
                if not writer.human:
                    logger.dump_sent(writer.name, utterance)

                if self._is_selection(utterance):
                    self.metrics.record('%s_make_sel' % writer.name, 1)
                    self.metrics.record('%s_make_sel' % reader.name, 0)
                    break

                if words_left <= 1:
                    expired = True
                    break

                writer, reader = reader, writer

            choices = []
            for agent in self.agents:
                choice = agent.choose()
                choices.append(choice)

            adv_agree, rewards = self.domain.score_choices(choices)
            if expired:
                adv_agree = False

            self.metrics.record('target_change', int(selected_ent_id != choices[0]))
            self.metrics.record('adv_first_turn', int(adv_agree))
            self.metrics.record('adv_first_turn_shared_{}'.format(num_shared), int(adv_agree))
            self.metrics.record('utterance_increase', int(len(adv_conv) > len(conv)))
            self.metrics.record('utterance_decrease', int(len(adv_conv) < len(conv)))
            logger.dump('-' * 80)

            self.metrics.record('time')
            self.metrics.record('dialog_len', len(adv_conv))

            logger.dump('-' * 80)
            logger.dump(self.show_metrics())
            logger.dump('-' * 80)

            conv = adv_conv
        else:
            adv_agree = False

        return conv, agree, adv_agree, rewards
