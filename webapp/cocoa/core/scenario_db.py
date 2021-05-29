def add_scenario_arguments(parser):
    parser.add_argument('--schema_path', help='Input path that describes the schema of the domain')
    parser.add_argument('--scenarios_path', help='Input path for the scenarios')
    parser.add_argument('--scenario_svgs_path', help='Input path for the scenario svgs')

class Scenario(object):
    '''
    A scenario represents a situation to be played out where each agent has a KB.
    '''
    def __init__(self, uuid, attributes, kbs):
        self.uuid = uuid
        self.attributes = attributes
        self.kbs = kbs

    @staticmethod
    def from_dict(schema, scenario_id, scenario):
        raise NotImplementedError

    def to_dict(self):
        return {'uuid': self.uuid,
                'attributes': [attr.to_json() for attr in self.attributes],
                'kbs': [kb.to_dict() for kb in self.kbs]
                }

    def get_kb(self, agent):
        return self.kbs[agent]

class ScenarioDB(object):
    '''
    Consists a list of scenarios (specifies the pair of KBs).
    '''
    def __init__(self, scenarios_map):
        self.scenarios_map = scenarios_map
        self.size = len(self.scenarios_map)

    def get(self, uuid):
        return self.scenarios_map[uuid]

    def select_random(self, exclude_seen=True):
        scenarios = set(self.scenarios_map.keys())

        if exclude_seen:
            scenarios = scenarios - self.selected_scenarios
            if len(scenarios) == 0:
                scenarios = set(self.scenarios_map.keys())
                self.selected_scenarios = set()
        uuid = np.random.choice(list(scenarios))

        return self.scenarios_map[uuid]

    @staticmethod
    def from_dict(schema, scenarios, scenario_class):
        return ScenarioDB({scenario_id: scenario_class.from_dict(schema, scenario_id, scenario) for scenario_id, scenario in scenarios.items()})

    def to_dict(self):
        return [s.to_dict() for s in self.scenarios_list]
