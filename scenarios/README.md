# Generating Scenarios

To generate new scenarios with the default world parameters, specify a random seed and the number of scenarios:

```
python generate_scenarios.py --seed 2021 --num_scenarios 10
```

You can unify the generated scenarios (based on different random seeds) into one file. To unify the scenarios (`data/scenario_seed_*.json`) and compute overall statistics, run

```
python scenario_statistics.py --unify_scenarios
python scenario_statistics.py --compute_statistics
```

The generated scenarios should be in the following format:

```
scenarios.json
|--scenarios
|  |--scenario_id
|    |--agents
|      |--[i]
|        |--xs
|        |--ys
|        |--r
|    |--entities
|      |--entity_id
|        |--xs
|        |--ys
|        |--thetas
|        |--moves
|        |--color
|        |--size

[optional]

|--world_parameters
|  |--color_range
|  |--min_dist_ent
|  |--min_shared
|  |--max_shared
|  |--num_agt_view
|  |--max_timesteps
|  |--num_scenarios
|  |--max_dist_agt
|  |--delta_theta_range
|  |--delta_dist_range
|  |--seed
```

# Generating Scenario SVGs

To convert scenarios into the SVG format, run

```
python scenario_to_svg.py --input_file scenarios.json --output_file scenario_svgs.json
```

By default, this will only generate SVGs based on each agent's perspective. To add the entire world SVGs, use `--add_world_svgs`.

Generated scenario SVGs should be in the following format:

```
scenario_svgs.json
|--scenario_id
|   |--agents
|     |--[i]
|       |--static_svgs
|       |--animation_svgs
|       |--reverse_animation_svgs

[optional]

|   |--world
|     |--static_svgs
|     |--animation_svgs
|     |--reverse_animation_svgs
```

To visualize the scenarios in HTML, run

```
python scenario_to_svg.py --in_html
```

This will generate simple HTML files for previewing scenario SVGs. You can check some examples in the [data/html](https://github.com/Alab-NII/dynamic-onecommon/tree/master/scenarios/data/html) directory.

# Pregenerated Scenarios and Scenario SVGs

Pregenerated scenarios used in our experiments are available in [data/pregenerated](https://github.com/Alab-NII/dynamic-onecommon/tree/master/scenarios/data/pregenerated). Please unzip and copy them to the other directories:

```
unzip data/pregenerated/\*.zip -d data/pregenerated
cp data/pregenerated/*.json ../webapp/src/data
cp data/pregenerated/*.json ../data/scenarios
cp data/pregenerated/*.json ../experiments/onecommon/data/dynamic-onecommon
cp data/pregenerated/*.json ../experiments/dynamic-onecommon/data/dynamic-onecommon
```

All scenarios can be reproduced using the default world parameters with the random seeds of 0 & 1 for `scenarios.json`, 2 & 3 for `scenarios_2.json`, 4 & 5 for `scenarios_3.json` and 6 & 7 for `scenarios_4.json`.

(`scenarios.json`, `scenarios_2.json` and `scenarios_3.json` are used for dataset collection and `scenarios_4.json` for selfplay & human evaluation.)

