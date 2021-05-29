# Dataset Analysis

Basic functionalities for analyzing the dataset is available by running

```
python simple_analysis.py
```

with the following arguments:

* `--basic_statistics`: compute basic statistics of the dataset
* `--plot_success`: compute and plot success rates at each turn
* `--count_scenarios`: compute the number of unique scenarios
* `--plot_selection`: compute and plot the selection bias
* `--vocab_analysis`: conduct lexical analysis (with onecommon)
* `--utterance_analysis`: conduct utterance-level analysis (with onecommon)
* `--nuance_analysis`: conduct nuanced expressions analysis (with onecommon)
* `--grounding_analysis`: conduct further analyses of grounding success rates
* `--worker_analysis`: analyze success rates at the worker-level
* `--anonymize_workers`: hash original worker IDs for privacy
* `--output_text`: output dialogue data in text format


# Manual Annotation

To conduct manual annotation of **target selection** and **spatio-temporal expressions**, first run

```
python annotation_analysis.py --output_raw
```

to output the raw (unannotated) files.

Next, conduct manual annotation using our `annotation_guideline.md` and [webapp](https://github.com/Alab-NII/dynamic-onecommon/tree/master/webapp).

The raw/annotated files should be structured like this:

```
target_selection_annotation.json
|--chat_id
|  |--agent_id

spatio_temporal_annotation.json
|--chat_id
|  |--utterance_id
|    |--previous
|    |--movement
|    |--current
|    |--none
```

Finally, place the annotated files in the `annotated` directory. You can check the annotation results by

```
python annotation_analysis.py --annotation_statistics
python annotation_analysis.py --annotation_agreement
```


# Model Format

To preprocess the dataset for model training/evaluation, run

```
python transform_to_model_format.py
```

You can add `--for_repeat` to split the data for repeated experiments, `--for_selfplay` for selfplay evaluation, and `--for_onecommon` to convert to the OneCommon format (c.f. [onecommon](https://github.com/Alab-NII/onecommon)). For instance, the output files should be in the following format:

```
train.json, valid.json, test.json
|--chat_id
|  |--scenario_id
|  |--agents
|    |--[i] (agent_id)
|      |--[j] (turn_id)
|        |--context
|          |--entity_ids
|            |--xs
|              |--[k]
|            |--ys
|              |--[k]
|            |--visible
|              |--[k]
|            |--color
|            |--size
|            |--selectable
|            |--previous_selectable
|            |--previous_selected
|        |--utterances
|        |--selection

selfplay.json
|--scenario_id
|  |--agents
|    |--[i] (agent_id)
|      |--[j] (turn_id)
|        |--context
|          |--entity_ids
|            |--xs
|              |--[k]
|            |--ys
|              |--[k]
|            |--visible
|              |--[k]
|            |--color
|            |--size
|            |--selectable
|            |--previous_selectable
|            |--previous_selected
```