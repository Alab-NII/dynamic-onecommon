# Setup

Install and activate [poetry](https://github.com/python-poetry/poetry).

```
poetry install
poetry shell
```

Add `dynamic-onecommon/webapp` and `dynamic-onecommon/webapp/src` to your PYTHONPATH, e.g.

```
export PYTHONPATH=$PYTHONPATH:~/dynamic-onecommon/webapp/src:~/dynamic-onecommon/webapp
```

To run the web server, move to the [src](https://github.com/Alab-NII/dynamic-onecommon/tree/master/webapp/src) directory and run `run_sample.sh`. Add `--reuse` option if you do not want to initialize or overwrite existing data.

By default, some URLs are password protected (username and password are both *sample*).

# Dataset Visualization

All human-human dialogues collected in our paper can be seen from the URL: <http://localhost:5000/sample/dataset>. You can also visualize the selfplay dialogues generated by the models (c.f. [experiments](https://github.com/Alab-NII/dynamic-onecommon/tree/master/experiments)) from <http://localhost:5000/sample/selfplay>.

# Dataset Collection

By default, <http://localhost:5000/sample> is used for dataset collection. When more than two people are connected to this URL, we create pairs to start playing the dialogue task.

After collecting the dialogues, you can use <http://localhost:5000/sample/admin> to view the collected dialogues and decide whether to accept or reject the dialogues.

Collected dialogues should be in the following format:

```
final_transcripts.json
|--chat_id
|  |--scenario_id
|  |--agents_info
|    |--[i]
|      |--agent_id
|      |--agent_type
|  |--outcome
|  |--events
|  |--time
|      |--start_time
|      |--duration
```

We also manually conducted the annotation of target selection and spatio-temporal expressions (c.f. [data](https://github.com/Alab-NII/dynamic-onecommon/tree/master/data)). To this end, we used the interface provided in <http://localhost:5000/sample/annotation?annotator_id=admin>.

# Credit

The overall framework is built upon StanfordNLP's [CoCoA framework](https://github.com/stanfordnlp/cocoa).
