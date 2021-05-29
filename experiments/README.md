# Setup

Please unzip pre-generated files or place the preprocessed files (c.f. [data](https://github.com/Alab-NII/dynamic-onecommon/tree/master/data)) in the `onecommon/data/dynamic-onecommon` and `dynamic-onecommon/data/dynamic-onecommon` directories:

```
unzip onecommon/data/dynamic-onecommon/\*.zip -d onecommon/data/dynamic-onecommon
unzip dynamic-onecommon/data/dynamic-onecommon/\*.zip -d dynamic-onecommon/data/dynamic-onecommon
```

# Model Pretraining on OneCommon

In the `onecommon` directory, you can pretrain the model on OneCommon Corpus by running `run_rnn_ref_model_for_multitask.sh`. This will train the TSEL-REF-DIAL model (from the [onecommon](https://github.com/Alab-NII/onecommon) repository) with the shared word embeddings for OneCommon and Dynamic-OneCommon.

To test the model on OneCommon, you can run

```
python test_reference.py --model_file rnn_ref_model_for_multitask --cuda
```

To use the trained models for pretraining, you need to copy them to the `dynamic-onecommon/saved_models` directory:

```
cp saved_models/rnn_ref_model_for_multitask_*.th ../dynamic-onecommon/saved_models
```

# Model Training/Testing on Dynamic-OneCommon

In the `dynamic-onecommon` directory, you can train the baseline models described in the paper.

To be specific, you can use the following scripts to train each model:

* `--run_rnn_dial_model_pretrained.sh`: train the baseline model with the module parameters initialized based TSEL-REF-DIAL.
* `--run_rnn_dial_model.sh`: ablation of pretraining (parameter initialization based on TSEL-REF-DIAL).
* `--run_rnn_dial_model_pretrained_abl_color.sh`: ablation of color features.
* `--run_rnn_dial_model_pretrained_abl_size.sh`: ablation of size features.
* `--run_rnn_dial_model_pretrained_abl_location.sh`: ablation of locational features.
* `--run_rnn_dial_model_pretrained_abl_previous_selected.sh`: ablation of previous target.
* `--run_rnn_dial_model_pretrained_abl_dynamics.sh`: ablation of dynamics information.

Here are some examples of the available arguments:

* `--from_pretrained`: initialize modules from the pretrained model (TSEL-REF-DIAL).
* `--abl_features`: ablate spatial/temporal/meta features.
* `--share_attn`: share attention modules for language generation and target selection.
* `--repeat_train`: repeat training 5 times with different random seeds.

After the model training, you can evaluate its performance on language modelling and target selection based on the heldout test set, e.g.:

```
python test.py --model_file rnn_dial_model_pretrained --cuda
```

You can also evaluate its performance based on the selfplay dialogue task, e.g.:

```
python selfplay.py --alice_model_file rnn_dial_model_pretrained --bob_model_file rnn_dial_model_pretrained --temperature 0.25 --cuda
```

The generated transcripts from selfplay dialogue will be in the following format. You can visualize them using our [webapp](https://github.com/Alab-NII/dynamic-onecommon/tree/master/webapp) directory.

```
selfplay_transcripts.json
|--scenario_id
|  |--agents_info
|    |--[i]
|  |--outcome
|  |--events
|    |--[i]
|      |--action
|      |--agent (0,1)
|      |--data
|      |--turn
```

After major refactoring (and small bug fixes), the results may be slightly different but comparable to the original paper.

# Additional Functionalities

We also provide the code for training hierarchical recurrent encoder-decoder (HRED) models ([paper](https://arxiv.org/abs/1507.04808)). Based on our current implementation, the results are similar but the flat encoder-decoder models (our original baselines) generally perform better.

It is also straightforward to pretrain the models on Dynamic-OneCommon and fine-tune them on OneCommon. You simply need to train the Dynamic-OneCommon models with the `--for_multitask` argument, save the trained models to `onecommon/saved_models` directory, and use `--from_pretrained` when fine-tuning models on OneCommon.
