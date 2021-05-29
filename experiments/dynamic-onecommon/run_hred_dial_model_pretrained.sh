poetry run python train.py \
	--ctx_encoder_type attn_encoder \
	--bsz 16 \
	--max_epoch 30 \
	--model_type hierarchical_model \
	--model_file hred_dial_model_pretrained \
	--pretrained_model_file hred_ref_model_for_multitask \
	--nembed_word 256 \
	--nembed_ctx 256 \
	--nhid_lang 256 \
	--nhid_attn 256 \
	--lang_weight 1.0 \
	--sel_weight 0.5 \
	--from_pretrained \
	--cuda \