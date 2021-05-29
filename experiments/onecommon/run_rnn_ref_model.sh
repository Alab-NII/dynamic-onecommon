poetry run python train_reference.py \
	--ctx_encoder_type attn_encoder \
	--bsz 16 \
	--max_epoch 30 \
	--model_type rnn_reference_model \
	--model_file rnn_ref_model \
	--nembed_word 256 \
	--nembed_ctx 256 \
	--nhid_lang 256 \
	--nhid_attn 256 \
	--lang_weight 1.0 \
	--sel_weight 0.03125 \
	--ref_weight 1.0 \
	--cuda \
	--share_attn \