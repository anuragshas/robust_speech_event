python run_speech_recognition_ctc.py \
--dataset_name="mozilla-foundation/common_voice_7_0" \
--model_name_or_path="facebook/wav2vec2-xls-r-1b" \
--dataset_config_name="hi" \
--output_dir="./" \
--overwrite_output_dir \
--num_train_epochs="100" \
--per_device_train_batch_size="32" \
--per_device_eval_batch_size="16" \
--gradient_accumulation_steps="1" \
--learning_rate="7.5e-5" \
--lr_scheduler_type="linear" \
--warmup_steps="2000" \
--length_column_name="input_length" \
--evaluation_strategy="steps" \
--text_column_name="sentence" \
--chars_to_ignore , ? . ! \- \; \: \" “ % ‘ ” � — ’ … – । \| \& \
--save_steps="400" \
--eval_steps="400" \
--logging_steps="100" \
--layerdrop="0.0" \
--activation_dropout="0.1" \
--save_total_limit="1" \
--freeze_feature_encoder \
--feat_proj_dropout="0.0" \
--mask_time_prob="0.75" \
--mask_time_length="10" \
--mask_feature_prob="0.25" \
--mask_feature_length="64" \
--seed="42" \
--gradient_checkpointing \
--use_auth_token \
--fp16 \
--group_by_length \
--do_train --do_eval \
--push_to_hub