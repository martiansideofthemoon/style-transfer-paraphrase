export DATA_DIR=formality/formality_prior_detokenize
rm -rf gpt2_finetune/saved_models/test_inverse_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 gpt2_finetune/run_lm_finetuning_dynamic.py \
    --output_dir=gpt2_finetune/saved_models/test_dynamic_roberta \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-medium \
    --data_dir=$DATA_DIR \
    --do_eval \
    --roberta_pretrained=$ROBERTA_BASE \
    --content_aggregation=1000 \
    --content_aggregation_type=bow \
    --extra_embedding_dim=768 \
    --save_steps 1000 \
    --logging_steps 1 \
    --save_total_limit 5 \
    --evaluate_during_training \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 8 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 10 \
    --roberta_ckpt_file model.pt \
    --context_type content \
    --extra_embedding_dim 768 \
    --roberta_layer 10 \
    --roberta_weights fixed \
    --do_train \
    --switch_type constant \
    --learning_rate 5e-5 \
    --context_input_type "variable_roberta_input" \
    --roberta_input_type "variable_paraphrase_250" \
    --context_noise "none" \
    --global_dense_feature_list "none" \
    --eval_all_checkpoints \
    --limit_examples 300 \
    --specific_author_train 0 \
    --optimizer adam
