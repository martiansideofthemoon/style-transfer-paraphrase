export DATA_DIR=paranmt/filter_and_kt_less_precision_less_lendiff_less_0.0_0.5_5_english_only
rm -rf gpt2_finetune/saved_models/test_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 gpt2_finetune/run_lm_finetuning_dynamic.py \
    --output_dir=gpt2_finetune/saved_models/test_dynamic_paraphrase \
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
    --num_train_epochs 2 \
    --gradient_accumulation_steps 10 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 10 \
    --roberta_ckpt_file model.pt \
    --context_type content \
    --extra_embedding_dim 20 \
    --roberta_layer 10 \
    --roberta_weights fixed \
    --do_train \
    --switch_type constant \
    --learning_rate 5e-5 \
    --context_input_type "variable_paraphrase" \
    --roberta_input_type "variable_nofilter" \
    --context_noise "none" \
    --global_dense_feature_list "none" \
    --eval_all_checkpoints \
    --limit_examples 300
