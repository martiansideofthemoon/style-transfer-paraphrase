#!/usr/bin/env bash

rm -rf gpt2_finetune/saved_models/test_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 gpt2_finetune/run_lm_finetuning_dynamic.py \
    --output_dir=gpt2_finetune/saved_models/test_dynamic_paraphrase \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --data_dir=datasets/paranmt_filtered \
    --do_eval \
    --extra_embedding_dim=768 \
    --save_steps 1000 \
    --logging_steps 1 \
    --save_total_limit 5 \
    --evaluate_during_training \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 10 \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 10 \
    --roberta_ckpt_file model.pt \
    --extra_embedding_dim 20 \
    --do_train \
    --switch_type constant \
    --learning_rate 5e-5 \
    --prefix_input_type "original" \
    --context_noise "none" \
    --global_dense_feature_list "none" \
    --eval_all_checkpoints
