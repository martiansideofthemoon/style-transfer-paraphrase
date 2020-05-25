#!/bin/sh
#SBATCH --job-name=eval_gpt2_{job_id}
#SBATCH -o /gpt2_finetune/logs/log_eval_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=300GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

export DATA_DIR={dataset}

BASE_DIR=gpt2_finetune

python -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/{module}.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=gpt2 \
    --model_name_or_path={model_name} \
    --data_dir=$DATA_DIR \
    --do_eval \
    --do_delete_old \
    --roberta_pretrained={roberta_dir} \
    --content_aggregation={content_aggregation} \
    --content_aggregation_type={content_aggregation_type} \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 5 \
    --eval_all_checkpoints \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --roberta_ckpt_file {roberta_ckpt_file} \
    --extra_embedding_dim {extra_embedding_dim} \
    --context_type {context_type} \
    --roberta_layer {roberta_layer} \
    --roberta_weights {roberta_weights} \
    --limit_examples 100 \
    --job_id {job_id} \
    --switch_type {switch_type} \
    --learning_rate {learning_rate} \
    --context_input_type variable_{context_input_type} \
    --roberta_input_type variable_{roberta_input_type} \
    --global_dense_feature_list {global_dense_feature_list} \
    --specific_author_train {specific_author_train}
