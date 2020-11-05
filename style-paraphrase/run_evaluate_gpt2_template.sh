#!/bin/sh
#SBATCH --job-name=eval_gpt2_{job_id}
#SBATCH -o style_paraphrase/logs/log_eval_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=titanx-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

export DATA_DIR={dataset}

BASE_DIR=style_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=gpt2 \
    --model_name_or_path={model_name} \
    --data_dir=$DATA_DIR \
    --do_eval \
    --do_delete_old \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 3 \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --limit_examples 1000 \
    --job_id {job_id} \
    --learning_rate {learning_rate} \
    --prefix_input_type {prefix_input_type} \
    --global_dense_feature_list {global_dense_feature_list} \
    --specific_style_train {specific_style_train} \
    --eval_frequency_min 30
