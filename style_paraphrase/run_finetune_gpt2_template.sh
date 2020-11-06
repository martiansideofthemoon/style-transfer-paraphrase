#!/bin/sh
#SBATCH --job-name=finetune_gpt2_{job_id}
#SBATCH -o style_paraphrase/logs/log_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition={gpu}-long
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

export DATA_DIR={dataset}

BASE_DIR=style_paraphrase

# Snapshot code used for the run
mkdir -p $BASE_DIR/saved_models/model_{job_id}_code

cp $BASE_DIR/*.py $BASE_DIR/saved_models/model_{job_id}_code

echo $HOSTNAME

python -m torch.distributed.launch --nproc_per_node={ngpus} $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=gpt2 \
    --model_name_or_path={model_name} \
    --do_train \
    --data_dir=$DATA_DIR \
    --save_steps {save_steps} \
    --logging_steps 20 \
    --save_total_limit {save_total_limit} \
    --evaluate_during_training \
    --num_train_epochs {num_epochs} \
    --gradient_accumulation_steps {accumulation} \
    --per_gpu_train_batch_size {batch_size} \
    --job_id {job_id} \
    --learning_rate {learning_rate} \
    --prefix_input_type {prefix_input_type} \
    --global_dense_feature_list {global_dense_feature_list} \
    --specific_style_train {specific_style_train} \
    --optimizer {optimizer}
