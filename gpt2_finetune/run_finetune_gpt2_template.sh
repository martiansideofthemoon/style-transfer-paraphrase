#!/bin/sh
#SBATCH --job-name=finetune_gpt2_{job_id}
#SBATCH -o gpt2_finetune/logs/log_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition={gpu}-long
#SBATCH --gres=gpu:{ngpus}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

export DATA_DIR={dataset}

BASE_DIR=gpt2_finetune

# Snapshot code used for the run
mkdir -p $BASE_DIR/saved_models/model_{job_id}_code

cp $BASE_DIR/*.py $BASE_DIR/saved_models/model_{job_id}_code

echo $HOSTNAME

python -m torch.distributed.launch --nproc_per_node={ngpus} $BASE_DIR/{module}.py \
    --output_dir=$BASE_DIR/saved_models/model_{job_id} \
    --model_type=gpt2 \
    --model_name_or_path={model_name} \
    --do_train \
    --data_dir=$DATA_DIR \
    --do_eval \
    --roberta_pretrained={roberta_dir} \
    --content_aggregation={content_aggregation} \
    --content_aggregation_type={content_aggregation_type} \
    --save_steps {save_steps} \
    --logging_steps 20 \
    --save_total_limit {save_total_limit} \
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
    --job_id {job_id} \
    --switch_type {switch_type} \
    --learning_rate {learning_rate} \
    --generator_loss_constants {generator_loss_constants} \
    --roberta_input_type variable_{roberta_input_type} \
    --context_input_type variable_{context_input_type} \
    --context_noise {context_noise} \
    --global_dense_feature_list {global_dense_feature_list} \
    --specific_author_train {specific_author_train} \
    --optimizer {optimizer}
