#!/bin/sh
#SBATCH --job-name=eval_gpt2_0
#SBATCH -o style_paraphrase/logs/log_eval_0.txt
#SBATCH --time=167:00:00
#SBATCH --partition=1080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=300GB
#SBATCH -d singleton

# Experiment Details :- GPT2-large model for paraphrasing.
# Run Details :- accumulation = 2, batch_size = 5, beam_size = 1, cpus = 3, dataset = datasets/paranmt_filtered, eval_batch_size = 1, global_dense_feature_list = none, gpu = m40, learning_rate = 5e-5, memory = 50, model_name = gpt2-large, ngpus = 1, num_epochs = 3, optimizer = adam, prefix_input_type = original, save_steps = 500, save_total_limit = -1, specific_style_train = -1, stop_token = eos

export DATA_DIR=datasets/paranmt_filtered

BASE_DIR=style_paraphrase

python -m torch.distributed.launch --nproc_per_node=1 $BASE_DIR/run_lm_finetuning.py \
    --output_dir=$BASE_DIR/saved_models/test_paraphrase \
    --model_type=gpt2 \
    --model_name_or_path=gpt2-large \
    --data_dir=$DATA_DIR \
    --do_eval \
    --do_delete_old \
    --save_steps 1000 \
    --logging_steps 1000 \
    --save_total_limit 3 \
    --evaluate_during_training \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 2 \
    --per_gpu_train_batch_size 5 \
    --limit_examples 1000 \
    --job_id paraphraser_test \
    --learning_rate 5e-5 \
    --prefix_input_type original \
    --global_dense_feature_list none \
    --specific_style_train -1

