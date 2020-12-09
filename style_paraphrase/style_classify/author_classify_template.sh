#!/bin/sh
#SBATCH --job-name=style_classify_{job_id}
#SBATCH -o style_paraphrase/style_classify/logs/log_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=45GB
#SBATCH -d singleton
#SBATCH --exclude=node172,node181

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

TOTAL_NUM_UPDATES={total_updates}
WARMUP_UPDATES={warmup}
LR={learning_rate}
NUM_CLASSES={num_classes}
MAX_SENTENCES={max_sentences}
ROBERTA_MODEL={roberta_model}
MAX_POSITIONS={max_positions}


if [ "$ROBERTA_MODEL" = "LARGE" ]; then
   ROBERTA_PATH=$ROBERTA_LARGE/model.pt
   ROBERTA_ARCH=roberta_large
else
   ROBERTA_PATH=$ROBERTA_BASE/model.pt
   ROBERTA_ARCH=roberta_base
fi

python fairseq/train.py {base_dataset}/{dataset}-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions $MAX_POSITIONS \
    --no-epoch-checkpoints \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch $ROBERTA_ARCH \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --max-epoch {num_epochs} \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters \
    --update-freq 4 \
    --save-dir style_paraphrase/style_classify/saved_models/save_{job_id} \
    --log-interval 100