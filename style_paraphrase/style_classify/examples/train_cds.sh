#!/bin/sh
#SBATCH --job-name=style_classify_cds
#SBATCH -o style_paraphrase/style_classify/logs/log_cds.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=45GB
#SBATCH -d singleton
#SBATCH --exclude=node172,node181

# Experiment Details :- CDS style classification.
# Run Details :- base_dataset = datasets, dataset = cds, learning_rate = 1e-5, max_positions = 512, max_sentences = 4, num_classes = 11, num_epochs = 10, roberta_model = LARGE, size = 273372, total_updates = 85428, update_freq = 8, warmup = 5125

TOTAL_NUM_UPDATES=85428
WARMUP_UPDATES=5125
LR=1e-5
NUM_CLASSES=11
MAX_SENTENCES=4
ROBERTA_MODEL=LARGE
MAX_POSITIONS=512


if [ "$ROBERTA_MODEL" = "LARGE" ]; then
   ROBERTA_PATH=$ROBERTA_LARGE/model.pt
   ROBERTA_ARCH=roberta_large
else
   ROBERTA_PATH=$ROBERTA_BASE/model.pt
   ROBERTA_ARCH=roberta_base
fi

python fairseq/train.py datasets/cds-bin/ \
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
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters \
    --update-freq 4 \
    --save-dir style_paraphrase/style_classify/saved_models/save_cds \
    --log-interval 100
