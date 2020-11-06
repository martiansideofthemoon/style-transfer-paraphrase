#!/bin/sh
#SBATCH --job-name=generate_gpt2_{job_id}
#SBATCH -o style_paraphrase/logs/log_generate_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=15
#SBATCH --mem=300GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

GPT2_MODEL_DIR=style_paraphrase/saved_models/model_{job_id}
DATA_DIR={dataset}
BEAM_SIZE={beam_size}
TOP_P=0.0

declare -a arr=("reconstruction" "class_fixed_0" "class_fixed_1")

for gtype in "${arr[@]}"
do
    printf "%0.s-" {1..40}
    printf "\n"
    printf "Running $gtype experiment"
    printf "\n"
    printf "%0.s-" {1..40}
    printf "\n"

    if [ "$gtype" == "reconstruction" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE="none"

    elif [ "$gtype" == "class_fixed_0" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_0

    elif [ "$gtype" == "class_fixed_1" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_1

    elif [ "$gtype" == "class_fixed_2" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_2

    elif [ "$gtype" == "class_fixed_3" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_3

    elif [ "$gtype" == "class_fixed_4" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_4

    elif [ "$gtype" == "class_fixed_5" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_5

    elif [ "$gtype" == "class_fixed_6" ] ; then
        PREFIX_INPUT_TYPE={prefix_input_type}
        TARGET_STYLE_OVERRIDE=class_fixed_6

    fi

    if (( $(echo "$TOP_P > 0.0" |bc -l) )); then
        GENERATION_OUTPUT_DIR=$GPT2_MODEL_DIR/generation_nucleus_${TOP_P}_$gtype
    else
        GENERATION_OUTPUT_DIR=$GPT2_MODEL_DIR/generation_beam_${BEAM_SIZE}_$gtype
    fi

    mkdir -p $GENERATION_OUTPUT_DIR
    mkdir -p $GENERATION_OUTPUT_DIR/final
    mkdir -p $GENERATION_OUTPUT_DIR/results
    mkdir -p $GENERATION_OUTPUT_DIR/html

    # kernprof -l style_paraphrase/run_generation.py \
    # python style_paraphrase/run_generation.py \
    python -m torch.distributed.launch --nproc_per_node=8 style_paraphrase/run_generation.py \
        --model_type=gpt2 \
        --data_dir=$DATA_DIR \
        --model_name_or_path=$GPT2_MODEL_DIR \
        --roberta_pretrained={roberta_dir} \
        --content_aggregation={content_aggregation} \
        --content_aggregation_type={content_aggregation_type} \
        --top_k=1 \
        --top_p=${TOP_P} \
        --temperature=0 \
        --beam_size=${BEAM_SIZE} \
        --generation_output_dir=$GENERATION_OUTPUT_DIR \
        --per_gpu_eval_batch_size {eval_batch_size} \
        --eval_split dev \
        --job_id {job_id} \
        --target_style_override $TARGET_STYLE_OVERRIDE \
        --prefix_input_type $PREFIX_INPUT_TYPE \
        --limit_examples 300 \
        --fixed_example_number 10 \
        --global_dense_feature_list {global_dense_feature_list} \
        --stop_token {stop_token} \
        --specific_style_train {specific_style_train}

    killall python

    # Sort the files and concatenate them
    cat $(ls $GENERATION_OUTPUT_DIR/generated_?.txt | sort) > $GENERATION_OUTPUT_DIR/generated_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/reference_?.txt | sort) > $GENERATION_OUTPUT_DIR/reference_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/context_?.txt | sort) > $GENERATION_OUTPUT_DIR/context_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/roberta_?.txt | sort) > $GENERATION_OUTPUT_DIR/roberta_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/context_suffix_styles_?.txt | sort) > $GENERATION_OUTPUT_DIR/context_suffix_styles_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/original_styles_?.txt | sort) > $GENERATION_OUTPUT_DIR/original_styles_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/metadata_?.txt | sort) > $GENERATION_OUTPUT_DIR/metadata_all.txt

    # TODO = Remove duplicate lines
    # awk '!seen[$0]++' $GENERATION_OUTPUT_DIR/generated_all.txt > $GENERATION_OUTPUT_DIR/final/generated_unique.txt
    # awk '!seen[$0]++' $GENERATION_OUTPUT_DIR/reference_all.txt > $GENERATION_OUTPUT_DIR/final/reference_unique.txt

    cp $GENERATION_OUTPUT_DIR/generated_all.txt $GENERATION_OUTPUT_DIR/final/generated_unique.txt
    cp $GENERATION_OUTPUT_DIR/reference_all.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt
    cp $GENERATION_OUTPUT_DIR/context_all.txt $GENERATION_OUTPUT_DIR/final/context_unique.txt
    cp $GENERATION_OUTPUT_DIR/roberta_all.txt $GENERATION_OUTPUT_DIR/final/roberta_unique.txt
    cp $GENERATION_OUTPUT_DIR/context_suffix_styles_all.txt $GENERATION_OUTPUT_DIR/final/context_suffix_styles_unique.txt
    cp $GENERATION_OUTPUT_DIR/original_styles_all.txt $GENERATION_OUTPUT_DIR/final/original_styles_unique.txt
    cp $GENERATION_OUTPUT_DIR/metadata_all.txt $GENERATION_OUTPUT_DIR/final/metadata_unique.txt

    MOSESDECODER="mosesdecoder" style_paraphrase/scripts/multi-bleu-wrapper.sh $GENERATION_OUTPUT_DIR/final/generated_unique.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/multibleu.txt

    MOSESDECODER="mosesdecoder" style_paraphrase/scripts/sacrebleu-wrapper.sh $GENERATION_OUTPUT_DIR/final/generated_unique.txt en en $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/sacrebleu.txt

    rm $GENERATION_OUTPUT_DIR/final/generated_unique.txt.*
    rm $GENERATION_OUTPUT_DIR/final/reference_unique.txt.*

    python style_paraphrase/scripts/evaluate_em_f1.py $GENERATION_OUTPUT_DIR/final/generated_unique.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/f1_em.txt

    python preprocess/data_filter_generated.py --data_dir $GENERATION_OUTPUT_DIR/final
    # Classify the generated text using a pretrained classifier and compare the labels to the expected labels
    python style_paraphrase/scripts/classify_generations.py --data_dir $GENERATION_OUTPUT_DIR/final --output_file $GENERATION_OUTPUT_DIR/results/classifier.txt --original_data_dir $DATA_DIR

    # Calculate paraphrase similarity
    python scripts/get_paraphrase_similarity.py --generated_path $GENERATION_OUTPUT_DIR/final/generated_unique.txt --reference_strs original,paraphrase --reference_paths $GENERATION_OUTPUT_DIR/final/reference_unique.txt,$GENERATION_OUTPUT_DIR/final/context_unique.txt

    # Measure the % of word copying from the roberta_sentence as well as the GPT2 context, also output ANSI colored sequences
    python style_paraphrase/scripts/evaluate_text_overlap.py $GENERATION_OUTPUT_DIR/final $GENERATION_OUTPUT_DIR/results/overlap | tee $GENERATION_OUTPUT_DIR/results/overlap_results.txt

done