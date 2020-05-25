#!/bin/sh
#SBATCH --job-name=generate_gpt2_{job_id}
#SBATCH -o gpt2_finetune/logs/log_generate_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=15
#SBATCH --mem=300GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

GPT2_MODEL_DIR=gpt2_finetune/saved_models/model_{job_id}
DATA_DIR={dataset}
BEAM_SIZE={beam_size}
TOP_P=0.0

declare -a arr=("reconstruction" "class_fixed_0_context_srl_input" "class_fixed_1_context_srl_input")

for gtype in "${arr[@]}"
do
    printf "%0.s-" {1..40}
    printf "\n"
    printf "Running $gtype experiment"
    printf "\n"
    printf "%0.s-" {1..40}
    printf "\n"

    if [ "$gtype" == "reconstruction" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=variable_{context_input_type}

    elif [ "$gtype" == "fixed_roberta" ] ; then
        ROBERTA_INPUT_TYPE=fixed_{roberta_input_type}
        CONTEXT_INPUT_TYPE=variable_{context_input_type}

    elif [ "$gtype" == "fixed_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=fixed_{context_input_type}

    elif [ "$gtype" == "mixed_roberta_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=mixed_{roberta_input_type}
        CONTEXT_INPUT_TYPE=variable_{context_input_type}

    elif [ "$gtype" == "class_fixed_0_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_0_{context_input_type}

    elif [ "$gtype" == "class_fixed_1_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_1_{context_input_type}

    elif [ "$gtype" == "class_fixed_2_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_2_{context_input_type}

    elif [ "$gtype" == "class_fixed_3_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_3_{context_input_type}

    elif [ "$gtype" == "class_fixed_4_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_4_{context_input_type}

    elif [ "$gtype" == "class_fixed_5_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_5_{context_input_type}

    elif [ "$gtype" == "class_fixed_6_context_srl_input" ] ; then
        ROBERTA_INPUT_TYPE=variable_{roberta_input_type}
        CONTEXT_INPUT_TYPE=class_fixed_6_{context_input_type}

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

    # kernprof -l gpt2_finetune/run_generation.py \
    # python gpt2_finetune/run_generation.py \
    python -m torch.distributed.launch --nproc_per_node=8 gpt2_finetune/{generation_module}.py \
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
        --roberta_ckpt_file {roberta_ckpt_file} \
        --extra_embedding_dim {extra_embedding_dim} \
        --context_type {context_type} \
        --roberta_layer {roberta_layer} \
        --roberta_weights {roberta_weights} \
        --job_id {job_id} \
        --context_input_type $CONTEXT_INPUT_TYPE \
        --roberta_input_type $ROBERTA_INPUT_TYPE \
        --limit_examples 300 \
        --fixed_example_number 10 \
        --global_dense_feature_list {global_dense_feature_list} \
        --stop_token {stop_token} \
        --specific_author_train {specific_author_train}

    killall python

    # Sort the files and concatenate them
    cat $(ls $GENERATION_OUTPUT_DIR/generated_?.txt | sort) > $GENERATION_OUTPUT_DIR/generated_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/reference_?.txt | sort) > $GENERATION_OUTPUT_DIR/reference_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/context_?.txt | sort) > $GENERATION_OUTPUT_DIR/context_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/roberta_?.txt | sort) > $GENERATION_OUTPUT_DIR/roberta_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/context_author_targets_?.txt | sort) > $GENERATION_OUTPUT_DIR/context_author_targets_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/roberta_author_targets_?.txt | sort) > $GENERATION_OUTPUT_DIR/roberta_author_targets_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/original_author_targets_?.txt | sort) > $GENERATION_OUTPUT_DIR/original_author_targets_all.txt
    cat $(ls $GENERATION_OUTPUT_DIR/metadata_?.txt | sort) > $GENERATION_OUTPUT_DIR/metadata_all.txt

    # TODO = Remove duplicate lines
    # awk '!seen[$0]++' $GENERATION_OUTPUT_DIR/generated_all.txt > $GENERATION_OUTPUT_DIR/final/generated_unique.txt
    # awk '!seen[$0]++' $GENERATION_OUTPUT_DIR/reference_all.txt > $GENERATION_OUTPUT_DIR/final/reference_unique.txt

    cp $GENERATION_OUTPUT_DIR/generated_all.txt $GENERATION_OUTPUT_DIR/final/generated_unique.txt
    cp $GENERATION_OUTPUT_DIR/reference_all.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt
    cp $GENERATION_OUTPUT_DIR/context_all.txt $GENERATION_OUTPUT_DIR/final/context_unique.txt
    cp $GENERATION_OUTPUT_DIR/roberta_all.txt $GENERATION_OUTPUT_DIR/final/roberta_unique.txt
    cp $GENERATION_OUTPUT_DIR/context_author_targets_all.txt $GENERATION_OUTPUT_DIR/final/context_author_targets_unique.txt
    cp $GENERATION_OUTPUT_DIR/roberta_author_targets_all.txt $GENERATION_OUTPUT_DIR/final/roberta_author_targets_unique.txt
    cp $GENERATION_OUTPUT_DIR/original_author_targets_all.txt $GENERATION_OUTPUT_DIR/final/original_author_targets_unique.txt
    cp $GENERATION_OUTPUT_DIR/metadata_all.txt $GENERATION_OUTPUT_DIR/final/metadata_unique.txt

    MOSESDECODER="mosesdecoder" gpt2_finetune/scripts/multi-bleu-wrapper.sh $GENERATION_OUTPUT_DIR/final/generated_unique.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/multibleu.txt

    MOSESDECODER="mosesdecoder" gpt2_finetune/scripts/sacrebleu-wrapper.sh $GENERATION_OUTPUT_DIR/final/generated_unique.txt en en $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/sacrebleu.txt

    rm $GENERATION_OUTPUT_DIR/final/generated_unique.txt.*
    rm $GENERATION_OUTPUT_DIR/final/reference_unique.txt.*

    python gpt2_finetune/scripts/evaluate_em_f1.py $GENERATION_OUTPUT_DIR/final/generated_unique.txt $GENERATION_OUTPUT_DIR/final/reference_unique.txt 2>&1 | tee $GENERATION_OUTPUT_DIR/results/f1_em.txt

    python preprocess/data_filter_generated.py --data_dir $GENERATION_OUTPUT_DIR/final
    # Classify the generated text using a pretrained classifier and compare the labels to the expected labels
    python gpt2_finetune/scripts/classify_generations.py --data_dir $GENERATION_OUTPUT_DIR/final --output_file $GENERATION_OUTPUT_DIR/results/classifier.txt --original_data_dir $DATA_DIR

    # Calculate paraphrase similarity
    python scripts/get_paraphrase_similarity.py --generated_path $GENERATION_OUTPUT_DIR/final/generated_unique.txt --reference_strs original,paraphrase --reference_paths $GENERATION_OUTPUT_DIR/final/reference_unique.txt,$GENERATION_OUTPUT_DIR/final/context_unique.txt

    # Measure the % of word copying from the roberta_sentence as well as the GPT2 context, also output ANSI colored sequences
    python gpt2_finetune/scripts/evaluate_text_overlap.py $GENERATION_OUTPUT_DIR/final $GENERATION_OUTPUT_DIR/results/overlap | tee $GENERATION_OUTPUT_DIR/results/overlap_results.txt

done