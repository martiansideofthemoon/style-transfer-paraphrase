#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

declare -a arr=(0.0 0.6 0.9)

gtype="nucleus_paraphrase"
split="test"

for top_p in "${arr[@]}"
do
    printf "\n-------------------------------------------\n"
    printf "Mode $gtype  --- "
    printf "top-p ${top_p}, split ${split}"
    printf "\n-------------------------------------------\n\n"

    mkdir -p $1/eval_${gtype}_${top_p}

    path0=$1/eval_${gtype}_${top_p}/transfer_modern_${split}.txt
    path1=$1/eval_${gtype}_${top_p}/transfer_shakespeare_${split}.txt
    base_path0=$1/eval_${gtype}_${top_p}


    printf "\ntranslate shakespeare to modern\n"
    python style_paraphrase/evaluation/scripts/style_transfer.py \
        --style_transfer_model $1 \
        --input_file datasets/shakespeare/raw/${split}_0.txt \
        --output_file transfer_modern_${split}.txt \
        --generation_mode $gtype \
        --detokenize \
	--output_class 1 \
        --post_detokenize \
        --paraphrase_model $2 \
        --top_p ${top_p}

    printf "\ntranslate modern to shakespeare\n"
    python style_paraphrase/evaluation/scripts/style_transfer.py \
        --style_transfer_model $1 \
        --input_file datasets/shakespeare/raw/${split}_1.txt \
        --output_file transfer_shakespeare_${split}.txt \
        --generation_mode $gtype \
        --detokenize \
	--output_class 0 \
        --post_detokenize \
        --paraphrase_model $2 \
        --top_p ${top_p}

    cat $path0 $path1 > ${base_path0}/all_${split}_generated.txt
    cat datasets/shakespeare/raw/${split}_0.txt datasets/shakespeare/raw/${split}_1.txt > ${base_path0}/all_${split}_input.txt
    cat datasets/shakespeare/raw/${split}_1.txt datasets/shakespeare/raw/${split}_0.txt > ${base_path0}/all_${split}_gold.txt
    cat datasets/shakespeare/raw/${split}_0.attr datasets/shakespeare/raw/${split}_1.attr > ${base_path0}/all_${split}_labels.txt

    python style_paraphrase/evaluation/scripts/flip_labels.py \
        --file1 datasets/shakespeare/raw/${split}_0.attr \
        --file2 datasets/shakespeare/raw/${split}_1.attr \
        --output_file ${base_path0}/all_${split}_transfer_labels.txt

    printf "\nRoBERTa ${split} classification\n\n"
    python style_paraphrase/evaluation/scripts/roberta_classify.py --input_file ${base_path0}/all_${split}_generated.txt --label_file ${base_path0}/all_${split}_transfer_labels.txt --model_dir style_paraphrase/evaluation/accuracy/shakespeare_classifier --model_data_dir style_paraphrase/evaluation/accuracy/shakespeare_classifier/shakespeare-data-bin

    printf "\nRoBERTa acceptability classification\n\n"
    python style_paraphrase/evaluation/scripts/acceptability.py --input_file ${base_path0}/all_${split}_generated.txt

    printf "\nParaphrase scores --- generated vs inputs..\n\n"
    python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${base_path0}/all_${split}_generated.txt --reference_strs reference --reference_paths ${base_path0}/all_${split}_input.txt --output_path ${base_path0}/generated_vs_inputs.txt

    printf "\nParaphrase scores --- generated vs gold..\n\n"
    python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${base_path0}/all_${split}_generated.txt --reference_strs reference --reference_paths ${base_path0}/all_${split}_gold.txt --output_path ${base_path0}/generated_vs_gold.txt --store_scores

    printf "\n final normalized scores vs gold..\n\n"
    python style_paraphrase/evaluation/scripts/micro_eval.py --classifier_file ${base_path0}/all_${split}_generated.txt.roberta_labels --paraphrase_file ${base_path0}/all_${split}_generated.txt.pp_scores --generated_file ${base_path0}/all_${split}_generated.txt --acceptability_file ${base_path0}/all_${split}_generated.txt.acceptability_labels

done
