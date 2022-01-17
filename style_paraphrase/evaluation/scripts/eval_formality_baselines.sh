# export CUDA_VISIBLE_DEVICES=0

split="test"
path=$1/transfer_entire_${split}.txt

printf "\nRoBERTa ${split} classification\n\n"
python style_paraphrase/evaluation/scripts/roberta_classify.py --input_file $path --label_file $1/all_test_transfer_labels.txt --model_dir style_paraphrase/evaluation/accuracy/formality_classifier --model_data_dir style_paraphrase/evaluation/accuracy/formality_classifier/formality-data-bin

printf "\nRoBERTa acceptability classification\n\n"
python style_paraphrase/evaluation/scripts/acceptability.py --input_file $path

printf "\nParaphrase scores --- generated vs gold..\n\n"
python style_paraphrase/evaluation/scripts/get_paraphrase_similarity.py --generated_path ${path} --reference_strs ref0,ref1,ref2,ref3 --reference_paths datasets/formality/raw/${split}.ref0,datasets/formality/raw/${split}.ref1,datasets/formality/raw/${split}.ref2,datasets/formality/raw/${split}.ref3 --output_path ${1}/generated_vs_gold.txt --store_scores

printf "\nnormalized paraphrase score vs gold..\n\n"
python style_paraphrase/evaluation/scripts/micro_eval.py --classifier_file ${path}.roberta_labels --paraphrase_file ${path}.pp_scores --generated_file ${path} --acceptability_file ${path}.acceptability_labels
