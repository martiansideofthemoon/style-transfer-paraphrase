# Appendix, Source Code and Data Samples

This is a part of the supplementary material accompanying our EMNLP 2020 submission ("<title>").

## Appendix

We provided a detailed appendix in `appendix.pdf` in the root folder of this paper. This appendix has important hyperparameter choices, our survey of evaluations used in prior work, more comparisons with prior work, details of our dataset collection process, style-wise results of our system on our proposed dataset and lots of generated outputs.

## Dataset Samples

We present 1000 sentences from each our of our eleven diverse styles in `data_samples`.

## Source Code

Note: **This source code is very preliminary and is for reference only**. This codebase was used to train all the GPT2 models in our paper (both paraphrase and inverse paraphrase models). While the codebase has configurations for many complex preliminary experiments (which did not work out), we've set the configurations to the models eventually presented in the paper.

Note that this codebase does not contain dataset preprocessing or postprocessing scripts which are necessary to run experiments in the paper. We plan to make a proper Github release of the source code, datasets, model checkpoints, model outputs and a live demo of our system after the acceptance of the paper to ensure the results are fully reproducible and extendable.

Our source code is located in `gpt2_finetune`. Additionally, we use a modified version of the Transformers library in `transformers`. The only modification made is to run the style code model ablation study, which is a minor modification to `transformers/transformers/modeling_gpt2.py`.

#### Installation

This codebase uses PyTorch 1.2 installed with CUDA 9.2. All libraries can be installed from `pip` except `transformers`, for which we've used a local fork (install it using `pip install --editable .`).

### Description of Files

These descriptions are for files inside our main source code folder `gpt2_finetune/`. The best starting points are `schedule.py` and `hyperparameters_config.py`.

1. `args.py` - List of argument flags used for finetuning GPT2 with default values.
2. `data_utils.py`, `dataset_config.py` - Methods and classes to configure dataset and decide data preprocessing steps. 
3. `hyperparameter_config.py` - The list of hyperparameters used for training our diverse paraphrase model as well as the inverse paraphrase models.
5. `run_generation_dynamic.py` - The less preferred method to generate text using our models which integrates with our `Dataset` classes in `style_dataset.py`. The recommended way is using the `GPT2Generator` API in `utils.py`. This script is called in `run_generation_gpt2_template.sh`.
6. `run_lm_finetuning_dynamic.py` - The primary script for finetuning models and evaluating models (in terms of perplexity). This script is called in `run_finetune_gpt2_template.sh` as well as `run_evaluate_gpt2_template.sh`.
7. `schedule.py` - Use the hyperparameters in `hyperparameter_config.py` to fill in bash scripts `run_*.sh` and schedule them on a SLURM cluster. Filled in examples of finetuning scripts are present in `examples/`
8. `style_dataset.py` - The Pytorch `Dataset` objects used for our paraphrase dataset and inverse paraphrase dataset.
9. `utils.py` - An important file containing the lowest level details of the training process and most importantly the `GPT2Generator` API which is used to perform generation. This API is very easy to use and loads any pretrained model. A user can easily specify the inputs as text in a list and the API will minibatch it and perform generation taking care of all the necessary preprocessing.

Finally, `saved_models`, `slurm-schedulers`, `logs`, `runs` are folders which contain experiment logs.
