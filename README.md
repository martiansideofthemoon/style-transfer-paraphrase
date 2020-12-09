# Reformulating Unsupervised Style Transfer as Paraphrase Generation (EMNLP 2020)

This is the official repository accompanying the EMNLP 2020 long paper [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). This repository contains the accompanying dataset and codebase.

This repository is a work-in-progress, but we have released several models, evaluation scripts, training code and demos.

## Demos

The web demo for the system can be found [here](http://arkham.cs.umass.edu:8553). The code and setup for the webpage can be found in [`web-demo/README.md`](web-demo/README.md). We also have a command-line demo for the paraphrase model. For more details, check [`README_terminal_demo.md`](README_terminal_demo.md).

## Setup

The code uses PyTorch 1.4+, HuggingFace's [`transformers`](https://github.com/huggingface/transformers) library for training GPT2 models, and Facebook AI Research's [`fairseq`](https://github.com/facebookresearch/fairseq) for evaluation using RoBERTa classifiers. To install PyTorch, look for the Python package compatible with your local CUDA setup [here](https://pytorch.org).

```
virtualenv style-venv
source style-venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
pip install --editable .

cd fairseq
pip install --editable .
```

## Datasets

All datasets will be added to this [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing). Download the datasets and place them under `datasets`. The datasets currently available are (with their folder names),

1. ParaNMT-50M filtered down to 75k pairs - `datasets/paranmt_filtered`
2. Shakespeare style transfer - `datasets/shakespeare`
3. Formality transfer - Please follow the instructions [here](https://github.com/raosudha89/GYAFC-corpus). Once you have access to the corpus, you could email me ([kalpesh@cs.umass.edu](mailto:kalpesh@cs.umass.edu)) to get access to the preprocessed version. We will also add scripts to preprocess the raw data.
4. Corpus of Diverse Styles - `datasets/cds`. Samples can be found in [`samples/data_samples`](samples/data_samples). Please cite the [original sources](https://arxiv.org/pdf/2010.05700.pdf#page=24) as well if you plan to use this dataset.

## Training

1. To train the paraphrase model, run [`style_paraphrase/examples/run_finetune_paraphrase.sh`](style_paraphrase/examples/run_finetune_paraphrase.sh).

2. To train the inverse paraphrasers for Shakespeare, check the two scripts in [`style_paraphrase/examples/shakespeare`](style_paraphrase/examples/shakespeare).

3. To train the inverse paraphrasers for Formality, check the two scripts in [`style_paraphrase/examples/formality`](style_paraphrase/examples/formality). Note that you will need to email me asking for the preprocessed dataset once you have access to the GYAFC corpus (see instructions in Datasets section).

All the main pretrained models have been added to the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing).

**Instructions on hyperparameter grid search & automation coming soon!** In the meantime you can find the code in `style_paraphrase/schedule.py`.

## Classifier Training

Classifiers are needed to evaluate style transfer performance. To train the classifiers follow the steps:

1. Install the local fork of `fairseq`, as discussed above in "Setup".
2. Download the RoBERTa checkpoints from [here](https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models). Alternatively, you could follow the commands below. If you want a smaller model, you can also setup a ROBERTA_BASE variable using a similar process.

```
wget https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xzvf roberta.large.tar.gz

# Add the following to your .bashrc file, feel free to store the model elsewhere on the hard disk
export ROBERTA_LARGE=$PWD/roberta.large
```

3. For training classifiers on Shakespeare, CoLA or CDS datasets, download the `shakespeare-bin`, `cola-bin` or `cds-bin` folders from the Drive link [here](https://drive.google.com/drive/folders/1Y54r47VSXP0Bo1h2cHhTMOLCH-YsP112?usp=sharing) and place them under `datasets`. I can provide similar files for the Formality dataset once you have access to the original corpus.

4. To train the classifiers, see the examples in [`style_paraphrase/style_classify/examples`](style_paraphrase/style_classify/examples). You can also use a grid search (with a Slurm scheduler) by using the code in [`style_paraphrase/style_classify/schedule.py`](style_paraphrase/style_classify/schedule.py). We also have a light-weight Flask interface to plot performance with epochs which works well with the Slurm grid search automation, check [`style_paraphrase/style_classify/webapp/run.sh`](style_paraphrase/style_classify/webapp/run.sh).


## Evaluation

Please check [`style_paraphrase/evaluation/README.md`](style_paraphrase/evaluation/README.md) for more details.

## Outputs from STRAP

All outputs generated by our model in the Corpus of Diverse Styles: [`samples/outputs`](samples/outputs). Output from Shakespeare & Formality datasets are coming soon!

## Custom Datasets

**Instructions for training the model on custom data coming soon!**

## Citation

If you find this repository useful, please cite us:

```
@inproceedings{style20,
author={Kalpesh Krishna and John Wieting and Mohit Iyyer},
Booktitle = {Empirical Methods in Natural Language Processing},
Year = "2020",
Title={Reformulating Unsupervised Style Transfer as Paraphrase Generation},
}
```
