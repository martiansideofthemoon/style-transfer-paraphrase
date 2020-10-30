# Reformulating Unsupervised Style Transfer as Paraphrase Generation (EMNLP 2020)

This is the official repository accompanying the EMNLP 2020 long paper [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). This repository contains the accompanying dataset and codebase.

This repository is a work-in-progress, but we have released our human evaluation templates, the filtered paraNMT dataset, a pretrained model for our diverse paraphrase generation system, along with a command-line demo to play with it! For more details, check [`README_DEMO.md`](README_DEMO.md).

## Dataset

All datasets will be added to this [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing). Download the datasets and place them under `datasets`. The datasets currently available are (with their folder names),

1. ParaNMT-50M filtered down to 75k pairs - `paranmt_filtered`
2. Shakespeare style transfer - `shakespeare`
3. Formality transfer - Please follow the instructions [here](https://github.com/raosudha89/GYAFC-corpus). Once you have access to the corpus, you could email me ([kalpesh@cs.umass.edu](mailto:kalpesh@cs.umass.edu)) to get access to the preprocessed version. We will also add scripts to preprocess the raw data.

## Training

To train the paraphrase model, run [`gpt2_finetune/examples/run_finetune_paraphrase.sh`](gpt2_finetune/examples/run_finetune_paraphrase.sh). To train the inverse paraphrasers for Shakespeare, check the two scripts in [`gpt2_finetune/examples/shakespeare`](gpt2_finetune/examples/shakespeare).

## Evaluation

Please check [`evaluation/README.md`](evaluation/README.md) for more details.

## Outputs from our model

coming soon!

## Citation

If you find this code or dataset useful, please cite us:

```
@inproceedings{style20,
author={Kalpesh Krishna and John Wieting and Mohit Iyyer},
Booktitle = {Empirical Methods in Natural Language Processing},
Year = "2020",
Title={Reformulating Unsupervised Style Transfer as Paraphrase Generation},
}
```
