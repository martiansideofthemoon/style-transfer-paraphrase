# Reformulating Unsupervised Style Transfer as Paraphrase Generation (EMNLP 2020)

This is the official repository accompanying the EMNLP 2020 long paper [Reformulating Unsupervised Style Transfer as Paraphrase Generation](https://arxiv.org/abs/2010.05700). This repository contains the accompanying dataset and codebase.

This repository is a work-in-progress, but we have released our human evaluation templates, the filtered paraNMT dataset, a pretrained model for our diverse paraphrase generation system, along with a command-line demo to play with it! For more details, check [`README_DEMO.md`](README_DEMO.md).

## Dataset

All datasets will be added to this [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing). For now, you can download and place the `paranmt_filtered` dataset as `datasets/paranmt_filtered`.

## Training

To train the paraphrase model, run [`gpt2_finetune/examples/run_finetune_paraphrase.sh`](gpt2_finetune/examples/run_finetune_paraphrase.sh).

## Evaluation

Please check [`evaluation/README.md`](evaluation/README.md) for details on our human evaluation setup.

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
