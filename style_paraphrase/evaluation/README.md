## Style Transfer Evaluation

### Accuracy

We use RoBERTa-large classifiers to check style transfer accuracy. Check the pretrained models in this [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place them under `accuracy`. Consider using [`gdown`](https://github.com/wkentaro/gdown) for downloading large files easily. Your final folder structure should look like (depending on the datasets you are interested in),

* `accuracy/shakespeare_classifier`
* `accuracy/formality_classifier`
* `accuracy/cds_classifier`

### Similarity

We use the SIM model from Wieting et al. 2019 ([paper](https://www.aclweb.org/anthology/P19-1427/)) for our evaluation. The code for similarity can be found under `similarity`. Make sure to download the `sim` model from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `similarity/sim`.

### Fluency

We use a RoBERTa-large classifier trained on the [CoLA corpus](https://nyu-mll.github.io/CoLA) to evaluate fluency of generations. Make sure to download the `cola_classifier` model from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `fluency/cola_classifier`.

### Running Evaluation

For Shakespeare evaluation from the root folder `style-transfer-paraphrase` run,

```
style_paraphrase/evaluation/scripts/evaluate_shakespeare.sh shakespeare_models/model_300 shakespeare_models/model_299 paraphrase_gpt2_large
```

For Formality evaluation from the root folder `style-transfer-paraphrase` run,

```
style_paraphrase/evaluation/scripts/evaluate_shakespeare.sh formality_models/model_314 formality_models/model_313 paraphrase_gpt2_large
```

### Running Evaluation on Conditional Models (-Multi PP. ablation in Section 5)

1. Make sure to install the local fork of `transformers` provided in this repository ([link](https://github.com/martiansideofthemoon/style-transfer-paraphrase/tree/master/transformers)), since it contains some modifications necessary to run this script.

2. You will need to edit `get_logits` to `get_logits_old` [here](https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/62e953d833f7d75c826b59d5ab5bf7f2b689ba45/style_paraphrase/utils.py#L281).

3. Download `model_305` from the Shakespeare folder of the Google Drive, and `model_315` from the Formality folder. Run the following commands,

```
style_paraphrase/evaluation/scripts/evaluate_shakespeare.sh shakespeare_models/model_305 paraphrase_gpt2_large
style_paraphrase/evaluation/scripts/evaluate_shakespeare.sh formality_models/model_315 paraphrase_gpt2_large
```

### Human Evaluation

We used Amazon Mechanical Turk for our evaluation. Please check the [`human/paraphrase_amt_template.html`](human/paraphrase_amt_template.html) and the attached screenshots (`human/crowdsourcing*.png`) for details on setting up the Mechanical Turk jobs.
