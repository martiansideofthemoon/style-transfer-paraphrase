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

### Human Evaluation

We used Amazon Mechanical Turk for our evaluation. Please check the [`human/paraphrase_amt_template.html`](human/paraphrase_amt_template.html) and the attached screenshots (`human/crowdsourcing*.png`) for details on setting up the Mechanical Turk jobs.