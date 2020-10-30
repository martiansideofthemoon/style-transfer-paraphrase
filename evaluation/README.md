## Style Transfer Evaluation

### Accuracy

We use a RoBERTa-large classifier to check style transfer accuracy. Check the `shakespeare_classifier` model from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `accuracy/shakespeare_classifier`.

### Similarity

We use the SIM model from Wieting et al. 2019 ([paper](https://www.aclweb.org/anthology/P19-1427/)) for our evaluation. The code for similarity can be found under `similarity`. Make sure to download the `sim` model from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `similarity/sim`.

## Fluency

We use a RoBERTa-large classifier trained on the [CoLA corpus](https://nyu-mll.github.io/CoLA) to evaluate fluency of generations. Make sure to download the `cola_classifier` model from the [Google Drive link](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing) and place it as `fluency/cola_classifier`.

### Human Evaluation

We used Amazon Mechanical Turk for our evaluation. Please check the [`human/paraphrase_amt_template.html`](human/paraphrase_amt_template.html) and the attached screenshots (`human/crowdsourcing*.png`) for details on setting up the Mechanical Turk jobs.