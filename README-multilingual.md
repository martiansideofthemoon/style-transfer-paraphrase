We are releasing multilingual formality classifiers by fine-tuning large multilingual language models on English GYAFC, to facilitate zero-shot cross-lingual transfer. We evaluated these classifiers on [XFORMAL](https://arxiv.org/abs/2104.04108).

For each language, we **lower-case sentences and remove trailing punctuation** to stop the model from latching onto easy indicators of formality.

Our results seem to indicate that XLM MADX is the best model, followed by XLM.

MAD-X and XLM-R base classifier checkpoints --- [link](https://drive.google.com/drive/folders/1EUYKeFslhSb_po6jwb7Pqkny5_zNsct6?usp=sharing)

The MAD-X adapter is also available on AdapterHub [here](https://adapterhub.ml/adapters/martiansideofthemoon/xlm-roberta-base_formality_classify_gyafc_pfeiffer/).

**Italian**

| Model    | relative Accuracy | absolute Accuracy |
|----------|-------------------|-------------------|
| mBERT    | 87.9              | 72.7              |
| XLM      | 88.1              | 75.0              |
| XLM MADX | 92.5              | 78.8              |

**French**

| Model      | relative Accuracy | absolute Accuracy |
|------------|-------------------|-------------------|
| mBERT      | 87.7              | 72.4              |
| mBERT MADX | 88.0              | 63.5              |
| XLM        | 90.3              | 78.9              |

**Brazilian Portuguese**

| Model | relative Accuracy | absolute Accuracy |
|-------|-------------------|-------------------|
| mBERT | 86.4              | 72.6              |
| XLM   | 89.3              | 78.1              |
