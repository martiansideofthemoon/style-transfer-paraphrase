We are releasing multilingual formality classifiers by fine-tuning large multilingual language models on English GYAFC, to facilitate zero-shot cross-lingual transfer. We evaluated these classifiers on [XFORMAL](https://arxiv.org/abs/2104.04108). For each language, we lower-case sentences and remove trailing punctuation to stop the model from latching onto easy indicators of formality.

Italian

| Model    | relative Accuracy | absolute Accuracy |
|----------|-------------------|-------------------|
| mBERT    | 87.9              | 72.7              |
| XLM      | 88.1              | 75.0              |
| XLM MADX | 92.5              | 78.8              |

French



