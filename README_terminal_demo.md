## Paraphrase Model Demo

### Installation

In a new virtual environment run the following commands,

```
virtualenv -p python3 style-venv
source style-venv/bin/activate
pip install transformers
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

These commands [might differ](https://pytorch.org/get-started/locally) if your CUDA installation is different or you are not using `virtualenv`.

Download the pretrained model `paraphraser_gpt2_large` from [https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing](https://drive.google.com/drive/folders/12ImHH2kJKw1Vs3rDUSRytP3DZYcHdsZw?usp=sharing).

### Running

```
python demo_paraphraser.py --model_dir paraphraser_gpt2_large --top_p_value 0.6
```

1. Make sure you point `--model_dir` to the path you used to store the model in the previous step.
2. Ignore the weight mismatch error message, it's due to a `transformers` version mismatch + some minor edits I did myself. It won't affect paraphrase generation.
3. Make sure you can reproduce the following greedy decoding generations, to confirm the model works as expected.

```
Input: Political and currency gyrations can whipsaw the funds.
Greedy: the money can be affected by political and currency fluctuations.

Input: So can a magazine survive by downright thumbing its nose at major advertisers?
Greedy: so can a magazine survive by completely ignoring the major advertising?

Input: The documents also said that Cray Computer anticipates needing perhaps another $120 million in financing beginning next September.
Greedy: the documents also said that Cray's computer is likely to need another $120 million in funding next September.
```

### Using the API

The demo shows how to use greedy decoding as well as nucleus sampling with `p = 0.6`. Samples tend to be more diverse with `p = 0.6`, sometimes at the expense of semantic accuracy. Note that sentences should be in their raw form (detokenized) and upto 50 subwords in length.

If you want to use this at a large scale, consider using `paraphrase_many.py` which accepts a file as input and outputs paraphrases of it.

You can also try out other paraphrase models by modifying the model ID at the end of the file path input to `GPT2Generator`. Use `model_249` for a GPT2-medium paraphraser, `model_317` for a non-diverse paraphraser and `model_219` for another GPT2-large paraphraser with lesser dataset filtering.
