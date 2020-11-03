from scipy.stats import kendalltau
import tqdm
import collections
import itertools
import numpy as np
import torch
import argparse
import subprocess
import re

from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

from utils import Bcolors

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default=None, type=str)
parser.add_argument('--batch_size', default=16, type=int)
args = parser.parse_args()

roberta = RobertaModel.from_pretrained(
    'style_paraphrase/evaluation/fluency/cola_classifier',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='style_paraphrase/evaluation/fluency/cola_classifier/cola-bin'
)

def detokenize(x):
    x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    return x

def label_fn(label):
    return roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )

ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()


with open(args.input_file, "r") as f:
    author_data = f.read().strip().split("\n")

unk_bpe = roberta.bpe.encode(" <unk>").strip()

argmax_results = []
prediction_data = {}
for label in ["acceptable", "unacceptable"]:
    prediction_data[label.lower()] = []

label_data = ["acceptable" for _ in author_data]

for i in tqdm.tqdm(range(0, len(author_data), args.batch_size), total=len(author_data) // args.batch_size):
    sds = author_data[i:i + args.batch_size]
    lds = label_data[i:i + args.batch_size]

    # detokenize and BPE encode input
    sds = [roberta.bpe.encode(detokenize(sd)) for sd in sds]

    batch = collate_tokens(
        [roberta.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in sds], pad_idx=1
    )

    batch = batch[:, :512]

    with torch.no_grad():
        predictions = roberta.predict('sentence_classification_head', batch.long())

    prediction_probs = [torch.exp(x).max(axis=0)[0].item() for x in predictions]
    prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]

    ncorrect += sum([1 if l1.lower() == l2.lower() else 0 for l1, l2 in zip(prediction_labels, lds)])

    nsamples += len(sds)

    for sd, ld, pld, ppd in zip(sds, lds, prediction_labels, prediction_probs):
        sd1 = sd.strip()
        sd1 = sd1.replace("<unk>", unk_bpe).strip()
        argmax_results.append(pld.lower())
        prediction_data[ld.lower()].append({
            "sentence": roberta.bpe.decode(sd1),
            "prediction": pld.lower(),
            "prediction_prob": ppd,
            "correct": ld.lower() == pld.lower()
        })

overall_accuracy = "<b>{: <31}</> = <b><green>{:6.2f}</> ({:3d} / {:3d})\n\n".format("overall accuracy", float(ncorrect) * 100 / float(nsamples), ncorrect, nsamples)

print("")
output = ""

# First compute a qualitative summary
author_str = {}
for label in roberta.task.label_dictionary.symbols:
    if label.lower() not in prediction_data:
        continue
    ncorrect = sum([x["correct"] for x in prediction_data[label.lower()]])
    ntotal = max(len(prediction_data[label.lower()]), 1)
    author_str[label.lower()] = "author <b>{: <24}</> = <b><green>{:6.2f}</> ({:3d} / {:3d})\n".format(label, float(ncorrect) * 100 / ntotal, ncorrect, ntotal)
    output += author_str[label.lower()]

output += "{}\n".format("".join("─" for _ in range(60)))
output += overall_accuracy
output += "{}\n\n".format("".join("=" for _ in range(60)))

print(Bcolors.postprocess(output))

with open(args.input_file + ".acceptability_labels", "w") as f:
    f.write("\n".join(argmax_results) + "\n")
