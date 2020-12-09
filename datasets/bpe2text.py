import argparse
import os
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default=None)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

with open(args.input, "r") as f:
    data = f.read().strip().split("\n")

data = [roberta.bpe.decode(x) for x in tqdm.tqdm(data)]

with open(args.output, "w") as f:
    f.write("\n".join(data) + "\n")
