import argparse
import os
import torch
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default=None)
args = parser.parse_args()

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

for split in ["train", "dev", "test"]:
    data_path = os.path.join(args.dataset, split) + ".txt"
    label_path = os.path.join(args.dataset, split) + ".label"

    with open(data_path, "r") as f:
        data = f.read().strip().split("\n")

    with open(label_path, "r") as f:
        labels = f.read().strip().split("\n")

    assert len(data) == len(labels)

    data = [roberta.bpe.encode(x) for x in tqdm.tqdm(data)]

    output_path = os.path.join(args.dataset, split) + ".input0.bpe"

    with open(output_path, "w") as f:
        f.write("\n".join(data) + "\n")
