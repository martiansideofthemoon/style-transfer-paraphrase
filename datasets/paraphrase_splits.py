import argparse
import os
import sys
import torch
import tqdm

from style_paraphrase.inference_utils import GPT2Generator


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size for inference.")
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--model_dir', default="paraphraser_gpt2_large", type=str)
parser.add_argument('--paraphrase_str', default="paraphrase_250", type=str)
parser.add_argument('--top_p_value', default=0.0, type=float)
args = parser.parse_args()

roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir, upper_length="same_5")
paraphraser.modify_p(top_p=args.top_p_value)


for split in ["train", "dev", "test"]:
    data_path = os.path.join(args.dataset, split) + ".txt"
    label_path = os.path.join(args.dataset, split) + ".label"

    with open(data_path, "r") as f:
        data = f.read().strip().split("\n")

    with open(label_path, "r") as f:
        labels = f.read().strip().split("\n")

    assert len(data) == len(labels)

    outputs = []
    for i in tqdm.tqdm(range(0, len(data), args.batch_size), desc="minibatches of {} split done...".format(split)):
        generations, _ = paraphraser.generate_batch(data[i:i + args.batch_size])
        outputs.extend(generations)

    outputs = [roberta.bpe.encode(x) for x in outputs]

    output_path = os.path.join(args.dataset, split) + ".{}_input0.bpe".format(args.paraphrase_str)

    with open(output_path, "w") as f:
        f.write("\n".join(outputs) + "\n")
