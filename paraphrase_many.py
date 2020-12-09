import argparse
import sys
import torch
import tqdm

from style_paraphrase.inference_utils import GPT2Generator

parser = argparse.ArgumentParser()

# Base parameters
parser.add_argument("--batch_size", default=64, type=int,
                    help="Batch size for inference.")
parser.add_argument('--model_dir', default="paraphraser_gpt2_large", type=str)
parser.add_argument('--top_p_value', default=0.6, type=float)
parser.add_argument("--input", default=None, type=str, required=True,
                    help="The input file.")
parser.add_argument("--output", default=None, type=str, required=True,
                    help="The output file.")

args = parser.parse_args()

if not torch.cuda.is_available():
    print("Please check if a GPU is available or your Pytorch installation is correct.")
    sys.exit()

with open(args.input, "r") as f:
    data = f.read().strip().split("\n")

print("Loading paraphraser...")
paraphraser = GPT2Generator(args.model_dir, upper_length="same_5")
paraphraser.modify_p(top_p=args.top_p_value)

outputs = []
for i in tqdm.tqdm(range(0, len(data), args.batch_size), desc="minibatches done..."):
    generations, _ = paraphraser.generate_batch(data[i:i + args.batch_size])
    outputs.extend(generations)

with open(args.output, "w") as f:
    f.write("\n".join(outputs) + "\n")
