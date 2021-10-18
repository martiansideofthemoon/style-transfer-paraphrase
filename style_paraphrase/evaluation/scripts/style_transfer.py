import argparse
import os
import random
import sys
import torch
import tqdm

from style_paraphrase.inference_utils import GPT2Generator

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input_file', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--generation_mode', type=str, default="nucleus_paraphrase")
parser.add_argument('--paraphrase_model', type=str, default="../../style_paraphrase/saved_models/model_250")
parser.add_argument('--style_transfer_model', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--output_class', type=int, default=None)
parser.add_argument("--detokenize", dest="detokenize", action="store_true")
parser.add_argument("--post_detokenize", dest="post_detokenize", action="store_true")
parser.add_argument("--lowercase", dest="lowercase", action="store_true")
parser.add_argument("--post_lowercase", dest="post_lowercase", action="store_true")
args = parser.parse_args()

if "greedy" in args.generation_mode:
    args.top_p = 0.0

def detokenize(x):
    x = x.replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?").replace(" )", ")").replace("( ", "(")
    return x

def tokenize(x):
    x = x.replace(".", " .").replace(",", " ,").replace("!", " !").replace("?", " ?").replace(")", " )").replace("(", "( ")
    return x

with open(args.input_file, "r") as f:
    input_data = f.read().strip().split("\n")

if os.path.exists("{}/eval_{}_{}/{}".format(args.style_transfer_model, args.generation_mode, args.top_p, args.output_file)):
    print("Output already exists...")
    sys.exit(0)

if args.detokenize:
    input_data = [detokenize(x) for x in input_data]

if args.lowercase:
    input_data = [x.lower() for x in input_data]

st_input_data = []
if "paraphrase" in args.generation_mode:
    pp_filename = args.input_file + ".paraphrase"
    # if not args.paraphrase_model.endswith("250"):
    #     pp_filename += "_" + args.paraphrase_model.split("_")[-1]
    print(pp_filename)
    if os.path.exists(pp_filename):
        with open(pp_filename, "r") as f:
            st_input_data = f.read().strip().split("\n")
    else:
        paraphrase_model = GPT2Generator(
            args.paraphrase_model, upper_length="same_5"
        )
        for i in tqdm.tqdm(range(0, len(input_data), args.batch_size), desc="paraphrasing dataset..."):
            st_input_data.extend(
                paraphrase_model.generate_batch(input_data[i:i + args.batch_size])[0]
            )
        with open(pp_filename, "w") as f:
            f.write("\n".join(st_input_data) + "\n")
else:
    st_input_data = input_data

if args.output_class is not None:
    vec_data_dir = os.path.dirname(os.path.dirname(args.input_file))
else:
    vec_data_dir = os.path.dirname(os.path.dirname(args.input_file))
if "nucleus" in args.generation_mode:
    style_transfer_model = GPT2Generator(
        args.style_transfer_model, upper_length="same_10", top_p=args.top_p, data_dir=vec_data_dir
    )
else:
    style_transfer_model = GPT2Generator(
        args.style_transfer_model, upper_length="same_10", data_dir=vec_data_dir
    )

transferred_data = []

for i in tqdm.tqdm(range(0, len(st_input_data), args.batch_size), desc="transferring dataset..."):
    if args.output_class:
        transferred_data.extend(
            style_transfer_model.generate_batch(
                contexts=st_input_data[i:i + args.batch_size],
                global_dense_features=[args.output_class for _ in st_input_data[i:i + args.batch_size]]
            )[0]
        )
    else:
        transferred_data.extend(
            style_transfer_model.generate_batch(st_input_data[i:i + args.batch_size])[0]
        )

if args.post_detokenize:
    transferred_data = [tokenize(x) for x in transferred_data]

if args.post_lowercase:
    transferred_data = [x.lower() for x in transferred_data]

transferred_data = [" ".join(x.split()) for x in transferred_data]

all_data = [(x, y, z) for x, y, z in zip(input_data, st_input_data, transferred_data)]

with open("{}/eval_{}_{}/{}".format(args.style_transfer_model, args.generation_mode, args.top_p, args.output_file), "w") as f:
    f.write("\n".join(transferred_data) + "\n")
