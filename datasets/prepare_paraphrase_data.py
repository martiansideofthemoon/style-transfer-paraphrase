import argparse
import pickle
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=None, type=str,
                    help="Input TSV file.")
parser.add_argument("--output_folder", default=None, type=str,
                    help="Output folder.")
parser.add_argument("--train_fraction", default=0.95, type=float,
                    help="Fraction of pairs to put in training split.")
args = parser.parse_args()

with open(args.input_file, "r") as f:
    data = [x.split("\t") for x in f.read().strip().split("\n")]

output_data = []

for dd in data:
    output_data.append((
        None, None, None, dd[0], dd[1], None, None, None, None
    ))

random.seed(43)
random.shuffle(output_data)

num_train = int(args.train_fraction * len(output_data))
train_data = output_data[:num_train]
dev_data = output_data[num_train:]

os.makedirs(args.output_folder, exist_ok=True)

with open(os.path.join(args.output_folder, "train.pickle"), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(args.output_folder, "dev.pickle"), "wb") as f:
    pickle.dump(dev_data, f)
