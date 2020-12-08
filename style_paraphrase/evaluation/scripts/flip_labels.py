import argparse

from style_paraphrase.evaluation.similarity.test_sim import find_similarity

parser = argparse.ArgumentParser()
parser.add_argument('--file1', type=str, default=None)
parser.add_argument('--file2', type=str, default=None)
parser.add_argument('--output_file', type=str, default=None)
args = parser.parse_args()

with open(args.file1, "r") as f:
    data1 = f.read().strip().split("\n")

with open(args.file2, "r") as f:
    data2 = f.read().strip().split("\n")

label1 = data1[0]
label2 = data2[0]

data1 = [label2 for _ in data1]
data2 = [label1 for _ in data2]

concat = data1 + data2

with open(args.output_file, "w") as f:
    f.write("\n".join(concat) + "\n")
