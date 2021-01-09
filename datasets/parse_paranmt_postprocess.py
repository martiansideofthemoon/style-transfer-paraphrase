"""
python preprocess/parse_paranmt_postprocess.py --filtering_method and_kt_less_precision_less_lendiff_less_0.0_0.5_5 --english_only
"""

import argparse
import os
import pickle
import tqdm
import random


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_dir', type=str, default="/mnt/nfs/work1/miyyer/datasets/paranmt")
parser.add_argument('--total', type=int, default=40)
parser.add_argument('--filtering_method', type=str, default="and_kt_less_precision_less_0.0_0.5")
parser.add_argument('--train_split', type=float, default=0.98)
parser.add_argument('--no_czech', dest='no_czech', action='store_true')
parser.add_argument('--english_only', dest='english_only', action='store_true')
parser.add_argument('--truncate', type=int, default=None)

args = parser.parse_args()

all_temp1, all_temp2, all_equality, all_sent1, all_sent2 = [], [], [], [], []
all_f1_scores, all_kt_scores, all_ed_scores, all_langids = [], [], [], []

for i in tqdm.tqdm(range(args.total)):
    if args.data_dir == "/mnt/nfs/work1/miyyer/datasets/paranmt":
        with open(os.path.join(args.data_dir, "template_%d.pickle" % i), "rb") as f:
            temp1, temp2, equality, sent1, sent2, f1_scores, kt_scores, ed_scores = pickle.load(f)

        all_temp1.extend(temp1)
        all_temp2.extend(temp2)
        all_equality.extend(equality)
        all_sent1.extend(sent1)
        all_sent2.extend(sent2)
        all_f1_scores.extend(f1_scores)
        all_kt_scores.extend(kt_scores)
        all_ed_scores.extend(ed_scores[:-1])
        all_langids.extend(ed_scores[-1])
    else:
        with open(os.path.join(args.data_dir, "template_%d.pickle" % i), "rb") as f:
            temp1, temp2, equality, sent1, sent2, f1_scores, kt_scores, ed_scores, langid_scores = pickle.load(f)

        all_temp1.extend(temp1)
        all_temp2.extend(temp2)
        all_equality.extend(equality)
        all_sent1.extend(sent1)
        all_sent2.extend(sent2)
        all_f1_scores.extend(f1_scores)
        all_kt_scores.extend(kt_scores)
        all_ed_scores.extend(ed_scores)
        all_langids.extend(langid_scores)

filtered_dataset = []

all_lists = [
    all_temp1, all_temp2, all_equality, all_sent1, all_sent2, all_f1_scores, all_kt_scores, all_ed_scores, all_langids
]

print("Original length = %d" % len(all_sent1))

for element in tqdm.tqdm(zip(*all_lists), total=len(all_sent1)):
    assert element[2] == (element[0] == element[1])

    if args.filtering_method == "parse_eq":
        if element[2] is False:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("lendiff_less_"):
        len_diff = int(args.filtering_method.split("_")[-1])

        if abs(len(element[3].split()) - len(element[4].split())) <= len_diff:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("kt_less_"):
        kt_upper = float(args.filtering_method.split("_")[-1])
        if element[6][0] < kt_upper:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("precision_less"):
        precision_upper = float(args.filtering_method.split("_")[-1])
        if element[5][0] < precision_upper:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("and_kt_less_precision_less_lendiff_less"):
        kt_upper = float(args.filtering_method.split("_")[-3])
        precision_upper = float(args.filtering_method.split("_")[-2])
        len_diff = int(args.filtering_method.split("_")[-1])

        if element[6][0] < kt_upper and \
           element[5][0] < precision_upper and \
           abs(len(element[3].split()) - len(element[4].split())) <= len_diff:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("and_precision_less_lendiff_less"):
        precision_upper = float(args.filtering_method.split("_")[-2])
        len_diff = int(args.filtering_method.split("_")[-1])

        if element[5][0] < precision_upper and \
           abs(len(element[3].split()) - len(element[4].split())) <= len_diff:
            filtered_dataset.append(element)

    elif args.filtering_method.startswith("and_kt_less_precision_less"):
        kt_upper = float(args.filtering_method.split("_")[-2])
        precision_upper = float(args.filtering_method.split("_")[-1])
        if element[6][0] < kt_upper and element[5][0] < precision_upper:
            filtered_dataset.append(element)

print("length of filtered = %d" % len(filtered_dataset))

if args.no_czech:
    args.filtering_method = args.filtering_method + "_no_czech"
    print("Size of filtered dataset = {:d}".format(len(filtered_dataset)))
    filtered_dataset = list(filter(lambda x: x[-1][0] != "cs" and x[-1][1] != "cs", filtered_dataset))
    filtered_dataset = list(filter(lambda x: x[-1][0] != "xx" and x[-1][1] != "xx", filtered_dataset))
    print("Size of no-czech filtered dataset = {:d}".format(len(filtered_dataset)))

elif args.english_only:
    args.filtering_method = args.filtering_method + "_english_only"
    print("Size of filtered dataset = {:d}".format(len(filtered_dataset)))
    filtered_dataset = list(filter(lambda x: x[-1][0] != "cs" and x[-1][1] != "cs", filtered_dataset))
    filtered_dataset = list(filter(lambda x: x[-1][0] != "xx" and x[-1][1] != "xx", filtered_dataset))
    filtered_dataset = list(filter(lambda x: x[-1][0] == "en" or x[-1][1] == "en", filtered_dataset))
    print("Size of english-only filtered dataset = {:d}".format(len(filtered_dataset)))

random.shuffle(filtered_dataset)

if args.truncate:
    args.filtering_method = args.filtering_method + "_truncate_{}".format(args.truncate)
    filtered_dataset = filtered_dataset[:args.truncate]

train_size = int(args.train_split * len(filtered_dataset))

train_data = filtered_dataset[:train_size]
dev_data = filtered_dataset[train_size:]

print("Statistics = {:d} train examples, {:d} dev examples".format(len(train_data), len(dev_data)))

os.makedirs(os.path.join(args.data_dir, "filter_{}".format(args.filtering_method)), exist_ok=True)

with open(os.path.join(args.data_dir, "filter_{}".format(args.filtering_method), "train.pickle"), "wb") as f:
    pickle.dump(train_data, f)

with open(os.path.join(args.data_dir, "filter_{}".format(args.filtering_method), "dev.pickle"), "wb") as f:
    pickle.dump(dev_data, f)
