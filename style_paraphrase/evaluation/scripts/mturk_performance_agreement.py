import argparse
import csv
import glob
import numpy as np
from collections import defaultdict, Counter

from nltk import agreement

parser = argparse.ArgumentParser()
parser.add_argument('--input_folder', default=None, type=str)
parser.add_argument('--key', default="text1,text2", type=str)
parser.add_argument("--add_header", dest="add_header", action="store_true")
parser.add_argument('--expected_label', default="incorrect", type=str)
args = parser.parse_args()

def print_counter(counts, prefix=""):
    keys = list(counts.keys())
    keys.sort()
    for key in keys:
        print("{}{} = {:d} / {:d} ({:.2f}%)".format(prefix, key, counts[key], sum(counts.values()), counts[key] * 100 / sum(counts.values())))


def most_common(lst):
    return max(set(lst), key=lst.count)

def build_key(hit_id, text1, text2):
    if args.key == "hit_id":
        return hit_id
    else:
        return text1 + "," + text2


hit_ratings = defaultdict(list)
hit_sentences = defaultdict(list)
all_labels = []

mturk_files = glob.glob("{}/Batch_*".format(args.input_folder))
mturk_files.sort()

for mf in mturk_files:
    header = None
    data = []

    with open(mf, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                header = row
            else:
                data.append(row)

    TEXT1_INDEX = header.index("Input.text1")
    TEXT2_INDEX = header.index("Input.text2")
    ANSWER_INDEX = header.index("Answer.semantic-similarity.label")

    for hit in data:
        key = build_key(hit[0], hit[TEXT1_INDEX], hit[TEXT2_INDEX])

        if key in hit_ratings and len(hit_ratings[key]) == 3:
            if len(set(hit_ratings[key])) == 3:
                # this is due to the duplicates case, remove old ratings
                hit_ratings[key] = []
            else:
                continue

        hit_ratings[key].append(hit[ANSWER_INDEX])
        all_labels.append(hit[ANSWER_INDEX])
        hit_sentences[key].append(
            (hit[TEXT1_INDEX], hit[TEXT2_INDEX])
        )

hit_ratings = [(k, v) for k, v in hit_ratings.items()]

ratings_augmented = 0
for k, v in hit_ratings:
    while len(v) < 3:
        print(k)
        ratings_augmented += 1
        v.append(v[-1])

print("Ratings augmented = {:d}".format(ratings_augmented))

for k, v in hit_ratings:
    assert len(v) == 3

taskdata = []
most_common_ratings = {}
fully_uncertain = []
for i, (hit, rating) in enumerate(hit_ratings):
    if len(set(rating)) < 3:
        most_common_ratings[hit] = most_common(rating)
    else:
        fully_uncertain.append(hit_sentences[hit][0])
    taskdata.extend([
        [j, i, rating[j]] for j in range(len(rating))
    ])

print(fully_uncertain)

count_mcr = Counter(most_common_ratings.values())
print_counter(count_mcr)
ratingtask = agreement.AnnotationTask(data=taskdata)
print("\nfleiss " + str(ratingtask.multi_kappa()))
# print(Counter([x[-1] for x in data]))
print_counter(Counter([len(set(x[1])) for x in hit_ratings]), prefix="number of unique ratings are ")

if args.add_header:
    fully_uncertain = [("text1", "text2")] + fully_uncertain

with open("{}/full_disagreement.csv".format(args.input_folder), "w") as f:
    writer = csv.writer(f, delimiter=",")
    writer.writerows(fully_uncertain)

# Computing micro-eval stats

all_input_files = glob.glob("{}/input_all_*".format(args.input_folder))
all_label_files = glob.glob("{}/label_all_*".format(args.input_folder))

all_input_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
all_label_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
assert len(all_input_files) == len(all_label_files)

all_input_data, all_label_data = [], []

for aif, alf in zip(all_input_files, all_label_files):
    with open(alf, "r") as f:
        all_label_data.extend(
            [x.split(",") for x in f.read().strip().split("\n")]
        )
    with open(aif, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            else:
                all_input_data.append(row)
    assert len(all_input_data) == len(all_label_data)


for eval_mode in ["without_duplicates", "with duplicates"]:
    acc = []
    sim = []
    j_acc_sim = []
    j_acc_sim_fl = []

    for aid, ald in zip(all_input_data, all_label_data):
        key = build_key("", aid[0], aid[1])

        if eval_mode == "without_duplicates" and key not in most_common_ratings:
            continue
        if ald[0] == args.expected_label:
            curr_acc = 1
        else:
            curr_acc = 0
        acc.append(curr_acc)

        if key in most_common_ratings:
            curr_sim = 0 if most_common_ratings[key] == "no paraphrase relationship" else 1
        else:
            curr_sim = 1
        sim.append(curr_sim)

        if curr_sim == 1 and curr_acc == 1:
            j_acc_sim.append(1)
        else:
            j_acc_sim.append(0)

        if key in most_common_ratings:
            curr_sim_fl = 1 if most_common_ratings[key] == "approximately the same meaning and the rewritten sentence is grammatical" else 0
        else:
            curr_sim_fl = 1

        if curr_sim_fl == 1 and curr_acc == 1:
            j_acc_sim_fl.append(1)
        else:
            j_acc_sim_fl.append(0)

    print("\nMicro-eval mode = {}".format(eval_mode))
    print("Accuracy = {:.1f}% ({:d} / {:d})".format(np.mean(acc) * 100, np.sum(acc), len(acc)))
    print("Similarity = {:.1f}% ({:d} / {:d})".format(np.mean(sim) * 100, np.sum(sim), len(sim)))
    print("J(A, S) = {:.1f}% ({:d} / {:d})".format(np.mean(j_acc_sim) * 100, np.sum(j_acc_sim), len(j_acc_sim)))
    print("J(A, S, F) = {:.1f}% ({:d} / {:d})".format(np.mean(j_acc_sim_fl) * 100, np.sum(j_acc_sim_fl), len(j_acc_sim_fl)))
