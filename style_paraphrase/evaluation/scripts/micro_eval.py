import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--generated_file', default=None, type=str)
parser.add_argument('--reference_file', default=None, type=str)
parser.add_argument('--classifier_file', default=None, type=str)
parser.add_argument('--paraphrase_file', default=None, type=str)
parser.add_argument('--acceptability_file', default=None, type=str)
parser.add_argument('--expected_classifier_value', default="correct", type=str)
args = parser.parse_args()

with open(args.generated_file, "r") as f:
    generated_data = f.read().strip().split("\n")

with open(args.classifier_file, "r") as f:
    classifier_data = f.read().strip().split("\n")

with open(args.paraphrase_file, "r") as f:
    paraphrase_data = f.read().strip().split("\n")

with open(args.acceptability_file, "r") as f:
    acceptability_data = f.read().strip().split("\n")

assert len(classifier_data) == len(paraphrase_data)
assert len(paraphrase_data) == len(acceptability_data)

scores = {
    "acc_sim": [],
    "cola_sim": [],
    "acc_cola": [],
    "acc_cola_sim": []
}

valid_count = {
    "acc_sim": 0,
    "cola_sim": 0,
    "acc_cola": 0,
    "acc_cola_sim": 0
}
normalized_generated_data = {
    "acc_sim": [],
    "cola_sim": [],
    "acc_cola_sim": []
}

for cd, pd, gd, ad in zip(classifier_data, paraphrase_data, generated_data, acceptability_data):
    curr_scores = max([float(x) for x in pd.split(",")])

    # check acc_sim
    if cd.split(",")[0] == args.expected_classifier_value:
        valid_count["acc_sim"] += 1
        scores["acc_sim"].append(curr_scores)
        normalized_generated_data["acc_sim"].append(gd)
    else:
        scores["acc_sim"].append(0)
        normalized_generated_data["acc_sim"].append("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    # check cola_sim
    if ad == "acceptable":
        valid_count["cola_sim"] += 1
        scores["cola_sim"].append(curr_scores)
        normalized_generated_data["cola_sim"].append(gd)
    else:
        scores["cola_sim"].append(0)
        normalized_generated_data["cola_sim"].append("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    # check acc_cola
    if ad == "acceptable" and cd.split(",")[0] == args.expected_classifier_value:
        valid_count["acc_cola"] += 1
        scores["acc_cola"].append(1)
        valid_count["acc_cola_sim"] += 1
        scores["acc_cola_sim"].append(curr_scores)
        normalized_generated_data["acc_cola_sim"].append(gd)
    else:
        scores["acc_cola"].append(0)
        scores["acc_cola_sim"].append(0)
        normalized_generated_data["acc_cola_sim"].append("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

for metric in ["acc_sim", "cola_sim", "acc_cola", "acc_cola_sim"]:
    print(
        "Normalized pp score ({}) = {:.4f} ({:d} / {:d} valid)".format(
            metric, np.mean(scores[metric]), valid_count[metric], len(scores[metric])
        )
    )

    if metric in normalized_generated_data:
        with open(args.generated_file + ".{}_normalized".format(metric), "w") as f:
            f.write("\n".join(normalized_generated_data[metric]) + "\n")
