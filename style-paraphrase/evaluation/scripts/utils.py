import re
import string
import collections as cll
from scipy.stats import kendalltau
import math
import subprocess


class Bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @classmethod
    def postprocess(cls, input_str):
        input_str = input_str.replace("<h>", cls.HEADER)
        input_str = input_str.replace("<blue>", cls.OKBLUE)
        input_str = input_str.replace("<green>", cls.OKGREEN)
        input_str = input_str.replace("<yellow>", cls.WARNING)
        input_str = input_str.replace("<red>", cls.FAIL)
        input_str = input_str.replace("</>", cls.ENDC)
        input_str = input_str.replace("<b>", cls.BOLD)
        input_str = input_str.replace("<u>", cls.UNDERLINE)
        return input_str


def print_counter(counts, prefix=""):
    keys = list(counts.keys())
    keys.sort()
    for key in keys:
        print("{}{} = {:d} / {:d} ({:.2f}%)".format(prefix, key, counts[key], sum(counts.values()), counts[key] * 100 / sum(counts.values())))

def get_bucket(x, thresholds):
    bucket = -1
    for flt in thresholds:
        if x >= flt:
            bucket += 1
        else:
            break
    return bucket


def get_kendall_tau(x1, x2):
    x1 = normalize_answer(x1)
    x2 = normalize_answer(x2)

    x1_tokens = x1.split()
    x2_tokens = x2.split()

    for x1_index, tok in enumerate(x1_tokens):
        try:
            x2_index = x2_tokens.index(tok)
            x1_tokens[x1_index] = "<match-found>-{:d}".format(x1_index + 1)
            x2_tokens[x2_index] = "<match-found>-{:d}".format(x1_index + 1)
        except ValueError:
            pass

    common_seq_x1 = [int(x1_tok_flag.split("-")[-1]) for x1_tok_flag in x1_tokens if x1_tok_flag.startswith("<match-found>")]
    common_seq_x2 = [int(x2_tok_flag.split("-")[-1]) for x2_tok_flag in x2_tokens if x2_tok_flag.startswith("<match-found>")]

    assert len(common_seq_x1) == len(common_seq_x2)

    ktd = kendalltau(common_seq_x1, common_seq_x2).correlation
    anomaly = False

    if math.isnan(ktd):
        ktd = -1.0
        anomaly = True

    return ktd, anomaly


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """Calculate word level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if not prediction_tokens and not ground_truth_tokens:
        return 1.0, 1.0, 1.0
    common = cll.Counter(prediction_tokens) & cll.Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def export_server(output, filename, server_folder, server="azkaban"):
    with open("{}.txt".format(filename), "w") as f:
        f.write(Bcolors.postprocess(output) + "\n")
    print("Exporting {} to {}...".format(filename, server))
    subprocess.check_output("cat {0}.txt | ansi2html.sh --palette=linux --bg=dark > {0}.html".format(filename), shell=True)
    subprocess.check_output("scp {}.html {}:/scratch/kalpesh/style_transfer_overlap/data_logs/{}".format(filename, server, server_folder), shell=True)


hp_feature_sets = [
    ("original", "input0", "save_62", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1"),
    ("punctuation", "punctuation_input0", "save_98", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_punctuation"),
    ("pos_tags", "pos_tags_input0", "save_45", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_pos_tag"),
    ("shuffle", "shuffle_input0", "save_60", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_shuffle"),
    ("top_100", "top_100_input0", "save_89", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_top_100"),
    ("top_10", "top_10_input0", "save_88", "author_data/authors_1M_tokens_37_classes_srl_arg0_arg1_top_10"),
]
shakespeare_feature_sets = [
    ("original", "input0", "save_110", "shakespeare/unsupervised_filtered"),
]
shakespeare_prior_feature_sets = [
    ("original", "input0", "save_151", "shakespeare/unsupervised_prior")
]
shakespeare_prior_detokenized_feature_sets = [
    ("original", "input0", "save_149", "shakespeare/unsupervised_prior_detokenize")
]
formality_prior_feature_sets = [
    ("original", "input0", "save_159", "formality/formality_prior")
]
formality_prior_detokenized_feature_sets = [
    ("original", "input0", "save_157", "formality/formality_prior_detokenize")
]
shakespeare_aae_tweets_feature_sets = [
    ("original", "input0", "save_112", "dataset_pools/shakespeare_aae_tweets"),
]
seven_styles_feature_sets = [
    ("original", "input0", "save_116", "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_joyce_congress-bills"),
]
six_styles_feature_sets = [
    ("original", "input0", "save_123", "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_switchboard"),
]
politeness_feature_sets = [
    ("original", "input0", "save_147", "politeness/politeness"),
]
gender_feature_sets = [
    ("original", "input0", "save_139", "gender/gender"),
]
formality_feature_sets = [
    ("original", "input0", "save_143", "formality/formality"),
]
political_feature_sets = [
    ("original", "input0", "save_145", "political-slant/political-slant"),
]
ten_styles_feature_sets = [
    ("original", "input0", "save_133", "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_switchboard_coha_3_bins_lyrics_full"),
]
twelve_styles_feature_sets = [
    ("original", "input0", "save_161", "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_congress-bills_joyce_switchboard_coha_3_bins_lyrics_full"),
]
eleven_styles_feature_sets = [
    ("original", "input0", "save_170", "dataset_pools/shakespeare_aae_tweets_bible_romantic-poetry_joyce_switchboard_coha_3_bins_lyrics_full"),
]

all_ft_sets = [
    hp_feature_sets, shakespeare_feature_sets, shakespeare_aae_tweets_feature_sets, seven_styles_feature_sets,
    six_styles_feature_sets, politeness_feature_sets, gender_feature_sets, formality_feature_sets,
    political_feature_sets, ten_styles_feature_sets, shakespeare_prior_detokenized_feature_sets, shakespeare_prior_feature_sets,
    formality_prior_feature_sets, formality_prior_detokenized_feature_sets, twelve_styles_feature_sets, eleven_styles_feature_sets
]


def choose_classifier_feat_sets_from_dir(data_dir):
    stripped_data_dir = data_dir.replace("/mnt/nfs/work1/miyyer/kalpesh/projects/style-embeddings", "").strip("/")
    for ft_set in all_ft_sets:
        if ft_set[0][-1] == stripped_data_dir:
            return ft_set
    raise ValueError("ft_set not found")

def choose_classifier_feat_sets(ft_set):
    if ft_set == "shakespeare":
        feature_sets = shakespeare_feature_sets
    elif ft_set == "shakespeare_aae_tweets":
        feature_sets = shakespeare_aae_tweets_feature_sets
    elif ft_set == "seven_styles_feature_sets":
        feature_sets = seven_styles_feature_sets
    elif ft_set == "six_styles_feature_sets":
        feature_sets = six_styles_feature_sets
    elif ft_set == "politeness_feature_sets":
        feature_sets = politeness_feature_sets
    elif ft_set == "ten_styles_feature_sets":
        feature_sets = ten_styles_feature_sets
    elif ft_set == "twelve_styles_feature_sets":
        feature_sets = twelve_styles_feature_sets
    elif ft_set == "eleven_styles_feature_sets":
        feature_sets = eleven_styles_feature_sets
    elif ft_set == "formality_feature_sets":
        feature_sets = formality_feature_sets
    elif ft_set == "gender_feature_sets":
        feature_sets = gender_feature_sets
    elif ft_set == "political_feature_sets":
        feature_sets = political_feature_sets
    elif ft_set == "harry_potter":
        feature_sets = hp_feature_sets
    elif ft_set == "shakespeare_prior":
        feature_sets = shakespeare_prior_feature_sets
    elif ft_set == "shakespeare_prior_detokenize":
        feature_sets = shakespeare_prior_detokenized_feature_sets
    elif ft_set == "formality_prior":
        feature_sets = formality_prior_feature_sets
    elif ft_set == "formality_prior_detokenize":
        feature_sets = formality_prior_detokenized_feature_sets
    else:
        raise ValueError("Invalid value for --ft_set")

    return feature_sets
