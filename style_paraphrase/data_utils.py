import logging
import numpy as np
import pickle
import random

MAX_ROBERTA_LENGTH = 502

random.seed(12)
logger = logging.getLogger(__name__)


class Instance(object):
    def __init__(self, args, config, instance_dict):
        self.dict = instance_dict
        self.args = args
        self.config = config
        self.truncated = False
        self.sent1_tokens = np.array(instance_dict["sent1_tokens"], dtype=np.int32)
        self.sent2_tokens = np.array(instance_dict["sent2_tokens"], dtype=np.int32)
        self.init_context_size = config["max_prefix_length"] + 1

    def preprocess(self, tokenizer):
        # shorten the very long sequences in the instance based on DATASET_CONFIG
        self.truncate()
        # whenever args.prefix_input_type has "original_shuffle" or "original_reverse"
        # exchange prefix/suffix with 50% probability or 100% probability
        self.shuffle_prefix_suffix()
        # Finally, perform prefix and suffix padding to build the sentence, label and segments
        self.build_sentence(tokenizer)
        self.build_label(tokenizer)
        self.build_segment(tokenizer)
        # check if the padding worked out correctly and all the lengths are aligned
        self.check_constraints()

    def truncate(self):
        config = self.config
        max_prefix_length = config["max_prefix_length"]
        max_suffix_length = config["max_suffix_length"]
        if len(self.sent1_tokens) > max_prefix_length:
            self.truncated = True
            self.sent1_tokens = self.sent1_tokens[:max_prefix_length]
        if len(self.sent2_tokens) > max_suffix_length:
            self.truncated = True
            self.sent2_tokens = self.sent2_tokens[:max_suffix_length]

    def shuffle_prefix_suffix(self):
        if not hasattr(self.args, "prefix_input_type"):
            # Keeping this check for backward compatibility with previous models
            return
        if self.args.prefix_input_type == "original_shuffle":
            # shuffle with 50% probability
            if random.random() <= 0.5:
                self.sent1_tokens, self.sent2_tokens = self.sent2_tokens, self.sent1_tokens

        elif self.args.prefix_input_type == "original_reverse":
            self.sent1_tokens, self.sent2_tokens = self.sent2_tokens, self.sent1_tokens

    def build_sentence(self, tokenizer):
        self.sent_prefix = left_padding(
            self.sent1_tokens, tokenizer.pad_token_id, self.config["max_prefix_length"]
        )

        self.sent_suffix = right_padding(
            np.append(self.sent2_tokens, tokenizer.eos_token_id),
            tokenizer.pad_token_id,
            self.config["max_suffix_length"] + 1
        )
        self.sentence = np.concatenate(
            [self.sent_prefix, [tokenizer.bos_token_id], self.sent_suffix]
        )

    def build_label(self, tokenizer):
        dense_length = self.config["global_dense_length"]
        self.label_suffix = right_padding(
            np.append(self.sent2_tokens, tokenizer.eos_token_id),
            -100,
            self.config["max_suffix_length"] + 1
        )
        self.label = np.concatenate([
            [-100 for _ in range(dense_length)],
            [-100 for _ in self.sent_prefix],
            [-100],
            self.label_suffix
        ]).astype(np.int64)

    def build_segment(self, tokenizer):
        dense_length = self.config["global_dense_length"]
        prefix_segment = [tokenizer.additional_special_tokens_ids[1] for _ in self.sent_prefix]
        suffix_segment_tag = tokenizer.additional_special_tokens_ids[2]

        self.segment = np.concatenate([
            [tokenizer.additional_special_tokens_ids[0] for _ in range(dense_length)],
            prefix_segment,
            [suffix_segment_tag],
            [suffix_segment_tag for _ in self.sent_suffix],
        ]).astype(np.int64)

    def check_constraints(self):
        dense_length = self.config["global_dense_length"]
        assert len(self.sentence) == len(self.label) - dense_length
        assert len(self.sentence) == len(self.segment) - dense_length


class InverseInstance(Instance):
    def __init__(self, args, config, instance_dict):
        self.dict = instance_dict
        self.args = args
        self.config = config
        self.truncated = False
        self.init_context_size = config["max_prefix_length"] + 1

        self.original_sentence = instance_dict["sentence"]
        self.prefix_sentence = instance_dict["prefix_sentence"]
        self.suffix_style = instance_dict["suffix_style"]
        self.original_style = instance_dict["original_style"]

        self.sent1_tokens = np.array(
            [int(x) for x in self.prefix_sentence.split()],
            dtype=np.int32
        )
        self.sent2_tokens = np.array(self.original_sentence, dtype=np.int32)


def np_prepend(array, value):
    return np.insert(array, 0, value)


def left_padding(data, pad_token, total_length):
    tokens_to_pad = total_length - len(data)
    return np.pad(data, (tokens_to_pad, 0), constant_values=pad_token)


def right_padding(data, pad_token, total_length):
    tokens_to_pad = total_length - len(data)
    return np.pad(data, (0, tokens_to_pad), constant_values=pad_token)


def string_to_ids(text, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


def get_label_dict(data_dir):
    label_dict = {}
    with open("{}/dict.txt".format(data_dir)) as f:
        label_dict_lines = f.read().strip().split("\n")
    for i, x in enumerate(label_dict_lines):
        if x.startswith("madeupword"):
            continue
        label_dict[x.split()[0]] = i
    reverse_label_dict = {v: k for k, v in label_dict.items()}

    return label_dict, reverse_label_dict


def get_global_dense_features(data_dir, global_dense_feature_list, label_dict):
    """Get dense style code vectors for the style code model."""

    global_dense_features = []
    if global_dense_feature_list != "none":
        logger.info("Using global dense vector features = %s" % global_dense_feature_list)
        for gdf in global_dense_feature_list.split(","):
            with open("{}/{}_dense_vectors.pickle".format(data_dir, gdf), "rb") as f:
                vector_data = pickle.load(f)

            final_vectors = {}
            for k, v in vector_data.items():
                final_vectors[label_dict[k]] = v["sum"] / v["total"]

            global_dense_features.append((gdf, final_vectors))
    return global_dense_features


def limit_dataset_size(dataset, limit_examples):
    """Limit the dataset size to a small number for debugging / generation."""

    if limit_examples:
        logger.info("Limiting dataset to {:d} examples".format(limit_examples))
        dataset = dataset[:limit_examples]

    return dataset


def limit_styles(dataset, specific_style_train, split, reverse_label_dict):
    """Limit the dataset size to a certain author."""
    specific_style_train = [int(x) for x in specific_style_train.split(",")]

    original_dataset_size = len(dataset)
    if split in ["train", "test"] and -1 not in specific_style_train:
        logger.info("Preserving authors = {}".format(", ".join([reverse_label_dict[x] for x in specific_style_train])))
        dataset = [x for x in dataset if x["suffix_style"] in specific_style_train]
        logger.info("Remaining instances after author filtering = {:d} / {:d}".format(len(dataset), original_dataset_size))
    return dataset


def datum_to_dict(config, datum, tokenizer):
    """Convert a data point to the instance dictionary."""

    instance_dict = {"metadata": ""}

    for key in config["keys"]:
        element_value = datum[key["position"]]
        instance_dict[key["key"]] = string_to_ids(element_value, tokenizer) if key["tokenize"] else element_value
        if key["metadata"]:
            instance_dict["metadata"] += "%s = %s, " % (key["key"], str(element_value))
    # strip off trailing , from metadata
    instance_dict["metadata"] = instance_dict["metadata"][:-2]
    return instance_dict


def update_config(args, config):
    if args.global_dense_feature_list != "none":
        global_dense_length = len(args.global_dense_feature_list.split(","))
        logger.info("Using {:d} dense feature vectors.".format(global_dense_length))
    else:
        global_dense_length = 0

    assert global_dense_length <= config["max_dense_length"]
    config["global_dense_length"] = global_dense_length
