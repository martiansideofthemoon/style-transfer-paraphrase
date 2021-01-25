import logging
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset

from data_utils import (
    Instance,
    InverseInstance,
    get_label_dict,
    get_global_dense_features,
    datum_to_dict,
    limit_styles,
    limit_dataset_size,
    string_to_ids,
    update_config,
)
from dataset_config import (
    BASE_CONFIG,
    DATASET_CONFIG,
    MAX_PARAPHRASE_LENGTH,
)

logger = logging.getLogger(__name__)


class ParaphraseDatasetText(Dataset):
    def __init__(self, tokenizer, args, evaluate=False, split="train"):
        data_dir = args.data_dir
        self.args = args

        if data_dir in DATASET_CONFIG:
            self.config = DATASET_CONFIG[data_dir]
        else:
            self.config = BASE_CONFIG

        update_config(self.args, self.config)
        logger.info(self.config)

        self.examples = []

        cached_features_file = os.path.join(
            data_dir, args.model_type + "_cached_lm_" + split
        )
        # Caching is important since it can avoid slow tokenization
        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            with open("{}/{}.pickle".format(data_dir, split), "rb") as handle:
                parse_data = pickle.load(handle)

            self.examples = [
                datum_to_dict(self.config, datum, tokenizer)
                for datum in tqdm.tqdm(parse_data)
            ]

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # in case we are using a fraction of the dataset, reduce the size of the dataset here
        self.examples = limit_dataset_size(self.examples, args.limit_examples)

        # convert the dataset into the Instance class
        self.examples = [
            Instance(args, self.config, datum_dict) for datum_dict in self.examples
        ]

        for instance in self.examples:
            # perform truncation, padding, label and segment building
            instance.preprocess(tokenizer)

        num_truncated = sum([x.truncated for x in self.examples])
        logger.info(
            "Total truncated instances due to length limit = {:d} / {:d}".format(num_truncated, len(self.examples))
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sentence = self.examples[item].sentence
        label = self.examples[item].label
        segment = self.examples[item].segment
        suffix_style = 0
        original_style = 0
        init_context_size = self.examples[item].init_context_size
        global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

        return {
            "sentence": torch.tensor(sentence),
            "instance_number": item,
            "label": torch.tensor(label),
            "segment": torch.tensor(segment),
            "suffix_style": suffix_style,
            "original_style": original_style,
            "init_context_size": init_context_size,
            "global_dense_vectors": global_dense_vectors,
            "metadata": self.examples[item].dict["metadata"],
        }


class InverseParaphraseDatasetText(Dataset):
    def __init__(self, tokenizer, args, evaluate=False, split="train"):
        self.tokenizer = tokenizer
        self.evaluate = evaluate
        self.args = args
        data_dir = args.data_dir
        prefix_input_type = args.prefix_input_type

        self.config = DATASET_CONFIG[data_dir]

        update_config(self.args, self.config)
        config = self.config
        logger.info(self.config)

        cached_features_file = os.path.join(
            data_dir, args.model_type + "_cached_lm_" + split
        )
        cached_features_file += "_prefix_{}".format(prefix_input_type)

        # Read cached list of labels, which is used for style code models
        self.label_dict, self.reverse_label_dict = get_label_dict(data_dir)

        # For the style code model, get the cached dense style code vectors
        self.global_dense_features = get_global_dense_features(data_dir,
                                                               args.global_dense_feature_list,
                                                               self.label_dict)

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            with open("%s/%s.input0.bpe" % (data_dir, split)) as f:
                author_input_data = f.read().strip().split("\n")

            with open("%s/%s.label" % (data_dir, split)) as f:
                suffix_styles = f.read().strip().split("\n")

            with open("%s/%s.%s_input0.bpe" % (data_dir, split, prefix_input_type)) as f:
                prefix_data = f.read().strip().split("\n")

            assert len(author_input_data) == len(suffix_styles)
            assert len(author_input_data) == len(prefix_data)

            self.examples = []

            for i, (inp, suffix_style) in tqdm.tqdm(enumerate(zip(author_input_data, suffix_styles)),
                                                    total=len(author_input_data)):
                self.examples.append({
                    "prefix_sentence": prefix_data[i],
                    "sentence": [int(x) for x in inp.split()],
                    "suffix_style": self.label_dict[suffix_style],
                })

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # in case this is training time and only a certain style's data is to be used
        self.examples = limit_styles(self.examples, args.specific_style_train, split, self.reverse_label_dict)

        # in case we are using a fraction of the dataset, reduce the size of the dataset here
        self.examples = limit_dataset_size(self.examples, args.limit_examples)

        for eg in self.examples:
            eg["original_style"] = eg["suffix_style"]

        # Override the suffix / target style for conditional generation
        if args.target_style_override.startswith("class_fixed_interpolate_"):
            class_number = args.target_style_override.replace("class_fixed_interpolate_", "")
            for eg in self.examples:
                eg["suffix_style"] = class_number

        elif args.target_style_override.startswith("class_fixed_"):
            class_number = int(args.target_style_override.split("_")[2])
            for eg in self.examples:
                eg["suffix_style"] = class_number

        # convert the dataset into the InverseInstance class
        self.examples = [InverseInstance(args, self.config, datum_dict) for datum_dict in self.examples]

        for instance in tqdm.tqdm(self.examples):
            # perform truncation, padding, label and segment building
            instance.preprocess(tokenizer)

        num_truncated = sum([x.truncated for x in self.examples])
        logger.info(
            "Total truncated instances due to length limit = {:d} / {:d}".format(num_truncated, len(self.examples))
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sentence = self.examples[item].sentence
        label = self.examples[item].label
        segment = self.examples[item].segment
        suffix_style = self.examples[item].suffix_style
        original_style = self.examples[item].original_style
        init_context_size = self.examples[item].init_context_size

        if len(self.global_dense_features) > 0:
            if self.args.target_style_override.startswith("class_fixed_interpolate_"):
                authors = [
                    (float(x.split("-")[0]), int(x.split("-")[1]))
                    for x in suffix_style.split("_")
                ]
                global_dense_vectors = []
                for feat_name, feat_vecs in self.global_dense_features:
                    global_dense_vectors.append(
                        # Interpolate feature vectors according to weights
                        [ath[0] * feat_vecs[ath[1]] for ath in authors]
                    )
                global_dense_vectors = np.sum(
                    np.array(global_dense_vectors, dtype=np.float32), axis=1
                )
            else:
                global_dense_vectors = np.array(
                    [x[1][suffix_style] for x in self.global_dense_features],
                    dtype=np.float32
                )
        else:
            global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

        metadata = "suffix_style = {}, original_style = {}".format(
            suffix_style, original_style
        )

        return {
            "sentence": torch.tensor(sentence),
            "instance_number": item,
            "label": torch.tensor(label),
            "segment": torch.tensor(segment),
            "suffix_style": suffix_style,
            "original_style": original_style,
            "init_context_size": init_context_size,
            "global_dense_vectors": global_dense_vectors,
            "metadata": metadata
        }
