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
    HPInstance,
    Instance,
    aggregate_content_information,
    datum_to_dict,
    limit_authors,
    limit_dataset_size,
    string_to_ids,
    update_config,
)
from dataset_config import (
    BASE_CONFIG,
    BASE_HP_CONFIG,
    DATASET_CONFIG,
    MAX_GPT2_LENGTH,
    MAX_PARAPHRASE_LENGTH,
)
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

logger = logging.getLogger(__name__)


class ParaphraseDatasetText(Dataset):
    def __init__(
        self, tokenizer, args, model_type, roberta, evaluate=False, split="train"
    ):
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
            data_dir, model_type + "_cached_lm_" + split
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

        logger.info(
            "Total truncated instances due to length limit = {:d} / {:d}".format(
                sum([x.truncated for x in self.examples]), len(self.examples)
            )
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sentence = self.examples[item].sentence
        roberta_sentence = np.zeros((1, 512), dtype=np.float32)
        label = self.examples[item].label
        segment = self.examples[item].segment
        author_target = 0
        roberta_author_target = 0
        original_author_target = 0
        init_context_size = self.examples[item].init_context_size

        if self.args.global_dense_feature_list != "none":
            global_dense_vectors = np.zeros((2, 20), dtype=np.float32)
            global_dense_vectors[0, self.examples[item].dict["f1_bucket"]] = 1
            global_dense_vectors[1, self.examples[item].dict["ed_bucket"] + 10] = 1
        else:
            global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

        return {
            "sentence": torch.tensor(sentence),
            "roberta_sentence": torch.tensor(roberta_sentence),
            "instance_number": item,
            "label": torch.tensor(label),
            "segment": torch.tensor(segment),
            "author_target": author_target,
            "roberta_author_target": roberta_author_target,
            "original_author_target": original_author_target,
            "init_context_size": init_context_size,
            "global_dense_vectors": global_dense_vectors,
            "metadata": self.examples[item].dict["metadata"],
        }


class InverseParaphraseDatasetText(Dataset):
    def __init__(
        self, tokenizer, args, model_type, roberta, evaluate=False, split="train"
    ):
        self.tokenizer = tokenizer
        self.evaluate = evaluate
        self.args = args
        data_dir = args.data_dir

        if data_dir in DATASET_CONFIG:
            self.config = DATASET_CONFIG[data_dir]
        else:
            self.config = BASE_HP_CONFIG
        update_config(self.args, self.config)
        config = self.config
        logger.info(self.config)

        roberta_input_type = "_".join(args.roberta_input_type.split("_")[1:])

        global_dense_feature_list = args.global_dense_feature_list

        if evaluate is False and args.context_noise != "none":
            self.context_noise_fn = partial(
                np.random.binomial, n=1, p=float(args.context_noise.split("_")[1])
            )
        else:
            self.context_noise_fn = None

        cached_features_file = os.path.join(
            data_dir, model_type + "_cached_lm_" + split
        )

        if roberta_input_type != "nofilter":
            cached_features_file += "_roberta_filter_{}".format(roberta_input_type)

        if roberta is not None:
            cached_features_file += "_roberta_input_encoded"

        if os.path.exists(
            "{}/{}_input0_mapped_tokens.txt".format(data_dir, roberta_input_type)
        ):
            with open(
                "{}/{}_input0_mapped_tokens.txt".format(data_dir, roberta_input_type),
                "r",
            ) as f:
                self.roberta_filter_map = f.read().strip().split("\n")
                self.roberta_filter_map = {
                    x.split()[1]: x.split()[0] for x in self.roberta_filter_map
                }
        else:
            self.roberta_filter_map = None

        with open("{}-bin/label/dict.txt".format(data_dir)) as f:
            author_target_dict = f.read().strip().split("\n")
            author_target_dict = {
                x.split()[0]: i
                for i, x in enumerate(author_target_dict)
                if not x.startswith("madeupword")
            }

        self.author_target_dict = author_target_dict
        self.reverse_author_target_dict = {
            v: k for k, v in self.author_target_dict.items()
        }

        self.global_dense_features = []
        if global_dense_feature_list != "none":
            logger.info(
                "Using global dense vector features = %s" % global_dense_feature_list
            )
            for gdf in global_dense_feature_list.split(","):
                with open(
                    "{}/{}_dense_vectors.pickle".format(data_dir, gdf), "rb"
                ) as f:
                    vector_data = pickle.load(f)

                final_vectors = {}
                for k, v in vector_data.items():
                    final_vectors[self.author_target_dict[k]] = v["sum"] / v["total"]

                self.global_dense_features.append((gdf, final_vectors))

        if os.path.exists(cached_features_file):
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            with open("%s/%s.input0.bpe" % (data_dir, split)) as f:
                author_input_data = f.read().strip().split("\n")

            if os.path.exists("{}/{}.tag_str.bpe".format(data_dir, split)):
                with open("%s/%s.tag_str.bpe" % (data_dir, split)) as f:
                    author_tag_str_data = f.read().split("\n")[:-1]
                with open("%s/%s.tag_ids.bpe" % (data_dir, split)) as f:
                    author_tag_id_data = f.read().split("\n")[:-1]
            else:
                author_tag_str_data = [x for x in author_input_data]
                author_tag_id_data = [x for x in author_input_data]

            with open("%s/%s.label" % (data_dir, split)) as f:
                author_targets = f.read().strip().split("\n")

            if roberta_input_type != "nofilter":
                with open(
                    "%s/%s.%s_input0.bpe" % (data_dir, split, roberta_input_type)
                ) as f:
                    roberta_filtered_data = f.read().strip().split("\n")

            # assert len(author_target_dict) == int(data_dir.split("_")[data_dir.split("_").index("classes") - 1])
            assert len(author_input_data) == len(author_targets)
            assert len(author_input_data) == len(author_tag_str_data)
            assert len(author_input_data) == len(author_tag_id_data)
            assert len(author_input_data) == len(roberta_filtered_data)

            self.examples = []

            for i, (inp, tag_str, tag_ids, author_target) in tqdm.tqdm(
                enumerate(
                    zip(
                        author_input_data,
                        author_tag_str_data,
                        author_tag_id_data,
                        author_targets,
                    )
                ),
                total=len(author_input_data),
            ):
                # check whether a filter needs to be applied before feeding data to RoBERTa
                roberta_sentence = (
                    roberta_filtered_data[i]
                    if roberta_input_type != "nofilter"
                    else inp
                )
                # if RoBERTa is being used, encode sentence using fairseq RoBERTa vocabulary
                if roberta is not None:
                    roberta_sentence = roberta.task.source_dictionary.encode_line(
                        "<s> " + roberta_sentence + " </s>", append_eos=False
                    )

                assert len(tag_ids.split()) == len(tag_str.split())
                self.examples.append(
                    {
                        "roberta_sentence": roberta_sentence,
                        "sentence": [int(x) for x in inp.split()],
                        "author_tag_str": tag_str.split(),
                        "author_tag_ids": tag_ids.split(),
                        "author_target": author_target_dict[author_target],
                    }
                )

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # in case this is training time and only a certain author's data is to be used
        self.examples = limit_authors(self.examples, args.specific_author_train, split, self.reverse_author_target_dict)

        # in case we are using a fraction of the dataset, reduce the size of the dataset here
        self.examples = limit_dataset_size(self.examples, args.limit_examples)

        for eg in self.examples:
            eg["original_target"] = eg["author_target"]

        if args.context_input_type.startswith("fixed_"):
            for eg in self.examples:
                eg["sentence"] = self.examples[args.fixed_example_number]["sentence"]
                eg["author_tag_str"] = self.examples[args.fixed_example_number][
                    "author_tag_str"
                ]
                eg["author_tag_ids"] = self.examples[args.fixed_example_number][
                    "author_tag_ids"
                ]
                eg["roberta_author_target"] = eg["author_target"]
                eg["author_target"] = self.examples[args.fixed_example_number][
                    "author_target"
                ]

        elif args.roberta_input_type.startswith("fixed_"):
            for eg in self.examples:
                eg["roberta_sentence"] = self.examples[args.fixed_example_number][
                    "roberta_sentence"
                ]
                eg["roberta_author_target"] = self.examples[args.fixed_example_number][
                    "author_target"
                ]

        elif args.roberta_input_type.startswith("mixed_"):
            random.seed(args.seed)
            new_order_examples = random.sample(self.examples, len(self.examples))
            for eg, neg in zip(self.examples, new_order_examples):
                eg["roberta_sentence"] = neg["roberta_sentence"]
                eg["roberta_author_target"] = neg["author_target"]

        elif args.context_input_type.startswith("class_fixed_interpolate_"):
            class_number = args.context_input_type.replace(
                "_roberta_input", ""
            ).replace("class_fixed_interpolate_", "")
            for eg in self.examples:
                eg["author_target"] = class_number

        elif args.context_input_type.startswith("class_fixed_"):
            class_number = int(args.context_input_type.split("_")[2])
            for eg in self.examples:
                eg["author_target"] = class_number

        # convert the dataset into the Instance class
        self.examples = [
            HPInstance(args, self.config, datum_dict) for datum_dict in self.examples
        ]

        for instance in tqdm.tqdm(self.examples):
            # perform truncation, padding, label and segment building
            instance.preprocess(tokenizer)

        logger.info(
            "Total truncated instances due to length limit = {:d} / {:d}".format(
                sum([x.truncated for x in self.examples]), len(self.examples)
            )
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sentence = self.examples[item].sentence
        roberta_sentence = self.examples[item].roberta_sentence
        label = self.examples[item].label
        segment = self.examples[item].segment
        author_target = self.examples[item].author_target
        roberta_author_target = self.examples[item].roberta_author_target
        original_author_target = self.examples[item].original_author_target
        init_context_size = self.examples[item].init_context_size

        if len(self.global_dense_features) > 0:
            if self.args.context_input_type.startswith("class_fixed_interpolate_"):
                authors = [
                    (float(x.split("-")[0]), int(x.split("-")[1]))
                    for x in author_target.split("_")
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

            elif self.args.context_input_type.endswith("_no_srl_input"):
                global_dense_vectors = np.array(
                    [x[1][author_target] for x in self.global_dense_features],
                    dtype=np.float32
                )
            else:
                global_dense_vectors = np.array(
                    [x[1][roberta_author_target] for x in self.global_dense_features],
                    dtype=np.float32,
                )
        else:
            global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

        if self.context_noise_fn is not None:
            to_replace = self.context_noise_fn(size=init_context_size)
            random_tokens = np.random.randint(0, 50262, size=init_context_size)
            sentence[:init_context_size] = (
                sentence[:init_context_size] * (1 - to_replace)
                + random_tokens * to_replace
            )

        return {
            "sentence": torch.tensor(sentence),
            "roberta_sentence": torch.tensor(roberta_sentence),
            "instance_number": item,
            "label": torch.tensor(label),
            "segment": torch.tensor(segment),
            "author_target": author_target,
            "roberta_author_target": roberta_author_target,
            "original_author_target": original_author_target,
            "init_context_size": init_context_size,
            "global_dense_vectors": global_dense_vectors,
            "metadata": "author_target = {}, roberta_author_target = {}, original_author_target = {}, style_code = {}".format(
                author_target,
                roberta_author_target,
                original_author_target,
                "author_target"
                if self.args.context_input_type.endswith("_no_srl_input")
                else "author_target",
            ),
        }


class AuthorDatasetText(Dataset):
    def __init__(
        self,
        tokenizer,
        args,
        model_type,
        roberta,
        evaluate=False,
        split="train",
        block_size=512,
    ):
        raise NotImplementedError(
            "This Dataset is moved to gpt2_finetune.legacy.style_dataset"
        )

    def __len__(self):
        raise NotImplementedError(
            "This Dataset is moved to gpt2_finetune.legacy.style_dataset"
        )

    def __getitem__(self, item):
        raise NotImplementedError(
            "This Dataset is moved to gpt2_finetune.legacy.style_dataset"
        )
