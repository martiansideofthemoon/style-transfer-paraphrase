#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from args import get_parser
from data_utils import MAX_ROBERTA_LENGTH
from fairseq.models.roberta import RobertaModel
from style_dataset import (InverseParaphraseDatasetText,
                           ParaphraseDatasetText)
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from utils import bpe_to_srl, class_number_to_str, decode_roberta, init_roberta_gpt2, recall

logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config,)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(args, tokenizer, roberta):
    if args.context_input_type.endswith("_srl_input") or args.context_input_type.endswith("_roberta_input"):
        dataset = InverseParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
            roberta=roberta,
            evaluate=True,
            split=args.eval_split
        )
    elif args.context_input_type.endswith("_paraphrase") or args.context_input_type.endswith("_simplewiki"):
        dataset = ParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
            roberta=roberta,
            evaluate=True,
            split=args.eval_split
        )
    return dataset


def main():
    parser = get_parser("generation", MODEL_CLASSES, ALL_MODELS)
    args = parser.parse_args()

    if args.local_rank == -1:
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1))

    set_seed(args)

    # Load a pretrained style-disentangling RoBERTa model, which will be used to create the features
    if args.content_aggregation > MAX_ROBERTA_LENGTH and args.roberta_weights == "fixed":
        roberta = None
    else:
        roberta = RobertaModel.from_pretrained(
            args.roberta_pretrained,
            checkpoint_file=args.roberta_ckpt_file,
            data_name_or_path=args.data_dir + "-bin"
        )
        roberta.cuda()
        roberta.eval()

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    roberta_gpt2, tokenizer = init_roberta_gpt2(roberta=roberta,
                                                checkpoint_dir=args.model_name_or_path,
                                                args=args,
                                                model_class=model_class,
                                                tokenizer_class=tokenizer_class,
                                                evaluation=True)
    roberta_gpt2.eval()

    if args.local_rank == 0:
        torch.distributed.barrier()

    config = roberta_gpt2.gpt2.config

    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(model,
    #                                                       device_ids=[args.local_rank],
    #                                                       output_device=args.local_rank,
    #                                                       find_unused_parameters=True)

    if args.length < 0 and config.max_position_embeddings > 0:
        args.length = config.max_position_embeddings
    elif 0 < config.max_position_embeddings < args.length:
        args.length = config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    eval_dataset = load_and_cache_examples(args, tokenizer, roberta)

    if args.local_rank == 0:
        torch.distributed.barrier()

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

    output_log = {
        "true_text": [],
        "generated_text": [],
        "context": [],
        "recall_score": [],
        "roberta_text": [],
        "context_author_targets": [],
        "roberta_author_targets": [],
        "original_author_targets": [],
        "metadata": []
    }

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        roberta_sentences = batch["roberta_sentence"].to(args.device)
        sentences = batch["sentence"].to(args.device)
        segments = batch["segment"].to(args.device)
        global_dense_vectors = batch["global_dense_vectors"].to(args.device)
        author_targets = batch["author_target"]
        roberta_author_targets = batch["roberta_author_target"]
        original_author_targets = batch["original_author_target"]
        metadata = batch["metadata"]

        # Assume init_context_size is same for all examples in minibatch
        init_context_size = batch["init_context_size"][0].item()

        out, dense_length, scores = roberta_gpt2.generate(roberta_sentences=roberta_sentences,
                                                          gpt2_sentences=sentences,
                                                          segments=segments,
                                                          eos_token_id=tokenizer.eos_token_id,
                                                          global_dense_vectors=global_dense_vectors,
                                                          init_context_size=init_context_size)

        for sent_num in range(sentences.shape[0]):
            output_sequence = out[sent_num][init_context_size:].tolist()

            if tokenizer.eos_token_id in output_sequence:
                output_sequence = output_sequence[:output_sequence.index(tokenizer.eos_token_id)]

            true_text = tokenizer.decode(sentences[sent_num, init_context_size:].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
            generated_text = tokenizer.decode(output_sequence, clean_up_tokenization_spaces=True, skip_special_tokens=True)

            if roberta is None:
                roberta_text = "<none>"
            else:
                roberta_text = decode_roberta(roberta, eval_dataset.roberta_filter_map, roberta_sentences[sent_num])

            if args.context_input_type.endswith("_srl_input"):
                context = bpe_to_srl(sentences[sent_num, :init_context_size].tolist(), segments[sent_num, dense_length:dense_length + init_context_size].tolist(), tokenizer)
                recall_score = recall(true_text, context)
            elif args.context_input_type.endswith("_paraphrase") or args.context_input_type.endswith("_roberta_input") or args.context_input_type.endswith("_simplewiki"):
                context = tokenizer.decode(sentences[sent_num, :init_context_size].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
                recall_score = recall(true_text, context)
            else:
                context = ""
                recall_score = 0.0
            # print("True Text = %s" % true_text)
            # print("Generated Text = %s" % generated_text)

            output_log["true_text"].append(true_text)
            output_log["generated_text"].append(generated_text)
            output_log["roberta_text"].append(roberta_text)
            output_log["context"].append(context)
            output_log["recall_score"].append("%.4f" % recall_score)
            output_log["metadata"].append(metadata[sent_num])

            if hasattr(eval_dataset, "reverse_author_target_dict"):
                output_log["context_author_targets"].append(
                    class_number_to_str(eval_dataset, author_targets[sent_num])
                )
                output_log["roberta_author_targets"].append(
                    class_number_to_str(eval_dataset, roberta_author_targets[sent_num])
                )
                output_log["original_author_targets"].append(
                    class_number_to_str(eval_dataset, original_author_targets[sent_num])
                )
            else:
                output_log["context_author_targets"].append("<none>")
                output_log["roberta_author_targets"].append("<none>")
                output_log["original_author_targets"].append("<none>")

    with open(os.path.join(args.generation_output_dir, "reference_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["true_text"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "generated_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["generated_text"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "roberta_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["roberta_text"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "context_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["context"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "recall_score_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["recall_score"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "context_author_targets_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["context_author_targets"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "roberta_author_targets_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["roberta_author_targets"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "original_author_targets_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["original_author_targets"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "metadata_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["metadata"]) + "\n")


if __name__ == '__main__':
    main()
