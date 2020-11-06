#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Kalpesh Krishna.
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
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)."""
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
from style_dataset import (InverseParaphraseDatasetText,
                           ParaphraseDatasetText)
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from utils import class_number_to_str, init_gpt2_model, recall

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


def load_and_cache_examples(args, tokenizer):
    if not args.prefix_input_type.startswith("original"):
        dataset = InverseParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
            evaluate=True,
            split=args.eval_split
        )
    else:
        dataset = ParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
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

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    gpt2_model, tokenizer = init_gpt2_model(checkpoint_dir=args.model_name_or_path,
                                            args=args,
                                            model_class=model_class,
                                            tokenizer_class=tokenizer_class)
    gpt2_model.eval()

    if args.local_rank == 0:
        torch.distributed.barrier()

    config = gpt2_model.gpt2.config

    if args.length < 0 and config.max_position_embeddings > 0:
        args.length = config.max_position_embeddings
    elif 0 < config.max_position_embeddings < args.length:
        args.length = config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

    eval_dataset = load_and_cache_examples(args, tokenizer)

    if args.local_rank == 0:
        torch.distributed.barrier()

    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

    output_log = {
        "true_text": [],
        "generated_text": [],
        "context": [],
        "recall_score": [],
        "context_suffix_styles": [],
        "original_styles": [],
        "metadata": []
    }

    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        sentences = batch["sentence"].to(args.device)
        segments = batch["segment"].to(args.device)
        global_dense_vectors = batch["global_dense_vectors"].to(args.device)
        suffix_styles = batch["suffix_style"]
        original_styles = batch["original_style"]
        metadata = batch["metadata"]

        # Assume init_context_size is same for all examples in minibatch
        init_context_size = batch["init_context_size"][0].item()

        out, dense_length, scores = gpt2_model.generate(gpt2_sentences=sentences,
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

            context = tokenizer.decode(sentences[sent_num, :init_context_size].tolist(), clean_up_tokenization_spaces=True, skip_special_tokens=True)
            recall_score = recall(true_text, context)

            output_log["true_text"].append(true_text)
            output_log["generated_text"].append(generated_text)
            output_log["context"].append(context)
            output_log["recall_score"].append("%.4f" % recall_score)
            output_log["metadata"].append(metadata[sent_num])

            if hasattr(eval_dataset, "reverse_label_dict"):
                output_log["context_suffix_styles"].append(
                    class_number_to_str(eval_dataset, suffix_styles[sent_num])
                )
                output_log["original_styles"].append(
                    class_number_to_str(eval_dataset, original_styles[sent_num])
                )
            else:
                output_log["context_suffix_styles"].append("<none>")
                output_log["original_styles"].append("<none>")

    with open(os.path.join(args.generation_output_dir, "reference_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["true_text"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "generated_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["generated_text"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "context_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["context"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "recall_score_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["recall_score"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "context_suffix_styles_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["context_suffix_styles"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "original_styles_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["original_styles"]) + "\n")

    with open(os.path.join(args.generation_output_dir, "metadata_%d.txt" % max(args.local_rank, 0)), "w") as f:
        f.write("\n".join(output_log["metadata"]) + "\n")


if __name__ == '__main__':
    main()
