# coding=utf-8
# Copyright 2020 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Fine-tuning GPT2 for conditional generation tasks."""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import re
import shutil
import subprocess

import numpy as np
import time
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from collections import defaultdict

from args import get_parser
from data_utils import MAX_ROBERTA_LENGTH
from style_dataset import (InverseParaphraseDatasetText,
                           ParaphraseDatasetText)
from transformers import (WEIGHTS_NAME, AdamW, GPT2Config, GPT2LMHeadModel,
                          GPT2Tokenizer, get_linear_schedule_with_warmup)

from utils import GPT2ParentModule, init_gpt2_model

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}

SPECIAL_TOKENS = {
    "additional_special_tokens": ["<dense-vectors>", "<tokens>", "<verb>", "<ARG0>", "<ARG1>", "<global-dense-vectors>"],
    "pad_token": "<pad>",
    "bos_token": "<bos>",
    "eos_token": "<eos>"
}


def load_and_cache_examples(args, tokenizer, evaluate=False):
    if not args.prefix_input_type.startswith("original"):
        dataset = InverseParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            evaluate=evaluate,
            split="dev" if evaluate else "train"
        )
    else:
        dataset = ParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            evaluate=evaluate,
            split="dev" if evaluate else "train"
        )
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def save_model(gpt2_model, output_dir, args, global_step, tokenizer=None):
    # Take care of distributed/parallel training
    model_to_save = gpt2_model.gpt2
    model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save

    model_to_save.save_pretrained(output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    with open(os.path.join(output_dir, "global_step.txt"), "w") as f:
        f.write(str(global_step) + "\n")

    if tokenizer:
        tokenizer.save_pretrained(output_dir)


def train(args, gpt2_model, train_dataset, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        try:
            tb_writer = SummaryWriter(logdir="runs/summary_%s" % args.job_id)
        except:
            tb_writer = SummaryWriter(log_dir="runs/summary_%s" % args.job_id)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Update the model definition in case RoBERTa is training
    model = gpt2_model.gpt2

    # Prepare optimizer and schedule (linear warmup and decay)
    # extra layer_norm.weight for com
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        }
    ]

    optimizer = AdamW(grouped_parameters,
                      lr=float(args.learning_rate),
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # this is necessary to ensure multi-GPU training happens since the gpt2_model.gpt2 pointer has been set to the model without the DDP wrapper
    gpt2_model.gpt2 = model

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    loss_metrics = {
        "lm": {"current": 0.0, "previous": 0.0}
    }
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            loss = gpt2_model(batch)

            if args.n_gpu > 1:
                for k, v in loss.items():
                    loss[k] = v.mean()

            if args.gradient_accumulation_steps > 1:
                for k, v in loss.items():
                    loss[k] = v / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss["lm"], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss["lm"].backward()

            # Update the metrics for Tensorboard logging
            for metric_type, metric_vals in loss_metrics.items():
                metric_vals["current"] += loss[metric_type].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update the generator or the discriminator optimizer
                optimizer.step()
                scheduler.step()

                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, gpt2_model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    tb_writer.add_scalar('learning_rate', scheduler.get_lr()[0], global_step)

                    for metric_type, metric_vals in loss_metrics.items():
                        tb_writer.add_scalar(
                            '%s_loss' % metric_type,
                            (metric_vals["current"] - metric_vals["previous"]) / args.logging_steps,
                            global_step
                        )
                        metric_vals["previous"] = metric_vals["current"]

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    save_model(gpt2_model, output_dir, args, global_step, tokenizer=tokenizer)

                    _rotate_checkpoints(args, checkpoint_prefix)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, loss_metrics["lm"]["current"] / global_step


def evaluate(args, gpt2_model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        gpt2_model = torch.nn.DataParallel(gpt2_model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    total_instances = 0

    gpt2_model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        curr_loss = gpt2_model.evaluate(batch)
        eval_loss += curr_loss
        total_instances += batch["suffix_style"].shape[0]
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = get_parser("finetuning")
    args = parser.parse_args()

    if (os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Barrier to make sure only the first process in distributed training download model & vocab
        torch.distributed.barrier()

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    # Adding an extra embedding dimension for style/content vectors
    config.extra_embedding_dim = args.extra_embedding_dim
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    gpt2_model = GPT2ParentModule(args=args, gpt2=model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, gpt2_model, train_dataset, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)
        save_model(gpt2_model, output_dir, args, global_step, tokenizer)

        gpt2_model, tokenizer = init_gpt2_model(checkpoint_dir=args.output_dir,
                                                args=args,
                                                model_class=model_class,
                                                tokenizer_class=tokenizer_class)

    # Evaluation
    if args.do_eval and args.local_rank in [-1, 0]:
        eval_done = False
        all_results = {}
        top_checkpoint = None
        patience = 0

        while not eval_done:
            checkpoints = []
            if not args.evaluate_specific:
                checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/checkpoint-*/' + WEIGHTS_NAME, recursive=True)))
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
                # Sort checkpoints according to the step number
                if len(checkpoints) > 0:
                    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            else:
                checkpoints.append(args.evaluate_specific)

            checkpoints = [x for x in checkpoints if x not in all_results]

            # Count the number of while loop iterations no new checkpoints were found
            if len(checkpoints) == 0:
                patience += 1
            else:
                patience = 0

            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            for checkpoint in checkpoints:
                prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

                gpt2_model, _ = init_gpt2_model(checkpoint_dir=checkpoint,
                                                args=args,
                                                model_class=model_class)

                result = evaluate(args, gpt2_model, tokenizer, prefix=prefix)
                all_results[checkpoint] = result["perplexity"]

            sorted_results = [(k, v) for k, v in all_results.items()]
            sorted_results.sort(key=lambda x: x[1].item())

            if not args.evaluate_specific and args.do_delete_old and len(sorted_results) > args.save_total_limit:
                logger.info("Deleting worse checkpoints...")
                # delete all but the top save_total_limit checkpoints
                for res in sorted_results[args.save_total_limit:]:
                    if os.path.exists(res[0]):
                        logger.info("Deleting {}...".format(res[0]))
                        shutil.rmtree(res[0])

            # move top checkpoint to root directory
            if not args.evaluate_specific and len(sorted_results) > 0 and sorted_results[0][0] != top_checkpoint:
                command = "cp {}/* {}".format(sorted_results[0][0], args.output_dir)
                logger.info("executing {}...".format(command))
                subprocess.check_output(command, shell=True)
                top_checkpoint = sorted_results[0][0]

            sorted_results_summary = "\n".join(["{} = {:.4f}".format(x[0], x[1]) for x in sorted_results])
            logger.info("Top checkpoints:\n{}".format(sorted_results_summary))

            if args.eval_frequency_min == 0 or args.evaluate_specific or patience > args.eval_patience:
                eval_done = True
            else:
                logger.info("Sleeping for {:d} minutes...zzzz...".format(args.eval_frequency_min))
                time.sleep(args.eval_frequency_min * 60)

    return all_results


if __name__ == "__main__":
    main()
