# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import re
import shutil
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from args import get_parser
from data_utils import MAX_ROBERTA_LENGTH
from fairseq.models.roberta import RobertaModel
from fairseq.optim.adafactor import Adafactor
from style_dataset import (InverseParaphraseDatasetText,
                           ParaphraseDatasetText)
from transformers import (WEIGHTS_NAME, AdamW, GPT2Config, GPT2LMHeadModel,
                          GPT2Tokenizer, get_linear_schedule_with_warmup)

from utils import RobertaToGPT2, get_new_update_type, init_roberta_gpt2

try:
    from torch.utils.tensorboard import SummaryWriter
except:
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


def load_and_cache_examples(args, tokenizer, roberta, evaluate=False):
    if args.context_input_type.endswith("_srl_input") or args.context_input_type.endswith("_roberta_input"):
        dataset = InverseParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
            roberta=roberta,
            evaluate=evaluate,
            split="dev" if evaluate else "train"
        )
    elif args.context_input_type.endswith("_paraphrase") or args.context_input_type.endswith("_simplewiki"):
        dataset = ParaphraseDatasetText(
            tokenizer=tokenizer,
            args=args,
            model_type=args.model_type,
            roberta=roberta,
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


def save_model(roberta_gpt2, output_dir, args, tokenizer=None):
    # Take care of distributed/parallel training
    if roberta_gpt2.roberta_training:
        model_to_save = roberta_gpt2.module if hasattr(roberta_gpt2, 'module') else roberta_gpt2
        model_to_save = model_to_save.gpt2
    else:
        model_to_save = roberta_gpt2.gpt2
        model_to_save = model_to_save.module if hasattr(model_to_save, 'module') else model_to_save

    model_to_save.save_pretrained(output_dir)
    logger.info("Saving model checkpoint to %s", output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))

    if tokenizer:
        tokenizer.save_pretrained(output_dir)

    if roberta_gpt2.roberta_training:
        # save the RoBERTa weights in the same directory as well
        state_dict = {
            "args": roberta_gpt2.roberta_extractor.roberta.args,
            "model": roberta_gpt2.roberta_extractor.roberta.model.state_dict(),
            # For compatability with fairseq
            "best_loss": 0,
            "epoch": 10,
            "optimizer": "AdamW",
            "batch_offset": 0,
            "val_loss": 0
        }
        torch.save(state_dict, os.path.join(output_dir, 'roberta.pt'))
        logger.info("RoBERTa weights saved to %s" % os.path.join(output_dir, 'roberta.pt'))


def train(args, roberta_gpt2, train_dataset, tokenizer):
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
    if roberta_gpt2.roberta_training:
        model = roberta_gpt2
    else:
        model = roberta_gpt2.gpt2

    # Prepare optimizer and schedule (linear warmup and decay)
    # extra layer_norm.weight for com
    no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']

    if not args.context_input_type.endswith("_none"):
        def parameter_filter(x):
            return True
    else:
        def parameter_filter(x):
            lambda x: "style_from_content" not in x

    is_adversarial_training = roberta_gpt2.roberta_training and args.context_input_type.endswith("_none") and args.switch_type != "constant"

    generator_grouped_parameters = [
        {
            'params': [
                p for n, p in model.named_parameters()
                if parameter_filter(n) and (not any(nd in n for nd in no_decay))
            ],
            'weight_decay': args.weight_decay
        },
        {
            'params': [
                p for n, p in model.named_parameters()
                if parameter_filter(n) and (any(nd in n for nd in no_decay))
            ],
            'weight_decay': 0.0
        }
    ]

    if args.optimizer == "adam":
        optimizer = {
            "generator": AdamW(generator_grouped_parameters, lr=float(args.learning_rate.split(",")[0]), eps=args.adam_epsilon)
        }
        scheduler = {
            "generator": get_linear_schedule_with_warmup(optimizer["generator"],
                                                        num_warmup_steps=args.warmup_steps // 2 if is_adversarial_training else args.warmup_steps,
                                                        num_training_steps=t_total // 2 if is_adversarial_training else t_total)
        }
    elif args.optimizer == "adafactor":
        optimizer = {
            "generator": Adafactor(generator_grouped_parameters, warmup_init=True)
        }
        scheduler = None
    elif args.optimizer == "adafactor-external":
        optimizer = {
            "generator": Adafactor(
                generator_grouped_parameters,
                lr=float(args.learning_rate.split(",")[0])
            )
        }
        scheduler = None
    elif args.optimizer == "adafactor-external2":
        optimizer = {
            "generator": Adafactor(
                generator_grouped_parameters,
                lr=float(args.learning_rate.split(",")[0]),
                warmup_init=True
            )
        }
        scheduler = None
        # scheduler = {
        #     "generator": get_linear_schedule_with_warmup(optimizer["generator"],
        #                                                 num_warmup_steps=args.warmup_steps // 2 if is_adversarial_training else args.warmup_steps,
        #                                                 num_training_steps=t_total // 2 if is_adversarial_training else t_total)
        # }

    if is_adversarial_training:
        # Setup new optimizer and scheduler here
        discriminator_parameters = [p for n, p in model.named_parameters() if "style_from_content" in n]
        optimizer["discriminator"] = AdamW(discriminator_parameters, lr=float(args.learning_rate.split(",")[1]), eps=args.adam_epsilon)
        scheduler["discriminator"] = get_linear_schedule_with_warmup(optimizer["discriminator"],
                                                                     num_warmup_steps=args.warmup_steps // 2,
                                                                     num_training_steps=t_total // 2)

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

    # this is necessary to ensure multi-GPU training happens since the roberta_gpt2.gpt2 pointer has been set to the model without the DDP wrapper
    if not roberta_gpt2.roberta_training:
        roberta_gpt2.gpt2 = model

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
        "lm": {"current": 0.0, "previous": 0.0},
        "style_from_style": {"current": 0.0, "previous": 0.0},
        "style_from_content_generator": {"current": 0.0, "previous": 0.0},
        "style_from_content_discriminator": {"current": 0.0, "previous": 0.0},
        "total": {"current": 0.0, "previous": 0.0}
    }
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)

    update_type = "generator"

    generator_loss_constants = args.generator_loss_constants.split(",")
    generator_loss_constants = [float(x) for x in generator_loss_constants]

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            loss = roberta_gpt2(batch, update_type=update_type)

            if roberta_gpt2.roberta_training and args.context_input_type.endswith("_none"):
                if update_type == "generator":
                    loss["total"] = \
                        generator_loss_constants[0] * loss["lm"] + \
                        generator_loss_constants[1] * loss["style_from_style"] + \
                        generator_loss_constants[2] * loss["style_from_content_generator"]
                else:
                    loss["total"] = loss["style_from_content_discriminator"]
            elif roberta_gpt2.roberta_training and not args.context_input_type.endswith("_none"):
                loss["total"] = \
                    generator_loss_constants[0] * loss["lm"] + \
                    generator_loss_constants[1] * loss["style_from_content_discriminator"]
            else:
                loss["total"] = loss["lm"]

            if args.n_gpu > 1:
                for k, v in loss.items():
                    loss[k] = v.mean()

            if args.gradient_accumulation_steps > 1:
                for k, v in loss.items():
                    loss[k] = v / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss["total"], optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss["total"].backward()

            # Update the metrics for Tensorboard logging
            for metric_type, metric_vals in loss_metrics.items():
                metric_vals["current"] += loss[metric_type].item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Update the generator or the discriminator optimizer
                optimizer[update_type].step()
                if scheduler:
                    scheduler[update_type].step()

                model.zero_grad()
                global_step += 1

                # Check if there is a modification for the update_type, generator --> discriminator
                update_type = get_new_update_type(args, global_step, update_type)

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, roberta_gpt2, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)

                    if scheduler:
                        tb_writer.add_scalar('lr_generator', scheduler["generator"].get_lr()[0], global_step)
                    else:
                        pass

                    if is_adversarial_training:
                        tb_writer.add_scalar('lr_discriminator', scheduler["discriminator"].get_lr()[0], global_step)

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

                    save_model(roberta_gpt2, output_dir, args, tokenizer=tokenizer)

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


def evaluate(args, roberta_gpt2, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, roberta=roberta_gpt2.roberta_extractor.roberta, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        roberta_gpt2 = torch.nn.DataParallel(roberta_gpt2)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    total_correct_style = 0
    total_instances = 0

    total_correct_content = 0
    total_token_instances = 0

    roberta_gpt2.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        curr_loss, style_from_style_info, style_from_content_info = roberta_gpt2.evaluate(batch)
        eval_loss += curr_loss

        total_correct_style += style_from_style_info["correct"]
        total_instances += batch["author_target"].shape[0]

        total_correct_content += style_from_content_info["discriminator_correct"]
        total_token_instances += style_from_content_info["discriminator_total"]

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": perplexity
    }

    result["accuracy"] = "%.4f, (%d / %d)" % (float(total_correct_style) / float(total_instances), total_correct_style, total_instances)
    result["accuracy_tokens"] = "%.4f, (%d / %d)" % (float(total_correct_content) / float(total_token_instances), total_correct_content, total_token_instances)

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    # Adding an extra embedding dimension for style/content vectors
    config.extra_embedding_dim = args.extra_embedding_dim
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    roberta_gpt2 = RobertaToGPT2(args=args, roberta=roberta, gpt2=model)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, roberta=roberta, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, roberta_gpt2, train_dataset, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        save_model(roberta_gpt2, args.output_dir, args, tokenizer)
        roberta_gpt2, tokenizer = init_roberta_gpt2(roberta=roberta_gpt2.roberta_extractor.roberta,
                                                    checkpoint_dir=args.output_dir,
                                                    args=args,
                                                    model_class=model_class,
                                                    tokenizer_class=tokenizer_class,
                                                    evaluation=True)

    # Evaluation
    results = []
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = []
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/checkpoint-*/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            # Sort checkpoints according to the step number
            if len(checkpoints) > 0:
                checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

        if os.path.exists("{}/pytorch_model.bin".format(args.output_dir)):
            checkpoints.append(args.output_dir)

        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            roberta_gpt2, _ = init_roberta_gpt2(roberta=roberta_gpt2.roberta_extractor.roberta,
                                                checkpoint_dir=checkpoint,
                                                args=args,
                                                model_class=model_class,
                                                evaluation=True)

            result = evaluate(args, roberta_gpt2, tokenizer, prefix=prefix)
            results.append((checkpoint, result))

        results.sort(key=lambda x: x[1]["perplexity"].item())

        if args.do_train or args.do_delete_old:
            logger.info("Deleting other checkpoints...")
            # delete all but the top-3 checkpoints
            for res in results[3:]:
                if res[0].split("/")[-1].startswith("checkpoint"):
                    logger.info("Deleting {}...".format(res[0]))
                    shutil.rmtree(res[0])
            # move top checkpoint to root directory
            if results[0][0].split("/")[-1].startswith("checkpoint"):
                command = "mv {}/* {}".format(results[0][0], args.output_dir)
                logger.info("executing {}...".format(command))
                subprocess.check_output(command, shell=True)
                command = "rm -rf {}".format(results[0][0])
                logger.info("executing {}...".format(command))
                subprocess.check_output(command, shell=True)

        logger.info(
            "Top checkpoints:\n{}".format(
                "\n".join(["{:.4f} = {}".format(res[1]['perplexity'].item(), res[0].split("/")[-1]) for res in results])
            )
        )

    return results


if __name__ == "__main__":
    main()
