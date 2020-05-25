import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from fairseq.models.roberta import RobertaModel
from collections import namedtuple

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from functools import partial

from gpt2_finetune.dataset_config import DATASET_CONFIG
from gpt2_finetune.data_utils import update_config, Instance

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}


logger = logging.getLogger(__name__)


def class_number_to_str(eval_dataset, class_number):
    if isinstance(class_number, str):
        return ", ".join(["{} {}".format(x.split("-")[0], x.split("-")[1]) for x in class_number.split("_")])
    else:
        return eval_dataset.reverse_author_target_dict[class_number.item()]

def recall(sentence, srl_string):
    matches = 0
    for word in sentence.split():
        if word in srl_string:
            matches += 1

    if len(sentence.split()) > 0:
        return float(matches) / len(sentence.split())
    else:
        return 0


def decode_roberta(roberta, roberta_filter_map, sentence):
    bpe_sequence = roberta.task.source_dictionary.string(sentence).replace("<pad>", "")

    if "<unk>" in bpe_sequence:
        bpe_sequence = bpe_sequence.replace(" <unk>", " " + roberta.bpe.encode(" <unk>").strip())
        bpe_sequence = bpe_sequence.replace("<unk>", roberta.bpe.encode("<unk>").strip())

    sentence = roberta.bpe.decode(bpe_sequence)

    if roberta_filter_map is not None:
        sentence = " ".join([roberta_filter_map.get(token, token) for token in sentence.split()])
    return sentence


def rindex(mylist, myvalue):
    return len(mylist) - mylist[::-1].index(myvalue) - 1


def bpe_to_srl(tag_str, tag_ids, tokenizer):
    assert len(tag_ids) == len(tag_str)

    curr_tag_id = tag_ids[0]
    output_str = ""
    curr_tag_str = []

    for tid, ts in zip(tag_ids, tag_str):
        if tid != curr_tag_id:
            output_str += " " + tokenizer.decode(curr_tag_id) + " " + tokenizer.decode(curr_tag_str)
            curr_tag_id = tid
            curr_tag_str = [ts]
        else:
            curr_tag_str.append(ts)
    return " ".join(output_str.split())


def get_new_update_type(args, global_step, update_type):
    switch_type = args.switch_type

    if switch_type == "constant":
        return "generator"
    elif switch_type.startswith("every_"):
        frequency = int(switch_type[switch_type.index("_") + 1:])

        # Switch the update type if it's the correct moment to do so
        if global_step % frequency == 0:
            new_update_type = "generator" if update_type == "discriminator" else "discriminator"
            return new_update_type
        else:
            return update_type

    else:
        raise ValueError("Invalid value for args.switch_type")


def init_roberta_gpt2(roberta, checkpoint_dir, args, model_class, tokenizer_class=None, evaluation=True):
    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(checkpoint_dir)
    model.to(args.device)

    if tokenizer_class:
        tokenizer = tokenizer_class.from_pretrained(checkpoint_dir, do_lower_case=args.do_lower_case)
    else:
        tokenizer = None

    # Reload the new RoBERTa checkpoint in case RoBERTa weights were not fixed
    if args.roberta_weights != "fixed":
        logger.info("Reloading new RoBERTa checkpoint in %s" % checkpoint_dir)
        roberta = RobertaModel.from_pretrained(
            checkpoint_dir,
            checkpoint_file="roberta.pt",
            data_name_or_path=args.data_dir + "-bin"
        )
        roberta.cuda()
        roberta.eval()

    return RobertaToGPT2(args=args, roberta=roberta, gpt2=model, evaluation=evaluation), tokenizer


class GPT2Generator(object):
    def __init__(self, model_path, upper_length="eos", beam_size=1, top_p=0.0):
        model_class, tokenizer_class = MODEL_CLASSES["gpt2"]
        self.model_path = model_path
        self.args = torch.load("{}/training_args.bin".format(self.model_path))
        self.modify_args(upper_length, beam_size, top_p)
        self.config = DATASET_CONFIG[self.args.data_dir]
        update_config(self.args, self.config)

        if self.args.global_dense_feature_list != "none" and not self.args.context_input_type.endswith("_paraphrase"):
            with open("{}-bin/label/dict.txt".format(self.args.data_dir)) as f:
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
            for gdf in self.args.global_dense_feature_list.split(","):
                with open(
                    "{}/{}_dense_vectors.pickle".format(self.args.data_dir, gdf), "rb"
                ) as f:
                    vector_data = pickle.load(f)

                final_vectors = {}
                for k, v in vector_data.items():
                    final_vectors[self.author_target_dict[k]] = v["sum"] / v["total"]

                self.global_dense_features.append((gdf, final_vectors))

        self.roberta_gpt2, self.tokenizer = init_roberta_gpt2(roberta=None,
                                                              checkpoint_dir=model_path,
                                                              args=self.args,
                                                              model_class=model_class,
                                                              tokenizer_class=tokenizer_class,
                                                              evaluation=True)

    def modify_args(self, upper_length, beam_size, top_p):
        args = self.args
        args.upper_length = upper_length
        args.roberta_weights = "fixed"
        args.stop_token = "eos" if upper_length == "eos" else None
        args.beam_size = beam_size
        args.num_samples = 1
        args.temperature = 0
        args.top_p = top_p
        args.top_k = 1
        args.device = torch.cuda.current_device()

    def generate_batch(self, contexts, global_dense_features=None, get_scores=False, interpolation=None):
        args = self.args
        tokenizer = self.tokenizer
        instances = []

        if global_dense_features is None:
            global_dense_features = [None for _ in contexts]

        for context, gdf in zip(contexts, global_dense_features):
            context_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(context))

            # NOTE - For model_110, use the older version of the code
            # The following code is only compatible with the newer models
            instance = Instance(
                self.args, self.config,
                {"sent1_tokens": context_ids, "sent2_tokens": context_ids}
            )
            instance.preprocess(tokenizer)

            if gdf is not None and self.args.global_dense_feature_list != "none":
                if self.global_dense_features:
                    global_dense_vectors = np.array(
                        [x[1][gdf] for x in self.global_dense_features],
                        dtype=np.float32,
                    )
                else:
                    global_dense_vectors = np.zeros((2, 20), dtype=np.float32)
                    global_dense_vectors[0, gdf["f1_bucket"]] = 1
                    global_dense_vectors[1, gdf["ed_bucket"] + 10] = 1
            else:
                global_dense_vectors = np.zeros((1, 768), dtype=np.float32)

            instance.gdv = global_dense_vectors
            instances.append(instance)

        output, _, scores = self.roberta_gpt2.generate(
            roberta_sentences=torch.tensor([np.zeros((1, 512), dtype=np.float32)]).to(args.device),
            gpt2_sentences=torch.tensor([inst.sentence for inst in instances]).to(args.device),
            segments=torch.tensor([inst.segment for inst in instances]).to(args.device),
            global_dense_vectors=torch.tensor([inst.gdv for inst in instances]).to(args.device),
            init_context_size=instances[0].init_context_size,
            eos_token_id=tokenizer.eos_token_id,
            get_scores=get_scores,
            interpolation=interpolation
        )

        all_output = []
        for out_num in range(len(output)):
            instance = instances[out_num]
            curr_out = output[out_num, instance.init_context_size:].tolist()

            if tokenizer.eos_token_id in curr_out:
                curr_out = curr_out[:curr_out.index(tokenizer.eos_token_id)]

            if self.args.upper_length.startswith("same"):
                extra = int(self.args.upper_length.split("_")[-1])
                curr_out = curr_out[:len(instance.sent1_tokens) + extra]

            all_output.append(
                tokenizer.decode(curr_out, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            )

        return all_output, scores

    def generate(self, context, global_dense_features=None, get_scores=False, interpolation=None):
        return self.generate_batch([context],
                                   [global_dense_features],
                                   get_scores=get_scores,
                                   interpolation=interpolation)[0]

class RobertaToGPT2(nn.Module):
    def __init__(self, args, roberta, gpt2, evaluation=False):
        super(RobertaToGPT2, self).__init__()
        self.args = args
        self.roberta_extractor = RobertaExtractor(args, roberta, evaluation)
        self.roberta_training = False

        if args.roberta_weights != "fixed":
            logger.info("RoBERTa weights will be finetuned during training")
            self.roberta_training = True
            self.roberta_extractor.setup_roberta_training()

        self.gpt2 = gpt2

    def forward(self, batch, update_type="generator"):
        args = self.args
        gpt2 = self.gpt2
        roberta_extractor = self.roberta_extractor

        roberta_sentences = batch["roberta_sentence"].to(args.device)
        sentences = batch["sentence"].to(args.device)
        labels = batch["label"].to(args.device)
        segments = batch["segment"].to(args.device)
        author_targets = batch["author_target"].to(args.device)
        global_dense_vectors = batch["global_dense_vectors"].to(args.device)

        roberta_outputs = roberta_extractor.get_style_content(
            roberta_sentences, global_dense_vectors, author_targets, update_type
        )

        gpt2.train()
        with torch.set_grad_enabled(update_type == "generator"):
            outputs = gpt2(
                input_ids=sentences,
                token_type_ids=segments,
                labels=labels,
                prefix_input_vectors=roberta_outputs[0]
            )

        loss = {
            "lm": outputs[0],
            "style_from_style": roberta_outputs[1]["loss"],
            "style_from_content_generator": roberta_outputs[2]["generator_loss"],
            "style_from_content_discriminator": roberta_outputs[2]["discriminator_loss"],
        }

        # to see if gradients are flowing, check this
        # torch.autograd.grad(outputs[0], roberta_extractor.roberta.model.decoder.sentence_encoder.layers[0].fc1.weight)

        return loss

    def evaluate(self, batch):
        args = self.args
        gpt2 = self.gpt2
        roberta_extractor = self.roberta_extractor

        roberta_sentences = batch["roberta_sentence"].to(args.device)
        sentences = batch["sentence"].to(args.device)
        labels = batch["label"].to(args.device)
        segments = batch["segment"].to(args.device)
        author_targets = batch["author_target"].to(args.device)
        global_dense_vectors = batch["global_dense_vectors"].to(args.device)

        roberta_outputs = roberta_extractor.get_style_content(roberta_sentences, global_dense_vectors, author_targets)

        with torch.no_grad():
            outputs = gpt2(
                input_ids=sentences,
                token_type_ids=segments,
                labels=labels,
                prefix_input_vectors=roberta_outputs[0]
            )
            lm_loss = outputs[0]

        return lm_loss.mean().item(), roberta_outputs[1], roberta_outputs[2]

    def generate(self, roberta_sentences, gpt2_sentences, segments, global_dense_vectors=None,
                 init_context_size=1, eos_token_id=None, get_scores=False, interpolation=None):
        args = self.args
        gpt2 = self.gpt2
        roberta_extractor = self.roberta_extractor

        style_content_vectors, _, _ = roberta_extractor.get_style_content(roberta_sentences, global_dense_vectors)

        generation_length = None if self.args.stop_token == "eos" else len(gpt2_sentences[0]) - init_context_size
        dense_length = 0 if style_content_vectors is None else len(style_content_vectors[0])

        if args.beam_size > 1:
            out, scores = beam_search(
                model=gpt2,
                length=generation_length,
                context=gpt2_sentences[:, 0:init_context_size],
                style_content_vectors=style_content_vectors,  # mixed_style_content,
                segments=segments[:, 0:dense_length + init_context_size],
                eos_token_id=eos_token_id,
                beam_size=args.beam_size,
                beam_search_scoring=args.beam_search_scoring
            )
        else:
            out, scores = sample_sequence(
                model=gpt2,
                context=gpt2_sentences[:, 0:init_context_size],
                style_content_vectors=style_content_vectors,  # mixed_style_content,
                segments=segments[:, 0:dense_length + init_context_size],
                eos_token_id=eos_token_id,
                num_samples=args.num_samples,
                length=generation_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                get_scores=True,
                interpolation=interpolation
            )
        return out, dense_length, scores


class RobertaExtractor(nn.Module):
    def __init__(self, args, roberta, evaluation):
        super(RobertaExtractor, self).__init__()
        self.args = args
        self.roberta = roberta
        # Cache variables to speed up some computations
        self.agg_indices = None
        self.extra_seq_vectors = {}

        self.extra_style = {}
        self.extra_content = {}

        self.roberta_training = False
        self.evaluation = evaluation
        self.func_mapping = {
            "zero": torch.zeros_like,
            "random": torch.rand_like,
            "first": lambda x: x[0:1].expand(x.shape)
        }
        self.default_style_from_style = {
            "loss": torch.tensor(0.0),
            "correct": 0
        }
        self.default_style_from_content = {
            "generator_loss": torch.tensor(0.0),
            "discriminator_loss": torch.tensor(0.0),
            "discriminator_correct": 0,
            "discriminator_total": 1
        }

    def setup_roberta_training(self):
        args = self.args

        if not self.evaluation:
            self.roberta.train()
            num_classes = int(args.data_dir.split("_")[args.data_dir.split("_").index("classes") - 1])
            self.roberta.model.register_classification_head(
                "style_from_style_classification_head",
                num_classes=num_classes
            )
            self.roberta.model.register_token_classification_head(
                "style_from_content_token_classification_head",
                num_classes=num_classes
            )
            self.roberta.cuda()

        self.roberta_training = True

    def aggregate(self, content_vectors):
        content_aggregation = self.args.content_aggregation
        content_aggregation_type = self.args.content_aggregation_type

        if content_aggregation_type == "single":
            # Pick out evenly spaced vectors as an estimate of content
            if self.agg_indices is None:
                self.agg_indices = [
                    i for i in range(content_vectors.shape[1]) if i % content_aggregation == 0
                ]
            content_agg_vectors = content_vectors[:, self.agg_indices, :]
        elif content_aggregation_type == "bow":
            # Use bag of words as an estimate of content
            batch_size, seq_length, hidden_size = content_vectors.shape

            if seq_length % content_aggregation != 0:
                desired_shape = (batch_size, content_aggregation - seq_length % content_aggregation, hidden_size)

                if desired_shape not in self.extra_seq_vectors:
                    # pad content vectors with extra zeros
                    self.extra_seq_vectors[desired_shape] = torch.zeros(
                        size=desired_shape,
                        dtype=content_vectors.dtype,
                        device=content_vectors.device
                    )
                content_vectors = torch.cat([content_vectors, self.extra_seq_vectors[desired_shape]], dim=1)

            # reshape the content_vectors tensor and use BoW style summation
            reshaped_content_vectors = content_vectors.reshape(batch_size, -1, content_aggregation, hidden_size)
            content_agg_vectors = reshaped_content_vectors.sum(dim=2)

        return content_agg_vectors

    def choose_context(self, style_vector, content_agg_vectors):
        context_type = self.args.context_type
        if context_type == "style":
            # style-only context
            style_content_vectors = style_vector.expand(content_agg_vectors.shape)
        elif context_type == "content":
            # content-only context
            style_content_vectors = content_agg_vectors
        elif context_type in ["first_content", "zero_content", "random_content"]:
            # content-only context
            if content_agg_vectors.shape not in self.extra_content:
                content_type = context_type.split("_")[0]
                self.extra_content[content_agg_vectors.shape] = self.func_mapping[content_type](content_agg_vectors)

            style_content_vectors = self.extra_content[content_agg_vectors.shape]
        elif context_type.endswith("_style_real_content"):
            # real content vectors but modified and fixed style vectors
            # useful for ablation experiments checking if style vectors are being utilized
            expanded_style_vector = style_vector.expand(content_agg_vectors.shape)
            if expanded_style_vector.shape not in self.extra_style:
                # check if the fixed style vector has already been cached
                style_type = context_type.split("_")[0]
                assert style_type in ["zero", "random", "first"]
                self.extra_style[expanded_style_vector.shape] = self.func_mapping[style_type](expanded_style_vector)
            # concatenate fake style with real content
            style_content_vectors = torch.cat([
                self.extra_style[expanded_style_vector.shape],
                content_agg_vectors
            ], dim=2)
        elif context_type.startswith("real_style_"):
            # real style vectors but modified and fixed content vectors
            # useful for ablation experiments checking how the style vectors are being utilized
            if content_agg_vectors.shape not in self.extra_content:
                # check if the fixed content vector has already been cached
                content_type = context_type.split("_")[2]
                assert content_type in ["zero", "random", "first"]
                self.extra_content[content_agg_vectors.shape] = self.func_mapping[content_type](content_agg_vectors)
            # concatenate real style with fake content
            style_content_vectors = torch.cat([
                style_vector.expand(content_agg_vectors.shape),
                self.extra_content[content_agg_vectors.shape]
            ], dim=2)
        elif context_type == "style_content":
            style_content_vectors = torch.cat([
                style_vector.expand(content_agg_vectors.shape),
                content_agg_vectors
            ], dim=2)
        else:
            raise Exception("Invalid context type")
        return style_content_vectors

    def get_style_content(self, roberta_sentences, global_dense_vectors=None,
                          author_targets=None, update_type="generator"):
        args = self.args
        default_style_from_style = self.default_style_from_style
        default_style_from_content = self.default_style_from_content

        if args.content_aggregation >= roberta_sentences.shape[1]:
            if args.global_dense_feature_list == "none":
                return None, default_style_from_style, default_style_from_content
            else:
                return global_dense_vectors, default_style_from_style, default_style_from_content

        roberta_layer = args.roberta_layer

        with torch.set_grad_enabled(self.roberta_training and not self.evaluation and update_type == "generator"):
            features, extra = self.roberta.model(
                roberta_sentences.long(),
                features_only=True,
                return_all_hiddens=roberta_layer != -1
            )

        if roberta_layer != -1:
            features = extra["inner_states"][roberta_layer].transpose(0, 1)

        style_vector = features[:, 0:1, :]
        content_vectors = features[:, 1:, :]

        content_agg_vectors = self.aggregate(content_vectors)
        style_content_vectors = self.choose_context(style_vector, content_agg_vectors)

        if (self.roberta_training or self.evaluation) and (author_targets is not None):
            with torch.set_grad_enabled(update_type == "generator"):
                style_from_style_info = self.get_style_style_info(features, author_targets)
        else:
            style_from_style_info = default_style_from_style

        if self.roberta_training and (author_targets is not None):
            style_from_content_info = self.get_style_content_info(content_agg_vectors, author_targets)
        else:
            style_from_content_info = default_style_from_content

        if args.global_dense_feature_list == "none":
            return style_content_vectors, style_from_style_info, style_from_content_info
        else:
            all_dense_vectors = torch.cat([global_dense_vectors, style_content_vectors], dim=1)
            return all_dense_vectors, style_from_style_info, style_from_content_info

    def get_style_style_info(self, features, author_targets):
        model = self.roberta.model

        style_from_style_logits = model.classification_heads["style_from_style_classification_head"](features)
        style_from_style_loss = F.nll_loss(
            F.log_softmax(style_from_style_logits, dim=-1, dtype=torch.float32),
            author_targets,
            reduction='mean',
        )

        accuracy = torch.sum(torch.argmax(style_from_style_logits, dim=1) == author_targets).item()

        style_from_style_info = {
            "loss": style_from_style_loss,
            "correct": accuracy
        }

        return style_from_style_info

    def get_style_content_info(self, content_agg_vectors, author_targets):
        model = self.roberta.model

        style_from_content_logits = model.classification_heads["style_from_content_token_classification_head"](content_agg_vectors)

        style_from_content_logprobs = F.log_softmax(
            style_from_content_logits.view(-1, style_from_content_logits.size(-1)),
            dim=-1,
            dtype=torch.float32
        )
        author_targets_expanded = author_targets.unsqueeze(-1).expand(style_from_content_logits.shape[:2]).reshape(-1)

        style_from_content_info = {}

        style_from_content_info["discriminator_loss"] = F.nll_loss(
            style_from_content_logprobs,
            author_targets_expanded,
            reduction='mean',
        )

        style_from_content_info["discriminator_correct"] = \
            torch.sum(torch.argmax(style_from_content_logprobs, dim=1) == author_targets_expanded).item()
        style_from_content_info["discriminator_total"] = author_targets_expanded.shape[0]

        if self.args.adversarial_loss_type == "negative_cross_entropy":
            style_from_content_info["generator_loss"] = -1 * style_from_content_info["discriminator_loss"]
        else:
            # maximize all log probabilities equally
            style_from_content_info["generator_loss"] = -1 * style_from_content_logprobs.mean()

        return style_from_content_info


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    elif top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    return logits


def get_logits(model, iteration, generated, segments, style_content_vectors, past):
    if iteration == 0:
        logits, past = model(
            input_ids=generated,
            token_type_ids=segments,
            prefix_input_vectors=style_content_vectors
        )
    else:
        # used the cached representations to speed up decoding
        logits, past = model(
            input_ids=generated[:, -1:],
            token_type_ids=segments[:, -1:],
            past=past
        )
    return logits, past

def sample_sequence(model, length, context, style_content_vectors, segments, eos_token_id,
                    num_samples=1, temperature=1, top_k=0, top_p=0.0, get_scores=False,
                    interpolation=None):

    if length is None and style_content_vectors is not None:
        new_length = 1024 - style_content_vectors.shape[1] - context.shape[1]
    elif length is None and style_content_vectors is None:
        new_length = 1024 - context.shape[1]
    else:
        new_length = length

    batch_size = context.shape[0]

    eos_emitted = [False for _ in range(batch_size)]

    generated = context
    scores = [{"score": 0, "sequence": []} for _ in range(batch_size)]

    with torch.no_grad():
        past = None
        past2 = None
        for i in range(new_length):
            logits, past = get_logits(
                model, i, generated, segments, style_content_vectors, past
            )
            if interpolation:
                logits2, past2 = get_logits(
                    model=interpolation["model"].roberta_gpt2.gpt2,
                    iteration=i,
                    generated=generated,
                    segments=segments,
                    style_content_vectors=style_content_vectors,
                    past=past2
                )
                probs = F.softmax(logits[:, -1, :], dim=-1)
                probs2 = F.softmax(logits2[:, -1, :], dim=-1)
                final_probs = interpolation["weight"] * probs2 + (1 - interpolation["weight"]) * probs
                next_token_logits = torch.log(final_probs) / (temperature if temperature > 0 else 1.)
                original_scores = next_token_logits.clone()
            else:
                next_token_logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.)
                original_scores = F.log_softmax(next_token_logits, dim=-1)

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0 and top_k in [0, 1] and top_p == 0.0:
                # greedy sampling
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            if get_scores:
                for batch_elem in range(batch_size):
                    if eos_emitted[batch_elem]:
                        continue
                    scores[batch_elem]["score"] += original_scores[batch_elem, next_token[batch_elem].item()].item()
                    scores[batch_elem]["sequence"].append("token")

            generated = torch.cat((generated, next_token), dim=1)
            segments = torch.cat((segments, segments[:, -1:]), dim=1)

            for batch_elem in range(batch_size):
                if next_token[batch_elem].item() == eos_token_id:
                    eos_emitted[batch_elem] = True

            if length is None and all(eos_emitted):
                break

    if get_scores:
        scores = [score_fn(x, True) for x in scores]

    return generated, scores


def score_fn(x, length_normalize):
    if length_normalize:
        return x["score"] / len(x["sequence"])
    else:
        return x["score"]


def beam_search(model, length, context, style_content_vectors, segments, eos_token_id,
                beam_size=1, beam_search_scoring="normalize"):

    def merge_pasts(all_beams, prev_past):
        past_indices = [beam["past"] for element in all_beams for beam in element]
        return [pp[:, past_indices, :, :, :] for pp in prev_past]

    def merge_input_ids(all_beams):
        input_ids = [beam["sequence"][-1] for element in all_beams for beam in element]
        return torch.cat(input_ids, dim=0)

    if beam_search_scoring == "normalize":
        _score_fn = partial(score_fn, length_normalize=True)
    else:
        _score_fn = partial(score_fn, length_normalize=False)

    if length is None and style_content_vectors is not None:
        new_length = 1024 - style_content_vectors.shape[1] - context.shape[1]
    elif length is None and style_content_vectors is None:
        new_length = 1024 - context.shape[1]
    else:
        new_length = length

    with torch.no_grad():
        logits, past = model(
            input_ids=context,
            token_type_ids=segments,
            prefix_input_vectors=style_content_vectors
        )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
        top_scores, top_indices = torch.topk(input=log_probs, k=beam_size, dim=-1)

        all_beams = []
        for elem_num, (ts, ti) in enumerate(zip(top_scores, top_indices)):
            curr_element = []
            for bs in range(beam_size):
                curr_element.append({
                    "score": ts[bs],
                    "past": elem_num,
                    "sequence": [x.unsqueeze(0).unsqueeze(0) for x in context[elem_num]] + [ti[bs].unsqueeze(0).unsqueeze(0)],
                    "eos_emitted": False
                })
            all_beams.append(curr_element)

        # one time step here since segment IDs remain constant during generation
        tiled_segments = torch.cat([segments[:, -1:] for _ in range(beam_size)], dim=-1)

        for i in range(1, new_length):
            # check if all beams have emitted an EOS token
            all_eos = all([beam["eos_emitted"] for element in all_beams for beam in element])
            if all_eos:
                break

            latest_input_ids = merge_input_ids(all_beams)
            past = merge_pasts(all_beams, past)

            logits, past = model(
                input_ids=latest_input_ids,  # input_ids[:, -1:],
                token_type_ids=tiled_segments,
                past=past
            )
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            top_scores, top_indices = torch.topk(input=log_probs, k=beam_size, dim=-1)

            new_beams = []
            curr_element = []
            for mb_num, (ts, ti) in enumerate(zip(top_scores, top_indices)):
                current_elem_num = mb_num // beam_size
                current_elem_beam_num = mb_num % beam_size
                old_beam = all_beams[current_elem_num][current_elem_beam_num]

                if old_beam["eos_emitted"]:
                    curr_element.append(old_beam)
                else:
                    for bs in range(beam_size):
                        token = ti[bs].unsqueeze(0).unsqueeze(0)
                        curr_element.append({
                            "score": old_beam["score"] + ts[bs],
                            "past": mb_num,
                            "sequence": old_beam["sequence"] + [token],
                            "eos_emitted": token.item() == eos_token_id
                        })
                if current_elem_beam_num == beam_size - 1:
                    new_beams.append(curr_element)
                    curr_element = []

            # Sort the beams by score and keep only top scoring elements
            all_beams = []
            for elem in new_beams:
                elem.sort(key=lambda x: _score_fn(x), reverse=True)
                all_beams.append(elem[:beam_size])

        final_beams = []
        for elem in all_beams:
            elem.sort(key=lambda x: _score_fn(x), reverse=True)
            # just return the highest scoring prediction
            final_beams.append(elem[:1])

        final_input_ids = [
            torch.cat(elem[0]["sequence"], dim=1).squeeze(0) for elem in final_beams
        ]

        return final_input_ids, [_score_fn(fb[0]) for fb in final_beams]


# https://github.com/ngram-lab/storyteller/blob/master/data/parallel.py#L103
class StaticDataParallel(nn.DataParallel):
    """
    A modified version of torch.nn.DataParallel which replicates to the devices
    only once at the beginning, reducing the overhead of each batch. Named
    StaticDataParallel since the model is static.
    This optimization is only useful during inference. If you need similar
    functionality during training, then torch.nn.DataParallel should be used
    instead.
    """

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(StaticDataParallel, self).__init__(
            module, device_ids=device_ids, output_device=output_device, dim=dim
        )

        self.replicas = []

    def replicate(self, module, device_ids):
        if not self.replicas:
            # Create the replicas once
            self.replicas = super().replicate(self.module, self.device_ids)  # type: ignore

        return self.replicas[: len(device_ids)]
