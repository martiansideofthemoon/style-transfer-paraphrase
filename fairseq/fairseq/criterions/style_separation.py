# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


def set_requires_grad(module, val):
    for p in module.parameters():
        p.requires_grad = val


def aggregate_content_information(content_vectors, content_aggregation):
    # select vectors at every content_aggregation interval
    content_agg_indices = [i for i in range(content_vectors.shape[1]) if i % content_aggregation == 0]
    content_agg_vectors = content_vectors[:, content_agg_indices, :]
    # copy the selected vectors across content_aggregation positions
    cav_expanded = content_agg_vectors.unsqueeze(2)
    cav_expanded = cav_expanded.expand(
        cav_expanded.shape[0], cav_expanded.shape[1], content_aggregation, cav_expanded.shape[3]
    )
    cav_expanded = cav_expanded.reshape(
        cav_expanded.shape[0], cav_expanded.shape[1] * cav_expanded.shape[2], cav_expanded.shape[3]
    )
    # remove the extra vectors from the last chunk
    cav_expanded = cav_expanded[:, :content_vectors.shape[1], :]
    return content_agg_vectors, cav_expanded


@register_criterion('style_separation')
class StyleSeparationCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True, num_updates=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # detach the LM head parameters temporarily since we want to freeze them
        # https://github.com/pytorch/pytorch/issues/2655#issuecomment-333501083
        if num_updates is None:
            # probably a validation part of the code, set all requires grad false
            # don't do anything since it's in a no_grad() block
            pass
        elif (num_updates // self.args.consecutive_updates) % 2 == 1:
            # if num_updates is an even number, it will be an update of the generators
            set_requires_grad(model.decoder, True)
            set_requires_grad(model.classification_heads["style_from_style_classification_head"], True)
            set_requires_grad(model.classification_heads["style_content_lm_head"], True)
            set_requires_grad(model.classification_heads["style_from_content_token_classification_head"], False)
        elif (num_updates // self.args.consecutive_updates) % 2 == 0:
            # if num_updates is an odd number, it will be an update of the adversarial classifiers
            set_requires_grad(model.decoder, False)
            set_requires_grad(model.classification_heads["style_from_style_classification_head"], False)
            set_requires_grad(model.classification_heads["style_content_lm_head"], False)
            set_requires_grad(model.classification_heads["style_from_content_token_classification_head"], True)
            pass

        features, _ = model(
            **sample['net_input'],
            features_only=True
        )

        # prepare the content-only and style-only embeddings for the subsequent losses
        style_vector = features[:, 0:1, :]
        content_vectors = features[:, 1:, :]

        # sample content vectors based on the provided aggregation amount
        content_agg_vectors, content_agg_vectors_expanded = aggregate_content_information(content_vectors, self.args.content_aggregation)

        style_vector_expanded = style_vector.expand(content_vectors.shape)

        # POS embeddings are used in the LM classifiers
        pos_embeddings = model.decoder.sentence_encoder.embed_positions(
            sample['net_input']['src_tokens'][:, 1:]
        )

        # L1 = author classification loss function
        style_from_style_logits = model.classification_heads["style_from_style_classification_head"](features)

        author_targets = model.get_targets(sample, [style_from_style_logits]).view(-1)
        sample_size = author_targets.numel()

        style_from_style_loss = F.nll_loss(
            F.log_softmax(style_from_style_logits, dim=-1, dtype=torch.float32),
            author_targets,
            reduction='sum',
        )

        # L2 = LM classification of the concatenation of <s>, v_i, p_i
        concat_feats = torch.cat([style_vector_expanded, content_agg_vectors_expanded, pos_embeddings], dim=2)
        lm_feats = model.classification_heads["style_content_lm_head"](concat_feats)
        lm_targets = sample['net_input']['src_tokens'][:, 1:].reshape(-1)

        num_tokens = torch.sum(lm_targets != self.padding_idx)

        lm_loss = F.nll_loss(
            F.log_softmax(
                lm_feats.view(-1, lm_feats.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            lm_targets,
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        lm_loss_normalized = lm_loss * sample_size / num_tokens

        # L3 - adversarial classifier for author prediction from content vectors
        style_from_content_logits = model.classification_heads["style_from_content_token_classification_head"](content_agg_vectors)

        style_from_content_logprobs = F.log_softmax(
            style_from_content_logits.view(-1, style_from_content_logits.size(-1)),
            dim=-1,
            dtype=torch.float32
        )
        author_targets_expanded = author_targets.unsqueeze(-1).expand(style_from_content_logits.shape[:2]).reshape(-1)

        style_from_content_loss = F.nll_loss(
            style_from_content_logprobs,
            author_targets_expanded,
            reduction='sum',
        )

        style_from_content_loss = style_from_content_loss * sample_size / author_targets_expanded.shape[0]

        if self.args.adversarial_loss_type == "negative_cross_entropy":
            adversarial_style_from_content_loss = -1 * style_from_content_loss
        else:
            # maximize all log probabilities equally
            adversarial_style_from_content_loss = -1 * style_from_content_logprobs.mean() * sample_size

        if num_updates is None or (num_updates // self.args.consecutive_updates) % 2 == 1:
            loss = \
                self.args.style_from_style_weight * style_from_style_loss + \
                self.args.lm_prediction_weight * lm_loss_normalized + \
                (1.0 - self.args.style_from_style_weight - self.args.lm_prediction_weight) * adversarial_style_from_content_loss

        elif (num_updates // self.args.consecutive_updates) % 2 == 0:
            loss = style_from_content_loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'style_from_style_loss': utils.item(style_from_style_loss.data) if reduce else style_from_style_loss.data,
            'lm_loss': utils.item(lm_loss_normalized.data) if reduce else lm_loss_normalized.data,
            'style_from_content_loss': utils.item(style_from_content_loss.data) if reduce else style_from_content_loss.data,
            'adversarial_style_from_content_loss': utils.item(adversarial_style_from_content_loss.data) if reduce else adversarial_style_from_content_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = style_from_style_logits.max(dim=1)[1]
        logging_output.update(
            ncorrect=(preds == author_targets).sum().item()
        )

        if num_updates is not None:
            # Set all requires grad true for correct gradient calculations
            set_requires_grad(model.decoder, True)
            set_requires_grad(model.classification_heads["style_from_style_classification_head"], True)
            set_requires_grad(model.classification_heads["style_content_lm_head"], True)
            set_requires_grad(model.classification_heads["style_from_content_token_classification_head"], True)

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)

        style_from_style_loss_sum = sum(log.get('style_from_style_loss', 0) for log in logging_outputs)
        lm_loss_sum = sum(log.get('lm_loss', 0) for log in logging_outputs)
        style_from_content_loss_sum = sum(log.get('style_from_content_loss', 0) for log in logging_outputs)
        adversarial_style_from_content_loss_sum = sum(log.get('adversarial_style_from_content_loss', 0) for log in logging_outputs)

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'style_from_style_loss': style_from_style_loss_sum / sample_size,
            'lm_loss': lm_loss_sum / sample_size,
            'style_from_content_loss': style_from_content_loss_sum / sample_size,
            'adversarial_style_from_content_loss': adversarial_style_from_content_loss_sum / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
