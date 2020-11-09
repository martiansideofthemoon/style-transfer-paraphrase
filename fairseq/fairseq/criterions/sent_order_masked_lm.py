# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


# create matrix mask from length vector
def compute_mask(lengths, max_len):
    range_row = torch.arange(0, max_len).long().cuda()[None, :].expand(lengths.size()[0], max_len)
    mask = lengths[:, None].expand_as(range_row)
    mask = range_row < mask
    return mask.float().cuda()


@register_criterion('sent_order_masked_lm')
class SentenceOrderMaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in sentence order + masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss + sentence order ranking loss
        masked_tokens = sample['target']['mlm_target'].ne(self.padding_idx)

        sent_vector_mask = sample['target']['order_target'].ne(self.padding_idx)
        num_sentences = sent_vector_mask.sum(axis=1)

        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits = model(**sample['net_input'],
                       masked_tokens=masked_tokens)[0]

        order_scores = model(**sample['net_input'],
                             classification_head_name='sentence_order_token_head')[0]

        # store the relevant sentence ordering scores
        # hard coding 8 for now since the maximum number of sentences is 6, and 2 more for padding
        relevant_scores = torch.zeros(sample['target']['order_target'].shape[0], 8).cuda()
        # scatter the scores depending on the ground truth ordering, so that relevant scores have the correct order
        relevant_scores.scatter_(dim=1, index=sample['target']['order_target'], src=order_scores.squeeze(dim=2).float())
        # remove the padding indices, 0 and 1
        # in the ground truth, the zeroth index is stored as 2
        relevant_scores = relevant_scores[:, 2:]

        # We want a loss function with a_1 - a0, a2 - a1 and so on
        # Shift the relevant scores to the left to create this effect
        shifted_relevant_scores = torch.zeros_like(relevant_scores)
        shifted_relevant_scores[:, :-1] = relevant_scores[:, 1:]

        # compute a mask to only keep relevant terms around
        num_terms_mask = compute_mask(num_sentences - 1, relevant_scores.shape[1])
        # compute a loss function based on the formulation in Bayesian Personalized Ranking
        diff_losses = -1 * num_terms_mask * torch.nn.functional.logsigmoid(shifted_relevant_scores - relevant_scores)

        # Renormalize the loss function to keep it compatible with rest of fairseq
        order_sample_size = torch.sum(num_sentences - 1)
        total_order_loss = diff_losses.sum()
        normalized_order_loss = total_order_loss * sample_size / order_sample_size

        targets = model.get_targets(sample, [logits])['mlm_target']

        if sample_size != 0:
            targets = targets[masked_tokens]

        mlm_loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        total_loss = self.args.interpolation * mlm_loss + (1 - self.args.interpolation) * normalized_order_loss

        logging_output = {
            'mlm_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'nll_loss': utils.item(mlm_loss.data) if reduce else mlm_loss.data,
            'order_loss': utils.item(normalized_order_loss.data) if reduce else normalized_order_loss.data,
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'order_sample_size': order_sample_size.item()
        }
        return total_loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        order_sample_size = sum(log.get('order_sample_size', 0) for log in logging_outputs)
        mlm_loss = sum(log.get('mlm_loss', 0) for log in logging_outputs)
        order_loss = sum(log.get('order_loss', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'mlm_loss': mlm_loss / sample_size,
            'order_loss': order_loss / sample_size,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'order_sample_size': order_sample_size
        }
        return agg_output
