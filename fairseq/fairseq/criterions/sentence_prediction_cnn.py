# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('sentence_prediction_cnn')
class SentencePredictionCNNCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
            'sequence_classification_head_cnn_probe' in model.classification_heads, \
            "model must provide sequence_classification_head_cnn_probe head for --criterion=sentence_prediction_cnn"

        model.eval()
        features, _ = model(
            **sample['net_input'],
            features_only=True
        )
        model.train()

        if "style" not in self.args.include_vectors:
            # ignore the first vector for convolution, since that contains stylistic information
            features = features[:, 1:, :]

        # apply a sequence classification removing the <s> style token
        logits = model.classification_heads["sequence_classification_head_cnn_probe"](features)

        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        loss = F.nll_loss(
            F.log_softmax(logits, dim=-1, dtype=torch.float32),
            targets,
            reduction='sum',
        )

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
        }

        preds = logits.max(dim=1)[1]
        logging_output.update(
            ncorrect=(preds == targets).sum().item()
        )
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
