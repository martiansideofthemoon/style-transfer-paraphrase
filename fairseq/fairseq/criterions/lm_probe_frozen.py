# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('lm_probe_frozen')
class LMProbeFrozenCriterion(FairseqCriterion):

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
            'lm_head_probe' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        model.eval()
        features, _ = model(
            **sample['net_input'],
            features_only=True
        )
        model.train()

        style_vector = features[:, 0:1, :]
        content_vectors = features[:, 1:, :]

        style_vector_expanded = style_vector.expand(content_vectors.shape)
        # POS embeddings are used in the LM classifiers
        pos_embeddings = model.decoder.sentence_encoder.embed_positions(
            sample['net_input']['src_tokens'][:, 1:]
        )

        if self.args.probe_features == "style,content" or self.args.probe_features == "content,style":
            concat_feats = torch.cat([style_vector_expanded, content_vectors, pos_embeddings], dim=2)
        elif self.args.probe_features == "style":
            concat_feats = torch.cat([style_vector_expanded, pos_embeddings], dim=2)
        elif self.args.probe_features == "content":
            concat_feats = torch.cat([content_vectors, pos_embeddings], dim=2)
        else:
            concat_feats = None

        lm_logits = model.classification_heads["lm_head_probe"](concat_feats)
        lm_targets = sample['net_input']['src_tokens'][:, 1:].reshape(-1)

        num_target_tokens = torch.sum(lm_targets != self.padding_idx)

        loss = F.nll_loss(
            F.log_softmax(
                lm_logits.view(-1, lm_logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            lm_targets,
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        loss = loss * features.shape[0] / num_target_tokens

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': features.shape[0],
            'sample_size': features.shape[0],
        }

        preds = lm_logits.max(dim=2)[1].view(-1)
        logging_output.update(
            ncorrect=(preds == lm_targets).sum().item()
        )
        logging_output.update(
            total=preds.shape[0]
        )
        return loss, features.shape[0], logging_output

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
            ntotal = sum(log.get('total', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect / ntotal)

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
