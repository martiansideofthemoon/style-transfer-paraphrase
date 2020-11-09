# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)

from . import FairseqTask, register_task


@register_task('style_separation')
class StyleSeparationTask(FairseqTask):
    """
    Separate the style and content representations of an input.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')

        parser.add_argument('--style-from-style-weight', type=float, default=0.33,
                            help='interpolation constant for classifier from <s> token')
        parser.add_argument('--lm-prediction-weight', type=float, default=0.33,
                            help='interpolation constant for LM prediction')
        parser.add_argument('--consecutive-updates', type=int, default=3,
                            help='number of consecutive adversarial or generative updates')
        parser.add_argument('--content-aggregation', type=int, default=20,
                            help='number of subwords to collect information from about content')
        parser.add_argument('--adversarial-loss-type', type=str, default='negative_cross_entrop11y',
                            help='type of loss for the adversarial training of generator updates')
        parser.add_argument('--lr-list', type=str, default=None,
                            help='Comma separated list of learning rates for different parameter groups')

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self.label_dictionary = label_dictionary
        self.num_updates = 0

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        args.tokens_per_sample = args.max_positions

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'input0', 'dict.txt'),
            source=True,
        )
        print('| [input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False,
            )
            print('| [label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return StyleSeparationTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset('input0', self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path(type, split))
        input1 = make_dataset('input1', self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if not self.args.regression_target:
            label_dataset = make_dataset('label', self.target_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.target_dictionary.eos(),
                        ),
                        offset=-self.target_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path('label', split))
            if os.path.exists(label_path):
                dataset.update(
                    target=RawLabelDataset([
                        float(x.strip()) for x in open(label_path).readlines()
                    ])
                )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        print("| Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            'style_from_style_classification_head',
            num_classes=self.args.num_classes,
        )

        model.register_lm_head(
            'style_content_lm_head',
            num_embeddings=3
        )

        model.register_token_classification_head(
            'style_from_content_token_classification_head',
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.label_dictionary

    def update_step(self, num_updates):
        """Task level update when number of update increases. This is called after optimization step and
           learning rate update of each step"""
        self.num_updates = num_updates

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        loss, sample_size, logging_output = criterion(model, sample, num_updates=self.num_updates)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def get_task_params(self, model, criterion):
        """Segregate the task parameters if needed. Return a list of parameter groups."""
        params1 = [x for x in model.decoder.parameters()] + \
            [x for x in model.classification_heads["style_from_style_classification_head"].parameters()] + \
            [x for x in model.classification_heads["style_content_lm_head"].parameters()]

        params2 = [x for x in model.classification_heads["style_from_content_token_classification_head"].parameters()]

        return [params1, params2]

    def choose_current_optimizer(self, optimizers):
        if (self.num_updates // self.args.consecutive_updates) % 2 == 1:
            return optimizers[0]
        elif (self.num_updates // self.args.consecutive_updates) % 2 == 0:
            return optimizers[1]
