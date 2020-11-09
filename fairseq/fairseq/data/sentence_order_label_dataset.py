# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import data_utils

from . import FairseqDataset


class SentenceOrderLabelDataset(FairseqDataset):

    def __init__(self, order_labels):
        super().__init__()
        self.order_labels = [[int(y) for y in x.strip().split(",")] for x in order_labels]

    def __getitem__(self, index):
        return self.order_labels[index]

    def __len__(self):
        return len(self.order_labels)
