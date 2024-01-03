"""
A much simplified and possibly more reasonable distributed sampler
for balancing dataset with fewer positive examples.
"""
import math
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler


class StratifiedSampler(DistributedSampler):
    """
    Equalize positive and negative examples in each load
    NOTE: This sampler assume the dataset contains no less negative
          examples than positive examples.
    """
    def __init__(self,
                 dataset,
                 strata,
                 num_replicas):

        super().__init__(dataset)

        self.pos_stratum = np.where(strata == 1)[0]
        self.neg_stratum = np.where(strata == 0)[0]

        self.num_pos = int(len(self.pos_stratum))
        self.num_neg = int(len(self.neg_stratum))
        assert self.num_neg >= self.num_pos, \
            "negative examples be no less than possitive examples"

        self.num_per_replica = int(math.ceil(self.num_pos / num_replicas))
        self.total = self.num_per_replica * num_replicas

    def __iter__(self):

        rng = torch.Generator()
        # stablize random number generator by epoch for all replicas
        rng.manual_seed(self.epoch)

        pos_indices = torch.randperm(self.num_pos, generator = rng).tolist()
        neg_indices = torch.randperm(self.num_neg, generator = rng).tolist()

        if len(pos_indices) < self.total:
            pos_indices += pos_indices[ : self.total - len(pos_indices)]

        if len(neg_indices) < self.total:
            neg_indices += neg_indices[ : self.total - len(neg_indices)]
        else:
            neg_indices = neg_indices[ : self.total]

        # Subsample data for a rank
        rank_pos_indices = pos_indices[self.rank : self.total : self.num_replicas]
        rank_neg_indices = neg_indices[self.rank : self.total : self.num_replicas]

        # interleave
        indices = []
        for pos_idx, neg_idx in zip(self.pos_stratum[rank_pos_indices],
                                    self.pos_stratum[rank_neg_indices]):
            indices += [pos_idx, neg_idx]

        return iter(indices)

    def __len__(self):
        return self.num_per_replica
